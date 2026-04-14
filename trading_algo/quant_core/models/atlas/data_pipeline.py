from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.features import ATLASFeatureComputer, RollingNormalizer


def download_universe(
    symbols: list[str], period: str = "10y", cache_dir: str = "data/atlas_cache"
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all symbols, cache as parquet."""
    import yfinance as yf

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        parquet_file = cache_path / f"{sym}.parquet"
        if parquet_file.exists():
            result[sym] = pd.read_parquet(parquet_file)
            continue

        df = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            continue
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        df.to_parquet(parquet_file)
        result[sym] = df

    return result


def _third_friday(year: int, month: int) -> int:
    """Return the day-of-month for the 3rd Friday of the given month."""
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    fridays = [
        d for d in c.itermonthdays2(year, month) if d[0] != 0 and d[1] == 4  # Friday=4
    ]
    return fridays[2][0]


def _is_opex_week(dt: datetime) -> bool:
    """True if dt falls in the week (Mon-Fri) containing the 3rd Friday of its month."""
    third_fri_day = _third_friday(dt.year, dt.month)
    third_fri = datetime(dt.year, dt.month, third_fri_day)
    # Monday of that week
    monday = third_fri_day - third_fri.weekday()
    friday = monday + 4
    return monday <= dt.day <= friday


def _is_quarter_end(dt: datetime, trading_dates: np.ndarray, idx: int) -> bool:
    """True if within 5 trading days of a quarter end (Mar 31, Jun 30, Sep 30, Dec 31)."""
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    for qm, qd in quarter_ends:
        try:
            qe_date = datetime(dt.year, qm, qd)
        except ValueError:
            qe_date = datetime(dt.year, qm, qd - 1)
        qe_ts = qe_date.timestamp()
        # Check if any of the 5 trading days before/after include the quarter end
        window_start = max(0, idx - 5)
        window_end = min(len(trading_dates), idx + 6)
        window_dates = trading_dates[window_start:window_end]
        for wd in window_dates:
            wd_dt = datetime.fromtimestamp(wd)
            if wd_dt.year == qe_date.year and wd_dt.month == qe_date.month and wd_dt.day == qe_date.day:
                return True
        # Also check by proximity: if the quarter end falls on a weekend, the nearest
        # trading day should still count
        dt_ts = dt.timestamp()
        if abs(dt_ts - qe_ts) <= 7 * 86400:
            # Within 7 calendar days — check trading day distance
            dist = 0
            if idx < len(trading_dates):
                for d in range(window_start, window_end):
                    wd_dt = datetime.fromtimestamp(trading_dates[d])
                    if (wd_dt.month == qm and wd_dt.day >= qd - 2) or (
                        wd_dt.month == qm + 1 if qm < 12 else 1
                    ) and wd_dt.day <= 3:
                        return True
    return False


class ATLASDataset(Dataset):
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        config: ATLASConfig | None = None,
        split: str = "train",
    ) -> None:
        if config is None:
            config = ATLASConfig()
        self.config = config
        self.context_len = config.context_len  # 90

        # Extract arrays
        closes = ohlcv_df["Close"].values.astype(np.float64)
        highs = ohlcv_df["High"].values.astype(np.float64)
        lows = ohlcv_df["Low"].values.astype(np.float64)
        volumes = ohlcv_df["Volume"].values.astype(np.float64)
        dates = ohlcv_df.index

        # Timestamps
        if hasattr(dates, "to_pydatetime"):
            dt_list = dates.to_pydatetime()
        else:
            dt_list = [datetime.fromtimestamp(d) if isinstance(d, (int, float)) else d for d in dates]

        timestamps = np.array([d.timestamp() for d in dt_list], dtype=np.float64)

        # Compute 12 market features
        fc = ATLASFeatureComputer()
        raw_features = fc.compute_features(closes, highs, lows, volumes)

        # Pad with 4 zeros for position state features -> (T, 16)
        T = len(closes)
        full_features = np.zeros((T, 16), dtype=np.float64)
        full_features[:, :12] = raw_features

        # Normalize with rolling z-score
        normalizer = RollingNormalizer()
        normed, mu, sigma = normalizer.normalize(full_features, lookback=252)

        # Calendar features
        day_of_week = np.array([d.weekday() for d in dt_list], dtype=np.int64)
        month = np.array([d.month - 1 for d in dt_list], dtype=np.int64)  # 0-indexed
        is_opex = np.array([1.0 if _is_opex_week(d) else 0.0 for d in dt_list], dtype=np.float32)
        is_quarter_end_arr = np.array(
            [1.0 if _is_quarter_end(d, timestamps, i) else 0.0 for i, d in enumerate(dt_list)],
            dtype=np.float32,
        )

        # Split by date
        split_dates = {
            "train": (None, datetime(2022, 1, 1)),
            "val": (datetime(2022, 1, 1), datetime(2024, 1, 1)),
            "test": (datetime(2024, 1, 1), None),
        }
        start_dt, end_dt = split_dates[split]

        # Minimum index: need 252 days for normalization + 89 for context window
        min_idx = 252 + self.context_len - 1  # 252 + 89 = 341

        # Build valid window indices
        self.windows: list[tuple[int, int]] = []
        for t in range(min_idx, T):
            dt_val = dt_list[t]
            if start_dt is not None and dt_val < start_dt:
                continue
            if end_dt is not None and dt_val >= end_dt:
                continue
            # Window: [t - context_len + 1, t + 1) = 90-day window
            w_start = t - self.context_len + 1
            self.windows.append((w_start, t + 1))

        # Store arrays
        self.normed = normed
        self.mu = mu
        self.sigma = sigma
        self.timestamps = timestamps
        self.day_of_week = day_of_week
        self.month = month
        self.is_opex = is_opex
        self.is_quarter_end = is_quarter_end_arr

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w_start, w_end = self.windows[idx]
        ctx = self.context_len

        feat = self.normed[w_start:w_end].copy()
        # Replace remaining NaN with 0
        feat = np.nan_to_num(feat, nan=0.0)

        mu_slice = self.mu[w_start:w_end].copy()
        mu_slice = np.nan_to_num(mu_slice, nan=0.0)
        sigma_slice = self.sigma[w_start:w_end].copy()
        sigma_slice = np.nan_to_num(sigma_slice, nan=1.0)

        return {
            "features": torch.tensor(feat, dtype=torch.float32),
            "timestamps": torch.tensor(self.timestamps[w_start:w_end], dtype=torch.float32),
            "day_of_week": torch.tensor(self.day_of_week[w_start:w_end], dtype=torch.long),
            "month": torch.tensor(self.month[w_start:w_end], dtype=torch.long),
            "is_opex": torch.tensor(self.is_opex[w_start:w_end], dtype=torch.float32),
            "is_quarter_end": torch.tensor(self.is_quarter_end[w_start:w_end], dtype=torch.float32),
            "pre_norm_mu": torch.tensor(mu_slice, dtype=torch.float32),
            "pre_norm_sigma": torch.tensor(sigma_slice, dtype=torch.float32),
            "action_label": torch.zeros(5, dtype=torch.float32),
            "return_to_go": torch.zeros(1, dtype=torch.float32),
        }
