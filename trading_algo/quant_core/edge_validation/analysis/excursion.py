from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from trading_algo.quant_core.edge_validation.types import (
    ExcursionComparison,
    ExcursionResult,
    KSTestResult,
    PatternOccurrence,
)


def _eod_timestamp(day: pd.Timestamp, eod_hour: int, eod_minute: int) -> pd.Timestamp:
    return day.normalize() + pd.Timedelta(hours=eod_hour, minutes=eod_minute)


def compute_excursions(
    occurrences: list[PatternOccurrence],
    bars: pd.DataFrame,
    eod_hour: int = 16,
    eod_minute: int = 0,
) -> ExcursionResult:
    """For each occurrence, compute MFE/MAE/PnL from entry to EOD.

    MFE = max favorable price movement in the direction of the trade (in points)
    MAE = max adverse price movement against the trade (in points)
    Realized PnL = price at EOD - entry price (adjusted for direction)
    """
    if not occurrences:
        empty = np.array([], dtype=np.float64)
        return ExcursionResult(
            mfe_points=empty,
            mae_points=empty,
            realized_pnl=empty,
            holding_bars=np.array([], dtype=np.int64),
            n_occurrences=0,
        )

    bars = bars.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()

    tz = bars.index.tz
    bar_dates = bars.index.date
    unique_dates = np.unique(bar_dates)
    date_to_mask: dict = {}
    for d in unique_dates:
        eod_ts = pd.Timestamp(d, tz=tz) + pd.Timedelta(hours=eod_hour, minutes=eod_minute)
        mask = (bar_dates == d) & (bars.index <= eod_ts)
        date_to_mask[d] = mask

    mfe_list: list[float] = []
    mae_list: list[float] = []
    pnl_list: list[float] = []
    holding_list: list[int] = []

    for occ in occurrences:
        entry_ts = pd.Timestamp(occ.timestamp, unit="s", tz="UTC").tz_convert(tz) if isinstance(occ.timestamp, (int, float)) else pd.Timestamp(occ.timestamp, tz=tz)
        day = occ.day

        if day not in date_to_mask:
            continue

        day_mask = date_to_mask[day]
        day_bars = bars.loc[day_mask]
        after_entry = day_bars.loc[day_bars.index >= entry_ts]

        if after_entry.empty:
            continue

        highs = after_entry["high"].values if "high" in after_entry.columns else after_entry["High"].values
        lows = after_entry["low"].values if "low" in after_entry.columns else after_entry["Low"].values
        closes = after_entry["close"].values if "close" in after_entry.columns else after_entry["Close"].values

        entry = occ.entry_price
        last_close = closes[-1]

        if occ.direction == "long":
            mfe = float(np.max(highs) - entry)
            mae = float(entry - np.min(lows))
            pnl = float(last_close - entry)
        else:
            mfe = float(entry - np.min(lows))
            mae = float(np.max(highs) - entry)
            pnl = float(entry - last_close)

        mfe_list.append(mfe)
        mae_list.append(mae)
        pnl_list.append(pnl)
        holding_list.append(len(after_entry))

    return ExcursionResult(
        mfe_points=np.array(mfe_list, dtype=np.float64),
        mae_points=np.array(mae_list, dtype=np.float64),
        realized_pnl=np.array(pnl_list, dtype=np.float64),
        holding_bars=np.array(holding_list, dtype=np.int64),
        n_occurrences=len(mfe_list),
    )


def compare_to_random(
    occurrences: list[PatternOccurrence],
    bars: pd.DataFrame,
    n_random: int = 10_000,
    seed: int = 42,
) -> ExcursionComparison:
    """Compare pattern excursions vs random entries on same days.

    Random entries: pick random bars during RTH on trading days that had pattern occurrences.
    Assign same direction as the pattern (so we're comparing timing, not direction).
    Run KS tests on MFE, MAE, and PnL distributions.
    """
    pattern_result = compute_excursions(occurrences, bars)

    if not occurrences:
        empty = np.array([], dtype=np.float64)
        empty_exc = ExcursionResult(
            mfe_points=empty, mae_points=empty, realized_pnl=empty,
            holding_bars=np.array([], dtype=np.int64), n_occurrences=0,
        )
        ks_default = KSTestResult(statistic=0.0, p_value=1.0, significant=False)
        return ExcursionComparison(
            pattern=pattern_result, random=empty_exc,
            ks_mfe=ks_default, ks_mae=ks_default, ks_pnl=ks_default,
        )

    bars = bars.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()

    trading_days = list({occ.day for occ in occurrences})
    directions = [occ.direction for occ in occurrences]
    direction_counts = {}
    for d in directions:
        direction_counts[d] = direction_counts.get(d, 0) + 1
    dominant_direction = max(direction_counts, key=direction_counts.get)

    day_bars_map: dict = {}
    bar_dates = bars.index.date
    for d in trading_days:
        mask = bar_dates == d
        day_df = bars.loc[mask]
        if len(day_df) > 1:
            day_bars_map[d] = day_df

    if not day_bars_map:
        empty = np.array([], dtype=np.float64)
        empty_exc = ExcursionResult(
            mfe_points=empty, mae_points=empty, realized_pnl=empty,
            holding_bars=np.array([], dtype=np.int64), n_occurrences=0,
        )
        ks_default = KSTestResult(statistic=0.0, p_value=1.0, significant=False)
        return ExcursionComparison(
            pattern=pattern_result, random=empty_exc,
            ks_mfe=ks_default, ks_mae=ks_default, ks_pnl=ks_default,
        )

    rng = np.random.default_rng(seed)
    available_days = list(day_bars_map.keys())
    random_occs: list[PatternOccurrence] = []

    for _ in range(n_random):
        day = available_days[rng.integers(0, len(available_days))]
        day_df = day_bars_map[day]
        idx = rng.integers(0, len(day_df) - 1)
        bar = day_df.iloc[idx]
        open_col = "open" if "open" in day_df.columns else "Open"
        random_occs.append(PatternOccurrence(
            timestamp=day_df.index[idx].timestamp(),
            pattern="random",
            direction=dominant_direction,
            entry_price=float(bar[open_col]),
            day=day,
        ))

    random_result = compute_excursions(random_occs, bars)

    def _ks_test(a: NDArray[np.float64], b: NDArray[np.float64], alpha: float = 0.05) -> KSTestResult:
        if len(a) < 2 or len(b) < 2:
            return KSTestResult(statistic=0.0, p_value=1.0, significant=False)
        stat, pval = ks_2samp(a, b)
        return KSTestResult(statistic=float(stat), p_value=float(pval), significant=pval < alpha)

    return ExcursionComparison(
        pattern=pattern_result,
        random=random_result,
        ks_mfe=_ks_test(pattern_result.mfe_points, random_result.mfe_points),
        ks_mae=_ks_test(pattern_result.mae_points, random_result.mae_points),
        ks_pnl=_ks_test(pattern_result.realized_pnl, random_result.realized_pnl),
    )
