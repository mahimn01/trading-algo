from __future__ import annotations

import time
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

from trading_algo.broker.base import Bar
from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec
from trading_algo.quant_core.edge_validation.config import EdgeValidationConfig

ET = pytz.timezone("US/Eastern")

QUARTERLY_MONTHS = {3: "H", 6: "M", 9: "U", 12: "Z"}
MONTH_CODES = {v: k for k, v in QUARTERLY_MONTHS.items()}


def _third_friday(year: int, month: int) -> date:
    first_day_weekday = date(year, month, 1).weekday()
    # Friday is weekday 4
    first_friday = 1 + (4 - first_day_weekday) % 7
    return date(year, month, first_friday + 14)


def _quarterly_expiries(start: date, end: date) -> list[tuple[str, date]]:
    """Return (YYYYMM expiry string, expiration date) for all quarterly contracts covering the range."""
    results: list[tuple[str, date]] = []
    d = date(start.year, 1, 1)
    while d <= end + timedelta(days=90):
        if d.month in QUARTERLY_MONTHS:
            exp_date = _third_friday(d.year, d.month)
            expiry_str = f"{d.year}{d.month:02d}"
            results.append((expiry_str, exp_date))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return results


class FuturesDataFetcher:
    def __init__(self, config: EdgeValidationConfig, offline: bool = False) -> None:
        self._config = config
        self._offline = offline
        self._cache_dir = Path(config.data_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_bars(self, symbol: str) -> pd.DataFrame:
        cached = self._load_cache(symbol)
        start = datetime.strptime(self._config.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self._config.end_date, "%Y-%m-%d").date()

        if cached is not None:
            cached_start = cached.index.min().date()
            cached_end = cached.index.max().date()
            if cached_start <= start and cached_end >= end:
                mask = (cached.index.date >= start) & (cached.index.date <= end)
                return cached.loc[mask]

        if self._offline:
            if cached is not None:
                return cached
            raise RuntimeError(
                f"No cache for {symbol} and --offline mode is enabled. "
                f"Expected cache at {self._cache_path(symbol)}"
            )

        df = self._build_continuous(symbol)
        self._save_cache(symbol, df)
        mask = (df.index.date >= start) & (df.index.date <= end)
        return df.loc[mask]

    def _cache_path(self, symbol: str) -> Path:
        return self._cache_dir / f"{symbol}_continuous_1min.parquet"

    def _load_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        return df

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol)
        df.to_parquet(path)

    def _get_front_month_expiry(self, symbol: str, as_of: date) -> str:
        expiries = _quarterly_expiries(as_of - timedelta(days=90), as_of + timedelta(days=120))
        for expiry_str, exp_date in expiries:
            roll_date = exp_date - timedelta(days=7)
            if as_of < roll_date:
                return expiry_str
        raise RuntimeError(f"Could not determine front month for {symbol} as of {as_of}")

    def _fetch_from_ibkr(self, symbol: str, expiry: str, fetch_date: date) -> pd.DataFrame:
        broker = IBKRBroker(
            config=IBKRConfig(
                host=self._config.ibkr_host,
                port=self._config.ibkr_port,
                client_id=self._config.ibkr_client_id,
            ),
            require_paper=False,
            allow_live=True,
        )
        broker.connect()
        try:
            instrument = InstrumentSpec(
                kind="FUT",
                symbol=symbol,
                exchange=self._config.exchange,
                currency=self._config.currency,
                expiry=expiry,
            )
            end_dt_str = (fetch_date + timedelta(days=1)).strftime("%Y%m%d-00:00:00")
            bars: list[Bar] = broker.get_historical_bars(
                instrument,
                end_datetime=end_dt_str,
                duration="1 D",
                bar_size=self._config.bar_size,
                what_to_show="TRADES",
                use_rth=self._config.use_rth_only,
            )
        finally:
            broker.disconnect()

        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        records = []
        for b in bars:
            records.append({
                "timestamp": pd.Timestamp.fromtimestamp(b.timestamp_epoch_s, tz=ET),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume or 0.0,
            })

        df = pd.DataFrame(records).set_index("timestamp")
        df.index.name = None
        return df

    def _build_continuous(self, symbol: str) -> pd.DataFrame:
        start = datetime.strptime(self._config.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self._config.end_date, "%Y-%m-%d").date()

        expiries = _quarterly_expiries(start, end)
        # Build schedule: for each trading day, which contract to use
        # Roll 1 week before 3rd Friday
        roll_schedule: list[tuple[date, date, str]] = []  # (start, end, expiry)
        for i, (expiry_str, exp_date) in enumerate(expiries):
            roll_date = exp_date - timedelta(days=7)
            if i == 0:
                seg_start = start
            else:
                prev_exp_date = expiries[i - 1][1]
                seg_start = prev_exp_date - timedelta(days=7)
            seg_end = roll_date - timedelta(days=1)
            if seg_end < start:
                continue
            if seg_start > end:
                break
            seg_start = max(seg_start, start)
            seg_end = min(seg_end, end)
            roll_schedule.append((seg_start, seg_end, expiry_str))

        # Fetch each contract segment
        segments: list[pd.DataFrame] = []
        total_days = sum((seg_end - seg_start).days + 1 for seg_start, seg_end, _ in roll_schedule)
        fetched = 0

        for seg_start, seg_end, expiry_str in roll_schedule:
            current = seg_start
            seg_frames: list[pd.DataFrame] = []
            while current <= seg_end:
                fetched += 1
                print(f"Fetching {symbol} {expiry_str} {current} ... {fetched}/{total_days}")
                try:
                    day_df = self._fetch_from_ibkr(symbol, expiry_str, current)
                    if not day_df.empty:
                        seg_frames.append(day_df)
                except Exception as e:
                    print(f"  Warning: failed to fetch {current}: {e}")
                current += timedelta(days=1)
                time.sleep(0.5)

            if seg_frames:
                segments.append(pd.concat(seg_frames))

        if not segments:
            raise RuntimeError(f"No data fetched for {symbol}")

        # Panama Canal adjustment: subtract gap at each roll point
        adjusted: list[pd.DataFrame] = [segments[-1]]
        cumulative_adjustment = 0.0

        for i in range(len(segments) - 2, -1, -1):
            prev_seg = segments[i]
            next_seg = segments[i + 1]
            if prev_seg.empty or next_seg.empty:
                adjusted.insert(0, prev_seg)
                continue
            gap = next_seg.iloc[0]["open"] - prev_seg.iloc[-1]["close"]
            cumulative_adjustment += gap
            adj_seg = prev_seg.copy()
            for col in ["open", "high", "low", "close"]:
                adj_seg[col] = adj_seg[col] + cumulative_adjustment
            adjusted.insert(0, adj_seg)

        df = pd.concat(adjusted).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
