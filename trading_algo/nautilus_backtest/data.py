"""Daily bar data for NautilusTrader backtests.

NautilusTrader has no "fill at the next bar's open" mode, so each trading
session is fed to the engine as three events:

* a synthetic opening ``TradeTick`` at 09:30 New York time. Orders decided at
  the prior close are submitted just after it and fill at this price.
* a close-stamped daily ``Bar`` at 16:00 New York time. It drives the
  strategies.
* a ``MarkPriceUpdate`` at the close. Without it the portfolio values open
  positions at the last trade (the open) instead of the close.

The same floats and naive session timestamps are handed to the repo's
``MultiStrategyBacktestRunner``, so both engines see identical inputs.

Early closes (13:00 half-days) are treated as normal 16:00 closes. That only
moves the timestamps, not the prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nautilus_trader.model import (
    AggressorSide,
    Bar,
    BarType,
    Currency,
    Equity,
    InstrumentId,
    MarkPriceUpdate,
    Price,
    Quantity,
    Symbol,
    TradeId,
    TradeTick,
    Venue,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "atlas_ibkr"
NEW_YORK = ZoneInfo("America/New_York")
PRICE_PRECISION = 4
OHLCV = ["open", "high", "low", "close", "volume"]
PRICES = ["open", "high", "low", "close"]
USD = Currency.from_str("USD")

# Open ticks carry a huge size so one order always fills at a single price.
_OPEN_TICK_SIZE = 10**12


class DataError(ValueError):
    """Raised when bar data fails validation."""


@dataclass(frozen=True)
class Session:
    """One trading session shared by every symbol."""

    day: date
    open_ns: int
    close_ns: int

    @property
    def timestamp(self) -> datetime:
        """Naive session timestamp handed to strategies (midnight of the day)."""
        return datetime(self.day.year, self.day.month, self.day.day)


def load_daily_bars(
    symbols: Iterable[str],
    data_dir: Path | str | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    drop_last_row: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load daily OHLCV bars from the IBKR parquet cache.

    The cache's final row is the session that was still trading when the file
    was downloaded, so it is dropped by default.

    Returns a dict of symbol -> DataFrame indexed by naive session date with
    float columns open/high/low/close/volume.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = root / f"{symbol}.parquet"
        if not path.exists():
            raise DataError(f"No daily bars for {symbol} at {path}")
        frame = pd.read_parquet(path)
        if drop_last_row:
            frame = frame.iloc[:-1]
        frame = frame[OHLCV].astype(float)
        # Prices are rounded to the instrument's tick so both engines trade
        # and mark at exactly the same values.
        frame[PRICES] = frame[PRICES].round(PRICE_PRECISION)
        frame.index = pd.DatetimeIndex(frame.index).normalize().as_unit("ns")
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        validate_frame(symbol, frame)
        frames[symbol] = frame
    return frames


def validate_frame(symbol: str, frame: pd.DataFrame) -> None:
    """Check a daily frame is sorted, unique, finite and OHLC-consistent."""
    missing = [c for c in OHLCV if c not in frame.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")
    if frame.empty:
        raise DataError(f"{symbol}: no bars in the requested range")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise DataError(f"{symbol}: dates must be strictly increasing")
    values = frame[OHLCV].to_numpy()
    if not np.isfinite(values).all():
        raise DataError(f"{symbol}: non-finite values")
    prices = frame[PRICES]
    if (prices <= 0).any().any():
        raise DataError(f"{symbol}: non-positive prices")
    if (frame["volume"] < 0).any():
        raise DataError(f"{symbol}: negative volume")
    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    if (frame["high"] < body_high).any() or (frame["low"] > body_low).any():
        raise DataError(f"{symbol}: high/low do not bracket open/close")


def build_sessions(frames: Mapping[str, pd.DataFrame]) -> list[Session]:
    """Return the union of all symbols' session dates, oldest first."""
    days = sorted({ts.date() for frame in frames.values() for ts in frame.index})
    return [_session(day) for day in days]


def _session(day: date) -> Session:
    open_ts = pd.Timestamp(day.year, day.month, day.day, 9, 30, tz=NEW_YORK)
    close_ts = pd.Timestamp(day.year, day.month, day.day, 16, 0, tz=NEW_YORK)
    return Session(
        day=day,
        open_ns=int(open_ts.tz_convert("UTC").as_unit("ns").value),
        close_ns=int(close_ts.tz_convert("UTC").as_unit("ns").value),
    )


def make_equity(symbol: str, venue: str, margin: Decimal) -> Any:
    """Build a whole-share NautilusTrader equity with 4-decimal prices.

    Four decimals keep basis-point slippage visible on low split-adjusted
    prices (NVDA traded near $0.89 in 2016 after adjustment).
    """
    return Equity(
        instrument_id=InstrumentId(Symbol(symbol), Venue(venue)),
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=PRICE_PRECISION,
        price_increment=Price.from_str("0.0001"),
        ts_event=0,
        ts_init=0,
        lot_size=Quantity.from_int(1),
        margin_init=margin,
        margin_maint=margin,
    )


def bar_type_for(instrument: Any) -> Any:
    """Daily last-price bar type, aggregated outside NautilusTrader."""
    return BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")


def to_nautilus_data(
    frames: Mapping[str, pd.DataFrame],
    instruments: Mapping[str, Any],
) -> list[Any]:
    """Convert daily frames into open ticks, close bars and close marks.

    A final mark is added just after the last close so time alerts set for the
    last session still fire before the engine runs out of data.
    """
    data: list[Any] = []
    last_close: dict[str, tuple[int, float]] = {}
    for symbol, frame in frames.items():
        instrument = instruments[symbol]
        bar_type = bar_type_for(instrument)
        for i, (ts, row) in enumerate(frame.iterrows()):
            session = _session(ts.date())
            data.append(
                TradeTick(
                    instrument_id=instrument.id,
                    price=Price(row["open"], PRICE_PRECISION),
                    size=Quantity.from_int(_OPEN_TICK_SIZE),
                    aggressor_side=AggressorSide.NO_AGGRESSOR,
                    trade_id=TradeId(f"{symbol}-OPEN-{i}"),
                    ts_event=session.open_ns,
                    ts_init=session.open_ns,
                ),
            )
            data.append(
                Bar(
                    bar_type=bar_type,
                    open=Price(row["open"], PRICE_PRECISION),
                    high=Price(row["high"], PRICE_PRECISION),
                    low=Price(row["low"], PRICE_PRECISION),
                    close=Price(row["close"], PRICE_PRECISION),
                    volume=Quantity(row["volume"], 0),
                    ts_event=session.close_ns,
                    ts_init=session.close_ns,
                ),
            )
            data.append(
                MarkPriceUpdate(
                    instrument.id,
                    Price(row["close"], PRICE_PRECISION),
                    session.close_ns,
                    session.close_ns,
                ),
            )
            last_close[symbol] = (session.close_ns, row["close"])

    end_ns = max(close_ns for close_ns, _ in last_close.values()) + 1_000
    for symbol, (_, close) in last_close.items():
        data.append(
            MarkPriceUpdate(instruments[symbol].id, Price(close, PRICE_PRECISION), end_ns, end_ns),
        )
    return data


def runner_bars(frames: Mapping[str, pd.DataFrame]) -> dict[str, list[SimpleNamespace]]:
    """Bars in the shape ``MultiStrategyBacktestRunner.run`` expects.

    Uses the same floats and the same naive session timestamps the
    NautilusTrader strategy hands to the controller.
    """
    return {
        symbol: [
            SimpleNamespace(
                timestamp=datetime(ts.year, ts.month, ts.day),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for ts, row in frame.iterrows()
        ]
        for symbol, frame in frames.items()
    }
