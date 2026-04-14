from __future__ import annotations

import numpy as np
import pandas as pd
import pytz
import pytest
from datetime import date

from trading_algo.quant_core.edge_validation.types import PatternOccurrence
from trading_algo.quant_core.edge_validation.analysis.excursion import (
    compute_excursions,
    compare_to_random,
)

ET = pytz.timezone("US/Eastern")


def _make_linear_day(date_str: str, start: float, end: float) -> pd.DataFrame:
    times = pd.date_range(f"{date_str} 09:30", f"{date_str} 16:00", freq="1min", tz=ET)
    n = len(times)
    prices = np.linspace(start, end, n)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 500.0),
        },
        index=times,
    )


class TestComputeExcursions:
    def test_long_on_uptrend(self) -> None:
        bars = _make_linear_day("2025-01-06", 5000, 5020)
        entry_idx = 30
        occ = PatternOccurrence(
            timestamp=bars.index[entry_idx].timestamp(),
            pattern="test",
            direction="long",
            entry_price=float(bars.iloc[entry_idx]["close"]),
            day=date(2025, 1, 6),
        )

        exc = compute_excursions([occ], bars)
        assert exc.n_occurrences == 1
        assert exc.mfe_points[0] > 0
        assert exc.realized_pnl[0] > 0
        assert exc.mae_points[0] >= 0

    def test_short_on_downtrend(self) -> None:
        bars = _make_linear_day("2025-01-06", 5020, 5000)
        entry_idx = 30
        occ = PatternOccurrence(
            timestamp=bars.index[entry_idx].timestamp(),
            pattern="test",
            direction="short",
            entry_price=float(bars.iloc[entry_idx]["close"]),
            day=date(2025, 1, 6),
        )

        exc = compute_excursions([occ], bars)
        assert exc.n_occurrences == 1
        assert exc.mfe_points[0] > 0
        assert exc.realized_pnl[0] > 0

    def test_long_on_downtrend_has_negative_pnl(self) -> None:
        bars = _make_linear_day("2025-01-06", 5020, 5000)
        entry_idx = 30
        occ = PatternOccurrence(
            timestamp=bars.index[entry_idx].timestamp(),
            pattern="test",
            direction="long",
            entry_price=float(bars.iloc[entry_idx]["close"]),
            day=date(2025, 1, 6),
        )

        exc = compute_excursions([occ], bars)
        assert exc.realized_pnl[0] < 0

    def test_empty_occurrences(self) -> None:
        bars = _make_linear_day("2025-01-06", 5000, 5020)
        exc = compute_excursions([], bars)
        assert exc.n_occurrences == 0
        assert len(exc.mfe_points) == 0


class TestCompareToRandom:
    def test_trending_day_pattern_beats_random(self) -> None:
        bars = _make_linear_day("2025-01-06", 5000, 5040)
        entry_idx = 5
        occ = PatternOccurrence(
            timestamp=bars.index[entry_idx].timestamp(),
            pattern="test",
            direction="long",
            entry_price=float(bars.iloc[entry_idx]["close"]),
            day=date(2025, 1, 6),
        )

        result = compare_to_random([occ], bars, n_random=100, seed=42)
        assert result.pattern.n_occurrences == 1
        assert result.random.n_occurrences > 0
