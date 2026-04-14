from __future__ import annotations

import numpy as np
import pandas as pd
import pytz
import pytest
from datetime import date

from trading_algo.quant_core.edge_validation.types import PatternOccurrence
from trading_algo.quant_core.edge_validation.analysis.regime import (
    classify_trading_days,
    regime_conditional_test,
    hurst_exponent_test,
    variance_ratio_test,
)

ET = pytz.timezone("US/Eastern")


def _make_bars(date_str: str, start: float, end: float) -> pd.DataFrame:
    times = pd.date_range(f"{date_str} 09:30", f"{date_str} 16:00", freq="1min", tz=ET)
    n = len(times)
    rng = np.random.default_rng(42)
    prices = np.linspace(start, end, n) + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(1, 5, n),
            "low": prices - rng.uniform(1, 5, n),
            "close": prices,
            "volume": rng.integers(100, 1000, n).astype(float),
        },
        index=times,
    )


class TestClassifyTradingDays:
    def test_classifies_up_day(self) -> None:
        bars = _make_bars("2025-01-06", 5000, 5050)
        labels = classify_trading_days(bars)
        assert date(2025, 1, 6) in labels
        assert "up" in labels[date(2025, 1, 6)]

    def test_classifies_down_day(self) -> None:
        bars = _make_bars("2025-01-06", 5050, 5000)
        labels = classify_trading_days(bars)
        assert "down" in labels[date(2025, 1, 6)]

    def test_multiple_days(self) -> None:
        bars1 = _make_bars("2025-01-06", 5000, 5050)
        bars2 = _make_bars("2025-01-07", 5050, 5000)
        bars = pd.concat([bars1, bars2]).sort_index()
        labels = classify_trading_days(bars)
        assert len(labels) == 2


class TestRegimeConditionalTest:
    def test_returns_breakdowns_per_regime(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        days = pd.bdate_range("2024-01-01", periods=n)
        returns = rng.normal(0.5, 2.0, n)
        occs = [
            PatternOccurrence(
                timestamp=0.0, pattern="test", direction="long",
                entry_price=5000, day=d.date(),
            )
            for d in days
        ]
        labels = {d.date(): ("high_vol_up" if i % 2 == 0 else "low_vol_down") for i, d in enumerate(days)}

        result = regime_conditional_test(occs, returns, labels)
        assert len(result.breakdowns) >= 2

        for bd in result.breakdowns:
            assert bd.n_occurrences > 0
            assert 0 <= bd.win_rate <= 1


class TestHurstExponent:
    def test_returns_float(self) -> None:
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, 500))
        h = hurst_exponent_test(series)
        assert isinstance(h, float)
        assert 0 < h < 2

    def test_trending_series_high_hurst(self) -> None:
        series = np.arange(1000, dtype=float)
        h = hurst_exponent_test(series)
        assert h > 0.5


class TestVarianceRatioTest:
    def test_returns_results_for_all_periods(self) -> None:
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, 500))
        results = variance_ratio_test(series, periods=(2, 5, 10))
        assert len(results) == 3

    def test_random_walk_near_one(self) -> None:
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, 2000))
        results = variance_ratio_test(series, periods=(5,))
        assert abs(results[0].vr - 1.0) < 0.3

    def test_trending_series_rejects_random_walk(self) -> None:
        series = np.arange(1000, dtype=float) + np.random.normal(0, 0.1, 1000)
        results = variance_ratio_test(series, periods=(5,))
        assert results[0].rejects_random_walk is True
