from __future__ import annotations

import numpy as np
import pandas as pd
import pytz
import pytest

from trading_algo.quant_core.edge_validation.patterns.orb import ORBDetector
from trading_algo.quant_core.edge_validation.patterns.gap_fade import GapFadeDetector
from trading_algo.quant_core.edge_validation.patterns.vwap_reversion import VWAPReversionDetector

ET = pytz.timezone("US/Eastern")


def _make_day_bars(
    date_str: str,
    start_price: float,
    price_func,
    n_minutes: int = 391,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{date_str} 09:30", periods=n_minutes, freq="1min", tz=ET)
    prices = np.array([price_func(i, start_price) for i in range(n_minutes)])
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.5, 1.5, n_minutes),
            "low": prices - rng.uniform(0.5, 1.5, n_minutes),
            "close": prices + rng.normal(0, 0.3, n_minutes),
            "volume": rng.integers(100, 1000, n_minutes).astype(float),
        },
        index=times,
    )


class TestORBDetector:
    def test_detects_upside_breakout(self) -> None:
        def price_fn(i: int, base: float) -> float:
            if i < 60:
                return base + 5 * np.sin(i * np.pi / 30)
            return base + 10 + (i - 60) * 0.1

        bars = _make_day_bars("2025-01-06", 5000, price_fn)
        detector = ORBDetector(range_minutes=60, min_range_points=1.0)
        occs = detector.detect(bars)

        assert len(occs) >= 1
        longs = [o for o in occs if o.direction == "long"]
        assert len(longs) == 1
        assert longs[0].metadata["range_size"] > 1.0

    def test_detects_downside_breakout(self) -> None:
        def price_fn(i: int, base: float) -> float:
            if i < 30:
                return base + 3 * np.sin(i * np.pi / 15)
            return base - 5 - (i - 30) * 0.1

        bars = _make_day_bars("2025-01-06", 5000, price_fn)
        detector = ORBDetector(range_minutes=30, min_range_points=0.5)
        occs = detector.detect(bars)

        shorts = [o for o in occs if o.direction == "short"]
        assert len(shorts) >= 1

    def test_skips_tiny_range(self) -> None:
        def price_fn(i: int, base: float) -> float:
            return base + 0.1 * np.sin(i * np.pi / 30)

        bars = _make_day_bars("2025-01-06", 5000, price_fn)
        detector = ORBDetector(range_minutes=60, min_range_points=50.0)
        occs = detector.detect(bars)
        assert len(occs) == 0

    def test_name_property(self) -> None:
        assert ORBDetector(range_minutes=15).name == "orb_15min"
        assert ORBDetector(range_minutes=60).name == "orb_60min"

    def test_max_one_per_direction_per_day(self) -> None:
        def price_fn(i: int, base: float) -> float:
            return base + 20 * np.sin(i * np.pi / 50)

        bars = _make_day_bars("2025-01-06", 5000, price_fn)
        detector = ORBDetector(range_minutes=15, min_range_points=0.1)
        occs = detector.detect(bars)

        longs = [o for o in occs if o.direction == "long"]
        shorts = [o for o in occs if o.direction == "short"]
        assert len(longs) <= 1
        assert len(shorts) <= 1


class TestGapFadeDetector:
    def test_detects_gap_down(self) -> None:
        def day1_fn(i: int, base: float) -> float:
            return base

        def day2_fn(i: int, base: float) -> float:
            return base

        bars1 = _make_day_bars("2025-01-06", 5000, day1_fn, seed=1)
        bars2 = _make_day_bars("2025-01-07", 4975, day2_fn, seed=2)
        bars = pd.concat([bars1, bars2]).sort_index()

        detector = GapFadeDetector(min_gap_pct=0.001, max_gap_pct=0.02)
        occs = detector.detect(bars)

        assert len(occs) >= 1
        assert occs[0].direction == "long"
        assert occs[0].metadata["gap_pct"] < 0

    def test_detects_gap_up(self) -> None:
        bars1 = _make_day_bars("2025-01-06", 5000, lambda i, b: b, seed=1)
        bars2 = _make_day_bars("2025-01-07", 5030, lambda i, b: b, seed=2)
        bars = pd.concat([bars1, bars2]).sort_index()

        detector = GapFadeDetector(min_gap_pct=0.001, max_gap_pct=0.02)
        occs = detector.detect(bars)

        assert len(occs) >= 1
        assert occs[0].direction == "short"

    def test_filters_tiny_gap(self) -> None:
        bars1 = _make_day_bars("2025-01-06", 5000, lambda i, b: b, seed=1)
        bars2 = _make_day_bars("2025-01-07", 5001, lambda i, b: b, seed=2)
        bars = pd.concat([bars1, bars2]).sort_index()

        detector = GapFadeDetector(min_gap_pct=0.01, max_gap_pct=0.05)
        occs = detector.detect(bars)
        assert len(occs) == 0

    def test_name_property(self) -> None:
        assert GapFadeDetector().name == "gap_fade"


class TestVWAPReversionDetector:
    def test_detects_deviation_above(self) -> None:
        def price_fn(i: int, base: float) -> float:
            if i < 90:
                return base
            if i < 150:
                return base + 30
            return base

        bars = _make_day_bars("2025-01-06", 5000, price_fn)
        detector = VWAPReversionDetector(sigma_threshold=1.5, min_bars_for_std=20)
        occs = detector.detect(bars)

        shorts = [o for o in occs if o.direction == "short"]
        assert len(shorts) >= 1

    def test_respects_time_window(self) -> None:
        def price_fn(i: int, base: float) -> float:
            if i < 10:
                return base + 50
            return base

        bars = _make_day_bars("2025-01-06", 5000, price_fn)
        detector = VWAPReversionDetector(sigma_threshold=1.0, min_bars_for_std=5)
        occs = detector.detect(bars)

        for occ in occs:
            ts = pd.Timestamp(occ.timestamp, unit="s", tz="UTC").tz_convert(ET)
            assert ts.hour >= 10 or (ts.hour == 10 and ts.minute >= 30)

    def test_name_property(self) -> None:
        assert VWAPReversionDetector().name == "vwap_reversion"
