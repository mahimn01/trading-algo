#!/usr/bin/env python3
"""Profile where time is spent in the backtest to find optimization targets."""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.WARNING)

from scripts.run_10yr_backtest import (
    load_bar_data, build_controller_and_runner, COMBO_B_STRATEGIES,
)

def main():
    print("Loading data...")
    bar_data, _ = load_bar_data()
    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    ref_bars = bar_data[ref_sym]

    # Use just 1 year (2024) for profiling
    target = date(2024, 1, 1)
    end = date(2024, 12, 31)
    start_idx = next(i for i, b in enumerate(ref_bars) if b.timestamp.date() >= target)
    end_idx = next(i for i in range(len(ref_bars)-1, -1, -1) if ref_bars[i].timestamp.date() <= end) + 1

    sliced = {sym: bars[start_idx:end_idx] for sym, bars in bar_data.items()}
    n_bars = end_idx - start_idx
    print(f"Profiling {n_bars:,} bars (year 2024)...")

    _, runner = build_controller_and_runner(COMBO_B_STRATEGIES)

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.time()
    result = runner.run(sliced)
    elapsed = time.time() - t0
    profiler.disable()

    print(f"\nCompleted in {elapsed:.1f}s ({n_bars/elapsed:.0f} bars/sec)")
    print(f"Sharpe: {result.sharpe_ratio:+.3f}")

    # Print top 40 by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n=== TOP 40 BY CUMULATIVE TIME ===")
    stats.print_stats(40)

    # Print top 40 by total time (self time)
    print("\n=== TOP 40 BY SELF TIME ===")
    stats.sort_stats('tottime')
    stats.print_stats(40)

if __name__ == "__main__":
    main()
