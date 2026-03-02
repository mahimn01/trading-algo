#!/usr/bin/env python3
"""
Quick diagnostic: runs a 2-year backtest (2024-2026) to verify baseline.
Should take ~5-10 minutes instead of 20 hours.
Expected baseline Sharpe: ~2.5 (from prior 2-year results)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.WARNING)

from scripts.run_10yr_backtest import (
    load_bar_data, build_controller_and_runner, COMBO_B_STRATEGIES,
    slice_bar_data, sharpe_from_daily, BarObject, RISK_FREE_RATE,
)
import numpy as np

def main():
    print("Loading data...")
    t0 = time.time()
    bar_data, _ = load_bar_data()
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    ref_bars = bar_data[ref_sym]
    total_bars = len(ref_bars)
    print(f"  Total bars: {total_bars:,}")

    # Find bar index for 2024-01-01
    from datetime import date
    targets = [
        ("Bear year (2022)", date(2022, 1, 1), date(2022, 12, 31)),
        ("AI Recovery (2023-2024)", date(2023, 1, 1), date(2024, 12, 31)),
        ("Recent 2yr (2024-2026)", date(2024, 1, 1), date(2026, 12, 31)),
    ]

    for label, start_dt, end_dt in targets:
        # Find indices
        start_idx = 0
        end_idx = total_bars
        for i, bar in enumerate(ref_bars):
            if bar.timestamp.date() >= start_dt:
                start_idx = i
                break
        for i in range(total_bars - 1, -1, -1):
            if ref_bars[i].timestamp.date() <= end_dt:
                end_idx = i + 1
                break

        n_bars = end_idx - start_idx
        if n_bars < 1000:
            print(f"\n  {label}: too few bars ({n_bars}), skipping")
            continue

        print(f"\n  Running {label} ({n_bars:,} bars)...", flush=True)
        sliced = {sym: bars[start_idx:end_idx] for sym, bars in bar_data.items()}

        _, runner = build_controller_and_runner(COMBO_B_STRATEGIES)
        t1 = time.time()
        result = runner.run(sliced)
        elapsed = time.time() - t1

        print(f"    Time: {elapsed:.1f}s ({n_bars/elapsed:.0f} bars/sec)")
        print(f"    Sharpe:       {result.sharpe_ratio:+.3f}")
        print(f"    Total Return: {result.total_return*100:+.2f}%")
        print(f"    Max Drawdown: {result.max_drawdown*100:.2f}%")
        print(f"    Trades:       {result.total_trades}")
        print(f"    Win Rate:     {result.win_rate*100:.1f}%")

        # Strategy attribution
        strat_signals = {}
        for k, v in result.strategy_attribution.items():
            if v.n_signals > 0:
                strat_signals[k] = v.n_signals
        if strat_signals:
            print(f"    Strategy signals: {strat_signals}")

    print("\nDone.")

if __name__ == "__main__":
    main()
