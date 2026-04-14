#!/usr/bin/env python3
"""Extract ATLAS features from ALL sources (IBKR + R3000 yfinance data).

Produces atlas_features_v4/ with the same .npz format as v3:
  - normed: (T, 16) z-scored features
  - mu/sigma: (T, 16) rolling stats for de-stationary module
  - timestamps, dow, month: time encodings
  - actions: (T, 5) hindsight-optimal actions
  - rtg: (T,) return-to-go
  - closes, ivs, iv_ranks: raw series

Sources:
  1. data/atlas_ibkr/{SYM}.parquet + {SYM}_iv.parquet  (real IV)
  2. data/atlas_r3000/{SYM}.parquet  (synthetic IV from realized vol)

Usage:
    .venv/bin/python scripts/atlas_extract_features_v4.py
    .venv/bin/python scripts/atlas_extract_features_v4.py --resume
    .venv/bin/python scripts/atlas_extract_features_v4.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.features import ATLASFeatureComputer, RollingNormalizer
from trading_algo.quant_core.models.atlas.fast_hindsight import compute_hindsight_fast, warmup as warmup_jit
from trading_algo.quant_core.strategies.options.iv_rank import iv_rank as compute_iv_rank

IBKR_DIR = "data/atlas_ibkr"
R3000_DIR = "data/atlas_r3000"
FEATURE_DIR = "data/atlas_features_v4"
MIN_BARS = 500  # Minimum trading days required


def process_symbol(sym: str, source: str, config: ATLASConfig) -> tuple[str, int, str]:
    """Process a single symbol. Returns (symbol, n_windows, status)."""
    feat_path = f"{FEATURE_DIR}/{sym}_features.npz"
    if os.path.exists(feat_path):
        data = np.load(feat_path)
        n_w = max(0, len(data["normed"]) - config.context_len + 1 - 252)
        return (sym, n_w, "cached")

    fc = ATLASFeatureComputer()
    norm = RollingNormalizer()

    try:
        # Load price data from appropriate source
        if source == "ibkr":
            df = pd.read_parquet(f"{IBKR_DIR}/{sym}.parquet")
        else:
            df = pd.read_parquet(f"{R3000_DIR}/{sym}.parquet")

        if len(df) < MIN_BARS:
            return (sym, 0, f"too_short ({len(df)} bars)")

        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)
        timestamps = np.array([ts.timestamp() for ts in df.index])
        dows = np.array([ts.weekday() for ts in df.index], dtype=np.int32)
        months = np.array([ts.month - 1 for ts in df.index], dtype=np.int32)

        # Compute 12 market features, pad to 16
        raw = fc.compute_features(closes, highs, lows, volumes)
        full = np.concatenate([raw, np.zeros((len(raw), 4))], axis=1)
        normed, mu_arr, sigma_arr = norm.normalize(full)

        # IV: use real IBKR IV if available, else synthetic from realized vol
        from trading_algo.quant_core.strategies.options.iv_rank import iv_series_from_prices
        if source == "ibkr":
            iv_path = f"{IBKR_DIR}/{sym}_iv.parquet"
            if os.path.exists(iv_path):
                iv_df = pd.read_parquet(iv_path)
                real_iv = np.full(len(closes), np.nan)
                price_dates = {ts.date(): idx for idx, ts in enumerate(df.index)}
                for ts, row in iv_df.iterrows():
                    d = ts.date() if hasattr(ts, 'date') else ts
                    if d in price_dates:
                        real_iv[price_dates[d]] = row["iv"]
                for j in range(1, len(real_iv)):
                    if np.isnan(real_iv[j]) and not np.isnan(real_iv[j-1]):
                        real_iv[j] = real_iv[j-1]
                synthetic_iv = iv_series_from_prices(closes, rv_window=30, dynamic=True)
                iv_series = np.where(np.isnan(real_iv), synthetic_iv, real_iv)
            else:
                iv_series = iv_series_from_prices(closes, rv_window=30, dynamic=True)
        else:
            iv_series = iv_series_from_prices(closes, rv_window=30, dynamic=True)

        iv_series = np.nan_to_num(iv_series, nan=0.25)
        iv_ranks = np.array([compute_iv_rank(iv_series, t, 252) for t in range(len(closes))])
        iv_ranks = np.nan_to_num(iv_ranks, nan=50.0)

        # Hindsight-optimal actions using numba-compiled BSM
        optimal = compute_hindsight_fast(closes, iv_series)
        optimal = np.nan_to_num(optimal, nan=0.0)

        # Return-to-go (45-day forward Sharpe)
        rtg = np.zeros(len(closes), dtype=np.float32)
        for t in range(len(closes) - 45):
            fr = np.diff(np.log(closes[t:t + 46]))
            if len(fr) > 0 and np.std(fr) > 1e-8:
                rtg[t] = float(np.mean(fr) / np.std(fr) * np.sqrt(252))

        np.savez_compressed(
            feat_path,
            normed=normed.astype(np.float32), mu=mu_arr.astype(np.float32),
            sigma=sigma_arr.astype(np.float32), timestamps=timestamps,
            actions=optimal.astype(np.float32), rtg=rtg,
            dow=dows, month=months,
            closes=closes.astype(np.float32), ivs=iv_series.astype(np.float32),
            iv_ranks=iv_ranks.astype(np.float32),
        )

        n_w = max(0, normed.shape[0] - config.context_len + 1 - 252)
        return (sym, n_w, "ok")

    except Exception as e:
        return (sym, 0, f"error: {str(e)[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ATLAS v4 features from all sources")
    parser.add_argument("--resume", action="store_true", help="Skip already extracted")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (careful with RAM)")
    args = parser.parse_args()

    os.makedirs(FEATURE_DIR, exist_ok=True)
    config = ATLASConfig()

    print("=" * 70)
    print("  ATLAS v4 Feature Extraction — IBKR + R3000")
    print("=" * 70)

    # Discover all symbols from both sources
    symbols: dict[str, str] = {}  # sym -> source

    # IBKR symbols (priority — they have real IV)
    if os.path.isdir(IBKR_DIR):
        for f in os.listdir(IBKR_DIR):
            if f.endswith(".parquet") and "_iv" not in f:
                sym = f.replace(".parquet", "")
                symbols[sym] = "ibkr"

    # R3000 symbols (supplement — synthetic IV)
    if os.path.isdir(R3000_DIR):
        for f in os.listdir(R3000_DIR):
            if f.endswith(".parquet") and f != "download_manifest.json":
                sym = f.replace(".parquet", "")
                if sym not in symbols:  # Don't overwrite IBKR source
                    symbols[sym] = "r3000"

    ibkr_count = sum(1 for s in symbols.values() if s == "ibkr")
    r3000_count = sum(1 for s in symbols.values() if s == "r3000")
    print(f"IBKR symbols (real IV): {ibkr_count}")
    print(f"R3000 symbols (synthetic IV): {r3000_count}")
    print(f"Total: {len(symbols)}")

    # Warmup numba JIT (first call compiles all functions)
    print("Warming up numba JIT...", flush=True)
    t_jit = time.time()
    warmup_jit()
    print(f"JIT compiled in {time.time() - t_jit:.1f}s", flush=True)

    # Filter already extracted
    if args.resume:
        existing = {f.replace("_features.npz", "") for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")}
        to_process = {s: src for s, src in symbols.items() if s not in existing}
        print(f"Already extracted: {len(existing)}, remaining: {len(to_process)}")
    else:
        to_process = symbols

    if not to_process:
        print("\nAll features already extracted!")
        # Count total windows
        total_w = 0
        for f in os.listdir(FEATURE_DIR):
            if f.endswith("_features.npz"):
                d = np.load(f"{FEATURE_DIR}/{f}")
                total_w += max(0, len(d["normed"]) - config.context_len + 1 - 252)
        print(f"Total training windows: {total_w:,}")
        return

    sorted_symbols = sorted(to_process.keys())
    t0 = time.time()
    total_windows = 0
    ok_count = 0
    fail_count = 0

    if args.workers > 1:
        # Parallel extraction
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_symbol, sym, to_process[sym], config): sym
                for sym in sorted_symbols
            }
            for i, future in enumerate(as_completed(futures)):
                sym, n_w, status = future.result()
                total_windows += n_w
                if status in ("ok", "cached"):
                    ok_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 25 == 0 or status not in ("ok", "cached"):
                    elapsed = time.time() - t0
                    per = elapsed / (i + 1)
                    eta = per * (len(sorted_symbols) - i - 1)
                    print(f"  [{i+1}/{len(sorted_symbols)}] {sym}: {status} "
                          f"({n_w} windows, {elapsed:.0f}s, ETA ~{eta/60:.0f}min)", flush=True)
    else:
        # Sequential extraction (safer for RAM)
        for i, sym in enumerate(sorted_symbols):
            sym, n_w, status = process_symbol(sym, to_process[sym], config)
            total_windows += n_w
            if status in ("ok", "cached"):
                ok_count += 1
            else:
                fail_count += 1
            if (i + 1) % 25 == 0 or i == 0 or status not in ("ok", "cached"):
                elapsed = time.time() - t0
                per = elapsed / (i + 1)
                eta = per * (len(sorted_symbols) - i - 1)
                print(f"  [{i+1}/{len(sorted_symbols)}] {sym}: {status} "
                      f"({n_w} windows, {elapsed:.0f}s, ETA ~{eta/60:.0f}min)", flush=True)

    # Also count previously extracted features (from --resume)
    for f in os.listdir(FEATURE_DIR):
        if f.endswith("_features.npz"):
            sym = f.replace("_features.npz", "")
            if sym not in to_process:
                d = np.load(f"{FEATURE_DIR}/{f}")
                total_windows += max(0, len(d["normed"]) - config.context_len + 1 - 252)

    elapsed = time.time() - t0
    n_features = len([f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")])

    print(f"\n{'='*70}")
    print(f"  Feature Extraction Complete")
    print(f"{'='*70}")
    print(f"  Symbols processed: {ok_count} OK, {fail_count} failed")
    print(f"  Total feature files: {n_features}")
    print(f"  Total training windows: {total_windows:,}")
    print(f"  Data:param ratio: {total_windows / 835_000:.1f}:1  (835K params)")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"\n  Next: train ATLAS v6 on {FEATURE_DIR}")


if __name__ == "__main__":
    main()
