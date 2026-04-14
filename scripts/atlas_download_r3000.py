#!/usr/bin/env python3
"""Download Russell 3000 + ETF OHLCV data for ATLAS v6 training.

Phase 1 of the data scaling pipeline:
  - Downloads Russell 3000 constituent tickers from iShares IWV holdings CSV
  - Adds ~150 liquid ETFs (sector, country, bond, commodity, volatility)
  - Bulk downloads daily OHLCV via yfinance (batched, with rate limiting)
  - Saves as parquet files compatible with atlas_train_v3_ibkr.py feature pipeline
  - Skips symbols already in data/atlas_ibkr/ (no duplicate work)

Output: data/atlas_r3000/{SYMBOL}.parquet  (same format as atlas_ibkr)

Usage:
    .venv/bin/python scripts/atlas_download_r3000.py
    .venv/bin/python scripts/atlas_download_r3000.py --resume      # Skip already downloaded
    .venv/bin/python scripts/atlas_download_r3000.py --batch-size 50
    .venv/bin/python scripts/atlas_download_r3000.py --start-date 2000-01-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "data/atlas_r3000"
IBKR_DIR = "data/atlas_ibkr"
MANIFEST_PATH = f"{DATA_DIR}/download_manifest.json"

# Liquid ETFs covering sectors, countries, bonds, commodities, volatility
# These add regime diversity the equity-only universe misses
EXTRA_ETFS = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "RSP",
    # Sector
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC",
    "VGT", "VHT", "VFH", "VDE", "VIS", "VNQ", "VCR", "VDC",
    # Size & style
    "IWF", "IWD", "IWN", "IWO", "MDY", "IJR", "IJH",
    # Country/region
    "EFA", "EEM", "VWO", "FXI", "EWJ", "EWZ", "EWG", "EWU", "EWY", "EWA",
    "INDA", "VEA", "IEMG", "EWT", "EWH", "EWC", "EWS", "EWP", "EWI", "EWQ",
    # Bond
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP", "MUB",
    "BNDX", "EMB", "GOVT", "VCSH", "VCIT", "VGLT",
    # Commodity
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "DBC", "DBA", "PALL", "PPLT",
    "COPX", "WEAT", "CORN", "SOYB",
    # Volatility
    "VIXY", "SVXY",
    # Thematic
    "ARKK", "ARKG", "ARKF", "SOXX", "SMH", "HACK", "BOTZ", "ROBO",
    "TAN", "ICLN", "LIT", "REMX",
    # Leveraged (for crash regime diversity)
    "TQQQ", "SQQQ", "SPXU", "UPRO", "SOXL", "SOXS",
    # Real estate
    "IYR", "SCHH", "REM",
    # Dividend
    "VIG", "SCHD", "DVY", "HDV", "NOBL",
]


def get_russell3000_tickers() -> list[str]:
    """Get Russell 3000 tickers from iShares IWV holdings or fallback sources."""

    # Method 1: Try to download IWV holdings CSV from iShares
    try:
        url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
        df = pd.read_csv(url, skiprows=9)
        tickers = df["Ticker"].dropna().tolist()
        tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip() and not t.startswith("-")]
        if len(tickers) > 2000:
            print(f"Got {len(tickers)} tickers from iShares IWV holdings CSV")
            return tickers
    except Exception as e:
        print(f"iShares CSV download failed: {e}")

    # Method 2: Try Wikipedia Russell 1000 + supplemental
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index", match="Ticker")
        if tables:
            tickers = tables[0]["Ticker"].dropna().tolist()
            if len(tickers) > 500:
                print(f"Got {len(tickers)} tickers from Wikipedia Russell 1000")
                return tickers
    except Exception as e:
        print(f"Wikipedia scrape failed: {e}")

    # Method 3: Construct from S&P 500 + S&P 400 + S&P 600 (covers ~1500 symbols)
    all_tickers = []
    for url_suffix, name in [
        ("List_of_S%26P_500_companies", "S&P 500"),
        ("List_of_S%26P_400_companies", "S&P 400"),
        ("List_of_S%26P_600_companies", "S&P 600"),
    ]:
        try:
            tables = pd.read_html(f"https://en.wikipedia.org/wiki/{url_suffix}")
            col = "Symbol" if "Symbol" in tables[0].columns else "Ticker symbol" if "Ticker symbol" in tables[0].columns else tables[0].columns[0]
            tickers = tables[0][col].dropna().tolist()
            tickers = [t.replace(".", "-").strip() for t in tickers if isinstance(t, str)]
            all_tickers.extend(tickers)
            print(f"  {name}: {len(tickers)} tickers")
        except Exception as e:
            print(f"  {name}: failed — {e}")

    if all_tickers:
        all_tickers = sorted(set(all_tickers))
        print(f"Combined S&P universe: {len(all_tickers)} tickers")
        return all_tickers

    raise RuntimeError("All ticker sources failed. Check internet connection.")


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def download_batch(
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    manifest: dict,
    batch_size: int = 20,
) -> tuple[int, int]:
    """Download OHLCV data in batches via yf.download().

    Returns (success_count, fail_count).
    """
    success, fail = 0, 0
    total = len(symbols)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = symbols[batch_start:batch_end]
        batch_str = " ".join(batch)

        print(f"\n  Batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size}: "
              f"downloading {len(batch)} symbols [{batch_start+1}-{batch_end}/{total}]...",
              flush=True)

        try:
            # yf.download returns MultiIndex (Price, Ticker) columns in v1.2+
            df = yf.download(
                batch_str,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                threads=True,
                progress=False,
            )

            if df.empty:
                print(f"    Empty result for entire batch", flush=True)
                fail += len(batch)
                continue

            for sym in batch:
                try:
                    # yfinance 1.2+ always uses MultiIndex (Price, Ticker)
                    if isinstance(df.columns, pd.MultiIndex):
                        tickers_in_df = df.columns.get_level_values(1).unique()
                        if sym not in tickers_in_df:
                            manifest[sym] = {"status": "no_data", "bars": 0}
                            fail += 1
                            continue
                        sym_df = pd.DataFrame({
                            col: df[(col, sym)] for col in ["Open", "High", "Low", "Close", "Volume"]
                            if (col, sym) in df.columns
                        }, index=df.index)
                    else:
                        sym_df = df.copy()

                    sym_df = sym_df.dropna(subset=["Close"])

                    if len(sym_df) < 500:
                        manifest[sym] = {"status": "too_short", "bars": len(sym_df)}
                        fail += 1
                        continue

                    # Normalize column names to match IBKR format
                    out = pd.DataFrame({
                        "open": sym_df["Open"].values,
                        "high": sym_df["High"].values,
                        "low": sym_df["Low"].values,
                        "close": sym_df["Close"].values,
                        "volume": sym_df["Volume"].values,
                    }, index=sym_df.index)
                    out.index.name = None

                    out_path = f"{output_dir}/{sym}.parquet"
                    out.to_parquet(out_path)

                    manifest[sym] = {
                        "status": "ok",
                        "bars": len(out),
                        "start": str(out.index[0].date()),
                        "end": str(out.index[-1].date()),
                    }
                    success += 1

                except Exception as e:
                    manifest[sym] = {"status": f"error: {str(e)[:100]}", "bars": 0}
                    fail += 1

            # Save manifest after each batch (resumable)
            save_manifest(manifest)

            # Rate limit: 0.5s between batches
            time.sleep(0.5)

        except Exception as e:
            print(f"    Batch download error: {e}", flush=True)
            fail += len(batch)
            time.sleep(2)

        # Progress report every 5 batches
        if (batch_start // batch_size + 1) % 5 == 0:
            print(f"  Progress: {success} OK, {fail} failed, "
                  f"{total - batch_end} remaining", flush=True)

    return success, fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Russell 3000 OHLCV for ATLAS")
    parser.add_argument("--start-date", default="2000-01-01", help="Download start date")
    parser.add_argument("--end-date", default="2026-04-07", help="Download end date")
    parser.add_argument("--batch-size", type=int, default=20, help="Symbols per yfinance batch")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded symbols")
    parser.add_argument("--etfs-only", action="store_true", help="Only download extra ETFs")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("  ATLAS Phase 1 — Russell 3000 + ETF OHLCV Download")
    print("=" * 70)

    # Get existing IBKR symbols (skip these — we already have them)
    ibkr_symbols = set()
    if os.path.isdir(IBKR_DIR):
        ibkr_symbols = {f.replace(".parquet", "") for f in os.listdir(IBKR_DIR)
                        if f.endswith(".parquet") and "_iv" not in f}
    print(f"Existing IBKR symbols: {len(ibkr_symbols)}")

    # Load or build manifest
    manifest = load_manifest()

    if args.etfs_only:
        all_symbols = sorted(set(EXTRA_ETFS))
    else:
        # Get Russell 3000
        print("\nFetching Russell 3000 ticker list...", flush=True)
        r3000 = get_russell3000_tickers()

        # Combine with ETFs and deduplicate
        all_symbols = sorted(set(r3000 + EXTRA_ETFS))
        print(f"\nTotal unique symbols: {len(all_symbols)}")

    # Filter out already-downloaded
    to_download = []
    skipped_ibkr = 0
    skipped_done = 0
    for sym in all_symbols:
        if sym in ibkr_symbols:
            skipped_ibkr += 1
            continue
        if args.resume and sym in manifest and manifest[sym].get("status") == "ok":
            skipped_done += 1
            continue
        # Skip known failures on resume too
        if args.resume and sym in manifest and manifest[sym].get("status") in ("too_short", "no_data"):
            skipped_done += 1
            continue
        to_download.append(sym)

    print(f"Skipping {skipped_ibkr} symbols already in IBKR data")
    if args.resume:
        print(f"Skipping {skipped_done} symbols already downloaded")
    print(f"Downloading: {len(to_download)} symbols")
    print(f"Date range: {args.start_date} → {args.end_date}")
    print(f"Batch size: {args.batch_size}")

    if not to_download:
        print("\nNothing to download!")
        return

    t0 = time.time()
    ok, fail = download_batch(
        to_download,
        args.start_date,
        args.end_date,
        DATA_DIR,
        manifest,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  Download Complete")
    print(f"{'='*70}")
    print(f"  Success: {ok}")
    print(f"  Failed: {fail}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  IBKR existing: {len(ibkr_symbols)}")

    # Count total data available
    r3000_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and f != "download_manifest.json"]
    total_symbols = len(ibkr_symbols) + len(r3000_files)
    print(f"\n  Total symbols available for training: {total_symbols}")
    print(f"    IBKR (with real IV): {len(ibkr_symbols)}")
    print(f"    R3000/ETF (synthetic IV): {len(r3000_files)}")
    print(f"\n  Next: .venv/bin/python scripts/atlas_extract_features_v4.py")


if __name__ == "__main__":
    main()
