#!/usr/bin/env python3
"""
Download 10Y daily bars from IBKR for ATLAS training.

Uses the existing download_10yr_data.py infrastructure but optimized for:
  - Daily bars (not 5-min) — 78x fewer requests per symbol
  - 2000+ symbols — pulled from IBKR scanner
  - Parquet output — 40% smaller, faster to load
  - Real historical IV data — whatToShow="OPTION_IMPLIED_VOLATILITY"

With daily bars, each symbol needs ~2 requests (10Y in "2 Y" chunks × 5).
At 12s pacing: ~60 seconds per symbol, ~33 hours for 2000 symbols.

Usage:
    .venv/bin/python scripts/download_atlas_data.py
    .venv/bin/python scripts/download_atlas_data.py --max-symbols 100
    .venv/bin/python scripts/download_atlas_data.py --resume
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

asyncio.set_event_loop(asyncio.new_event_loop())

from ib_async import IB, Stock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ATLAS_DIR = DATA_DIR / "atlas_ibkr"
MANIFEST_PATH = ATLAS_DIR / "manifest.json"

# Universe: S&P 500 components + Russell 1000 most liquid + major ETFs
# Start with a core set, expand via IBKR scanner
CORE_SYMBOLS = [
    # S&P 500 top by market cap
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "BRK B", "LLY", "TSM",
    "AVGO", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ",
    "ABBV", "WMT", "NFLX", "CRM", "BAC", "ORCL", "CVX", "MRK", "KO", "AMD",
    "ADBE", "PEP", "TMO", "ACN", "LIN", "MCD", "ABT", "CSCO", "DHR", "TXN",
    "WFC", "PM", "NEE", "INTU", "DIS", "ISRG", "RTX", "QCOM", "AMGN", "GE",
    "HON", "BKNG", "AMAT", "PFE", "SYK", "GS", "T", "BLK", "SPGI", "CAT",
    "MS", "AXP", "MDT", "CB", "DE", "GILD", "PLD", "C", "MMC", "SCHW",
    "VRTX", "SO", "BDX", "MO", "ADI", "CME", "CI", "ZTS", "CL", "DUK",
    "BMY", "ICE", "APD", "SHW", "NOC", "FCX", "EOG", "MCK", "EMR", "COP",
    "USB", "PNC", "TGT", "INTC", "ITW", "EL", "SLB", "PXD", "REGN", "D",
    # Mid-cap with liquid options
    "NKE", "BA", "SBUX", "PYPL", "SQ", "SNAP", "UBER", "ABNB", "DKNG", "RIVN",
    "LCID", "PLTR", "SOFI", "HOOD", "NIO", "MARA", "RIOT", "COIN", "ROKU", "PINS",
    "RBLX", "CRWD", "SNOW", "DDOG", "ZS", "NET", "PANW", "TTWO", "EA", "MTCH",
    "LYFT", "DASH", "UPST", "AFRM", "PATH", "U", "BILL", "HUBS", "OKTA", "MDB",
    "AAL", "UAL", "DAL", "LUV", "CCL", "RCL", "NCLH", "MAR", "HLT", "WYNN",
    "MGM", "LVS", "F", "GM", "TSLA", "RIVN", "LCID", "CHPT", "FSLR", "ENPH",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLU",
    "XLP", "XLY", "XLB", "XLC", "XLRE", "GLD", "SLV", "GDX", "USO", "UNG",
    "TLT", "IEF", "HYG", "LQD", "AGG", "EMB", "EEM", "EFA", "FXI", "EWJ",
    "VXX", "SQQQ", "TQQQ", "ARKK", "ARKG", "IBIT", "BITO",
    # REITs
    "O", "AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "DLR", "WELL", "AVB",
]

# Deduplicate
CORE_SYMBOLS = list(dict.fromkeys(CORE_SYMBOLS))


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"symbols": {}, "started": datetime.now().isoformat()}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


async def download_symbol(ib: IB, symbol: str, sleep_sec: float = 12) -> dict | None:
    """Download 10Y daily bars for one symbol. Returns bar data or None on failure."""
    try:
        contract = Stock(symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            print(f"    {symbol}: failed to qualify contract", flush=True)
            return None

        contract = qualified[0]
        all_bars = []
        end_dt = ""  # empty = now

        # Pull in 2-year chunks (IBKR limit for daily bars)
        for chunk in range(5):  # 5 × 2Y = 10Y
            await asyncio.sleep(sleep_sec)
            try:
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_dt,
                    durationStr="2 Y",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                )
            except Exception as e:
                print(f"    {symbol} chunk {chunk}: {e}", flush=True)
                break

            if not bars:
                break

            for b in bars:
                all_bars.append({
                    "timestamp": b.date.isoformat() if hasattr(b.date, 'isoformat') else str(b.date),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                })

            # Move end_dt to earliest bar's date for next chunk
            earliest = bars[0].date
            if hasattr(earliest, 'strftime'):
                end_dt = earliest.strftime("%Y%m%d %H:%M:%S")
            else:
                end_dt = str(earliest)

        if not all_bars:
            return None

        # Deduplicate by timestamp and sort
        seen = set()
        unique_bars = []
        for b in all_bars:
            if b["timestamp"] not in seen:
                seen.add(b["timestamp"])
                unique_bars.append(b)
        unique_bars.sort(key=lambda x: x["timestamp"])

        return {"symbol": symbol, "bars": unique_bars, "count": len(unique_bars)}

    except Exception as e:
        print(f"    {symbol}: ERROR — {e}", flush=True)
        return None


async def download_iv_data(ib: IB, symbol: str, sleep_sec: float = 12) -> dict | None:
    """Download historical implied volatility for a symbol."""
    try:
        contract = Stock(symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            return None
        contract = qualified[0]

        await asyncio.sleep(sleep_sec)
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="2 Y",
            barSizeSetting="1 day",
            whatToShow="OPTION_IMPLIED_VOLATILITY",
            useRTH=True,
            formatDate=2,
        )

        if not bars:
            return None

        iv_data = []
        for b in bars:
            iv_data.append({
                "timestamp": b.date.isoformat() if hasattr(b.date, 'isoformat') else str(b.date),
                "iv": float(b.close),
            })

        return {"symbol": symbol, "iv_bars": iv_data, "count": len(iv_data)}

    except Exception:
        return None


async def main(args):
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    # Load from universe file if available, else use CORE_SYMBOLS
    universe_path = ATLAS_DIR / "symbol_universe.txt"
    if universe_path.exists():
        with open(universe_path) as f:
            all_symbols = [l.strip() for l in f if l.strip()]
        symbols = all_symbols[:args.max_symbols]
    else:
        symbols = CORE_SYMBOLS[:args.max_symbols]
    print(f"ATLAS Data Download: {len(symbols)} symbols, 10Y daily bars")
    print(f"Output: {ATLAS_DIR}/")
    print(f"Pacing: {args.sleep}s between requests")

    # Filter already-completed symbols
    remaining = [s for s in symbols if s not in manifest.get("symbols", {}) or
                 not manifest["symbols"][s].get("completed")]
    print(f"Already done: {len(symbols) - len(remaining)}, remaining: {len(remaining)}")

    if not remaining:
        print("All symbols already downloaded.")
        return

    # Connect to IBKR
    ib = IB()
    await ib.connectAsync("127.0.0.1", args.port, clientId=args.client_id)
    print(f"Connected to IBKR on port {args.port}")

    start = time.time()
    for i, sym in enumerate(remaining):
        print(f"  [{i+1}/{len(remaining)}] {sym}...", end=" ", flush=True)

        result = await download_symbol(ib, sym, sleep_sec=args.sleep)
        if result is None:
            print("SKIP (no data)", flush=True)
            manifest.setdefault("symbols", {})[sym] = {"completed": False, "error": "no data"}
            save_manifest(manifest)
            continue

        # Save bars as parquet
        import pandas as pd
        df = pd.DataFrame(result["bars"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        parquet_path = ATLAS_DIR / f"{sym}.parquet"
        df.to_parquet(parquet_path)

        # Try to get IV data too
        iv_result = await download_iv_data(ib, sym, sleep_sec=args.sleep)
        if iv_result:
            iv_df = pd.DataFrame(iv_result["iv_bars"])
            iv_df["timestamp"] = pd.to_datetime(iv_df["timestamp"])
            iv_df = iv_df.set_index("timestamp").sort_index()
            iv_path = ATLAS_DIR / f"{sym}_iv.parquet"
            iv_df.to_parquet(iv_path)
            iv_count = iv_result["count"]
        else:
            iv_count = 0

        elapsed = time.time() - start
        per_sym = elapsed / (i + 1)
        eta = per_sym * (len(remaining) - i - 1)

        print(f"{result['count']} bars, {iv_count} IV bars ({elapsed:.0f}s, ETA ~{eta/60:.0f}min)", flush=True)

        manifest.setdefault("symbols", {})[sym] = {
            "completed": True,
            "bars": result["count"],
            "iv_bars": iv_count,
            "path": str(parquet_path),
            "downloaded": datetime.now().isoformat(),
        }
        save_manifest(manifest)

    ib.disconnect()
    total = time.time() - start
    n_done = sum(1 for v in manifest.get("symbols", {}).values() if v.get("completed"))
    print(f"\nDone: {n_done} symbols in {total/60:.1f} minutes")
    print(f"Data at: {ATLAS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=200, help="Max symbols to download")
    parser.add_argument("--port", type=int, default=4002, help="IBKR port")
    parser.add_argument("--client-id", type=int, default=30, help="IBKR client ID")
    parser.add_argument("--sleep", type=float, default=12, help="Seconds between requests")
    parser.add_argument("--resume", action="store_true", help="Resume from manifest")
    args = parser.parse_args()

    asyncio.run(main(args))
