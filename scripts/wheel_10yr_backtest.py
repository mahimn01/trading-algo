#!/usr/bin/env python3
"""
10-Year Wheel Strategy Backtest — 50 Stocks Across All Sectors

Runs the Wheel with FIXED a-priori config (delta=0.30, DTE=45, PT=50%, roll_dte=0,
stop_loss=0, SMA50) with volatility skew and dividend yields on 10Y yfinance data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_algo.broker.base import Bar
from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
from trading_algo.quant_core.strategies.options.options_backtester import run_options_backtest

# ---------------------------------------------------------------------------
# Universe: 50 stocks + 3 ETFs across sectors
# ---------------------------------------------------------------------------

SYMBOLS_BY_SECTOR: dict[str, list[str]] = {
    "Tech": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "INTC", "AMD", "CRM", "ADBE"],
    "Finance": ["JPM", "BAC", "GS", "C", "WFC", "MS", "BLK", "SCHW", "AXP", "V"],
    "Consumer": ["KO", "PG", "MCD", "WMT", "HD", "NKE", "SBUX", "TGT", "COST", "DIS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Health": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
    "Industrial": ["CAT", "BA", "GE", "HON", "UPS"],
    "REITs/Utils": ["O", "AMT", "NEE", "SO", "D"],
    "ETFs": ["SPY", "QQQ", "IWM"],
}

SYMBOL_TO_SECTOR = {sym: sector for sector, syms in SYMBOLS_BY_SECTOR.items() for sym in syms}
ALL_SYMBOLS = list(SYMBOL_TO_SECTOR.keys())

# Approximate annualized dividend yields (as of ~2024-2025)
DIVIDEND_YIELDS: dict[str, float] = {
    # Tech
    "AAPL": 0.005, "MSFT": 0.007, "GOOG": 0.0, "AMZN": 0.0, "META": 0.004,
    "NVDA": 0.0003, "INTC": 0.015, "AMD": 0.0, "CRM": 0.0, "ADBE": 0.0,
    # Finance
    "JPM": 0.022, "BAC": 0.025, "GS": 0.025, "C": 0.035, "WFC": 0.025,
    "MS": 0.035, "BLK": 0.022, "SCHW": 0.013, "AXP": 0.012, "V": 0.007,
    # Consumer
    "KO": 0.03, "PG": 0.025, "MCD": 0.022, "WMT": 0.013, "HD": 0.025,
    "NKE": 0.015, "SBUX": 0.025, "TGT": 0.032, "COST": 0.006, "DIS": 0.0,
    # Energy
    "XOM": 0.035, "CVX": 0.04, "COP": 0.02, "SLB": 0.022, "EOG": 0.03,
    # Health
    "JNJ": 0.032, "PFE": 0.055, "UNH": 0.015, "ABBV": 0.035, "MRK": 0.03,
    # Industrial
    "CAT": 0.016, "BA": 0.0, "GE": 0.006, "HON": 0.02, "UPS": 0.05,
    # REITs/Utils
    "O": 0.05, "AMT": 0.03, "NEE": 0.028, "SO": 0.035, "D": 0.045,
    # ETFs
    "SPY": 0.013, "QQQ": 0.006, "IWM": 0.013,
}

# Fixed a-priori config — NOT optimized
BASE_CONFIG = WheelConfig(
    initial_capital=100_000.0,
    put_delta=0.30,
    call_delta=0.30,
    target_dte=45,
    profit_target=0.50,
    roll_dte=0,
    stop_loss=0.0,
    trend_sma_period=50,
    min_iv_rank=25.0,
    min_premium_pct=0.005,
    cash_reserve_pct=0.20,
    risk_free_rate=0.045,
    skew_slope=0.8,
    commission_per_contract=0.90,
    commission_per_share=0.005,
    bid_ask_slip_per_share=0.05,
)


def yf_to_bars(ticker: str, period: str = "10y") -> list[Bar]:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return []
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    bars: list[Bar] = []
    for idx, row in df.iterrows():
        try:
            o = float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"])
            h = float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"])
            l = float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"])
            c = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
            v = float(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else float(row["Volume"])
        except Exception:
            continue
        if np.isnan(c):
            continue
        bars.append(Bar(timestamp_epoch_s=idx.timestamp(), open=o, high=h, low=l, close=c, volume=v))
    return bars


def run_single(symbol: str) -> dict | None:
    sector = SYMBOL_TO_SECTOR[symbol]
    try:
        bars = yf_to_bars(symbol, period="10y")
    except Exception as e:
        print(f"  {symbol}: DOWNLOAD ERROR — {e}")
        return None

    if len(bars) < 120:
        print(f"  {symbol}: SKIP — only {len(bars)} bars")
        return None

    div_yield = DIVIDEND_YIELDS.get(symbol, 0.0)
    cfg = WheelConfig(
        initial_capital=BASE_CONFIG.initial_capital,
        put_delta=BASE_CONFIG.put_delta,
        call_delta=BASE_CONFIG.call_delta,
        target_dte=BASE_CONFIG.target_dte,
        profit_target=BASE_CONFIG.profit_target,
        roll_dte=BASE_CONFIG.roll_dte,
        stop_loss=BASE_CONFIG.stop_loss,
        trend_sma_period=BASE_CONFIG.trend_sma_period,
        min_iv_rank=BASE_CONFIG.min_iv_rank,
        min_premium_pct=BASE_CONFIG.min_premium_pct,
        cash_reserve_pct=BASE_CONFIG.cash_reserve_pct,
        risk_free_rate=BASE_CONFIG.risk_free_rate,
        dividend_yield=div_yield,
        skew_slope=BASE_CONFIG.skew_slope,
        commission_per_contract=BASE_CONFIG.commission_per_contract,
        commission_per_share=BASE_CONFIG.commission_per_share,
        bid_ask_slip_per_share=BASE_CONFIG.bid_ask_slip_per_share,
    )
    strategy = WheelStrategy(cfg)
    try:
        report = run_options_backtest(strategy, bars, symbol)
    except Exception as e:
        print(f"  {symbol}: BACKTEST ERROR — {e}")
        return None

    s = report.summary
    return {
        "symbol": symbol,
        "sector": sector,
        "strat_ret": s["total_return_pct"],
        "bnh_ret": report.benchmark_return_pct,
        "vs_bnh": s["total_return_pct"] - report.benchmark_return_pct,
        "sharpe": s["sharpe_ratio"],
        "max_dd": s["max_drawdown_pct"],
        "win_rate": s["win_rate"],
        "trades": s["total_trades"],
        "cycles": s["wheel_cycles"],
        "final_eq": s["final_equity"],
        "bnh_sharpe": report.benchmark_sharpe,
        "bnh_dd": report.benchmark_max_dd_pct,
        "ann_ret": report.annualized_return(),
        "ann_bnh": report.benchmark_annualized(),
        "days": s["days"],
        "div_yield": div_yield,
    }


def main() -> None:
    print("=" * 140)
    print("  WHEEL STRATEGY — 10-YEAR BACKTEST (50 STOCKS + 3 ETFs ACROSS ALL SECTORS)")
    print("  Config: delta=0.30, DTE=45, PT=50%, roll_dte=0, stop_loss=0, SMA50 | Vol skew ON | Dividends ON")
    print(f"  Capital: $100,000 per symbol | Period: 10Y daily via yfinance")
    print("=" * 140)
    print()

    all_results: list[dict] = []
    total_t0 = time.time()

    for sector, tickers in SYMBOLS_BY_SECTOR.items():
        print(f"[{sector}]")
        for sym in tickers:
            t0 = time.time()
            result = run_single(sym)
            elapsed = time.time() - t0
            if result:
                all_results.append(result)
                arrow = "+" if result["strat_ret"] > 0 else ""
                print(f"  {sym:<6} {arrow}{result['strat_ret']:>8.2f}%  Sharpe: {result['sharpe']:>6.3f}  "
                      f"MaxDD: {result['max_dd']:>6.2f}%  BnH: {result['bnh_ret']:>+8.2f}%  [{elapsed:.1f}s]")
            else:
                print(f"  {sym:<6} FAILED [{elapsed:.1f}s]")
        print()

    total_elapsed = time.time() - total_t0
    print(f"Total runtime: {total_elapsed:.0f}s")
    print()

    if not all_results:
        print("No results.")
        return

    n = len(all_results)

    # ── Main results table sorted by Sharpe ──
    print("=" * 160)
    print(f"{'Symbol':<7} {'Sector':<14} {'Ret%':>8} {'Ann%':>8} {'Sharpe':>8} {'MaxDD%':>8} "
          f"{'WinRate':>8} {'Trades':>7} {'Cycles':>7} {'BnH Ret%':>10} {'BnH Ann%':>10} {'vs BnH':>8} {'DivYld':>7}")
    print("-" * 160)

    for r in sorted(all_results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['symbol']:<7} {r['sector']:<14} {r['strat_ret']:>8.2f} {r['ann_ret']:>8.2f} "
              f"{r['sharpe']:>8.3f} {r['max_dd']:>8.2f} {r['win_rate']:>8.1f} {r['trades']:>7d} "
              f"{r['cycles']:>7d} {r['bnh_ret']:>10.2f} {r['ann_bnh']:>10.2f} {r['vs_bnh']:>+8.2f} "
              f"{r['div_yield']*100:>6.1f}%")
    print("-" * 160)

    # ── Summary statistics ──
    rets = [r["strat_ret"] for r in all_results]
    sharpes = [r["sharpe"] for r in all_results]
    dds = [r["max_dd"] for r in all_results]
    wrs = [r["win_rate"] for r in all_results]
    bnh_rets = [r["bnh_ret"] for r in all_results]
    ann_rets = [r["ann_ret"] for r in all_results]

    print()
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"  {'Metric':<25} {'Mean':>10} {'Median':>10} {'StdDev':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Total Return %':<25} {np.mean(rets):>10.2f} {np.median(rets):>10.2f} {np.std(rets):>10.2f}")
    print(f"  {'Annualized Return %':<25} {np.mean(ann_rets):>10.2f} {np.median(ann_rets):>10.2f} {np.std(ann_rets):>10.2f}")
    print(f"  {'Sharpe Ratio':<25} {np.mean(sharpes):>10.3f} {np.median(sharpes):>10.3f} {np.std(sharpes):>10.3f}")
    print(f"  {'Max Drawdown %':<25} {np.mean(dds):>10.2f} {np.median(dds):>10.2f} {np.std(dds):>10.2f}")
    print(f"  {'Win Rate %':<25} {np.mean(wrs):>10.1f} {np.median(wrs):>10.1f} {np.std(wrs):>10.1f}")
    print(f"  {'BnH Return %':<25} {np.mean(bnh_rets):>10.2f} {np.median(bnh_rets):>10.2f} {np.std(bnh_rets):>10.2f}")
    print()

    # ── By sector ──
    print("BY SECTOR AVERAGES")
    print("=" * 130)
    print(f"  {'Sector':<14} {'N':>3} {'Avg Ret%':>10} {'Avg Ann%':>10} {'Avg Sharpe':>11} {'Avg MaxDD%':>11} "
          f"{'Avg WR%':>8} {'Avg BnH%':>10} {'vs BnH':>8}")
    print(f"  {'-'*14} {'-'*3} {'-'*10} {'-'*10} {'-'*11} {'-'*11} {'-'*8} {'-'*10} {'-'*8}")
    for sector in SYMBOLS_BY_SECTOR:
        sr = [r for r in all_results if r["sector"] == sector]
        if not sr:
            continue
        ar = np.mean([r["strat_ret"] for r in sr])
        aa = np.mean([r["ann_ret"] for r in sr])
        ash_ = np.mean([r["sharpe"] for r in sr])
        add = np.mean([r["max_dd"] for r in sr])
        awr = np.mean([r["win_rate"] for r in sr])
        abnh = np.mean([r["bnh_ret"] for r in sr])
        print(f"  {sector:<14} {len(sr):>3} {ar:>10.2f} {aa:>10.2f} {ash_:>11.3f} {add:>11.2f} "
              f"{awr:>8.1f} {abnh:>10.2f} {ar - abnh:>+8.2f}")
    print()

    # ── Key counts ──
    profitable = sum(1 for r in all_results if r["strat_ret"] > 0)
    beat_rf = sum(1 for r in all_results if r["ann_ret"] > 4.5)
    beat_bnh = sum(1 for r in all_results if r["strat_ret"] > r["bnh_ret"])
    positive_sharpe = sum(1 for r in all_results if r["sharpe"] > 0)

    print("KEY METRICS")
    print("=" * 60)
    print(f"  Profitable (>0%):           {profitable}/{n} ({profitable / n * 100:.0f}%)")
    print(f"  Beat risk-free (ann >4.5%): {beat_rf}/{n} ({beat_rf / n * 100:.0f}%)")
    print(f"  Beat buy-and-hold:          {beat_bnh}/{n} ({beat_bnh / n * 100:.0f}%)")
    print(f"  Positive Sharpe:            {positive_sharpe}/{n} ({positive_sharpe / n * 100:.0f}%)")
    print()

    # ── Best 5 and Worst 5 ──
    sorted_by_sharpe = sorted(all_results, key=lambda r: r["sharpe"], reverse=True)

    print("BEST 5 (by Sharpe)")
    print("-" * 100)
    for r in sorted_by_sharpe[:5]:
        print(f"  {r['symbol']:<7} ({r['sector']:<12}) Ret: {r['strat_ret']:>+8.2f}%  Ann: {r['ann_ret']:>+8.2f}%  "
              f"Sharpe: {r['sharpe']:.3f}  MaxDD: {r['max_dd']:.2f}%  WR: {r['win_rate']:.1f}%  "
              f"BnH: {r['bnh_ret']:>+.2f}%")
    print()

    print("WORST 5 (by Sharpe)")
    print("-" * 100)
    for r in sorted_by_sharpe[-5:]:
        print(f"  {r['symbol']:<7} ({r['sector']:<12}) Ret: {r['strat_ret']:>+8.2f}%  Ann: {r['ann_ret']:>+8.2f}%  "
              f"Sharpe: {r['sharpe']:.3f}  MaxDD: {r['max_dd']:.2f}%  WR: {r['win_rate']:.1f}%  "
              f"BnH: {r['bnh_ret']:>+.2f}%")
    print()

    # ── Simulation assumptions ──
    print("SIMULATION ASSUMPTIONS")
    print("=" * 60)
    print("  IV estimation:       Dynamic IV/RV ratio (regime-dependent)")
    print("  Vol model:           BSM with parametric volatility skew (slope=0.8)")
    print("  Dividends:           Per-symbol annualized yield in BSM pricing")
    print("  Bid-ask slippage:    Fixed $0.05/share")
    print("  Commission+exchange: $0.90/contract")
    print("  Assignment:          At expiry only (no early assignment)")
    print("  Data:                10Y daily via yfinance (auto-adjusted)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
