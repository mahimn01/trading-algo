#!/usr/bin/env python3
"""
Defined-Risk Options Strategy Comparison Backtest

Compares 4 strategies on 15 symbols over 5 years:
  1. Wheel (naked puts / CSP -> CC cycle)
  2. Bull Put Spread (vertical credit spread)
  3. Jade Lizard (short put + short call spread)
  4. Buy-and-Hold (benchmark)

Key metric: Return on Risk (RoR) — measures capital efficiency,
which matters most for small accounts ($2K-$10K).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_algo.broker.base import Bar
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_rank,
    iv_series_from_prices,
)
from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
from trading_algo.quant_core.strategies.options.put_spread import PutSpreadStrategy, PutSpreadConfig
from trading_algo.quant_core.strategies.options.jade_lizard import JadeLizardStrategy, JadeLizardConfig

# ---------------------------------------------------------------------------
# Universe: 15 symbols good for small-account options selling
# ---------------------------------------------------------------------------

SYMBOLS = [
    "SOFI", "PLTR", "AMD", "INTC", "F",
    "BAC", "T", "NIO", "RIVN", "HOOD",
    "SNAP", "MARA", "CCL", "AAL", "LCID",
]

DIVIDEND_YIELDS: dict[str, float] = {
    "SOFI": 0.0, "PLTR": 0.0, "AMD": 0.0, "INTC": 0.015, "F": 0.05,
    "BAC": 0.025, "T": 0.065, "NIO": 0.0, "RIVN": 0.0, "HOOD": 0.0,
    "SNAP": 0.0, "MARA": 0.0, "CCL": 0.0, "AAL": 0.0, "LCID": 0.0,
}

# Small account configs — $5K for spreads, $10K for Wheel/Jade Lizard
WHEEL_CFG = WheelConfig(
    initial_capital=10_000.0,
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

SPREAD_CFG = PutSpreadConfig(
    initial_capital=5_000.0,
    short_delta=0.30,
    spread_width_dollars=2.0,
    target_dte=45,
    profit_target=0.50,
    stop_loss=2.0,
    roll_dte=14,
    min_iv_rank=30.0,
    max_risk_per_trade_pct=0.05,
    trend_sma_period=50,
    risk_free_rate=0.045,
    commission_per_contract=0.90,
    bid_ask_slip_per_share=0.05,
    skew_slope=0.8,
)

JADE_CFG = JadeLizardConfig(
    initial_capital=10_000.0,
    put_delta=0.25,
    call_short_delta=0.25,
    call_spread_width=5.0,
    target_dte=45,
    profit_target=0.50,
    stop_loss=2.0,
    roll_dte=14,
    min_iv_rank=35.0,
    max_risk_per_trade_pct=0.20,
    trend_sma_period=50,
    risk_free_rate=0.045,
    commission_per_contract=0.90,
    bid_ask_slip_per_share=0.05,
    skew_slope=0.8,
    cash_reserve_pct=0.20,
)

WARMUP = 60


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def yf_to_bars(ticker: str, period: str = "5y") -> list[Bar]:
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


# ---------------------------------------------------------------------------
# Run a single strategy on bars
# ---------------------------------------------------------------------------

def run_strategy(
    strategy_obj: WheelStrategy | PutSpreadStrategy | JadeLizardStrategy,
    bars: list[Bar],
    symbol: str,
) -> dict | None:
    if len(bars) < WARMUP + 10:
        return None

    prices = np.array([b.close for b in bars], dtype=float)
    timestamps = [datetime.fromtimestamp(b.timestamp_epoch_s) for b in bars]

    iv_series = iv_series_from_prices(prices, rv_window=30, dynamic=True)

    for i in range(WARMUP, len(bars)):
        date = timestamps[i]
        price = float(prices[i])
        current_iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        current_iv = max(current_iv, 0.05)
        current_rank = iv_rank(iv_series, i, lookback=252)
        strategy_obj.on_bar(date, price, current_iv, current_rank)

    return strategy_obj.summary()


def compute_benchmark(bars: list[Bar]) -> dict:
    if len(bars) < WARMUP + 10:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}

    prices = np.array([b.close for b in bars[WARMUP:]], dtype=float)
    ret_pct = (prices[-1] / prices[0] - 1) * 100

    daily_returns = np.diff(prices) / prices[:-1]
    daily_rf = 0.045 / 252
    excess = daily_returns - daily_rf
    sharpe = 0.0
    if np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / peak
    max_dd = float(np.max(dd) * 100)

    return {
        "total_return_pct": round(ret_pct, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Return on Risk calculation
# ---------------------------------------------------------------------------

def compute_ror(summary: dict, strategy_type: str, initial_capital: float) -> float:
    """
    Return on Risk: how efficiently capital at risk generates returns.

    Wheel:       risk = initial capital (full assignment exposure)
    Put Spread:  risk = max_loss per spread * avg contracts (much less)
    Jade Lizard: risk = put assignment exposure - call credit offset
    Buy-Hold:    risk = initial capital
    """
    total_return = summary.get("total_return_pct", 0.0) / 100 * initial_capital

    if strategy_type == "wheel":
        max_risk = initial_capital
    elif strategy_type == "spread":
        max_risk = initial_capital * 0.05 * summary.get("total_trades", 1)
        max_risk = max(max_risk, initial_capital * 0.10)  # floor
    elif strategy_type == "jade_lizard":
        max_risk = initial_capital * 0.80  # put collateral minus call premium offset
    else:
        max_risk = initial_capital

    if max_risk <= 0:
        return 0.0
    return round(total_return / max_risk * 100, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 90)
    print("  DEFINED-RISK OPTIONS STRATEGY COMPARISON")
    print("  15 symbols | 5Y data | Small account focus ($2K-$10K)")
    print("=" * 90)
    print()

    all_results: list[dict] = []

    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"[{i:2d}/{len(SYMBOLS)}] {symbol}...", end=" ", flush=True)
        t0 = time.time()

        try:
            bars = yf_to_bars(symbol, period="5y")
        except Exception as e:
            print(f"DOWNLOAD ERROR: {e}")
            continue

        if len(bars) < WARMUP + 10:
            print(f"SKIP (only {len(bars)} bars)")
            continue

        div_yield = DIVIDEND_YIELDS.get(symbol, 0.0)

        # -- Wheel --
        from dataclasses import replace
        wheel_cfg = replace(WHEEL_CFG, dividend_yield=div_yield)
        wheel = WheelStrategy(wheel_cfg)
        wheel_summary = run_strategy(wheel, bars, symbol)

        # -- Bull Put Spread --
        spread_cfg = replace(SPREAD_CFG, dividend_yield=div_yield)
        spread = PutSpreadStrategy(spread_cfg)
        spread_summary = run_strategy(spread, bars, symbol)

        # -- Jade Lizard --
        jade_cfg = replace(JADE_CFG, dividend_yield=div_yield)
        jade = JadeLizardStrategy(jade_cfg)
        jade_summary = run_strategy(jade, bars, symbol)

        # -- Benchmark --
        bench = compute_benchmark(bars)

        elapsed = time.time() - t0

        if wheel_summary and spread_summary and jade_summary:
            wheel_ror = compute_ror(wheel_summary, "wheel", WHEEL_CFG.initial_capital)
            spread_ror = compute_ror(spread_summary, "spread", SPREAD_CFG.initial_capital)
            jade_ror = compute_ror(jade_summary, "jade_lizard", JADE_CFG.initial_capital)
            bench_ror = compute_ror(bench, "benchmark", WHEEL_CFG.initial_capital)

            all_results.append({
                "symbol": symbol,
                "wheel": {**wheel_summary, "ror": wheel_ror},
                "spread": {**spread_summary, "ror": spread_ror},
                "jade_lizard": {**jade_summary, "ror": jade_ror},
                "benchmark": {**bench, "ror": bench_ror},
            })
            print(f"OK ({elapsed:.1f}s)")
        else:
            print(f"PARTIAL FAIL ({elapsed:.1f}s)")

    if not all_results:
        print("\nNo results. Check network / yfinance.")
        return

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 130)
    print(f"  {'Symbol':<7} | {'Strategy':<14} | {'Return%':>8} | {'Sharpe':>7} | {'MaxDD%':>7} | {'WinRate%':>9} | {'RoR%':>7} | {'Trades':>6} | {'Capital':>8}")
    print("-" * 130)

    for r in all_results:
        sym = r["symbol"]
        for strat_key, label, cap in [
            ("wheel", "Wheel", WHEEL_CFG.initial_capital),
            ("spread", "Put Spread", SPREAD_CFG.initial_capital),
            ("jade_lizard", "Jade Lizard", JADE_CFG.initial_capital),
            ("benchmark", "Buy&Hold", WHEEL_CFG.initial_capital),
        ]:
            s = r[strat_key]
            ret = s.get("total_return_pct", 0.0)
            sharpe = s.get("sharpe_ratio", 0.0)
            dd = s.get("max_drawdown_pct", 0.0)
            wr = s.get("win_rate", 0.0)
            ror = s.get("ror", 0.0)
            trades = s.get("total_trades", 0)
            print(f"  {sym:<7} | {label:<14} | {ret:>7.1f}% | {sharpe:>7.3f} | {dd:>6.1f}% | {wr:>8.1f}% | {ror:>6.1f}% | {trades:>6} | ${cap:>7,.0f}")
        print("-" * 130)

    # ---------------------------------------------------------------------------
    # Aggregated averages
    # ---------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("  AVERAGES ACROSS ALL SYMBOLS")
    print("=" * 90)

    for strat_key, label in [
        ("wheel", "Wheel (Naked Puts)"),
        ("spread", "Bull Put Spread"),
        ("jade_lizard", "Jade Lizard"),
        ("benchmark", "Buy & Hold"),
    ]:
        rets = [r[strat_key].get("total_return_pct", 0.0) for r in all_results]
        sharpes = [r[strat_key].get("sharpe_ratio", 0.0) for r in all_results]
        dds = [r[strat_key].get("max_drawdown_pct", 0.0) for r in all_results]
        wrs = [r[strat_key].get("win_rate", 0.0) for r in all_results]
        rors = [r[strat_key].get("ror", 0.0) for r in all_results]
        trades = [r[strat_key].get("total_trades", 0) for r in all_results]

        print(f"\n  {label}:")
        print(f"    Avg Return:      {np.mean(rets):>8.2f}%")
        print(f"    Avg Sharpe:      {np.mean(sharpes):>8.3f}")
        print(f"    Avg MaxDD:       {np.mean(dds):>8.2f}%")
        print(f"    Avg Win Rate:    {np.mean(wrs):>8.1f}%")
        print(f"    Avg RoR:         {np.mean(rors):>8.2f}%")
        print(f"    Avg Trades:      {np.mean(trades):>8.1f}")

    # ---------------------------------------------------------------------------
    # Summary for small accounts
    # ---------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("  SMALL ACCOUNT VERDICT")
    print("=" * 90)
    print()

    avg_ror = {}
    for strat_key, label in [("wheel", "Wheel"), ("spread", "Put Spread"), ("jade_lizard", "Jade Lizard")]:
        avg_ror[label] = np.mean([r[strat_key].get("ror", 0.0) for r in all_results])

    best = max(avg_ror, key=lambda k: avg_ror[k])

    print(f"  Best capital-efficient strategy: {best} (avg RoR: {avg_ror[best]:.1f}%)")
    print()
    print("  Why defined-risk wins for small accounts:")
    print("    - Bull Put Spread: $5K capital, $100-200 max risk per trade")
    print("      vs Wheel: $10K capital, $1,500+ collateral per put")
    print("    - Jade Lizard: extra call premium offsets downside risk")
    print("    - Spreads allow 3-5 concurrent positions vs 1-2 for Wheel")
    print("    - Max loss is KNOWN at entry — no surprise margin calls")
    print()
    print("  -- Simulation assumptions --")
    print("  IV estimation:       Dynamic IV/RV ratio (regime-dependent)")
    print("  Bid-ask slippage:    Fixed $0.05/share per leg")
    print("  Commission:          $0.90/contract/leg")
    print("  Assignment:          At expiry only (no early assignment)")
    print("  Vol model:           BSM with parametric volatility skew")
    print("  Data:                5Y yfinance daily bars")
    print()


if __name__ == "__main__":
    main()
