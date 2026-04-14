#!/usr/bin/env python3
"""
Hybrid Regime Strategy Backtest — 20 symbols, 5Y, three-way comparison

Runs Buy-and-Hold vs Base Wheel vs Hybrid Regime on each symbol.
Reports per-symbol comparison and aggregate regime breakdown.
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

from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
from trading_algo.quant_core.strategies.options.hybrid_regime import HybridRegimeStrategy, HybridRegimeConfig
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_series_from_prices,
    iv_rank as compute_iv_rank,
)

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "JPM", "BAC", "GS", "KO", "PG", "WMT",
    "XOM", "JNJ", "T", "F", "INTC", "PYPL", "SPY", "QQQ", "AMD", "NIO",
]

DIVIDEND_YIELDS: dict[str, float] = {
    "AAPL": 0.005, "MSFT": 0.007, "NVDA": 0.0003, "GOOG": 0.0, "JPM": 0.022,
    "BAC": 0.025, "GS": 0.025, "KO": 0.03, "PG": 0.025, "WMT": 0.013,
    "XOM": 0.035, "JNJ": 0.032, "T": 0.065, "F": 0.05, "INTC": 0.015,
    "PYPL": 0.0, "SPY": 0.013, "QQQ": 0.006, "AMD": 0.0, "NIO": 0.0,
}

INITIAL_CAPITAL = 100_000.0


def _safe_float(val) -> float:
    if hasattr(val, "iloc"):
        return float(val.iloc[0])
    return float(val)


def download_prices(symbol: str, period: str = "5y") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[datetime]] | None:
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  {symbol}: DOWNLOAD ERROR — {e}")
        return None

    if df.empty:
        print(f"  {symbol}: NO DATA")
        return None

    # Handle multi-level columns from yfinance
    if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    dates_list: list[datetime] = []

    for idx, row in df.iterrows():
        try:
            c = _safe_float(row["Close"])
            h = _safe_float(row["High"])
            lo = _safe_float(row["Low"])
        except Exception:
            continue
        if np.isnan(c) or c <= 0:
            continue
        closes.append(c)
        highs.append(h if not np.isnan(h) else c)
        lows.append(lo if not np.isnan(lo) else c)
        dates_list.append(idx.to_pydatetime())

    if len(closes) < 120:
        print(f"  {symbol}: SKIP — only {len(closes)} bars")
        return None

    return np.array(closes), np.array(highs), np.array(lows), dates_list


def compute_benchmark(prices: np.ndarray, risk_free_rate: float = 0.045) -> dict:
    if len(prices) < 2:
        return {"ret": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    ret_pct = (prices[-1] / prices[0] - 1) * 100
    daily_returns = np.diff(prices) / prices[:-1]
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    sharpe = 0.0
    if len(excess) > 1 and np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / peak
    max_dd = float(np.max(dd) * 100)

    return {"ret": round(ret_pct, 2), "sharpe": round(sharpe, 3), "max_dd": round(max_dd, 2)}


def run_strategy(
    strategy,
    prices: np.ndarray,
    dates: list[datetime],
    warmup: int = 60,
    rv_window: int = 30,
    iv_lookback: int = 252,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
) -> dict:
    iv_series = iv_series_from_prices(prices, rv_window, dynamic=True)
    pass_hl = highs is not None and lows is not None and isinstance(strategy, HybridRegimeStrategy)

    for i in range(warmup, len(prices)):
        date = dates[i]
        price = float(prices[i])
        current_iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        current_iv = max(current_iv, 0.05)
        rank = compute_iv_rank(iv_series, i, iv_lookback)
        if pass_hl:
            strategy.on_bar(date, price, current_iv, rank, high=float(highs[i]), low=float(lows[i]))
        else:
            strategy.on_bar(date, price, current_iv, rank)

    return strategy.summary()


def run_symbol(symbol: str) -> dict | None:
    data = download_prices(symbol, period="5y")
    if data is None:
        return None

    prices, highs, lows, dates = data
    div_yield = DIVIDEND_YIELDS.get(symbol, 0.0)
    warmup = 60

    # 1. Buy-and-hold benchmark
    bnh = compute_benchmark(prices[warmup:])

    # 2. Base Wheel
    wheel_cfg = WheelConfig(
        initial_capital=INITIAL_CAPITAL,
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
        dividend_yield=div_yield,
        skew_slope=0.8,
        commission_per_contract=0.90,
        commission_per_share=0.005,
        bid_ask_slip_per_share=0.05,
    )
    wheel = WheelStrategy(wheel_cfg)
    wheel_summary = run_strategy(wheel, prices, dates, warmup=warmup, highs=highs, lows=lows)

    # 3. Hybrid Regime
    hybrid_cfg = HybridRegimeConfig(
        initial_capital=INITIAL_CAPITAL,
        put_delta=0.30,
        call_delta=0.30,
        target_dte=45,
        profit_target=0.50,
        roll_dte=0,
        stop_loss=0.0,
        min_iv_rank=25.0,
        min_premium_pct=0.005,
        cash_reserve_pct=0.20,
        risk_free_rate=0.045,
        dividend_yield=div_yield,
        skew_slope=0.8,
        commission_per_contract=0.90,
        commission_per_share=0.005,
        bid_ask_slip_per_share=0.05,
        adx_period=14,
        adx_trend_threshold=25.0,
        adx_range_threshold=20.0,
        sma_period=50,
        sma_slope_window=20,
        regime_stability_days=5,
        uptrend_delta=0.25,
        range_delta=0.35,
        allow_stock_purchase=True,
    )
    hybrid = HybridRegimeStrategy(hybrid_cfg)
    hybrid_summary = run_strategy(hybrid, prices, dates, warmup=warmup, highs=highs, lows=lows)

    return {
        "symbol": symbol,
        "bnh_ret": bnh["ret"],
        "bnh_sharpe": bnh["sharpe"],
        "bnh_dd": bnh["max_dd"],
        "wheel_ret": wheel_summary["total_return_pct"],
        "wheel_sharpe": wheel_summary["sharpe_ratio"],
        "wheel_dd": wheel_summary["max_drawdown_pct"],
        "hybrid_ret": hybrid_summary["total_return_pct"],
        "hybrid_sharpe": hybrid_summary["sharpe_ratio"],
        "hybrid_dd": hybrid_summary["max_drawdown_pct"],
        "regime_pct": hybrid_summary.get("regime_pct", {}),
        "regime_transitions": hybrid_summary.get("regime_transitions", 0),
        "hybrid_trades": hybrid_summary.get("total_trades", 0),
        "hybrid_win_rate": hybrid_summary.get("win_rate", 0.0),
        "hybrid_net_premium": hybrid_summary.get("net_premium", 0.0),
        "wheel_trades": wheel_summary.get("total_trades", 0),
        "wheel_win_rate": wheel_summary.get("win_rate", 0.0),
    }


def main() -> None:
    print("=" * 160)
    print("  HYBRID REGIME STRATEGY — 5-YEAR BACKTEST (20 SYMBOLS)")
    print("  Comparing: Buy-and-Hold vs Base Wheel vs Hybrid Regime")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f} per symbol")
    print("=" * 160)
    print()

    results: list[dict] = []
    t0_total = time.time()

    for sym in SYMBOLS:
        t0 = time.time()
        result = run_symbol(sym)
        elapsed = time.time() - t0
        if result:
            results.append(result)
            best = max(result["bnh_ret"], result["wheel_ret"], result["hybrid_ret"])
            winner = "BnH" if best == result["bnh_ret"] else ("Wheel" if best == result["wheel_ret"] else "Hybrid")
            print(f"  {sym:<6} BnH: {result['bnh_ret']:>+8.2f}%  Wheel: {result['wheel_ret']:>+8.2f}%  "
                  f"Hybrid: {result['hybrid_ret']:>+8.2f}%  Winner: {winner:<7} [{elapsed:.1f}s]")
        else:
            print(f"  {sym:<6} FAILED [{elapsed:.1f}s]")

    total_elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {total_elapsed:.0f}s\n")

    if not results:
        print("No results.")
        return

    n = len(results)

    # ── Per-symbol comparison table ──
    print("=" * 180)
    print(f"{'Symbol':<7} {'BnH Ret%':>9} {'Wheel Ret%':>11} {'Hybrid Ret%':>12} "
          f"{'BnH Shrp':>9} {'Whl Shrp':>9} {'Hyb Shrp':>9} "
          f"{'BnH DD%':>8} {'Whl DD%':>8} {'Hyb DD%':>8} "
          f"{'Regimes':>8} {'Winner':>8}")
    print("-" * 180)

    for r in sorted(results, key=lambda x: x["hybrid_ret"], reverse=True):
        rets = {"BnH": r["bnh_ret"], "Wheel": r["wheel_ret"], "Hybrid": r["hybrid_ret"]}
        winner = max(rets, key=rets.get)  # type: ignore
        print(f"{r['symbol']:<7} {r['bnh_ret']:>+9.2f} {r['wheel_ret']:>+11.2f} {r['hybrid_ret']:>+12.2f} "
              f"{r['bnh_sharpe']:>9.3f} {r['wheel_sharpe']:>9.3f} {r['hybrid_sharpe']:>9.3f} "
              f"{r['bnh_dd']:>8.2f} {r['wheel_dd']:>8.2f} {r['hybrid_dd']:>8.2f} "
              f"{r['regime_transitions']:>8d} {winner:>8}")
    print("-" * 180)
    print()

    # ── Summary: which strategy wins each metric ──
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 90)

    bnh_rets = [r["bnh_ret"] for r in results]
    wheel_rets = [r["wheel_ret"] for r in results]
    hybrid_rets = [r["hybrid_ret"] for r in results]
    bnh_sharpes = [r["bnh_sharpe"] for r in results]
    wheel_sharpes = [r["wheel_sharpe"] for r in results]
    hybrid_sharpes = [r["hybrid_sharpe"] for r in results]
    bnh_dds = [r["bnh_dd"] for r in results]
    wheel_dds = [r["wheel_dd"] for r in results]
    hybrid_dds = [r["hybrid_dd"] for r in results]

    print(f"  {'Metric':<25} {'Buy&Hold':>12} {'Wheel':>12} {'Hybrid':>12}  {'Best':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}  {'-'*8}")

    metrics = [
        ("Avg Return %", np.mean(bnh_rets), np.mean(wheel_rets), np.mean(hybrid_rets), "max"),
        ("Med Return %", np.median(bnh_rets), np.median(wheel_rets), np.median(hybrid_rets), "max"),
        ("Avg Sharpe", np.mean(bnh_sharpes), np.mean(wheel_sharpes), np.mean(hybrid_sharpes), "max"),
        ("Med Sharpe", np.median(bnh_sharpes), np.median(wheel_sharpes), np.median(hybrid_sharpes), "max"),
        ("Avg MaxDD %", np.mean(bnh_dds), np.mean(wheel_dds), np.mean(hybrid_dds), "min"),
        ("Med MaxDD %", np.median(bnh_dds), np.median(wheel_dds), np.median(hybrid_dds), "min"),
    ]

    for name, bv, wv, hv, direction in metrics:
        vals = {"BnH": bv, "Wheel": wv, "Hybrid": hv}
        if direction == "max":
            best = max(vals, key=vals.get)  # type: ignore
        else:
            best = min(vals, key=vals.get)  # type: ignore
        print(f"  {name:<25} {bv:>12.2f} {wv:>12.2f} {hv:>12.2f}  {best:>8}")

    print()

    # Win counts
    bnh_wins_ret = sum(1 for r in results if r["bnh_ret"] >= r["wheel_ret"] and r["bnh_ret"] >= r["hybrid_ret"])
    wheel_wins_ret = sum(1 for r in results if r["wheel_ret"] > r["bnh_ret"] and r["wheel_ret"] >= r["hybrid_ret"])
    hybrid_wins_ret = sum(1 for r in results if r["hybrid_ret"] > r["bnh_ret"] and r["hybrid_ret"] > r["wheel_ret"])

    bnh_wins_sharpe = sum(1 for r in results if r["bnh_sharpe"] >= r["wheel_sharpe"] and r["bnh_sharpe"] >= r["hybrid_sharpe"])
    wheel_wins_sharpe = sum(1 for r in results if r["wheel_sharpe"] > r["bnh_sharpe"] and r["wheel_sharpe"] >= r["hybrid_sharpe"])
    hybrid_wins_sharpe = sum(1 for r in results if r["hybrid_sharpe"] > r["bnh_sharpe"] and r["hybrid_sharpe"] > r["wheel_sharpe"])

    bnh_wins_dd = sum(1 for r in results if r["bnh_dd"] <= r["wheel_dd"] and r["bnh_dd"] <= r["hybrid_dd"])
    wheel_wins_dd = sum(1 for r in results if r["wheel_dd"] < r["bnh_dd"] and r["wheel_dd"] <= r["hybrid_dd"])
    hybrid_wins_dd = sum(1 for r in results if r["hybrid_dd"] < r["bnh_dd"] and r["hybrid_dd"] < r["wheel_dd"])

    print("WIN COUNTS (per symbol)")
    print("=" * 70)
    print(f"  {'Metric':<25} {'BnH':>8} {'Wheel':>8} {'Hybrid':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Best Return'::<25} {bnh_wins_ret:>8} {wheel_wins_ret:>8} {hybrid_wins_ret:>8}")
    print(f"  {'Best Sharpe'::<25} {bnh_wins_sharpe:>8} {wheel_wins_sharpe:>8} {hybrid_wins_sharpe:>8}")
    print(f"  {'Lowest MaxDD'::<25} {bnh_wins_dd:>8} {wheel_wins_dd:>8} {hybrid_wins_dd:>8}")
    print()

    # Hybrid vs BnH / Wheel
    hybrid_beats_bnh = sum(1 for r in results if r["hybrid_ret"] > r["bnh_ret"])
    hybrid_beats_wheel = sum(1 for r in results if r["hybrid_ret"] > r["wheel_ret"])
    avg_margin_vs_bnh = np.mean([r["hybrid_ret"] - r["bnh_ret"] for r in results])
    avg_margin_vs_wheel = np.mean([r["hybrid_ret"] - r["wheel_ret"] for r in results])

    print("HYBRID PERFORMANCE")
    print("=" * 70)
    print(f"  Hybrid beats Buy-and-Hold: {hybrid_beats_bnh}/{n} ({hybrid_beats_bnh/n*100:.0f}%)  avg margin: {avg_margin_vs_bnh:+.2f}%")
    print(f"  Hybrid beats Base Wheel:   {hybrid_beats_wheel}/{n} ({hybrid_beats_wheel/n*100:.0f}%)  avg margin: {avg_margin_vs_wheel:+.2f}%")
    print()

    # ── Regime breakdown ──
    print("REGIME BREAKDOWN (across all symbols)")
    print("=" * 90)

    agg_regime: dict[str, list[float]] = {
        "STRONG_UPTREND": [], "WEAK_UPTREND": [], "RANGE_BOUND": [], "DOWNTREND": [],
    }
    for r in results:
        rp = r.get("regime_pct", {})
        for regime in agg_regime:
            agg_regime[regime].append(rp.get(regime, 0.0))

    print(f"  {'Regime':<20} {'Avg Time%':>10} {'Min%':>8} {'Max%':>8} {'Med%':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for regime, pcts in agg_regime.items():
        if pcts:
            print(f"  {regime:<20} {np.mean(pcts):>10.1f} {np.min(pcts):>8.1f} {np.max(pcts):>8.1f} {np.median(pcts):>8.1f}")

    avg_transitions = np.mean([r["regime_transitions"] for r in results])
    print(f"\n  Avg regime transitions per symbol: {avg_transitions:.1f}")
    print()

    print("SIMULATION ASSUMPTIONS")
    print("=" * 60)
    print("  IV estimation:       Dynamic IV/RV ratio (regime-dependent)")
    print("  Vol model:           BSM with parametric volatility skew (slope=0.8)")
    print("  Bid-ask slippage:    Fixed $0.05/share")
    print("  Commission+exchange: $0.90/contract")
    print("  Assignment:          At expiry only (no early assignment)")
    print("  Data:                5Y daily via yfinance (auto-adjusted)")
    print("  Regime stability:    5-day confirmation filter")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
