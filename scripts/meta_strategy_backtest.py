#!/usr/bin/env python3
"""
Adaptive Meta-Strategy Backtest — 12 symbols, 10Y, five-way comparison

Compares Buy-and-Hold vs Pure Wheel vs Pure PMCC vs Hybrid Regime vs Adaptive Meta
on 10 years of daily data per symbol.

Reports:
  - Per-symbol comparison table
  - Year-by-year income for GLD and SIVR at $12,500 capital
  - Regime breakdown: % of time in each sub-strategy
  - P&L attribution: how much came from each sub-strategy
  - Monthly income figures
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
from trading_algo.quant_core.strategies.options.pmcc import PMCCStrategy, PMCCConfig
from trading_algo.quant_core.strategies.options.hybrid_regime import HybridRegimeStrategy, HybridRegimeConfig
from trading_algo.quant_core.strategies.options.meta_strategy import AdaptiveMetaStrategy, AdaptiveMetaConfig
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_series_from_prices,
    iv_rank as compute_iv_rank,
)


SYMBOLS = ["GLD", "SIVR", "SPY", "AAPL", "MSFT", "JPM", "XOM", "KO", "AMD", "INTC", "BAC", "T"]

DIVIDEND_YIELDS: dict[str, float] = {
    "GLD": 0.0, "SIVR": 0.0, "SPY": 0.013, "AAPL": 0.005, "MSFT": 0.007,
    "JPM": 0.022, "XOM": 0.035, "KO": 0.03, "AMD": 0.0, "INTC": 0.015,
    "BAC": 0.025, "T": 0.065,
}

INITIAL_CAPITAL = 12_500.0


def _safe_float(val) -> float:
    if hasattr(val, "iloc"):
        return float(val.iloc[0])
    return float(val)


def download_prices(
    symbol: str, period: str = "10y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[datetime]] | None:
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  {symbol}: DOWNLOAD ERROR -- {e}")
        return None

    if df.empty:
        print(f"  {symbol}: NO DATA")
        return None

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
        print(f"  {symbol}: SKIP -- only {len(closes)} bars")
        return None

    return np.array(closes), np.array(highs), np.array(lows), dates_list


def compute_benchmark(prices: np.ndarray, initial_capital: float = INITIAL_CAPITAL) -> dict:
    if len(prices) < 2:
        return {"ret": 0.0, "sharpe": 0.0, "max_dd": 0.0, "final_equity": initial_capital}

    ret_pct = (prices[-1] / prices[0] - 1) * 100
    final_eq = initial_capital * (1 + ret_pct / 100)
    daily_returns = np.diff(prices) / prices[:-1]
    daily_rf = 0.045 / 252
    excess = daily_returns - daily_rf
    sharpe = 0.0
    if len(excess) > 1 and np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / peak
    max_dd = float(np.max(dd) * 100)

    return {
        "ret": round(ret_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "final_equity": round(final_eq, 2),
    }


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
    has_hl = highs is not None and lows is not None
    accepts_hl = isinstance(strategy, (HybridRegimeStrategy, AdaptiveMetaStrategy))

    for i in range(warmup, len(prices)):
        date = dates[i]
        price = float(prices[i])
        current_iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        current_iv = max(current_iv, 0.05)
        rank = compute_iv_rank(iv_series, i, iv_lookback)
        if has_hl and accepts_hl:
            strategy.on_bar(date, price, current_iv, rank, high=float(highs[i]), low=float(lows[i]))
        else:
            strategy.on_bar(date, price, current_iv, rank)

    return strategy.summary()


def _annual_equity(equity_curve: list[tuple[datetime, float]]) -> dict[int, float]:
    """Return {year: equity_at_year_end} from equity curve."""
    by_year: dict[int, float] = {}
    for dt, eq in equity_curve:
        by_year[dt.year] = eq
    return by_year


def _monthly_equity(equity_curve: list[tuple[datetime, float]]) -> dict[str, float]:
    """Return {YYYY-MM: equity_at_month_end} from equity curve."""
    by_month: dict[str, float] = {}
    for dt, eq in equity_curve:
        key = dt.strftime("%Y-%m")
        by_month[key] = eq
    return by_month


def run_symbol(symbol: str) -> dict | None:
    data = download_prices(symbol, period="10y")
    if data is None:
        return None

    prices, highs, lows, dates = data
    div_yield = DIVIDEND_YIELDS.get(symbol, 0.0)
    warmup = 60

    # 1. Buy-and-hold
    bnh = compute_benchmark(prices[warmup:])

    # 2. Pure Wheel
    wheel_cfg = WheelConfig(
        initial_capital=INITIAL_CAPITAL,
        put_delta=0.30, call_delta=0.30,
        target_dte=45, profit_target=0.50, roll_dte=14,
        stop_loss=0.0, trend_sma_period=50, min_iv_rank=25.0,
        min_premium_pct=0.005, cash_reserve_pct=0.20,
        risk_free_rate=0.045, dividend_yield=div_yield,
        skew_slope=0.8, commission_per_contract=0.90,
        commission_per_share=0.005, bid_ask_slip_per_share=0.05,
    )
    wheel = WheelStrategy(wheel_cfg)
    wheel_sum = run_strategy(wheel, prices, dates, warmup=warmup, highs=highs, lows=lows)

    # 3. Pure PMCC (aggressive)
    pmcc_cfg = PMCCConfig(
        initial_capital=INITIAL_CAPITAL,
        leaps_delta=0.80, leaps_dte=270, leaps_roll_dte=90,
        leaps_max_capital_pct=0.65,
        short_delta=0.25, short_dte=21,
        short_profit_target=0.50, short_roll_dte=7,
        short_stop_loss=2.0,
        min_iv_rank=25.0, min_short_premium_pct=0.005,
        min_short_premium_abs=0.10,
        trend_sma_period=50, risk_free_rate=0.045,
        commission_per_contract=0.90,
        leaps_slip_per_share=0.15, short_slip_per_share=0.08,
        max_drawdown_pct=0.40, cooldown_days=30,
        max_notional_exposure=2.0,
    )
    pmcc = PMCCStrategy(pmcc_cfg)
    pmcc_sum = run_strategy(pmcc, prices, dates, warmup=warmup, highs=highs, lows=lows)

    # 4. Hybrid Regime
    hybrid_cfg = HybridRegimeConfig(
        initial_capital=INITIAL_CAPITAL,
        put_delta=0.30, call_delta=0.30,
        target_dte=45, profit_target=0.50, roll_dte=0,
        stop_loss=0.0, min_iv_rank=25.0, min_premium_pct=0.005,
        cash_reserve_pct=0.20, risk_free_rate=0.045,
        dividend_yield=div_yield, skew_slope=0.8,
        commission_per_contract=0.90, commission_per_share=0.005,
        bid_ask_slip_per_share=0.05,
        adx_period=14, adx_trend_threshold=25.0, adx_range_threshold=20.0,
        sma_period=50, sma_slope_window=20, regime_stability_days=5,
        uptrend_delta=0.25, range_delta=0.35, allow_stock_purchase=True,
    )
    hybrid = HybridRegimeStrategy(hybrid_cfg)
    hybrid_sum = run_strategy(hybrid, prices, dates, warmup=warmup, highs=highs, lows=lows)

    # 5. Adaptive Meta-Strategy
    meta_cfg = AdaptiveMetaConfig(
        initial_capital=INITIAL_CAPITAL,
        sma_period=50, sma_slope_window=20, adx_period=14,
        vol_window=30, vol_lookback=252, momentum_window=60,
        regime_stability_days=5,
        wheel_delta=0.30, wheel_wide_delta=0.20, wheel_dte=45,
        wheel_call_delta=0.30, wheel_profit_target=0.50, wheel_roll_dte=14,
        wheel_min_iv_rank=25.0, wheel_min_premium_pct=0.005,
        wheel_cash_reserve_pct=0.20,
        pmcc_leaps_delta=0.80, pmcc_short_delta=0.25, pmcc_short_dte=21,
        pmcc_leaps_dte=270, pmcc_leaps_roll_dte=90, pmcc_leaps_max_pct=0.65,
        pmcc_short_profit_target=0.50, pmcc_short_roll_dte=7,
        pmcc_short_stop_loss=2.0, pmcc_min_short_premium_abs=0.10,
        max_drawdown_pct=0.25, risk_free_rate=0.045,
        commission_per_contract=0.90, commission_per_share=0.005,
        bid_ask_slip_per_share=0.05, pmcc_leaps_slip=0.15,
        pmcc_short_slip=0.08, skew_slope=0.8, dividend_yield=div_yield,
    )
    meta = AdaptiveMetaStrategy(meta_cfg)
    meta_sum = run_strategy(meta, prices, dates, warmup=warmup, highs=highs, lows=lows)

    return {
        "symbol": symbol,
        "bars": len(prices) - warmup,
        # Buy-and-hold
        "bnh_ret": bnh["ret"],
        "bnh_sharpe": bnh["sharpe"],
        "bnh_dd": bnh["max_dd"],
        "bnh_final": bnh["final_equity"],
        # Wheel
        "wheel_ret": wheel_sum["total_return_pct"],
        "wheel_sharpe": wheel_sum["sharpe_ratio"],
        "wheel_dd": wheel_sum["max_drawdown_pct"],
        "wheel_final": wheel_sum["final_equity"],
        "wheel_trades": wheel_sum["total_trades"],
        # PMCC
        "pmcc_ret": pmcc_sum["total_return_pct"],
        "pmcc_sharpe": pmcc_sum["sharpe_ratio"],
        "pmcc_dd": pmcc_sum["max_drawdown_pct"],
        "pmcc_final": pmcc_sum["final_equity"],
        "pmcc_trades": pmcc_sum["total_trades"],
        # Hybrid
        "hybrid_ret": hybrid_sum["total_return_pct"],
        "hybrid_sharpe": hybrid_sum["sharpe_ratio"],
        "hybrid_dd": hybrid_sum["max_drawdown_pct"],
        "hybrid_final": hybrid_sum["final_equity"],
        "hybrid_trades": hybrid_sum.get("total_trades", 0),
        "hybrid_regime_pct": hybrid_sum.get("regime_pct", {}),
        # Meta
        "meta_ret": meta_sum["total_return_pct"],
        "meta_sharpe": meta_sum["sharpe_ratio"],
        "meta_dd": meta_sum["max_drawdown_pct"],
        "meta_final": meta_sum["final_equity"],
        "meta_trades": meta_sum.get("total_trades", 0),
        "meta_strategy_pct": meta_sum.get("strategy_pct", {}),
        "meta_strategy_pnl": meta_sum.get("strategy_pnl", {}),
        "meta_transitions": meta_sum.get("transition_count", 0),
        "meta_transition_cost": meta_sum.get("transition_cost", 0.0),
        "meta_net_premium": meta_sum.get("net_premium", 0.0),
        "meta_win_rate": meta_sum.get("win_rate", 0.0),
        # Equity curves for year-by-year
        "meta_equity_curve": meta.equity_curve,
        "wheel_equity_curve": wheel.equity_curve,
        "pmcc_equity_curve": pmcc.equity_curve,
    }


def print_comparison_table(results: list[dict]) -> None:
    print()
    print("=" * 200)
    print("  ADAPTIVE META-STRATEGY BACKTEST -- 10-YEAR, 5-WAY COMPARISON")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f} per symbol | Symbols: {len(results)}")
    print("=" * 200)
    print()

    hdr = (f"{'Symbol':<7} {'BnH':>9} {'Wheel':>9} {'PMCC':>9} {'Hybrid':>9} {'META':>9}  "
           f"{'BnH$':>10} {'Wheel$':>10} {'PMCC$':>10} {'Hybrid$':>10} {'META$':>10}  "
           f"{'MetaDD%':>8} {'MTrns':>6} {'Winner':>8}")
    print(hdr)
    print("-" * 200)

    for r in sorted(results, key=lambda x: x["meta_ret"], reverse=True):
        rets = {
            "BnH": r["bnh_ret"], "Wheel": r["wheel_ret"], "PMCC": r["pmcc_ret"],
            "Hybrid": r["hybrid_ret"], "Meta": r["meta_ret"],
        }
        winner = max(rets, key=rets.get)  # type: ignore
        print(
            f"{r['symbol']:<7} "
            f"{r['bnh_ret']:>+9.2f} {r['wheel_ret']:>+9.2f} {r['pmcc_ret']:>+9.2f} "
            f"{r['hybrid_ret']:>+9.2f} {r['meta_ret']:>+9.2f}  "
            f"${r['bnh_final']:>9,.0f} ${r['wheel_final']:>9,.0f} ${r['pmcc_final']:>9,.0f} "
            f"${r['hybrid_final']:>9,.0f} ${r['meta_final']:>9,.0f}  "
            f"{r['meta_dd']:>8.2f} {r['meta_transitions']:>6d} {winner:>8}"
        )

    print("-" * 200)
    print()


def print_summary_stats(results: list[dict]) -> None:
    n = len(results)
    strategies = ["bnh", "wheel", "pmcc", "hybrid", "meta"]
    labels = ["Buy&Hold", "Wheel", "PMCC", "Hybrid", "META"]

    print("AGGREGATE STATISTICS")
    print("=" * 120)
    print(f"  {'Metric':<25}", end="")
    for lbl in labels:
        print(f"{lbl:>14}", end="")
    print(f"  {'Best':>10}")
    print(f"  {'-'*25}", end="")
    for _ in labels:
        print(f"  {'-'*12}", end="")
    print(f"  {'-'*10}")

    metrics = [
        ("Avg Return %", "ret", "max"),
        ("Med Return %", "ret", "max"),
        ("Avg Sharpe", "sharpe", "max"),
        ("Avg MaxDD %", "dd", "min"),
    ]

    for name, key, direction in metrics:
        vals = {}
        for strat, lbl in zip(strategies, labels):
            data = [r[f"{strat}_{key}"] for r in results]
            if "Med" in name:
                vals[lbl] = float(np.median(data))
            else:
                vals[lbl] = float(np.mean(data))

        if direction == "max":
            best = max(vals, key=vals.get)  # type: ignore
        else:
            best = min(vals, key=vals.get)  # type: ignore

        print(f"  {name:<25}", end="")
        for lbl in labels:
            print(f"{vals[lbl]:>14.2f}", end="")
        print(f"  {best:>10}")

    print()

    # Win counts by return
    print("WIN COUNTS BY RETURN (%)")
    print("=" * 80)
    win_counts = {lbl: 0 for lbl in labels}
    for r in results:
        rets = {
            "Buy&Hold": r["bnh_ret"], "Wheel": r["wheel_ret"], "PMCC": r["pmcc_ret"],
            "Hybrid": r["hybrid_ret"], "META": r["meta_ret"],
        }
        winner = max(rets, key=rets.get)  # type: ignore
        win_counts[winner] += 1

    print(f"  ", end="")
    for lbl in labels:
        print(f"{lbl:>14}", end="")
    print()
    print(f"  ", end="")
    for lbl in labels:
        print(f"{win_counts[lbl]:>14d}", end="")
    print()

    # Meta beats each
    print()
    for other, key in [("Buy&Hold", "bnh_ret"), ("Wheel", "wheel_ret"), ("PMCC", "pmcc_ret"), ("Hybrid", "hybrid_ret")]:
        beats = sum(1 for r in results if r["meta_ret"] > r[key])
        avg_margin = float(np.mean([r["meta_ret"] - r[key] for r in results]))
        print(f"  Meta beats {other:<10}: {beats}/{n} ({beats/n*100:.0f}%)  avg margin: {avg_margin:+.2f}%")

    print()


def print_year_by_year(results: list[dict], symbols: list[str]) -> None:
    print("YEAR-BY-YEAR INCOME (Meta-Strategy)")
    print("=" * 120)

    for sym in symbols:
        r = next((x for x in results if x["symbol"] == sym), None)
        if r is None:
            continue

        meta_annual = _annual_equity(r["meta_equity_curve"])
        wheel_annual = _annual_equity(r["wheel_equity_curve"])
        pmcc_annual = _annual_equity(r["pmcc_equity_curve"])

        years = sorted(meta_annual.keys())
        if not years:
            continue

        print(f"\n  {sym} (${INITIAL_CAPITAL:,.0f} capital)")
        print(f"  {'Year':<6} {'Meta Eq':>12} {'Meta YoY':>10} {'Wheel Eq':>12} {'PMCC Eq':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

        prev_meta = INITIAL_CAPITAL
        for yr in years:
            meta_eq = meta_annual[yr]
            wh_eq = wheel_annual.get(yr, INITIAL_CAPITAL)
            pm_eq = pmcc_annual.get(yr, INITIAL_CAPITAL)
            yoy = meta_eq - prev_meta
            print(f"  {yr:<6} ${meta_eq:>11,.2f} ${yoy:>+9,.2f} ${wh_eq:>11,.2f} ${pm_eq:>11,.2f}")
            prev_meta = meta_eq


def print_regime_breakdown(results: list[dict]) -> None:
    print("\n\nREGIME / STRATEGY BREAKDOWN (Meta-Strategy)")
    print("=" * 120)

    agg: dict[str, list[float]] = defaultdict(list)
    for r in results:
        sp = r.get("meta_strategy_pct", {})
        for strat, pct in sp.items():
            agg[strat].append(pct)

    print(f"  {'Strategy':<15} {'Avg Time%':>10} {'Min%':>8} {'Max%':>8} {'Med%':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for strat in ["PMCC", "WHEEL", "WHEEL_WIDE", "HOLD", "CASH"]:
        pcts = agg.get(strat, [0.0])
        print(f"  {strat:<15} {np.mean(pcts):>10.1f} {np.min(pcts):>8.1f} {np.max(pcts):>8.1f} {np.median(pcts):>8.1f}")

    avg_transitions = np.mean([r["meta_transitions"] for r in results])
    avg_cost = np.mean([r["meta_transition_cost"] for r in results])
    print(f"\n  Avg transitions: {avg_transitions:.1f} per symbol")
    print(f"  Avg transition cost: ${avg_cost:,.2f} per symbol")
    print()


def print_pnl_attribution(results: list[dict]) -> None:
    print("P&L ATTRIBUTION BY SUB-STRATEGY")
    print("=" * 120)
    print(f"  {'Symbol':<7} {'PMCC':>12} {'WHEEL':>12} {'WHEEL_WIDE':>12} {'HOLD':>12} {'CASH':>12} {'Total':>12}")
    print(f"  {'-'*7} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for r in sorted(results, key=lambda x: x["meta_ret"], reverse=True):
        sp = r.get("meta_strategy_pnl", {})
        pmcc_pnl = sp.get("PMCC_pnl", 0.0)
        wheel_pnl = sp.get("WHEEL_pnl", 0.0)
        wide_pnl = sp.get("WHEEL_WIDE_pnl", 0.0)
        hold_pnl = sp.get("HOLD_pnl", 0.0)
        cash_pnl = sp.get("CASH_pnl", 0.0)
        total = pmcc_pnl + wheel_pnl + wide_pnl + hold_pnl + cash_pnl
        print(
            f"  {r['symbol']:<7} "
            f"${pmcc_pnl:>+11,.2f} ${wheel_pnl:>+11,.2f} ${wide_pnl:>+11,.2f} "
            f"${hold_pnl:>+11,.2f} ${cash_pnl:>+11,.2f} ${total:>+11,.2f}"
        )
    print()


def print_monthly_income(results: list[dict], symbols: list[str]) -> None:
    print("MONTHLY INCOME ESTIMATES (Meta-Strategy)")
    print("=" * 100)

    for sym in symbols:
        r = next((x for x in results if x["symbol"] == sym), None)
        if r is None:
            continue

        monthly = _monthly_equity(r["meta_equity_curve"])
        months = sorted(monthly.keys())
        if len(months) < 2:
            continue

        # Compute monthly P&L
        monthly_pnl: list[float] = []
        prev_eq = INITIAL_CAPITAL
        for m in months:
            eq = monthly[m]
            monthly_pnl.append(eq - prev_eq)
            prev_eq = eq

        arr = np.array(monthly_pnl)
        positive = arr[arr > 0]
        negative = arr[arr < 0]

        print(f"\n  {sym} (${INITIAL_CAPITAL:,.0f} capital, {len(months)} months)")
        print(f"    Avg monthly income:     ${np.mean(arr):>+9,.2f}")
        print(f"    Med monthly income:     ${np.median(arr):>+9,.2f}")
        print(f"    Best month:             ${np.max(arr):>+9,.2f}")
        print(f"    Worst month:            ${np.min(arr):>+9,.2f}")
        print(f"    Positive months:        {len(positive)}/{len(arr)} ({len(positive)/len(arr)*100:.0f}%)")
        if len(positive) > 0:
            print(f"    Avg positive month:     ${np.mean(positive):>+9,.2f}")
        if len(negative) > 0:
            print(f"    Avg negative month:     ${np.mean(negative):>+9,.2f}")

        # Annual income
        annual_income = float(np.mean(arr)) * 12
        monthly_income = float(np.mean(arr))
        print(f"    Expected annual income: ${annual_income:>+9,.2f}")
        print(f"    Expected monthly:       ${monthly_income:>+9,.2f}")
        ann_yield = annual_income / INITIAL_CAPITAL * 100
        print(f"    Annualized yield:       {ann_yield:>+.1f}%")


def print_simulation_assumptions() -> None:
    print("\n\nSIMULATION ASSUMPTIONS")
    print("=" * 70)
    print("  IV estimation:       Dynamic IV/RV ratio (regime-dependent)")
    print("  Vol model:           BSM with parametric volatility skew (slope=0.8)")
    print("  Bid-ask slippage:    Fixed $0.05/share (options), $0.15/share (LEAPS)")
    print("  Commission+exchange: $0.90/contract, $0.005/share")
    print("  Assignment:          At expiry only (no early assignment)")
    print("  Data:                10Y daily via yfinance (auto-adjusted)")
    print("  Regime stability:    5-day confirmation filter")
    print("  Circuit breaker:     25% max drawdown with 30-day cooldown")
    print(f"  Initial capital:     ${INITIAL_CAPITAL:,.0f}")
    print("=" * 70)
    print()


def main() -> None:
    print()
    print("#" * 120)
    print("##  ADAPTIVE META-STRATEGY: CAPSTONE BACKTEST")
    print("##  10-Year, 12 Symbols, 5-Way Comparison")
    print(f"##  Capital: ${INITIAL_CAPITAL:,.0f}")
    print("#" * 120)
    print()

    results: list[dict] = []
    t0_total = time.time()

    for sym in SYMBOLS:
        t0 = time.time()
        print(f"  Running {sym}...", end=" ", flush=True)
        result = run_symbol(sym)
        elapsed = time.time() - t0
        if result:
            results.append(result)
            print(
                f"BnH: {result['bnh_ret']:>+.1f}%  Wheel: {result['wheel_ret']:>+.1f}%  "
                f"PMCC: {result['pmcc_ret']:>+.1f}%  Hybrid: {result['hybrid_ret']:>+.1f}%  "
                f"META: {result['meta_ret']:>+.1f}%  [{elapsed:.1f}s]"
            )
        else:
            print(f"FAILED [{elapsed:.1f}s]")

    total_elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {total_elapsed:.0f}s")

    if not results:
        print("No results.")
        return

    # 1. Per-symbol comparison table
    print_comparison_table(results)

    # 2. Aggregate statistics
    print_summary_stats(results)

    # 3. Year-by-year for GLD and SIVR
    print_year_by_year(results, ["GLD", "SIVR"])

    # 4. Regime breakdown
    print_regime_breakdown(results)

    # 5. P&L attribution
    print_pnl_attribution(results)

    # 6. Monthly income
    print_monthly_income(results, ["GLD", "SIVR"])

    # 7. Assumptions
    print_simulation_assumptions()


if __name__ == "__main__":
    main()
