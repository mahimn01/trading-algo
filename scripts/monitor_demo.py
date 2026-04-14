#!/usr/bin/env python3
"""
Monitor & Auto-Optimizer Demo

Simulates 6 months of live trading using historical data, demonstrating:
  1. StrategyMonitor detecting regime changes and raising alerts
  2. AutoOptimizer detecting parameter drift and suggesting updates
  3. Symbol scorer ranking candidate stocks for the Wheel
  4. Sample daily/weekly/monthly report generation
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from trading_algo.quant_core.strategies.options.wheel import WheelConfig, WheelStrategy
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_series_from_prices,
    iv_rank as compute_iv_rank,
    realized_volatility,
)
from trading_algo.quant_core.strategies.options.strategy_monitor import (
    StrategyMonitor,
    TradeJournalEntry,
)
from trading_algo.quant_core.strategies.options.auto_optimizer import (
    AutoOptimizer,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_symbol_data(symbol: str, period: str = "2y") -> dict | None:
    """Download daily data from yfinance, compute IV and IV rank series."""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 120:
            print(f"  [SKIP] {symbol}: insufficient data ({len(df)} bars)")
            return None

        if hasattr(df.columns, "get_level_values") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])

        prices = df["Close"].values.astype(float)
        volumes = df["Volume"].values.astype(float) if "Volume" in df.columns else None
        dates = [
            datetime.combine(d.date(), datetime.min.time()) if hasattr(d, "date") else d
            for d in df.index
        ]

        iv_series = iv_series_from_prices(prices, rv_window=30, iv_premium=1.20, dynamic=True)

        iv_rank_series = np.full(len(prices), 50.0)
        for i in range(60, len(prices)):
            iv_rank_series[i] = compute_iv_rank(iv_series, i, lookback=252)

        rv_30d = realized_volatility(prices, 30)

        return {
            "prices": prices,
            "dates": dates,
            "iv": iv_series,
            "iv_rank": iv_rank_series,
            "rv_30d": rv_30d,
            "volume": volumes,
        }
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Demo 1: Live simulation with monitoring
# ---------------------------------------------------------------------------

def demo_live_monitoring(symbol: str, data: dict) -> StrategyMonitor:
    """Simulate 6 months of live trading with the monitor attached."""
    print(f"\n{'=' * 70}")
    print(f"  DEMO 1: Live Monitoring Simulation — {symbol}")
    print(f"{'=' * 70}")

    prices = data["prices"]
    dates = data["dates"]
    iv_series = data["iv"]
    iv_rank_series = data["iv_rank"]
    rv_30d = data["rv_30d"]

    warmup = 60
    sim_start = len(prices) - 126  # ~6 months
    if sim_start < warmup:
        sim_start = warmup

    # Pre-train: run strategy on data before sim window to get a "backtest" Sharpe
    cfg = WheelConfig(initial_capital=10_000.0, put_delta=0.30, call_delta=0.30, target_dte=45, trend_sma_period=50)
    pretrain_strategy = WheelStrategy(cfg)
    for i in range(warmup, sim_start):
        iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        iv = max(iv, 0.05)
        rank = float(iv_rank_series[i])
        pretrain_strategy.on_bar(dates[i], float(prices[i]), iv, rank)
    pretrain_summary = pretrain_strategy.summary()
    backtest_sharpe = pretrain_summary.get("sharpe_ratio", 0.5)
    backtest_wr = pretrain_summary.get("win_rate", 75.0) / 100

    print(f"  Pre-train period: {dates[warmup]:%Y-%m-%d} to {dates[sim_start - 1]:%Y-%m-%d}")
    print(f"  Backtest Sharpe: {backtest_sharpe:.2f}, Win rate: {backtest_wr:.0%}")

    # Live sim
    strategy = WheelStrategy(cfg)
    monitor = StrategyMonitor(
        strategy_name=f"Wheel_{symbol}",
        backtest_sharpe=backtest_sharpe,
        backtest_win_rate=backtest_wr,
        last_validation_date=dates[sim_start] - timedelta(days=100),
    )

    # Feed warmup bars to strategy (needed for SMA filter)
    feed_start = max(0, sim_start - warmup)
    for i in range(feed_start, sim_start):
        iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        iv = max(iv, 0.05)
        rank = float(iv_rank_series[i])
        strategy.on_bar(dates[i], float(prices[i]), iv, rank)

    print(f"\n  Simulating {dates[sim_start]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d} ({len(prices) - sim_start} days)")
    print(f"  {'-' * 60}")

    alert_count = 0
    for i in range(sim_start, len(prices)):
        iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        iv = max(iv, 0.05)
        rank = float(iv_rank_series[i])
        price = float(prices[i])

        events = strategy.on_bar(dates[i], price, iv, rank)
        equity = strategy.get_equity(price, iv, as_of=dates[i])

        rv_val = float(rv_30d[i]) if not np.isnan(rv_30d[i]) else None

        # Log trades to monitor journal
        for ev in events:
            if ev.pnl != 0 or "sell" in ev.event_type:
                journal_entry = TradeJournalEntry(
                    date=dates[i],
                    symbol=symbol,
                    action=ev.event_type,
                    strike=ev.details.get("strike", 0),
                    premium=ev.details.get("premium", 0),
                    underlying_price=price,
                    iv=iv,
                    iv_rank=rank,
                    regime=monitor.regime.current,
                    contracts=ev.details.get("contracts", 1),
                    pnl=ev.pnl,
                    attribution={"type": "premium" if ev.pnl > 0 else "loss"},
                )
                monitor.log_trade(journal_entry)

        new_alerts = monitor.update(dates[i], equity, price, rv_val)
        for a in new_alerts:
            alert_count += 1
            print(f"  {a}")

    final_summary = strategy.summary()
    print(f"\n  --- Results ---")
    print(f"  Final equity: ${final_summary['final_equity']:,.2f}  ({final_summary['total_return_pct']:+.2f}%)")
    print(f"  Sharpe: {final_summary['sharpe_ratio']:.2f}  |  Max DD: {final_summary['max_drawdown_pct']:.1f}%")
    print(f"  Trades: {final_summary['total_trades']}  |  Win rate: {final_summary['win_rate']:.1f}%")
    print(f"  Total alerts raised: {alert_count}")
    print(f"  Regime at end: {monitor.regime.current}")
    print(f"  Current drawdown: {monitor.current_drawdown:.1%}")
    if monitor.is_paused:
        print(f"  *** STRATEGY IS PAUSED (circuit breaker active) ***")

    return monitor


# ---------------------------------------------------------------------------
# Demo 2: Auto-optimizer / walk-forward validation
# ---------------------------------------------------------------------------

def demo_auto_optimizer(symbol: str, data: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  DEMO 2: Auto-Optimizer — {symbol}")
    print(f"{'=' * 70}")

    prices = data["prices"]
    dates = data["dates"]
    iv_series = data["iv"]
    iv_rank_series = data["iv_rank"]

    cfg = WheelConfig(initial_capital=10_000.0, put_delta=0.30, call_delta=0.30, target_dte=45)
    optimizer = AutoOptimizer(base_config=cfg)

    # Walk-forward validation
    print(f"\n  --- Walk-Forward Validation ---")
    result = optimizer.walk_forward_validate(prices, dates, iv_series, iv_rank_series, split_ratio=0.7)
    print(f"  Train Sharpe: {result.backtest_sharpe:.2f}  |  Test Sharpe: {result.live_sharpe:.2f}")
    print(f"  Train Return: {result.backtest_return_pct:.1f}%  |  Test Return: {result.live_return_pct:.1f}%")
    print(f"  Degradation: {result.degradation_pct:.1f}%  |  Needs review: {result.needs_review}")

    # Sensitivity analysis
    print(f"\n  --- Parameter Sensitivity Analysis ---")
    sensitivities = optimizer.sensitivity_analysis(prices, dates, iv_series, iv_rank_series)
    print(f"  {'Parameter':<20} {'Current':>10} {'Optimal':>10} {'Curr Sharpe':>12} {'Opt Sharpe':>12} {'Robust':>8} {'Recommendation'}")
    print(f"  {'-' * 95}")
    for s in sensitivities:
        print(f"  {s.parameter:<20} {s.current_value:>10} {s.optimal_value:>10} {s.current_sharpe:>12.3f} {s.optimal_sharpe:>12.3f} {str(s.robust):>8} {s.recommendation}")


# ---------------------------------------------------------------------------
# Demo 3: Symbol scoring
# ---------------------------------------------------------------------------

def demo_symbol_scoring(all_data: dict[str, dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  DEMO 3: Symbol Universe Scoring")
    print(f"{'=' * 70}")

    optimizer = AutoOptimizer()

    symbol_data = {}
    for sym, data in all_data.items():
        symbol_data[sym] = {
            "prices": data["prices"],
            "volume": data.get("volume"),
        }

    ranked = optimizer.rank_symbols(symbol_data, account_size=10_000.0, top_n=10)

    print(f"\n  {'Rank':<6} {'Symbol':<8} {'Total':>7} {'IV Rank':>9} {'Liquid':>8} {'Trend':>7} {'Price':>7} {'Corr':>6} {'Price':>10}")
    print(f"  {'-' * 75}")
    for i, score in enumerate(ranked, 1):
        print(
            f"  {i:<6} {score.symbol:<8} {score.total_score:>7.1f} "
            f"{score.iv_rank_score:>9.1f} {score.liquidity_score:>8.1f} "
            f"{score.trend_stability_score:>7.1f} {score.price_range_score:>7.1f} "
            f"{score.correlation_score:>6.1f} ${score.details.get('current_price', 0):>8.2f}"
        )


# ---------------------------------------------------------------------------
# Demo 4: Report generation
# ---------------------------------------------------------------------------

def demo_reports(monitor: StrategyMonitor, data: dict, symbol: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  DEMO 4: Report Generation — {symbol}")
    print(f"{'=' * 70}")

    optimizer = AutoOptimizer()

    # Daily report
    print(f"\n  --- Daily Report ---")
    daily = optimizer.generate_daily_report(monitor)
    print(f"  {daily.render()}")

    # Weekly report
    print(f"\n  --- Weekly Report ---")
    weekly = optimizer.generate_weekly_report(monitor)
    print(f"  {weekly.render()}")

    # Monthly report for the last full month
    dates = monitor._dates
    if dates:
        last_date = dates[-1]
        # Go to previous month
        if last_date.month == 1:
            report_year, report_month = last_date.year - 1, 12
        else:
            report_year, report_month = last_date.year, last_date.month - 1

        # Get benchmark prices for that month
        month_prices = [
            float(data["prices"][i])
            for i, d in enumerate(data["dates"])
            if d.year == report_year and d.month == report_month
        ]
        bench = np.array(month_prices) if month_prices else None

        print(f"\n  --- Monthly Report ({report_year}-{report_month:02d}) ---")
        monthly = optimizer.generate_monthly_report(monitor, report_year, report_month, benchmark_prices=bench)
        print(f"  {monthly.render()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Strategy Monitor & Auto-Optimizer Demo")
    print("  Loading market data from Yahoo Finance...")
    print("=" * 70)

    symbols = ["SOFI", "F", "PLTR", "T", "BAC", "NIO", "RIVN", "AMD", "INTC", "HOOD"]

    all_data: dict[str, dict] = {}
    for sym in symbols:
        print(f"  Loading {sym}...", end=" ", flush=True)
        data = load_symbol_data(sym, period="2y")
        if data:
            all_data[sym] = data
            print(f"{len(data['prices'])} bars")
        else:
            print("failed")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    # Pick the first symbol with enough data for the main demos
    primary_symbol = next(iter(all_data))
    primary_data = all_data[primary_symbol]

    # Demo 1: Live monitoring
    monitor = demo_live_monitoring(primary_symbol, primary_data)

    # Demo 2: Auto-optimizer on same symbol
    demo_auto_optimizer(primary_symbol, primary_data)

    # Demo 3: Symbol scoring across all loaded symbols
    demo_symbol_scoring(all_data)

    # Demo 4: Report generation
    demo_reports(monitor, primary_data, primary_symbol)

    print(f"\n{'=' * 70}")
    print("  Demo complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
