"""
Options Strategy Backtester

Runs Wheel, PMCC, or any options strategy on historical underlying-price bars.
Options prices are simulated via Black-Scholes with dynamic IV estimation.

Honesty/reliability features:
  - Dynamic IV/RV ratio (collapses during selloffs, high in calm markets)
  - Buy-and-hold benchmark included in every report
  - Fixed-dollar bid-ask slippage (not percentage)
  - Exchange fees + commissions
  - Proper Sharpe ratio (excess over risk-free)
  - Equity curve uses simulation date, not wall-clock
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from trading_algo.broker.base import Bar
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_percentile,
    iv_rank,
    iv_series_from_prices,
    realized_volatility,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class OptionsStrategy(Protocol):
    def on_bar(self, date: datetime, price: float, iv: float, iv_rank: float) -> list: ...
    def summary(self) -> dict: ...
    @property
    def events(self) -> list: ...
    @property
    def equity_curve(self) -> list[tuple[datetime, float]]: ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    symbol: str
    strategy_name: str
    summary: dict
    events: list
    equity_curve: list[tuple[datetime, float]]
    bars_used: int
    iv_premium_factor: float
    # Benchmark
    benchmark_return_pct: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_dd_pct: float = 0.0

    def annualized_return(self) -> float:
        days = self.summary.get("days", 0)
        ret = self.summary.get("total_return_pct", 0.0) / 100
        if days <= 0 or ret <= -1:
            return 0.0
        return float(((1 + ret) ** (365 / days) - 1) * 100)

    def benchmark_annualized(self) -> float:
        days = self.summary.get("days", 0)
        ret = self.benchmark_return_pct / 100
        if days <= 0 or ret <= -1:
            return 0.0
        return float(((1 + ret) ** (365 / days) - 1) * 100)


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def _compute_benchmark(prices: np.ndarray, risk_free_rate: float = 0.045) -> tuple[float, float, float]:
    """Compute buy-and-hold return, Sharpe, and max drawdown."""
    if len(prices) < 2:
        return 0.0, 0.0, 0.0
    ret_pct = (prices[-1] / prices[0] - 1) * 100
    daily_returns = np.diff(prices) / prices[:-1]
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    sharpe = 0.0
    if np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))
    peak = np.maximum.accumulate(prices)
    dd = (peak - prices) / peak
    max_dd = float(np.max(dd) * 100)
    return ret_pct, sharpe, max_dd


def run_options_backtest(
    strategy: OptionsStrategy,
    bars: list[Bar],
    symbol: str,
    iv_premium_factor: float = 1.20,
    rv_window: int = 30,
    iv_lookback: int = 252,
    warmup: int = 60,
    dividend_yields: dict[str, float] | None = None,
) -> BacktestReport:
    """
    Run an options strategy over historical bars.

    Args:
        strategy: An initialized strategy instance.
        bars: Daily OHLCV bars from IBKR (sorted chronologically).
        symbol: Underlying ticker (labelling only).
        iv_premium_factor: Base IV/RV premium (used when dynamic IV is off).
        rv_window: Window for realized vol calculation.
        iv_lookback: Window for IV rank/percentile.
        warmup: Skip this many bars to let vol estimates stabilize.
        dividend_yields: Mapping of symbol -> annualized dividend yield.

    Returns:
        BacktestReport with strategy + benchmark results.
    """
    if dividend_yields and symbol in dividend_yields:
        if hasattr(strategy, 'cfg') and hasattr(strategy.cfg, 'dividend_yield'):
            from dataclasses import replace
            strategy.cfg = replace(strategy.cfg, dividend_yield=dividend_yields[symbol])

    if len(bars) < warmup + 10:
        raise ValueError(f"Need at least {warmup + 10} bars, got {len(bars)}")

    prices = np.array([b.close for b in bars], dtype=float)
    timestamps = [datetime.fromtimestamp(b.timestamp_epoch_s) for b in bars]

    # Pre-compute IV series with DYNAMIC IV/RV ratio
    iv_series = iv_series_from_prices(prices, rv_window, iv_premium_factor, dynamic=True)

    for i in range(warmup, len(bars)):
        date = timestamps[i]
        price = float(prices[i])
        current_iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        current_iv = max(current_iv, 0.05)
        current_rank = iv_rank(iv_series, i, iv_lookback)

        strategy.on_bar(date, price, current_iv, current_rank)

    # Benchmark: buy and hold
    bench_prices = prices[warmup:]
    bench_ret, bench_sharpe, bench_dd = _compute_benchmark(bench_prices)

    strategy_name = type(strategy).__name__
    return BacktestReport(
        symbol=symbol,
        strategy_name=strategy_name,
        summary=strategy.summary(),
        events=strategy.events,
        equity_curve=strategy.equity_curve,
        bars_used=len(bars) - warmup,
        iv_premium_factor=iv_premium_factor,
        benchmark_return_pct=round(bench_ret, 2),
        benchmark_sharpe=round(bench_sharpe, 3),
        benchmark_max_dd_pct=round(bench_dd, 2),
    )


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(report: BacktestReport) -> str:
    s = report.summary
    lines = [
        "",
        f"{'=' * 65}",
        f"  {report.strategy_name} on {report.symbol}",
        f"  {s.get('start_date', '?')} -> {s.get('end_date', '?')}  ({s.get('days', 0)} trading days)",
        f"{'=' * 65}",
        "",
        f"                         {'Strategy':>12}   {'Buy&Hold':>12}",
        f"                         {'--------':>12}   {'--------':>12}",
        f"  Total Return:          {s['total_return_pct']:>11.2f} %   {report.benchmark_return_pct:>11.2f} %",
        f"  Annualized Return:     {report.annualized_return():>11.2f} %   {report.benchmark_annualized():>11.2f} %",
        f"  Sharpe Ratio:          {s['sharpe_ratio']:>11.3f}     {report.benchmark_sharpe:>11.3f}",
        f"  Max Drawdown:          {s['max_drawdown_pct']:>11.2f} %   {report.benchmark_max_dd_pct:>11.2f} %",
        "",
        f"  Initial Capital:     ${s['initial_capital']:>12,.2f}",
        f"  Final Equity:        ${s['final_equity']:>12,.2f}",
        "",
        f"  Total Trades:         {s['total_trades']:>11d}",
        f"  Wins / Losses:        {s['wins']:>5d} / {s['losses']:<5d}",
        f"  Win Rate:             {s['win_rate']:>11.1f} %",
        "",
    ]

    if "wheel_cycles" in s:
        lines.extend([
            f"  Wheel Cycles:         {s['wheel_cycles']:>11d}",
            f"  Net Premium:         ${s['net_premium']:>12,.2f}",
            f"  Premium Collected:   ${s['total_premium_collected']:>12,.2f}",
            f"  Premium Paid:        ${s['total_premium_paid']:>12,.2f}",
        ])
    if "short_cycles" in s:
        lines.extend([
            f"  Short Call Cycles:    {s['short_cycles']:>11d}",
            f"  LEAPS Rolls:          {s['leaps_rolls']:>11d}",
            f"  Total LEAPS Cost:    ${s['total_leaps_cost']:>12,.2f}",
            f"  Short Premium:       ${s['total_short_premium']:>12,.2f}",
        ])
    lines.extend([
        f"  Total Commissions:   ${s['total_commissions']:>12,.2f}",
    ])

    # Honesty flags
    lines.append("")
    lines.append("  -- Simulation assumptions --")
    lines.append("  IV estimation:       Dynamic IV/RV ratio (regime-dependent)")
    lines.append(f"  Bid-ask slippage:    Fixed ${0.05:.2f}-${0.15:.2f}/share (not % of mid)")
    lines.append(f"  Commission+exchange: ${0.90:.2f}/contract")
    lines.append("  Assignment:          At expiry only (no early assignment)")
    lines.append("  Vol model:           BSM with parametric volatility skew")
    lines.append("  Data:                IBKR historical daily bars")
    lines.append("")

    # Trade log (last 15 events)
    trade_events = [e for e in report.events if e.pnl != 0 or "sell" in e.event_type or "buy" in e.event_type or "assign" in e.event_type]
    if trade_events:
        lines.append(f"  Last {min(15, len(trade_events))} events:")
        lines.append(f"  {'Date':<12} {'Event':<35} {'PnL':>10} {'Details'}")
        lines.append(f"  {'-' * 80}")
        for ev in trade_events[-15:]:
            detail_str = ""
            d = ev.details
            if "strike" in d:
                detail_str += f"K={d['strike']}"
            if "premium" in d:
                detail_str += f" prem=${d['premium']:.2f}"
            if "underlying" in d:
                detail_str += f" @{d['underlying']}"
            if "contracts" in d:
                detail_str += f" x{d['contracts']}"
            pnl_str = f"${ev.pnl:>+9.2f}" if ev.pnl != 0 else ""
            lines.append(f"  {ev.date.strftime('%Y-%m-%d'):<12} {ev.event_type:<35} {pnl_str:>10} {detail_str}")

    lines.append(f"\n{'=' * 65}\n")
    return "\n".join(lines)
