"""
PMCC Parameter Sweep

Exhaustive grid search over PMCC config space, ranked by cross-symbol
average Sharpe.  Includes walk-forward validation on the top configs.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict
from datetime import datetime

import numpy as np

from trading_algo.broker.base import Bar
from trading_algo.quant_core.strategies.options.options_backtester import (
    BacktestReport,
    run_options_backtest,
)
from trading_algo.quant_core.strategies.options.pmcc import PMCCConfig, PMCCStrategy


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

SWEEP_GRID: dict[str, list] = {
    "leaps_delta": [0.70, 0.80, 0.90],
    "short_delta": [0.15, 0.20, 0.25, 0.30],
    "short_dte": [21, 28, 45],
    "short_profit_target": [0.0, 0.50],
    "max_drawdown_pct": [0.0, 0.30, 0.40],
    "trend_sma_period": [0, 50],
}


def _make_configs() -> list[PMCCConfig]:
    keys = list(SWEEP_GRID.keys())
    combos = list(itertools.product(*(SWEEP_GRID[k] for k in keys)))
    configs: list[PMCCConfig] = []
    for vals in combos:
        overrides = dict(zip(keys, vals))
        configs.append(PMCCConfig(**overrides))
    return configs


# ---------------------------------------------------------------------------
# Single config evaluation
# ---------------------------------------------------------------------------

def _run_single(
    cfg: PMCCConfig,
    all_bars: dict[str, list[Bar]],
    warmup: int = 60,
) -> dict[str, BacktestReport]:
    results: dict[str, BacktestReport] = {}
    for symbol, bars in all_bars.items():
        if len(bars) < warmup + 10:
            continue
        strat = PMCCStrategy(cfg)
        try:
            report = run_options_backtest(strat, bars, symbol, warmup=warmup)
            results[symbol] = report
        except Exception:
            continue
    return results


def _avg_sharpe(reports: dict[str, BacktestReport]) -> float:
    if not reports:
        return -999.0
    sharpes = [r.summary["sharpe_ratio"] for r in reports.values()]
    return float(np.mean(sharpes))


def _avg_max_dd(reports: dict[str, BacktestReport]) -> float:
    if not reports:
        return 999.0
    dds = [r.summary["max_drawdown_pct"] for r in reports.values()]
    return float(np.mean(dds))


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def _walk_forward(
    cfg: PMCCConfig,
    all_bars: dict[str, list[Bar]],
    warmup: int = 60,
) -> tuple[dict[str, BacktestReport], dict[str, BacktestReport]]:
    in_sample: dict[str, BacktestReport] = {}
    out_sample: dict[str, BacktestReport] = {}

    for symbol, bars in all_bars.items():
        mid = len(bars) // 2
        first_half = bars[:mid]
        second_half = bars[mid:]

        for half_bars, dest in [(first_half, in_sample), (second_half, out_sample)]:
            if len(half_bars) < warmup + 10:
                continue
            strat = PMCCStrategy(cfg)
            try:
                report = run_options_backtest(strat, half_bars, symbol, warmup=warmup)
                dest[symbol] = report
            except Exception:
                continue

    return in_sample, out_sample


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _cfg_label(cfg: PMCCConfig) -> str:
    return (
        f"ld={cfg.leaps_delta:.2f} sd={cfg.short_delta:.2f} "
        f"sdte={cfg.short_dte} pt={cfg.short_profit_target:.2f} "
        f"dd={cfg.max_drawdown_pct:.2f} sma={cfg.trend_sma_period}"
    )


def _format_top_configs(
    ranked: list[tuple[PMCCConfig, float, float, dict[str, BacktestReport]]],
    n: int = 10,
) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"  TOP {min(n, len(ranked))} PMCC CONFIGURATIONS (ranked by cross-symbol avg Sharpe)")
    lines.append("=" * 100)
    lines.append("")

    for rank, (cfg, avg_sh, avg_dd, reports) in enumerate(ranked[:n], 1):
        lines.append(f"  #{rank:>2d}  Sharpe={avg_sh:>+6.3f}  MaxDD={avg_dd:>6.2f}%  | {_cfg_label(cfg)}")

    return lines


def _format_top3_breakdown(
    ranked: list[tuple[PMCCConfig, float, float, dict[str, BacktestReport]]],
) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("  PER-SYMBOL BREAKDOWN (top 3 configs)")
    lines.append("=" * 100)

    for rank, (cfg, avg_sh, avg_dd, reports) in enumerate(ranked[:3], 1):
        lines.append("")
        lines.append(f"  --- Config #{rank}: {_cfg_label(cfg)} ---")
        lines.append(f"  {'Symbol':<8} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7} {'WinRate':>8}")
        lines.append(f"  {'-' * 60}")

        for sym in sorted(reports.keys()):
            s = reports[sym].summary
            lines.append(
                f"  {sym:<8} {s['total_return_pct']:>+9.2f}% "
                f"{s['sharpe_ratio']:>+7.3f} "
                f"{s['max_drawdown_pct']:>7.2f}% "
                f"{s['total_trades']:>6d} "
                f"{s['win_rate']:>7.1f}%"
            )

        lines.append(f"  {'AVG':<8} {'':>10} {avg_sh:>+7.3f} {avg_dd:>7.2f}%")

    return lines


def _format_walk_forward(
    ranked: list[tuple[PMCCConfig, float, float, dict[str, BacktestReport]]],
    all_bars: dict[str, list[Bar]],
) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("  WALK-FORWARD VALIDATION (top 3 configs: 1st half train, 2nd half test)")
    lines.append("=" * 100)

    for rank, (cfg, _, _, _) in enumerate(ranked[:3], 1):
        in_sample, out_sample = _walk_forward(cfg, all_bars)
        is_sharpe = _avg_sharpe(in_sample)
        os_sharpe = _avg_sharpe(out_sample)
        is_dd = _avg_max_dd(in_sample)
        os_dd = _avg_max_dd(out_sample)

        degradation = 0.0
        if is_sharpe != 0:
            degradation = (1 - os_sharpe / is_sharpe) * 100 if is_sharpe > 0 else 0.0

        lines.append("")
        lines.append(f"  --- Config #{rank}: {_cfg_label(cfg)} ---")
        lines.append(f"  {'':>20} {'In-Sample':>12} {'Out-Sample':>12} {'Degradation':>12}")
        lines.append(f"  {'Avg Sharpe':>20} {is_sharpe:>+11.3f} {os_sharpe:>+11.3f} {degradation:>+10.1f}%")
        lines.append(f"  {'Avg MaxDD':>20} {is_dd:>10.2f}% {os_dd:>10.2f}%")

        lines.append(f"  {'Symbol':<8} {'IS Sharpe':>10} {'OS Sharpe':>10} {'IS DD%':>8} {'OS DD%':>8}")
        lines.append(f"  {'-' * 55}")
        all_syms = sorted(set(list(in_sample.keys()) + list(out_sample.keys())))
        for sym in all_syms:
            is_s = in_sample[sym].summary["sharpe_ratio"] if sym in in_sample else float("nan")
            os_s = out_sample[sym].summary["sharpe_ratio"] if sym in out_sample else float("nan")
            is_d = in_sample[sym].summary["max_drawdown_pct"] if sym in in_sample else float("nan")
            os_d = out_sample[sym].summary["max_drawdown_pct"] if sym in out_sample else float("nan")
            lines.append(
                f"  {sym:<8} {is_s:>+9.3f} {os_s:>+9.3f} {is_d:>7.2f}% {os_d:>7.2f}%"
            )

    return lines


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pmcc_sweep(all_bars: dict[str, list[Bar]]) -> str:
    """
    Run exhaustive PMCC parameter sweep across all symbols.

    Args:
        all_bars: Dict of symbol -> list[Bar] (daily OHLCV).

    Returns:
        Formatted string report with top 10 configs, per-symbol breakdown,
        and walk-forward validation of top 3.
    """
    configs = _make_configs()
    total = len(configs)

    scored: list[tuple[PMCCConfig, float, float, dict[str, BacktestReport]]] = []

    for i, cfg in enumerate(configs):
        reports = _run_single(cfg, all_bars)
        if not reports:
            continue
        avg_sh = _avg_sharpe(reports)
        avg_dd = _avg_max_dd(reports)
        scored.append((cfg, avg_sh, avg_dd, reports))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    lines: list[str] = []
    lines.append("")
    lines.append(f"  PMCC Sweep: {total} configs x {len(all_bars)} symbols = {total * len(all_bars)} backtests")
    lines.append(f"  Symbols: {', '.join(sorted(all_bars.keys()))}")
    lines.append(f"  Configs with results: {len(ranked)}")
    lines.append("")

    lines.extend(_format_top_configs(ranked))
    lines.extend(_format_top3_breakdown(ranked))
    lines.extend(_format_walk_forward(ranked, all_bars))

    lines.append("")
    return "\n".join(lines)
