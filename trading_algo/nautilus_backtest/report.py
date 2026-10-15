"""Write a NautilusTrader backtest's results to disk.

Each run gets its own folder with:

* ``summary.json``: settings, data range, caveats, metrics, benchmark
  statistics, baseline and the parity check.
* ``nav.csv``, ``fills.csv``, ``signals.csv`` and ``positions.csv``.
* ``tearsheet.html``: NautilusTrader's tearsheet, built from session returns
  with the benchmark statistics in its table. Skipped when plotly is missing.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from trading_algo.nautilus_backtest.engine import NautilusBacktestResult
from trading_algo.nautilus_backtest.parity import ParityReport

logger = logging.getLogger(__name__)

_METRIC_FIELDS = (
    "total_return",
    "annualized_return",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "max_drawdown_duration_days",
    "var_95",
    "cvar_95",
    "skewness",
    "kurtosis",
    "profit_factor",
    "expectancy_per_trade",
    "win_rate",
    "total_trades",
    "annual_turnover",
)


def write_report(
    result: NautilusBacktestResult,
    out_dir: Path,
    *,
    run_info: dict[str, Any],
    controller_config: Any,
    benchmark: dict[str, float] | None = None,
    benchmark_label: str = "",
    benchmark_returns: pd.Series | None = None,
    baseline: dict[str, Any] | None = None,
    parity: ParityReport | None = None,
    tearsheet: bool = True,
) -> Path:
    """Write all artifacts for one run into ``out_dir`` and return it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {name: getattr(result.metrics, name) for name in _METRIC_FIELDS}
    summary = {
        "run": run_info,
        "config": dataclasses.asdict(result.config),
        "controller_config": dataclasses.asdict(controller_config),
        "costs": result.costs,
        "results": {
            "final_nav": result.final_nav,
            "total_return_full_period": result.total_return,
            "metrics_after_warmup": metrics,
            "sharpe_rf0": result.sharpe_rf0,
            "sortino_rf0": result.sortino_rf0,
            "cost_breakdown": result.cost_breakdown,
            "fills": len(result.fills),
            "signals": len(result.signals),
            "closed_trades": result.closed_trades,
            "winning_trades": result.winning_trades,
            "ledger_vs_nautilus_max_nav_diff": result.max_reconcile_diff,
        },
        "notes": [
            "Metrics exclude the strategies' warm-up sessions.",
            "Idle cash earns no interest here, so Sharpe after the risk-free rate "
            "understates the strategy. Sharpe at rf=0 is shown as well.",
            "Short borrow is not charged; an estimate is in cost_breakdown.",
            "Each fill that reduces or closes a position counts as one closed trade.",
        ],
        "benchmark": {"label": benchmark_label, "statistics": benchmark},
        "baseline": baseline,
        "parity": dataclasses.asdict(parity) if parity is not None else None,
        "nautilus_native_stats": result.native_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    pd.DataFrame({"nav": result.nav, "return": result.returns}).to_csv(
        out_dir / "nav.csv",
        index_label="date",
    )
    result.fills.to_csv(out_dir / "fills.csv", index=False)
    result.signals.to_csv(out_dir / "signals.csv", index=False)
    result.positions_report.to_csv(out_dir / "positions.csv")
    if tearsheet:
        _write_tearsheet(result, out_dir, run_info, benchmark, benchmark_label, benchmark_returns)
    return out_dir


def _write_tearsheet(
    result: NautilusBacktestResult,
    out_dir: Path,
    run_info: dict[str, Any],
    benchmark: dict[str, float] | None,
    benchmark_label: str,
    benchmark_returns: pd.Series | None,
) -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        logger.warning("plotly is not installed, skipping tearsheet.html")
        return
    from nautilus_trader.analysis import create_tearsheet_from_stats

    def utc_index(series: pd.Series) -> pd.Series:
        out = series.copy()
        out.index = pd.DatetimeIndex([pd.Timestamp(d) for d in out.index]).tz_localize("UTC")
        return out

    stats_returns = {
        "Sharpe Ratio (rf=0, sessions)": result.sharpe_rf0,
        "Sortino Ratio (rf=0, sessions)": result.sortino_rf0,
        "Annualized Return": result.metrics.annualized_return,
        "Volatility (annualized)": result.metrics.volatility,
        "Max Drawdown": result.metrics.max_drawdown,
        **{f"{k} vs {benchmark_label}": v for k, v in (benchmark or {}).items()},
    }
    stats_general = {
        "Fills": float(len(result.fills)),
        "Closed Trades": float(result.closed_trades),
        "Win Rate": result.metrics.win_rate,
    }
    stats_pnls = {
        "USD": {
            "Final NAV": result.final_nav,
            "Commission": result.cost_breakdown["commission"],
            "Slippage": result.cost_breakdown["slippage"],
        },
    }
    create_tearsheet_from_stats(
        stats_pnls=stats_pnls,
        stats_returns=stats_returns,
        stats_general=stats_general,
        returns=utc_index(result.returns),
        output_path=str(out_dir / "tearsheet.html"),
        title=run_info.get("title", "trading-algo on NautilusTrader"),
        benchmark_returns=utc_index(benchmark_returns) if benchmark_returns is not None else None,
        benchmark_name=benchmark_label or "Benchmark",
    )


def _jsonable(value: Any) -> Any:
    """Convert dataclass-derived values into strict JSON (NaN becomes null)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value
