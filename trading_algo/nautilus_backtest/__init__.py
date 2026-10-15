"""
NautilusTrader backtests for the repo's multi-strategy controller.

Runs the existing ``TradingStrategy`` adapters, unchanged, through
NautilusTrader's event-driven engine:

- Decisions at the session close, fills at the next session's open
- The repo's realistic cost model (spread, impact, IBKR tiered commission)
- A per-session ledger reconciled against NautilusTrader's account
- Benchmark statistics from NautilusTrader's analysis module
- A parity check against ``MultiStrategyBacktestRunner``

Requires ``nautilus_trader`` (see ``requirements-nautilus.txt``).
"""

from .benchmark import benchmark_statistics, buy_and_hold, price_returns
from .costs import LegacyCosts, RealisticCosts, lagged_cost_inputs
from .data import build_sessions, load_daily_bars
from .engine import (
    BacktestIntegrityError,
    NautilusBacktestConfig,
    NautilusBacktestResult,
    explicit_controller_config,
    run_nautilus_backtest,
)
from .parity import ParityReport, run_parity
from .report import write_report
from .strategy import SizingRules

__all__ = [
    "BacktestIntegrityError",
    "LegacyCosts",
    "NautilusBacktestConfig",
    "NautilusBacktestResult",
    "ParityReport",
    "RealisticCosts",
    "SizingRules",
    "benchmark_statistics",
    "build_sessions",
    "buy_and_hold",
    "explicit_controller_config",
    "lagged_cost_inputs",
    "load_daily_bars",
    "price_returns",
    "run_nautilus_backtest",
    "run_parity",
    "write_report",
]
