"""Run a ``MultiStrategyController`` backtest on NautilusTrader.

``run_nautilus_backtest`` builds a quiet ``BacktestEngine`` with one MARGIN
venue at 1x leverage, loads the daily data, runs ``ControllerStrategy`` and
returns a ``NautilusBacktestResult``. The headline numbers come from the
strategy's per-session ledger NAV. NautilusTrader's own statistics are kept
for reference, but they use calendar-day returns with zero rows on weekends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal, Mapping

import pandas as pd
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggerConfig
from nautilus_trader.model import AccountType, Money, OmsType, Venue

from backtest.metrics import compute_backtest_metrics
from trading_algo.backtest_v2.cost_model import CostModelConfig
from trading_algo.multi_strategy.backtest_runner import MultiStrategyBacktestResults
from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)
from trading_algo.nautilus_backtest.costs import (
    CostFeeModel,
    CostFillModel,
    ExecutionCosts,
    LegacyCosts,
    RealisticCosts,
    lagged_cost_inputs,
)
from trading_algo.nautilus_backtest.data import (
    USD,
    build_sessions,
    make_equity,
    to_nautilus_data,
)
from trading_algo.nautilus_backtest.strategy import (
    ControllerStrategy,
    ControllerStrategyConfig,
    Fill,
    SizingRules,
)

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class BacktestIntegrityError(RuntimeError):
    """Raised when an order is denied or the ledger and engine disagree."""


@dataclass
class NautilusBacktestConfig:
    """Settings for one NautilusTrader backtest."""

    initial_capital: float = 100_000.0
    venue: str = "XNAS"
    costs: Literal["realistic", "legacy"] = "realistic"
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    slippage_bps: float = 2.0
    commission_per_share: float = 0.0035
    max_position_pct: float = 0.25
    max_gross_exposure: float = 1.0
    cash_buffer: float = 0.95
    liquidate_at_end: bool = True
    risk_free_rate: float = 0.045
    # Leading session returns left out of reported metrics. None leaves out
    # every return up to and including the first session with a signal (the
    # strategies' warm-up).
    warmup_sessions: int | None = None
    short_borrow_rate: float = 0.003
    # A backstop that rejects grossly oversized orders. It cannot be 1.0:
    # NautilusTrader's free margin excludes unrealized gains, so fully
    # margining long-held winners would deny valid orders.
    margin_init: Decimal = Decimal("0.5")

    def sizing(self) -> SizingRules:
        return SizingRules(
            max_position_pct=self.max_position_pct,
            max_gross_exposure=self.max_gross_exposure,
            cash_buffer=self.cash_buffer,
        )


@dataclass
class NautilusBacktestResult:
    """Everything a run produces, with the ledger NAV as the source of truth."""

    config: NautilusBacktestConfig
    symbols: list[str]
    costs: dict[str, Any]
    nav: pd.Series
    returns: pd.Series
    reported_returns: pd.Series
    final_nav: float
    fills: pd.DataFrame
    positions_report: pd.DataFrame
    signals: pd.DataFrame
    metrics: MultiStrategyBacktestResults
    sharpe_rf0: float
    sortino_rf0: float
    cost_breakdown: dict[str, float]
    native_stats: dict[str, Any]
    # Fills in the reported window that reduced or closed a position.
    closed_trades: int
    winning_trades: int
    max_reconcile_diff: float

    @property
    def total_return(self) -> float:
        return self.final_nav / self.config.initial_capital - 1.0


def explicit_controller_config(
    strategy_names: list[str],
    n_symbols: int,
    *,
    vol_management: bool = True,
    risk_controller: bool = True,
) -> ControllerConfig:
    """A controller config whose limits do not fight the execution layer.

    The controller adds new targets on top of existing weights when it checks
    its single-symbol and gross caps, so a strategy re-sending the same target
    gets capped and its position shrinks. The caps here are wide enough never
    to bind. Position and gross limits are enforced on the resulting trades by
    ``SizingRules`` instead.
    """
    weight = 1.0 / len(strategy_names)
    return ControllerConfig(
        allocations={
            name: StrategyAllocation(weight=weight, max_positions=n_symbols)
            for name in strategy_names
        },
        max_gross_exposure=100.0,
        max_net_exposure=100.0,
        max_single_symbol_weight=100.0,
        max_portfolio_positions=max(40, n_symbols),
        enable_regime_adaptation=False,
        enable_risk_controller=risk_controller,
        enable_vol_management=vol_management,
        enable_entropy_filter=False,
    )


def make_costs(
    config: NautilusBacktestConfig, frames: Mapping[str, pd.DataFrame]
) -> ExecutionCosts:
    if config.costs == "realistic":
        return RealisticCosts(inputs=lagged_cost_inputs(frames), config=config.cost_model)
    if config.costs == "legacy":
        return LegacyCosts(
            slippage_bps=config.slippage_bps,
            commission_per_share=config.commission_per_share,
        )
    raise ValueError(f"unknown costs mode {config.costs!r}")


def build_engine(config: NautilusBacktestConfig, costs: ExecutionCosts) -> BacktestEngine:
    """A quiet engine with one MARGIN venue at 1x leverage.

    NautilusTrader's logging setup is process-wide and the first engine wins,
    so every engine in this package must be built here.
    """
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggerConfig(bypass_logging=True)))
    engine.add_venue(
        venue=Venue(config.venue),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(config.initial_capital, USD)],
        default_leverage=Decimal(1),
        fee_model=CostFeeModel(costs),
        fill_model=CostFillModel(costs),
    )
    return engine


def run_nautilus_backtest(
    controller: MultiStrategyController,
    frames: Mapping[str, pd.DataFrame],
    config: NautilusBacktestConfig | None = None,
) -> NautilusBacktestResult:
    """Run the controller's strategies over daily frames on NautilusTrader."""
    config = config or NautilusBacktestConfig()
    symbols = list(frames)
    sessions = build_sessions(frames)
    instruments = {s: make_equity(s, config.venue, config.margin_init) for s in symbols}
    raw_bars = {
        (symbol, ts.date()): (row.open, row.high, row.low, row.close, row.volume)
        for symbol, frame in frames.items()
        for ts, row in zip(frame.index, frame.itertuples(index=False), strict=True)
    }
    costs = make_costs(config, frames)

    engine = build_engine(config, costs)
    try:
        for instrument in instruments.values():
            engine.add_instrument(instrument)
        engine.add_data(to_nautilus_data(frames, instruments))
        strategy = ControllerStrategy(
            ControllerStrategyConfig(
                controller=controller,
                instruments=instruments,
                sessions=sessions,
                raw_bars=raw_bars,
                costs=costs,
                sizing=config.sizing(),
                initial_capital=config.initial_capital,
                venue=config.venue,
                liquidate_at_end=config.liquidate_at_end,
                order_id_tag="000",
            ),
        )
        engine.add_strategy(strategy)
        engine.run()

        if strategy.errors:
            raise BacktestIntegrityError("; ".join(strategy.errors[:5]))
        if strategy.final_nav is None:
            raise BacktestIntegrityError("backtest produced no sessions")
        if strategy.submitted_orders != len(strategy.ledger.fills):
            raise BacktestIntegrityError(
                f"{strategy.submitted_orders} orders submitted, "
                f"{len(strategy.ledger.fills)} fills received",
            )

        native = engine.get_result()
        native_stats = {
            "note": "NautilusTrader statistics use calendar-day returns, including zero weekends",
            "returns": dict(sorted(native.stats_returns.items())),
            "general": dict(sorted(native.stats_general.items())),
            "pnls": {k: dict(sorted(v.items())) for k, v in sorted(native.stats_pnls.items())},
        }
        positions_report = engine.generate_positions_report()
    finally:
        engine.dispose()

    return _build_result(config, symbols, costs, strategy, native_stats, positions_report)


def _build_result(
    config: NautilusBacktestConfig,
    symbols: list[str],
    costs: ExecutionCosts,
    strategy: ControllerStrategy,
    native_stats: dict[str, Any],
    positions_report: pd.DataFrame,
) -> NautilusBacktestResult:
    nav = pd.Series(dict(strategy.nav), name="nav", dtype=float)
    returns = pd.Series(dict(strategy.returns), name="return", dtype=float)
    fills = _fills_frame(strategy.ledger.fills)
    signals = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": sig.symbol,
                "strategy": sig.strategy_name,
                "direction": sig.direction,
                "target_weight": sig.target_weight,
                "confidence": sig.confidence,
            }
            for day, sig in strategy.signals
        ],
    )

    reported = _reported_returns(returns, strategy, config.warmup_sessions)
    if reported.empty:
        raise BacktestIntegrityError("no sessions left after the warm-up period")
    start_nav = float(nav.loc[: reported.index[0]].iloc[-2])
    # Like the runner, the curve ends with the NAV after the final liquidation.
    equity_curve = [start_nav, *nav.loc[reported.index].tolist(), float(strategy.final_nav)]
    window = [f for f in strategy.ledger.fills if f.day >= reported.index[0]]
    closed = [f for f in window if f.realized_pnl is not None]
    winners = sum(1 for f in closed if f.realized_pnl > 0)
    trades = _metric_trades(window)
    signal_counts: dict[str, int] = {}
    for _, sig in strategy.signals:
        signal_counts[sig.strategy_name] = signal_counts.get(sig.strategy_name, 0) + 1

    def metrics_at(rf: float) -> MultiStrategyBacktestResults:
        return compute_backtest_metrics(
            equity_curve=equity_curve,
            daily_returns=reported.tolist(),
            timestamps=list(reported.index),
            trades=trades,
            signals_by_strategy=signal_counts,
            closed_trades=len(closed),
            winning_trades=winners,
            initial_capital=start_nav,
            risk_free_rate=rf,
            trading_days_per_year=TRADING_DAYS,
        )

    metrics = metrics_at(config.risk_free_rate)
    at_zero = metrics_at(0.0)
    return NautilusBacktestResult(
        config=config,
        symbols=symbols,
        costs=costs.describe(),
        nav=nav,
        returns=returns,
        reported_returns=reported,
        final_nav=float(strategy.final_nav),
        fills=fills,
        positions_report=positions_report,
        signals=signals,
        metrics=metrics,
        sharpe_rf0=at_zero.sharpe_ratio,
        sortino_rf0=at_zero.sortino_ratio,
        cost_breakdown=_cost_breakdown(fills, strategy.short_notional, nav.index, config),
        native_stats=native_stats,
        closed_trades=len(closed),
        winning_trades=winners,
        max_reconcile_diff=strategy.max_reconcile_diff,
    )


def _reported_returns(
    returns: pd.Series,
    strategy: ControllerStrategy,
    warmup: int | None,
) -> pd.Series:
    if warmup is not None:
        return returns.iloc[warmup:]
    if not strategy.signals:
        return returns.iloc[0:0]
    first_signal_day = strategy.signals[0][0]
    return returns[returns.index > first_signal_day]


def _fills_frame(fills: list[Fill]) -> pd.DataFrame:
    columns = [
        "date",
        "symbol",
        "side",
        "quantity",
        "price",
        "reference_price",
        "commission",
        "slippage_cost",
        "realized_pnl",
        "strategy",
    ]
    rows = []
    for f in fills:
        signed = f.quantity if f.side == "BUY" else -f.quantity
        rows.append(
            {
                "date": f.day,
                "symbol": f.symbol,
                "side": f.side,
                "quantity": f.quantity,
                "price": f.price,
                "reference_price": f.reference_price,
                "commission": f.commission,
                "slippage_cost": signed * (f.price - f.reference_price),
                "realized_pnl": f.realized_pnl,
                "strategy": f.strategy,
            },
        )
    return pd.DataFrame(rows, columns=columns)


def _metric_trades(fills: list[Fill]) -> list[dict[str, Any]]:
    """Fills in the trade-dict shape ``compute_backtest_metrics`` reads.

    Fills that reduce or close a position carry their realized P&L (net of
    commission) as ``CLOSE_*`` entries, so profit factor and expectancy use the
    ledger's own P&L. Each such fill counts as one closed trade.
    """
    trades = []
    for f in fills:
        side = f.side
        record: dict[str, Any] = {
            "timestamp": f.day,
            "symbol": f.symbol,
            "shares": f.quantity if f.side == "BUY" else -f.quantity,
            "price": f.price,
            "strategy": f.strategy,
        }
        if f.realized_pnl is not None:
            side = "CLOSE_LONG" if f.side == "SELL" else "CLOSE_SHORT"
            record["pnl"] = f.realized_pnl
        record["side"] = side
        trades.append(record)
    return trades


def _cost_breakdown(
    fills: pd.DataFrame,
    short_notional: list[tuple[date, float]],
    days: pd.Index,
    config: NautilusBacktestConfig,
) -> dict[str, float]:
    """Commission and slippage actually charged, plus an estimated borrow cost.

    Borrow is not charged by the engine. The estimate accrues the configured
    annual rate on each session's short notional over the calendar days to the
    next session (ACT/360).
    """
    borrow = 0.0
    for (day, notional), next_day in zip(short_notional, [*days[1:], None], strict=True):
        if notional > 0 and next_day is not None:
            borrow += notional * config.short_borrow_rate * (next_day - day).days / 360.0
    return {
        "commission": float(fills["commission"].sum()) if len(fills) else 0.0,
        "slippage": float(fills["slippage_cost"].sum()) if len(fills) else 0.0,
        "short_borrow_estimate_not_charged": borrow,
        "traded_notional": float((fills["quantity"] * fills["price"]).sum()) if len(fills) else 0.0,
    }
