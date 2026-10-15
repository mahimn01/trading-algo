"""NautilusTrader strategy that runs the repo's ``MultiStrategyController``.

The controller and its strategy adapters are used unchanged. This class only
replaces the execution and accounting that ``MultiStrategyBacktestRunner`` does
by hand.

Timeline for each session (all times New York):

* 09:30 open tick, then the open alert: pending targets from the previous
  close become market orders. Reductions are sized first and sells are sent
  first, so their cash is available to buys. They fill at the open, moved by
  the cost model.
* 16:00 bar, then the close alert. It runs the runner's steps in the runner's
  order: ``generate_signals`` (only after every symbol's bar for the session
  has arrived, so there is no look-ahead), ``update_portfolio_state``, record
  the session return, ``add_return``, ``new_trading_day``, and
  ``detect_regime`` every 5th session when enabled. Like the runner, no
  signals are generated on the first session.

Behaviour kept from the runner on purpose, so the two engines stay comparable:

* Share counts are sized at the session's open price, as a market-on-open
  order filled at a known open would be.
* A signal for a symbol with no bar the next session waits until that
  symbol trades again.
* ``new_trading_day`` resets the controller's daily P&L right after it is
  set, so the controller's daily-loss stop cannot fire in either engine.

If the controller returns several signals for one symbol, the last one sets
the target and a single order trades to it (the runner trades each in turn).

The strategy keeps its own cash and position ledger from fill events. That
ledger is the official per-session NAV, and it is reconciled against
NautilusTrader's positions and account at every close. NautilusTrader logs and
swallows exceptions raised in strategy callbacks, so every callback records
its exception in ``errors`` and ``run_nautilus_backtest`` raises it.
"""

from __future__ import annotations

import functools
import logging
import math
import traceback
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import OrderSide, Quantity, Venue
from nautilus_trader.trading import Strategy

from trading_algo.multi_strategy.controller import MultiStrategyController
from trading_algo.multi_strategy.protocol import StrategySignal
from trading_algo.nautilus_backtest.costs import ExecutionCosts
from trading_algo.nautilus_backtest.data import USD, Session, bar_type_for

logger = logging.getLogger(__name__)

# NautilusTrader books every fill's cash in whole cents and the ledger does
# not, so each fill can move the two apart by up to half a cent.
_RECONCILE_CENTS_PER_FILL = 0.01
_RECONCILE_TOLERANCE_REL = 1e-9
# Sell estimates can size a short a few basis points past its target.
_GROSS_SLACK = 0.01


@dataclass(frozen=True)
class SizingRules:
    """Position sizing, matching ``MultiStrategyBacktestRunner._open_position``.

    ``max_gross_exposure`` only limits trades that increase exposure, so it
    never blocks a reduction (the runner's version does).
    """

    max_position_pct: float = 0.25
    max_gross_exposure: float = 1.0
    cash_buffer: float = 0.95


@dataclass
class Fill:
    """One fill as recorded by the ledger.

    ``realized_pnl`` is set on fills that reduce or close a position. It is net
    of the commission paid to open and to close the shares involved.
    """

    day: date
    ts_ns: int
    symbol: str
    side: str
    quantity: int
    price: float
    reference_price: float
    commission: float
    strategy: str
    realized_pnl: float | None = None


@dataclass
class Ledger:
    """Cash and whole-share positions, updated from fill events.

    ``cost`` is the average cost per share of each position, including the
    commission paid to open it (for shorts, the average proceeds net of it).
    """

    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def equity(self, marks: Mapping[str, float]) -> float:
        return self.cash + sum(qty * marks[sym] for sym, qty in self.positions.items())

    def apply(self, fill: Fill) -> None:
        symbol, quantity = fill.symbol, fill.quantity
        signed = quantity if fill.side == "BUY" else -quantity
        previous = self.positions.get(symbol, 0)
        updated = previous + signed
        fee_per_share = fill.commission / quantity

        closed = 0
        if previous != 0 and (previous > 0) != (signed > 0):
            closed = min(quantity, abs(previous))
            direction = 1 if previous > 0 else -1
            fill.realized_pnl = (fill.price - self.cost[symbol]) * closed * direction - (
                fee_per_share * closed
            )

        opened = quantity - closed
        if updated == 0:
            self.positions.pop(symbol)
            self.cost.pop(symbol)
        else:
            direction = 1 if updated > 0 else -1
            entry = fill.price + direction * fee_per_share
            if opened and abs(updated) == opened:  # new position, or the remainder of a flip
                self.cost[symbol] = entry
            elif opened:  # adding to a position in the same direction
                held = abs(previous)
                self.cost[symbol] = (held * self.cost[symbol] + opened * entry) / (held + opened)
            self.positions[symbol] = updated
        self.cash -= signed * fill.price + fill.commission
        self.fills.append(fill)


class ControllerStrategyConfig(StrategyConfig):
    """Inputs for ``ControllerStrategy``. Only ``config`` reaches the strategy."""

    def __init__(
        self,
        *,
        controller: MultiStrategyController,
        instruments: Mapping[str, Any],
        sessions: Sequence[Session],
        raw_bars: Mapping[tuple[str, date], tuple[float, float, float, float, float]],
        costs: ExecutionCosts,
        sizing: SizingRules,
        initial_capital: float,
        venue: str,
        liquidate_at_end: bool,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.instruments = dict(instruments)
        self.sessions = list(sessions)
        self.raw_bars = raw_bars
        self.costs = costs
        self.sizing = sizing
        self.initial_capital = initial_capital
        self.venue = venue
        self.liquidate_at_end = liquidate_at_end


def _guarded(handler: Callable[..., None]) -> Callable[..., None]:
    """Record a callback's exception and stop processing further events."""

    @functools.wraps(handler)
    def wrapper(self: ControllerStrategy, *args: Any) -> None:
        if self.errors:
            return
        try:
            handler(self, *args)
        except Exception:
            self.errors.append(traceback.format_exc())

    return wrapper


class ControllerStrategy(Strategy):
    """Drive an unchanged ``MultiStrategyController`` inside NautilusTrader."""

    def __init__(self, config: ControllerStrategyConfig) -> None:
        super().__init__(config)
        self.controller = config.controller
        self.symbols = list(config.instruments)
        self.ledger = Ledger(cash=config.initial_capital)
        self.marks: dict[str, float] = {}
        self.nav: list[tuple[date, float]] = []
        self.returns: list[tuple[date, float]] = []
        self.final_nav: float | None = None
        self.signals: list[tuple[date, StrategySignal]] = []
        self.errors: list[str] = []
        self.max_reconcile_diff = 0.0
        self.short_notional: list[tuple[date, float]] = []
        self.submitted_orders = 0
        self._symbol_by_bar_type = {
            bar_type_for(inst): sym for sym, inst in config.instruments.items()
        }
        self._pending: dict[str, StrategySignal] = {}
        self._open_price: dict[str, float] = {}
        self._session_index: dict[str, int] = {}
        self._day_open_equity = config.initial_capital
        self._boundaries = 0
        self._references: dict[str, tuple[float, str]] = {}

    # ── lifecycle ────────────────────────────────────────────────────────

    def on_start(self) -> None:
        for instrument in self.config.instruments.values():
            self.subscribe_bars(bar_type_for(instrument))
            self.subscribe_trades(instrument.id)
            self.subscribe_mark_prices(instrument.id)
        for i, session in enumerate(self.config.sessions):
            self.clock.set_time_alert_ns(f"open-{i}", session.open_ns + 1, self._on_open)
            self.clock.set_time_alert_ns(f"close-{i}", session.close_ns + 1, self._on_close)

    @_guarded
    def on_stop(self) -> None:
        if self.marks:
            # Covers the end-of-run liquidation fills, which happen after the
            # last close's reconciliation.
            self._reconcile(self.config.sessions[-1].day)
            if not self.errors:
                self.final_nav = self.ledger.equity(self.marks)

    # ── market data ──────────────────────────────────────────────────────

    @_guarded
    def on_trade(self, tick: Any) -> None:
        symbol = tick.instrument_id.symbol.value
        self._open_price[symbol] = float(tick.price)
        self._session_index[symbol] = self._current_session(tick.ts_event)

    @_guarded
    def on_bar(self, bar: Any) -> None:
        symbol = self._symbol_by_bar_type[bar.bar_type]
        session = self.config.sessions[self._current_session(bar.ts_event)]
        open_, high, low, close, volume = self.config.raw_bars[(symbol, session.day)]
        self.controller.update(symbol, session.timestamp, open_, high, low, close, volume)
        self.marks[symbol] = close

    # ── session alerts ───────────────────────────────────────────────────

    @_guarded
    def _on_open(self, event: Any) -> None:
        index = int(event.name.split("-")[1])
        session = self.config.sessions[index]
        tradable = {s for s, i in self._session_index.items() if i == index}
        ready = {s: sig for s, sig in self._pending.items() if s in tradable}
        if not ready:
            return
        for symbol in ready:
            del self._pending[symbol]

        equity = self.ledger.equity(self.marks)
        # Exposure is valued at today's open where a symbol has traded, else
        # at its last close.
        prices = {s: self._open_price[s] if s in tradable else c for s, c in self.marks.items()}
        # Fills arrive after this callback, so each order is sized against the
        # positions the earlier orders in this batch will leave behind.
        # Reductions go first so they free room for increases.
        planned = dict(self.ledger.positions)
        signals = sorted(
            ready.values(),
            key=lambda sig: not self._reduces(sig, planned, prices, equity),
        )
        orders = []
        for sig in signals:
            order = self._target_order(sig, session.day, equity, planned, prices)
            if order is not None:
                symbol, side, quantity, _ = order
                planned[symbol] = planned.get(symbol, 0) + (
                    quantity if side == "BUY" else -quantity
                )
                orders.append(order)
        self._check_gross(planned, prices, equity, session.day)

        cash = self.ledger.cash
        for symbol, side, quantity, tag in sorted(orders, key=lambda o: o[1] != "SELL"):
            if side == "BUY":
                quantity = self._cap_buy_to_cash(symbol, session.day, quantity, cash)
                if quantity < 1:
                    continue
            cash += self._submit(symbol, side, quantity, session.day, self._open_price[symbol], tag)

    @_guarded
    def _on_close(self, event: Any) -> None:
        index = int(event.name.split("-")[1])
        session = self.config.sessions[index]
        if not self.marks:
            return
        self._reconcile(session.day)
        equity = self.ledger.equity(self.marks)
        self.nav.append((session.day, equity))
        self.short_notional.append(
            (
                session.day,
                sum(-q * self.marks[s] for s, q in self.ledger.positions.items() if q < 0),
            ),
        )

        if index > 0:
            if equity > 0:
                signals = self.controller.generate_signals(self.symbols, session.timestamp, equity)
                for sig in signals:
                    self._pending[sig.symbol] = sig
                    self.signals.append((session.day, sig))
            weights = (
                {sym: qty * self.marks[sym] / equity for sym, qty in self.ledger.positions.items()}
                if equity > 0
                else {}
            )
            daily_return = equity / self._day_open_equity - 1 if self._day_open_equity > 0 else 0.0
            self.controller.update_portfolio_state(
                equity=equity,
                positions=weights,
                daily_pnl=daily_return,
            )
            self.returns.append((session.day, daily_return))
            self.controller.add_return(daily_return)
            self.controller.new_trading_day()
            self._day_open_equity = equity
            self._boundaries += 1
            if self._boundaries % 5 == 0 and self.controller.config.enable_regime_adaptation:
                self.controller.detect_regime()

        if index == len(self.config.sessions) - 1 and self.config.liquidate_at_end:
            self._pending.clear()
            for symbol, qty in sorted(self.ledger.positions.items(), key=lambda kv: kv[1] < 0):
                side = "SELL" if qty > 0 else "BUY"
                self._submit(symbol, side, abs(qty), session.day, self.marks[symbol], "liquidation")

    # ── order logic ──────────────────────────────────────────────────────

    def _target_value(self, sig: StrategySignal, equity: float) -> float:
        weight = min(abs(sig.target_weight), self.config.sizing.max_position_pct)
        return equity * weight * sig.direction

    def _reduces(
        self,
        sig: StrategySignal,
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        equity: float,
    ) -> bool:
        """True when the signal shrinks or flips the symbol's current position."""
        current = positions.get(sig.symbol, 0)
        if sig.is_exit or current == 0:
            return sig.is_exit
        target = self._target_value(sig, equity)
        held = current * prices[sig.symbol]
        return abs(target) < abs(held) or (target > 0) != (held > 0)

    def _target_order(
        self,
        sig: StrategySignal,
        day: date,
        equity: float,
        positions: Mapping[str, int],
        prices: Mapping[str, float],
    ) -> tuple[str, str, int, str] | None:
        """Turn a target-weight signal into one whole-share order, or None."""
        symbol = sig.symbol
        price = self._open_price[symbol]
        current = positions.get(symbol, 0)
        tag = sig.strategy_name
        if sig.is_exit:
            if current == 0:
                return None
            return (symbol, "SELL" if current > 0 else "BUY", abs(current), tag)
        if equity <= 0:
            return None

        target = self._target_value(sig, equity)
        other_gross = sum(abs(q * prices[s]) for s, q in positions.items() if s != symbol)
        room = max(0.0, equity * self.config.sizing.max_gross_exposure - other_gross)
        if abs(target) > room and abs(target) > abs(current * price):
            target = math.copysign(max(room, abs(current * price)), target)
        delta_value = target - current * price
        if delta_value == 0:
            return None
        side = "BUY" if delta_value > 0 else "SELL"
        estimate = self.config.costs.fill_price(
            symbol,
            day,
            side,
            abs(delta_value) / price,
            price,
        )
        quantity = int(abs(delta_value) / estimate)
        if quantity < 1:
            return None
        return (symbol, side, quantity, tag)

    def _check_gross(
        self,
        planned: Mapping[str, int],
        prices: Mapping[str, float],
        equity: float,
        day: date,
    ) -> None:
        """Fail if planned orders would push exposure past its cap.

        Reductions are allowed even when exposure is already above the cap
        after a price move.
        """
        current = sum(abs(q * prices[s]) for s, q in self.ledger.positions.items())
        after = sum(abs(q * prices[s]) for s, q in planned.items())
        limit = max(equity * self.config.sizing.max_gross_exposure, current)
        if after > limit * (1 + _GROSS_SLACK):
            raise RuntimeError(f"{day}: planned gross exposure {after:.2f} exceeds {limit:.2f}")

    def _cap_buy_to_cash(self, symbol: str, day: date, quantity: int, cash: float) -> int:
        """Shrink a buy to ``cash_buffer`` of available cash, as the runner does."""
        price = self._open_price[symbol]
        fill = self.config.costs.fill_price(symbol, day, "BUY", quantity, price)
        commission = self.config.costs.commission("BUY", quantity, fill)
        if quantity * fill + commission <= cash:
            return quantity
        budget = cash * self.config.sizing.cash_buffer - commission
        if budget <= 0:
            return 0
        return int(budget / fill)

    def _submit(
        self,
        symbol: str,
        side: str,
        quantity: int,
        day: date,
        reference: float,
        tag: str,
    ) -> float:
        """Submit a market order and return its expected cash change."""
        instrument = self.config.instruments[symbol]
        order = self.order_factory.market(
            instrument_id=instrument.id,
            order_side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            quantity=Quantity.from_int(quantity),
        )
        self._references[order.client_order_id.value] = (reference, tag)
        self.submitted_orders += 1
        self.submit_order(order)
        fill = self.config.costs.fill_price(symbol, day, side, quantity, reference)
        commission = self.config.costs.commission(side, quantity, fill)
        signed = quantity if side == "BUY" else -quantity
        return -(signed * fill + commission)

    # ── events ───────────────────────────────────────────────────────────

    @_guarded
    def on_order_filled(self, event: Any) -> None:
        reference, tag = self._references[event.client_order_id.value]
        self.ledger.apply(
            Fill(
                day=self.config.sessions[self._current_session(event.ts_event)].day,
                ts_ns=event.ts_event,
                symbol=event.instrument_id.symbol.value,
                side="BUY" if event.order_side == OrderSide.BUY else "SELL",
                quantity=int(event.last_qty),
                price=float(event.last_px),
                reference_price=reference,
                commission=float(event.commission),
                strategy=tag,
            ),
        )

    def on_order_denied(self, event: Any) -> None:
        self.errors.append(f"order denied: {event.client_order_id} {event.reason}")

    def on_order_rejected(self, event: Any) -> None:
        self.errors.append(f"order rejected: {event.client_order_id} {event.reason}")

    # ── helpers ──────────────────────────────────────────────────────────

    def _current_session(self, ts_ns: int) -> int:
        """Index of the latest session that has opened at or before ``ts_ns``."""
        sessions = self.config.sessions
        lo, hi = 0, len(sessions) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sessions[mid].open_ns <= ts_ns:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _reconcile(self, day: date) -> None:
        """Check the ledger's positions and NAV against NautilusTrader's."""
        account = self.portfolio.account(Venue(self.config.venue))
        engine_nav = account.balance_total(USD).as_double()
        for symbol, instrument in self.config.instruments.items():
            engine_qty = int(self.portfolio.net_position(instrument.id))
            ledger_qty = self.ledger.positions.get(symbol, 0)
            if engine_qty != ledger_qty:
                raise RuntimeError(
                    f"{day}: {symbol} position {ledger_qty} in the ledger, "
                    f"{engine_qty} in NautilusTrader",
                )
            if engine_qty:
                unrealized = self.portfolio.unrealized_pnl(instrument.id)
                if unrealized is None:
                    raise RuntimeError(f"{day}: no unrealized P&L for held {symbol}")
                engine_nav += unrealized.as_double()
        ledger_nav = self.ledger.equity(self.marks)
        diff = abs(engine_nav - ledger_nav)
        self.max_reconcile_diff = max(self.max_reconcile_diff, diff)
        tolerance = _RECONCILE_CENTS_PER_FILL * (
            len(self.ledger.fills) + 1
        ) + _RECONCILE_TOLERANCE_REL * abs(ledger_nav)
        if diff > tolerance:
            raise RuntimeError(
                f"{day}: ledger NAV {ledger_nav:.2f} != NautilusTrader NAV {engine_nav:.2f}",
            )
