from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from typing import Dict, List

import numpy as np

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.options.wheel import (
    TradeEvent,
    WheelConfig,
    WheelStrategy,
)

log = logging.getLogger(__name__)


class WheelStrategyAdapter(TradingStrategy):
    """Adapts the Wheel options strategy for the multi-strategy controller.

    Accumulates OHLCV bars, feeds daily close/iv/iv_rank into the Wheel,
    and translates trade events into StrategySignal objects.
    """

    def __init__(
        self,
        symbol: str,
        wheel_config: WheelConfig | None = None,
        warmup_bars: int = 60,
        iv_rv_window: int = 30,
    ) -> None:
        self._symbol = symbol.upper()
        self._wheel = WheelStrategy(wheel_config or WheelConfig())
        self._state = StrategyState.WARMING_UP
        self._warmup_bars = warmup_bars
        self._iv_rv_window = iv_rv_window

        self._bar_count = 0
        self._close_history: deque[float] = deque(maxlen=500)
        self._iv_series: deque[float] = deque(maxlen=500)
        self._pending_events: list[TradeEvent] = []
        self._last_bar_date: datetime | None = None
        self._capital_in_use: float = 0.0

    @property
    def name(self) -> str:
        return f"Wheel_{self._symbol}"

    @property
    def state(self) -> StrategyState:
        return self._state

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        if symbol.upper() != self._symbol:
            return

        self._close_history.append(close)
        self._bar_count += 1
        self._last_bar_date = timestamp

        if self._state == StrategyState.WARMING_UP and self._bar_count >= self._warmup_bars:
            self._state = StrategyState.ACTIVE
            log.info("Wheel_%s: warmup complete after %d bars", self._symbol, self._bar_count)

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        if self._symbol.upper() not in [s.upper() for s in symbols]:
            return []

        if len(self._close_history) < max(self._iv_rv_window, 2):
            return []

        price = self._close_history[-1]
        iv = self._compute_iv()
        self._iv_series.append(iv)
        iv_rank = self._compute_iv_rank()

        events = self._wheel.on_bar(timestamp, price, iv, iv_rank)
        self._pending_events.extend(events)
        self._update_capital_tracking()

        return self._events_to_signals(events, timestamp, price)

    def get_current_exposure(self) -> float:
        cfg = self._wheel.cfg
        if cfg.initial_capital <= 0:
            return 0.0
        cash_used = cfg.initial_capital - self._wheel.cash
        stock_value = self._wheel.stock_qty * (self._close_history[-1] if self._close_history else 0.0)
        return max(0.0, min(1.0, (cash_used + stock_value) / cfg.initial_capital))

    def get_performance_stats(self) -> Dict[str, float]:
        s = self._wheel.summary()
        return {
            "total_return_pct": s.get("total_return_pct", 0.0),
            "win_rate": s.get("win_rate", 0.0),
            "total_trades": float(s.get("total_trades", 0)),
            "wheel_cycles": float(s.get("wheel_cycles", 0)),
            "net_premium": s.get("net_premium", 0.0),
            "sharpe_ratio": s.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": s.get("max_drawdown_pct", 0.0),
        }

    def reset(self) -> None:
        self._wheel = WheelStrategy(self._wheel.cfg)
        self._state = StrategyState.WARMING_UP
        self._bar_count = 0
        self._close_history.clear()
        self._iv_series.clear()
        self._pending_events.clear()
        self._last_bar_date = None
        self._capital_in_use = 0.0

    def _compute_iv(self) -> float:
        prices = list(self._close_history)
        window = min(self._iv_rv_window, len(prices) - 1)
        if window < 2:
            return 0.30

        recent = prices[-(window + 1):]
        log_returns = [
            math.log(recent[i + 1] / recent[i])
            for i in range(len(recent) - 1)
            if recent[i] > 0 and recent[i + 1] > 0
        ]
        if len(log_returns) < 2:
            return 0.30

        rv = float(np.std(log_returns)) * math.sqrt(252)
        return max(rv, 0.05)

    def _compute_iv_rank(self) -> float:
        if len(self._iv_series) < 2:
            return 50.0

        current = self._iv_series[-1]
        iv_min = min(self._iv_series)
        iv_max = max(self._iv_series)
        iv_range = iv_max - iv_min
        if iv_range <= 0:
            return 50.0

        return max(0.0, min(100.0, (current - iv_min) / iv_range * 100.0))

    def _events_to_signals(
        self,
        events: list[TradeEvent],
        timestamp: datetime,
        price: float,
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []

        for ev in events:
            if ev.event_type == "sell_put":
                strike = ev.details.get("strike", price)
                premium = ev.details.get("premium", 0.0)
                contracts = ev.details.get("contracts", 1)
                capital_at_risk = strike * contracts * 100
                weight = capital_at_risk / self._wheel.cfg.initial_capital if self._wheel.cfg.initial_capital > 0 else 0.0
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=self._symbol,
                    direction=-1,
                    target_weight=min(weight, 1.0),
                    confidence=min(ev.details.get("iv_rank", 50.0) / 100.0, 1.0),
                    entry_price=premium,
                    trade_type="wheel_csp",
                    metadata={
                        "event": ev.event_type,
                        "strike": strike,
                        "dte": ev.details.get("dte", 0),
                        "contracts": contracts,
                        "iv": ev.details.get("iv", 0.0),
                        "phase": "CSP",
                    },
                ))

            elif ev.event_type == "sell_call":
                strike = ev.details.get("strike", price)
                premium = ev.details.get("premium", 0.0)
                contracts = ev.details.get("contracts", 1)
                weight = (self._wheel.stock_qty * price) / self._wheel.cfg.initial_capital if self._wheel.cfg.initial_capital > 0 else 0.0
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=self._symbol,
                    direction=-1,
                    target_weight=min(weight, 1.0),
                    confidence=0.7,
                    entry_price=premium,
                    trade_type="wheel_cc",
                    metadata={
                        "event": ev.event_type,
                        "strike": strike,
                        "contracts": contracts,
                        "cost_basis": ev.details.get("cost_basis", 0.0),
                        "phase": "CC",
                    },
                ))

            elif ev.event_type.startswith("close_put") or ev.event_type.startswith("close_call"):
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=self._symbol,
                    direction=0,
                    target_weight=0.0,
                    confidence=0.9,
                    trade_type="wheel_close",
                    metadata={
                        "event": ev.event_type,
                        "reason": ev.details.get("reason", ""),
                        "pnl": ev.pnl,
                    },
                ))

            elif ev.event_type == "put_assigned":
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=self._symbol,
                    direction=1,
                    target_weight=self.get_current_exposure(),
                    confidence=1.0,
                    entry_price=ev.details.get("cost_basis", price),
                    trade_type="wheel_assignment",
                    metadata={
                        "event": ev.event_type,
                        "shares": ev.details.get("shares", 0),
                        "strike": ev.details.get("strike", 0.0),
                    },
                ))

            elif ev.event_type == "called_away":
                signals.append(StrategySignal(
                    strategy_name=self.name,
                    symbol=self._symbol,
                    direction=0,
                    target_weight=0.0,
                    confidence=1.0,
                    trade_type="wheel_called_away",
                    metadata={
                        "event": ev.event_type,
                        "stock_pnl": ev.details.get("stock_pnl", 0.0),
                        "option_pnl": ev.details.get("option_pnl", 0.0),
                        "total_pnl": ev.pnl,
                    },
                ))

        return signals

    def _update_capital_tracking(self) -> None:
        if self._wheel.short_option is not None:
            opt = self._wheel.short_option
            self._capital_in_use = opt.strike * opt.contracts * 100
        elif self._wheel.stock_qty > 0 and self._close_history:
            self._capital_in_use = self._wheel.stock_qty * self._close_history[-1]
        else:
            self._capital_in_use = 0.0
