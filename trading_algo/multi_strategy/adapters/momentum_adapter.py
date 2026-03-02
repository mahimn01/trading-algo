"""
Adapter wrapping the Pure Momentum strategy.

Pure Momentum is a daily-rebalancing trend-following strategy.
It ranks assets by momentum, filters by trend (price > 200 MA),
sizes positions with inverse-volatility weighting, and targets
150% gross exposure.

The adapter maps MomentumSignal dicts into per-symbol
StrategySignals for the controller.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.quant_core.strategies.pure_momentum import (
    PureMomentumStrategy,
    MomentumConfig,
    TrendState,
)

logger = logging.getLogger(__name__)


class MomentumStrategyAdapter(TradingStrategy):
    """
    Wraps PureMomentumStrategy as a TradingStrategy.

    Needs at least ``trend_ma`` bars (default 200) before
    producing signals.  After warmup, it emits one signal per
    symbol with positive momentum above threshold.
    """

    def __init__(self, config: Optional[MomentumConfig] = None):
        self._momentum = PureMomentumStrategy(config)
        self._config = config or MomentumConfig()
        self._state = StrategyState.WARMING_UP

        # Price history (close prices) for building numpy arrays
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._bars_per_symbol: Dict[str, int] = defaultdict(int)

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "PureMomentum"

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
        # Feed the underlying strategy
        self._momentum.update(symbol, close)

        # Keep our own history for generate_signals
        self._price_history[symbol].append(close)
        self._bars_per_symbol[symbol] += 1

        # Trim to bounded length
        max_len = self._config.trend_ma + 20
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

        # Activate once any symbol has enough history
        if self._state == StrategyState.WARMING_UP:
            ready = sum(
                1 for n in self._bars_per_symbol.values()
                if n >= self._config.trend_ma
            )
            if ready >= 1:
                self._state = StrategyState.ACTIVE

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        if self._state != StrategyState.ACTIVE:
            return []

        # Build numpy price arrays from history
        prices: Dict[str, np.ndarray] = {}
        for s in symbols:
            hist = self._price_history.get(s)
            if hist and len(hist) >= self._config.trend_ma:
                prices[s] = np.array(hist, dtype=np.float64)

        if not prices:
            return []

        # Generate momentum signals
        mom_signals = self._momentum.generate_signals(
            symbols=list(prices.keys()),
            prices=prices,
        )

        signals: List[StrategySignal] = []
        for symbol, ms in mom_signals.items():
            if ms.position_size == 0:
                continue  # No position for this symbol

            # Determine direction from position_size sign
            if ms.position_size < 0:
                # Short signal
                direction = -1
                target_weight = abs(ms.position_size)
            else:
                # Long signal — use trend for direction
                if ms.trend in (TrendState.STRONG_UP, TrendState.UP):
                    direction = 1
                elif ms.trend in (TrendState.STRONG_DOWN, TrendState.DOWN):
                    direction = -1
                else:
                    continue  # NEUTRAL => skip
                target_weight = ms.position_size

            # Confidence from momentum score magnitude
            confidence = min(1.0, abs(ms.momentum_score))

            signals.append(StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                direction=direction,
                target_weight=target_weight,
                confidence=confidence,
                entry_price=ms.entry_price,
                trade_type="momentum",
                metadata={
                    "trend": ms.trend.name,
                    "momentum_score": ms.momentum_score,
                },
            ))

        return signals

    def get_current_exposure(self) -> float:
        # Use the underlying strategy's target weights
        prices: Dict[str, np.ndarray] = {}
        for s, hist in self._price_history.items():
            if len(hist) >= self._config.trend_ma:
                prices[s] = np.array(hist, dtype=np.float64)
        if not prices:
            return 0.0
        weights = self._momentum.get_target_weights(list(prices.keys()), prices)
        return sum(weights.values())

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self._momentum.reset()
        self._price_history.clear()
        self._bars_per_symbol.clear()
        self._state = StrategyState.WARMING_UP
