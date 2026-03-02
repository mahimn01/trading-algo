"""
Adapter wrapping the Regime Transition strategy.

Regime Transition exploits Hidden Markov Model regime TRANSITIONS
rather than regime states.  It monitors HMM transition probability
dynamics and trades on rising/falling transition probabilities toward
bullish or bearish regimes, capturing alpha from forward-looking
information embedded in the transition matrix.

The adapter maintains per-symbol price and return histories, lazily
creates and fits HMM models, and maps TransitionSignal objects into
per-symbol StrategySignals for the controller.
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
from trading_algo.quant_core.strategies.regime_transition import (
    RegimeTransitionStrategy,
    TransitionConfig,
)
from trading_algo.quant_core.models.hmm_regime import HiddenMarkovRegime
from trading_algo.quant_core.utils.math_utils import simple_returns

logger = logging.getLogger(__name__)

# Minimum bars before the strategy can activate.
_WARMUP_BARS = 60


class RegimeTransitionAdapter(TradingStrategy):
    """
    Wraps RegimeTransitionStrategy as a TradingStrategy.

    Aggregates intraday bars into daily OHLC before passing to the
    underlying strategy, since the HMM transition thresholds are
    calibrated for daily data (not 5-minute bars).

    Needs at least ~60 daily bars before the strategy begins producing
    signals.
    """

    def __init__(
        self,
        config: Optional[TransitionConfig] = None,
        hmm_n_states: int = 3,
        hmm_retrain_frequency: int = 21,
    ):
        self._config = config or TransitionConfig()
        self._state = StrategyState.WARMING_UP

        # Per-symbol strategy instances (avoids mixing transition matrices
        # from different symbols in the shared deque)
        self._strategies: Dict[str, RegimeTransitionStrategy] = {}

        # Per-symbol DAILY aggregated histories (not 5-min bars)
        self._daily_close: Dict[str, List[float]] = defaultdict(list)
        self._daily_high: Dict[str, List[float]] = defaultdict(list)
        self._daily_low: Dict[str, List[float]] = defaultdict(list)
        self._daily_count: Dict[str, int] = defaultdict(int)

        # Intraday accumulators for current day
        self._current_day: Dict[str, object] = {}  # symbol -> date
        self._intraday_high: Dict[str, float] = defaultdict(lambda: -np.inf)
        self._intraday_low: Dict[str, float] = defaultdict(lambda: np.inf)
        self._intraday_close: Dict[str, float] = {}

        # Per-symbol HMM models (lazily created)
        self._hmm_models: Dict[str, HiddenMarkovRegime] = {}
        self._hmm_n_states = hmm_n_states
        self._hmm_retrain_frequency = hmm_retrain_frequency

        # Track last signal per symbol for exposure calculation
        self._last_signals: Dict[str, float] = {}

    # ── Protocol implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "RegimeTransition"

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
        current_date = timestamp.date()

        # Aggregate intraday bars into daily OHLC
        prev_date = self._current_day.get(symbol)
        if prev_date is not None and current_date != prev_date:
            # New day: flush previous day's aggregated bar
            self._daily_close[symbol].append(self._intraday_close[symbol])
            self._daily_high[symbol].append(self._intraday_high[symbol])
            self._daily_low[symbol].append(self._intraday_low[symbol])
            self._daily_count[symbol] += 1

            # Trim histories to bounded length
            max_len = self._config.regime_return_lookback + 50
            for hist in (
                self._daily_close[symbol],
                self._daily_high[symbol],
                self._daily_low[symbol],
            ):
                if len(hist) > max_len:
                    del hist[: len(hist) - max_len]

            # Reset intraday accumulators
            self._intraday_high[symbol] = high
            self._intraday_low[symbol] = low
        else:
            # Same day: update intraday extremes
            self._intraday_high[symbol] = max(self._intraday_high[symbol], high)
            self._intraday_low[symbol] = min(self._intraday_low[symbol], low)

        self._intraday_close[symbol] = close
        self._current_day[symbol] = current_date

        # Transition from WARMING_UP to ACTIVE once any symbol has
        # enough daily history.
        if self._state == StrategyState.WARMING_UP:
            ready = sum(
                1
                for n in self._daily_count.values()
                if n >= _WARMUP_BARS
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

        signals: List[StrategySignal] = []

        for symbol in symbols:
            daily_closes = self._daily_close.get(symbol)
            if not daily_closes or len(daily_closes) < _WARMUP_BARS:
                continue

            prices = np.array(daily_closes, dtype=np.float64)
            returns = simple_returns(prices)
            if len(returns) < self._config.min_regime_history:
                continue

            high_prices = np.array(
                self._daily_high.get(symbol, []), dtype=np.float64
            )
            low_prices = np.array(
                self._daily_low.get(symbol, []), dtype=np.float64
            )

            # Lazy-create and fit HMM model
            hmm = self._hmm_models.get(symbol)
            if hmm is None:
                hmm = HiddenMarkovRegime(
                    n_states=self._hmm_n_states,
                    n_iter=20,  # 20 EM iters (vs default 100) for speed
                    retrain_frequency=self._hmm_retrain_frequency,
                )
                self._hmm_models[symbol] = hmm

            bar_idx = self._daily_count[symbol]
            if not hmm._is_fitted or hmm.should_retrain(bar_idx):
                try:
                    hmm.fit(prices)
                    hmm._last_train_bar = bar_idx
                except Exception:
                    logger.debug(
                        "HMM fit failed for %s, skipping signal generation",
                        symbol,
                    )
                    continue

            if not hmm._is_fitted:
                continue

            # Get or create per-symbol strategy instance
            strategy = self._strategies.get(symbol)
            if strategy is None:
                strategy = RegimeTransitionStrategy(self._config)
                self._strategies[symbol] = strategy

            # Generate signal from the per-symbol strategy
            try:
                ts_signal = strategy.generate_signal(
                    symbol=symbol,
                    prices=prices,
                    returns=returns,
                    hmm_model=hmm,
                    timestamp=timestamp,
                    high_prices=high_prices if len(high_prices) > 0 else None,
                    low_prices=low_prices if len(low_prices) > 0 else None,
                )
            except Exception:
                logger.debug(
                    "Regime transition signal generation failed for %s",
                    symbol,
                    exc_info=True,
                )
                continue

            if ts_signal is None:
                continue

            # Skip flat signals with zero position
            if abs(ts_signal.position_size) < 1e-9 and ts_signal.direction == 0.0:
                self._last_signals.pop(symbol, None)
                continue

            # Map direction to discrete -1/0/+1
            if ts_signal.direction > 0:
                direction = 1
            elif ts_signal.direction < 0:
                direction = -1
            else:
                direction = 0

            self._last_signals[symbol] = ts_signal.position_size

            signals.append(
                StrategySignal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=direction,
                    target_weight=abs(ts_signal.position_size),
                    confidence=ts_signal.confidence,
                    entry_price=float(prices[-1]),
                    trade_type="regime_transition",
                    metadata={
                        "current_regime": ts_signal.current_regime.name,
                        "predicted_regime": ts_signal.predicted_regime.name,
                        "transition_probability": ts_signal.transition_probability,
                        "transition_velocity": ts_signal.transition_velocity,
                        "expected_return": ts_signal.expected_return,
                    },
                )
            )

        return signals

    def get_current_exposure(self) -> float:
        return sum(abs(w) for w in self._last_signals.values())

    def get_performance_stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        for s in self._strategies.values():
            s.reset()
        self._strategies.clear()
        self._daily_close.clear()
        self._daily_high.clear()
        self._daily_low.clear()
        self._daily_count.clear()
        self._current_day.clear()
        self._intraday_high.clear()
        self._intraday_low.clear()
        self._intraday_close.clear()
        self._hmm_models.clear()
        self._last_signals.clear()
        self._state = StrategyState.WARMING_UP
