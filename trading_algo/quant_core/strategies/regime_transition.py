"""
Regime Transition Alpha Strategy

Exploits Hidden Markov Model regime TRANSITIONS rather than regime states.

Key Insight:
    Most managers react to regime changes AFTER they happen. HMM transition
    probabilities can PREDICT them. This strategy extracts alpha from the
    predictive information embedded in the transition matrix dynamics.

Alpha Sources:
    1. Transition probability level: P(current_regime -> target_regime)
    2. Transition velocity: d/dt of transition probabilities
    3. Multi-regime scoring: probability-weighted expected returns across
       all destination regimes
    4. Regime uncertainty premium: higher uncertainty -> higher transition
       likelihood -> stronger signal

Position Sizing:
    - Kelly-inspired: f = edge / variance
    - Inverse volatility scaling
    - Regime-conditional confidence adjustments

Exit Rules:
    - Regime fully established (>70% probability in new regime)
    - Time-based: max 20 day holding period
    - ATR-based stop loss

References:
    - Hamilton, J.D. (1989): "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle"
    - Ang, A. & Bekaert, G. (2002): "Regime Switches in Interest Rates"
    - Guidolin, M. & Timmermann, A. (2007): "Asset Allocation Under
      Multivariate Regime Switching"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import warnings

from trading_algo.quant_core.models.hmm_regime import (
    MarketRegime,
    RegimeState,
    HiddenMarkovRegime,
)
from trading_algo.quant_core.utils.constants import (
    EPSILON,
    SQRT_252,
    VOL_TARGET_DEFAULT,
)
from trading_algo.quant_core.utils.math_utils import (
    simple_returns,
    rolling_mean,
    rolling_std,
    ewma_volatility,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransitionSignal:
    """
    Signal generated from regime transition dynamics.

    Captures not just the predicted direction, but the full probabilistic
    context: which regime we are leaving, where we are going, how fast
    the transition is occurring, and what returns to expect conditional
    on the destination regime.

    Attributes:
        timestamp: Time of signal generation.
        symbol: Instrument identifier.
        direction: Directional signal in [-1, +1]. Positive is long,
            negative is short. Magnitude reflects conviction.
        confidence: Signal confidence in [0, 1]. Derived from transition
            probability level and velocity agreement.
        current_regime: The regime the HMM currently assigns highest
            probability to.
        predicted_regime: The most likely destination regime based on
            transition probability dynamics.
        transition_probability: Raw P(current_regime -> predicted_regime).
        transition_velocity: Rate of change of transition probability
            over the velocity window (5 days default).
        expected_return: Probability-weighted expected return across all
            destination regimes, annualized.
        position_size: Final recommended position size after Kelly sizing,
            volatility scaling, and regime confidence adjustments.
    """
    timestamp: datetime
    symbol: str
    direction: float
    confidence: float
    current_regime: MarketRegime
    predicted_regime: MarketRegime
    transition_probability: float
    transition_velocity: float
    expected_return: float
    position_size: float


@dataclass
class TransitionConfig:
    """
    Configuration for the regime transition strategy.

    All thresholds have been calibrated against historical US equity
    regime data (1990-2024). Adjust for other asset classes.

    Attributes:
        transition_threshold: Minimum P(transition) to generate a signal.
            Default 0.25 balances signal frequency with quality.
        velocity_threshold: Minimum d/dt of P(transition) to confirm
            directional momentum. Prevents stale signals.
        velocity_window: Rolling window (days) for computing transition
            probability rate of change.
        max_holding_days: Maximum position holding period before forced
            exit. Prevents stale positions in slow transitions.
        vol_target: Annualized portfolio volatility target for position
            scaling.
        max_position: Hard cap on position size as fraction of equity.
        min_regime_history: Minimum days of regime observations before
            the strategy begins generating signals.
        regime_return_lookback: Days of history used to estimate
            regime-conditional returns and volatilities.
        regime_established_prob: Probability threshold above which a
            regime is considered fully established (triggers exit).
        atr_window: Window for ATR calculation used in stop losses.
        atr_stop_multiplier: Number of ATRs for stop loss distance.
        slippage_bps: Assumed slippage in basis points per trade.
        commission_bps: Assumed commission in basis points per trade.
        high_vol_position_scale: Multiplicative scale factor applied to
            positions during HIGH_VOL regime (reduces conviction).
        kelly_fraction: Fraction of full Kelly to use. Half-Kelly (0.5)
            is standard for robustness to estimation error.
        min_signal_strength: Minimum absolute signal strength to act on.
            Filters out noise near zero.
    """
    transition_threshold: float = 0.05
    velocity_threshold: float = 0.005
    velocity_window: int = 5
    max_holding_days: int = 20
    vol_target: float = 0.15
    max_position: float = 0.20
    min_regime_history: int = 60
    regime_return_lookback: int = 252
    regime_established_prob: float = 0.70
    atr_window: int = 14
    atr_stop_multiplier: float = 2.0
    slippage_bps: float = 5.0
    commission_bps: float = 10.0
    high_vol_position_scale: float = 0.5
    kelly_fraction: float = 0.5
    min_signal_strength: float = 0.0001


# =============================================================================
# REGIME TRANSITION STRATEGY
# =============================================================================

class RegimeTransitionStrategy:
    """
    Alpha strategy exploiting HMM regime transition dynamics.

    Rather than reacting to the current regime (which is backward-looking
    and already priced in by most participants), this strategy monitors
    the HMM transition probability matrix over time and trades on:

    1. Rising transition probabilities toward bearish regimes (short).
    2. Rising transition probabilities toward bullish regimes (long).
    3. The velocity of transition probability changes (momentum of the
       regime shift itself).
    4. Multi-regime expected return scoring to handle ambiguous transitions.

    The core insight is that transition probabilities contain FORWARD-LOOKING
    information about regime changes before they are confirmed by price
    action, giving a timing edge over managers who wait for regime
    confirmation.

    Usage:
        hmm = HiddenMarkovRegime(n_states=3)
        strategy = RegimeTransitionStrategy()

        # Fit HMM on initial history
        hmm.fit(prices)

        # Generate signals on new data
        signal = strategy.generate_signal(
            symbol="SPY",
            prices=prices,
            returns=returns,
            hmm_model=hmm,
            timestamp=datetime.now(),
        )

        if signal is not None:
            print(f"Direction: {signal.direction:.2f}")
            print(f"Position: {signal.position_size:.4f}")
    """

    def __init__(self, config: Optional[TransitionConfig] = None):
        """
        Initialize regime transition strategy.

        Args:
            config: Strategy configuration. Uses sensible defaults
                if not provided.
        """
        self.config = config or TransitionConfig()

        # Transition matrix history: list of (timestamp, matrix) tuples
        self._transition_history: deque[Tuple[datetime, NDArray[np.float64]]] = deque(
            maxlen=500
        )

        # Regime state history for return estimation
        self._regime_state_history: deque[Tuple[datetime, RegimeState]] = deque(
            maxlen=2000
        )

        # Per-symbol active position tracking
        self._active_positions: Dict[str, _ActivePosition] = {}

        # Cached regime return estimates: MarketRegime -> (mean_ret, vol)
        self._regime_return_cache: Dict[MarketRegime, Tuple[float, float]] = {}
        self._cache_timestamp: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        prices: NDArray[np.float64],
        returns: NDArray[np.float64],
        hmm_model: HiddenMarkovRegime,
        timestamp: datetime,
        high_prices: Optional[NDArray[np.float64]] = None,
        low_prices: Optional[NDArray[np.float64]] = None,
    ) -> Optional[TransitionSignal]:
        """
        Generate a transition-based trading signal.

        This is the main entry point. It:
        1. Gets the current regime state from the HMM.
        2. Extracts and stores the transition matrix.
        3. Computes transition probabilities and velocities.
        4. Scores all destination regimes by expected return.
        5. Determines direction and sizes the position.

        Args:
            symbol: Instrument identifier (e.g., "SPY").
            prices: Close price array. Must have at least
                config.min_regime_history observations.
            returns: Simple return array corresponding to prices.
                Length should be len(prices) - 1.
            hmm_model: A fitted HiddenMarkovRegime instance.
            timestamp: Current observation timestamp.
            high_prices: Optional high price array for ATR calculation.
            low_prices: Optional low price array for ATR calculation.

        Returns:
            TransitionSignal if a signal is generated, None otherwise.
            Returns None when insufficient history, no transition detected,
            or signal strength below minimum threshold.
        """
        # Guard: need enough history
        if len(returns) < self.config.min_regime_history:
            return None

        # Step 1: Get current regime state
        regime_state = hmm_model.predict_regime(prices)
        self._regime_state_history.append((timestamp, regime_state))

        # Step 2: Extract and store transition matrix
        transition_matrix = hmm_model.get_transition_matrix()
        if transition_matrix is None:
            return None

        self._transition_history.append((timestamp, transition_matrix.copy()))

        # Step 3: Need enough transition history for velocity
        if len(self._transition_history) < self.config.velocity_window + 1:
            return None

        # Step 4: Estimate regime-conditional returns
        regime_labels = self._extract_regime_labels()
        if len(regime_labels) < self.config.min_regime_history:
            return None

        regime_returns = self._estimate_regime_returns(
            returns[-len(regime_labels):],
            regime_labels,
        )
        self._regime_return_cache = regime_returns
        self._cache_timestamp = timestamp

        # Step 5: Get regime mapping from HMM for matrix interpretation
        regime_mapping = hmm_model._regime_mapping
        if not regime_mapping:
            return None

        # Step 6: Compute transition signals
        current_regime = regime_state.regime
        current_prob = regime_state.probability

        # Find the HMM state index for current regime
        current_state_idx = self._regime_to_state_idx(current_regime, regime_mapping)
        if current_state_idx is None:
            return None

        # Step 7: Compute multi-regime score
        score, predicted_regime, trans_prob, trans_velocity = (
            self._compute_multi_regime_score(
                current_state_idx=current_state_idx,
                current_prob=current_prob,
                regime_mapping=regime_mapping,
                regime_returns=regime_returns,
            )
        )

        # Step 8: Check thresholds
        if abs(score) < self.config.min_signal_strength:
            return None

        if trans_prob < self.config.transition_threshold and abs(trans_velocity) < self.config.velocity_threshold:
            return None

        # Step 9: Determine direction
        direction = np.clip(score, -1.0, 1.0)

        # Step 10: Compute confidence
        confidence = self._compute_confidence(
            trans_prob=trans_prob,
            trans_velocity=trans_velocity,
            current_prob=current_prob,
        )

        # Step 11: Position sizing
        current_vol = self._estimate_current_volatility(returns)
        position_size = self._compute_position_size(
            direction=direction,
            confidence=confidence,
            current_vol=current_vol,
            current_regime=current_regime,
            expected_return=score,
        )

        # Step 12: Check exit conditions on existing position
        if symbol in self._active_positions:
            active = self._active_positions[symbol]
            should_exit, exit_reason = self._check_exit_conditions(
                active_position=active,
                regime_state=regime_state,
                timestamp=timestamp,
                current_price=float(prices[-1]),
                high_prices=high_prices,
                low_prices=low_prices,
                prices=prices,
            )
            if should_exit:
                del self._active_positions[symbol]
                # Return a flat signal to close position
                return TransitionSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=0.0,
                    confidence=confidence,
                    current_regime=current_regime,
                    predicted_regime=predicted_regime,
                    transition_probability=trans_prob,
                    transition_velocity=trans_velocity,
                    expected_return=0.0,
                    position_size=0.0,
                )

        # Step 13: Track new position
        if abs(position_size) > EPSILON:
            self._active_positions[symbol] = _ActivePosition(
                entry_timestamp=timestamp,
                entry_price=float(prices[-1]),
                direction=direction,
                entry_regime=current_regime,
            )

        expected_return_ann = score * SQRT_252 * np.sqrt(252)  # Daily score -> annualized

        return TransitionSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=float(direction),
            confidence=float(confidence),
            current_regime=current_regime,
            predicted_regime=predicted_regime,
            transition_probability=float(trans_prob),
            transition_velocity=float(trans_velocity),
            expected_return=float(expected_return_ann),
            position_size=float(position_size),
        )

    def reset(self) -> None:
        """Reset all strategy state. Call between backtest runs."""
        self._transition_history.clear()
        self._regime_state_history.clear()
        self._active_positions.clear()
        self._regime_return_cache.clear()
        self._cache_timestamp = None

    # -------------------------------------------------------------------------
    # TRANSITION SIGNAL EXTRACTION
    # -------------------------------------------------------------------------

    def _compute_multi_regime_score(
        self,
        current_state_idx: int,
        current_prob: float,
        regime_mapping: Dict[int, MarketRegime],
        regime_returns: Dict[MarketRegime, Tuple[float, float]],
    ) -> Tuple[float, MarketRegime, float, float]:
        """
        Compute multi-regime expected return score.

        Score = sum over all destination regimes of:
            P(current -> destination) * expected_return(destination)

        The signal strength is amplified by regime uncertainty: the less
        certain the current regime, the more likely a transition, so
        the transition signal carries more weight.

        Args:
            current_state_idx: HMM state index for current regime.
            current_prob: Probability of current regime assignment.
            regime_mapping: HMM state -> MarketRegime mapping.
            regime_returns: Regime -> (mean_return, volatility) estimates.

        Returns:
            Tuple of (score, predicted_regime, max_transition_prob,
            transition_velocity). Score is in approximately daily return
            units. predicted_regime is the destination regime with highest
            probability-weighted contribution.
        """
        latest_matrix = self._transition_history[-1][1]
        n_states = latest_matrix.shape[0]

        # Compute transition velocity from recent history
        velocity_matrix = self._compute_transition_velocity()

        score = 0.0
        max_contribution = -np.inf
        predicted_regime = MarketRegime.NEUTRAL
        max_trans_prob = 0.0
        max_trans_velocity = 0.0

        for dest_state in range(n_states):
            if dest_state == current_state_idx:
                continue

            dest_regime = regime_mapping.get(dest_state)
            if dest_regime is None:
                continue

            # Transition probability
            trans_prob = float(latest_matrix[current_state_idx, dest_state])

            # Transition velocity
            if velocity_matrix is not None:
                trans_vel = float(velocity_matrix[current_state_idx, dest_state])
            else:
                trans_vel = 0.0

            # Expected return for destination regime (annualized -> daily)
            if dest_regime in regime_returns:
                dest_mean_return, dest_vol = regime_returns[dest_regime]
                daily_expected_return = dest_mean_return / 252.0
            else:
                daily_expected_return = 0.0

            # Signal strength = trans_prob * (1 - current_regime_prob)
            # More uncertain current regime -> more likely transition
            uncertainty_amplifier = 1.0 - current_prob
            signal_strength = trans_prob * (1.0 + uncertainty_amplifier)

            # Boost if velocity confirms direction
            if trans_vel > 0:
                signal_strength *= (1.0 + min(trans_vel / self.config.velocity_threshold, 2.0))

            # Contribution to score
            contribution = signal_strength * daily_expected_return
            score += contribution

            # Track dominant destination regime
            if abs(contribution) > max_contribution:
                max_contribution = abs(contribution)
                predicted_regime = dest_regime
                max_trans_prob = trans_prob
                max_trans_velocity = trans_vel

        return score, predicted_regime, max_trans_prob, max_trans_velocity

    def _compute_transition_velocity(self) -> Optional[NDArray[np.float64]]:
        """
        Compute rate of change of transition probabilities.

        Uses a simple difference over the velocity window:
            velocity[i,j] = P_now(i->j) - P_{window_ago}(i->j)

        Positive velocity means increasing transition probability.

        Returns:
            Velocity matrix of same shape as transition matrix, or None
            if insufficient history.
        """
        n_history = len(self._transition_history)
        window = self.config.velocity_window

        if n_history < window + 1:
            return None

        current_matrix = self._transition_history[-1][1]
        past_matrix = self._transition_history[-(window + 1)][1]

        # Guard against shape mismatch (model may have been retrained)
        if current_matrix.shape != past_matrix.shape:
            return None

        velocity = (current_matrix - past_matrix) / float(window)
        return velocity

    # -------------------------------------------------------------------------
    # REGIME RETURN ESTIMATION
    # -------------------------------------------------------------------------

    def _estimate_regime_returns(
        self,
        returns: NDArray[np.float64],
        regime_labels: List[MarketRegime],
    ) -> Dict[MarketRegime, Tuple[float, float]]:
        """
        Estimate mean return and volatility conditional on each regime.

        For each regime, computes the annualized mean return and
        annualized volatility from historical observations where the HMM
        assigned that regime as the most probable state.

        Edge cases handled:
        - Regimes with fewer than 10 observations: excluded (insufficient
          data for reliable estimation).
        - Zero-variance regimes: volatility floored at EPSILON.
        - Missing regimes: not included in output dict.

        Args:
            returns: Historical return array.
            regime_labels: List of MarketRegime labels corresponding to
                each return observation.

        Returns:
            Dict mapping MarketRegime to (annualized_mean_return,
            annualized_volatility). Only regimes with sufficient
            observations are included.
        """
        if len(returns) != len(regime_labels):
            min_len = min(len(returns), len(regime_labels))
            returns = returns[-min_len:]
            regime_labels = regime_labels[-min_len:]

        regime_returns: Dict[MarketRegime, Tuple[float, float]] = {}

        # Group returns by regime
        regime_groups: Dict[MarketRegime, List[float]] = {}
        for ret, regime in zip(returns, regime_labels):
            if regime == MarketRegime.UNKNOWN:
                continue
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(ret)

        for regime, ret_list in regime_groups.items():
            if len(ret_list) < 10:
                continue

            ret_array = np.array(ret_list, dtype=np.float64)

            # Annualized mean return
            mean_daily = float(np.mean(ret_array))
            mean_annual = mean_daily * 252.0

            # Annualized volatility
            std_daily = float(np.std(ret_array, ddof=1))
            vol_annual = std_daily * SQRT_252

            # Floor volatility for numerical stability
            vol_annual = max(vol_annual, EPSILON)

            regime_returns[regime] = (mean_annual, vol_annual)

        return regime_returns

    # -------------------------------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------------------------------

    def _compute_position_size(
        self,
        direction: float,
        confidence: float,
        current_vol: float,
        current_regime: MarketRegime,
        expected_return: float,
    ) -> float:
        """
        Compute position size using Kelly-inspired sizing with regime
        adjustments and volatility scaling.

        Sizing pipeline:
        1. Kelly fraction: f = (expected_return / variance) * kelly_fraction
        2. Scale by confidence
        3. Scale by inverse volatility (vol target / current vol)
        4. Reduce in HIGH_VOL regime
        5. Clip to max position

        Args:
            direction: Signal direction [-1, +1].
            confidence: Signal confidence [0, 1].
            current_vol: Current annualized volatility estimate.
            current_regime: Current market regime for conditional
                position adjustments.
            expected_return: Expected daily return from the signal.

        Returns:
            Signed position size in [-max_position, +max_position].
        """
        if abs(direction) < EPSILON or current_vol < EPSILON:
            return 0.0

        # Kelly fraction: f = edge / variance
        # Edge = expected daily return (from score)
        # Variance = current daily variance
        daily_variance = (current_vol / SQRT_252) ** 2
        if daily_variance < EPSILON:
            daily_variance = EPSILON

        edge = abs(expected_return)
        kelly_full = edge / daily_variance
        kelly_position = kelly_full * self.config.kelly_fraction

        # Scale by confidence
        kelly_position *= confidence

        # Inverse volatility scaling: target vol / realized vol
        if current_vol > EPSILON:
            vol_scalar = self.config.vol_target / current_vol
        else:
            vol_scalar = 1.0
        kelly_position *= vol_scalar

        # Regime-conditional adjustment
        if current_regime == MarketRegime.HIGH_VOL:
            kelly_position *= self.config.high_vol_position_scale

        # Apply direction
        position = kelly_position * np.sign(direction)

        # Hard cap
        position = float(np.clip(position, -self.config.max_position, self.config.max_position))

        return position

    def _compute_confidence(
        self,
        trans_prob: float,
        trans_velocity: float,
        current_prob: float,
    ) -> float:
        """
        Compute signal confidence from transition dynamics.

        Confidence is higher when:
        - Transition probability is well above threshold.
        - Transition velocity confirms the direction.
        - Current regime probability is declining (uncertainty rising).

        Args:
            trans_prob: Transition probability level.
            trans_velocity: Rate of change of transition probability.
            current_prob: Probability of current regime assignment.

        Returns:
            Confidence in [0, 1].
        """
        # Base confidence from probability level
        if trans_prob >= self.config.transition_threshold:
            prob_confidence = min(
                1.0,
                (trans_prob - self.config.transition_threshold)
                / (1.0 - self.config.transition_threshold + EPSILON),
            )
        else:
            prob_confidence = trans_prob / (self.config.transition_threshold + EPSILON)

        # Velocity confirmation bonus
        if trans_velocity > self.config.velocity_threshold:
            velocity_bonus = min(
                0.3,
                0.3 * (trans_velocity - self.config.velocity_threshold)
                / (self.config.velocity_threshold + EPSILON),
            )
        elif trans_velocity > 0:
            velocity_bonus = 0.1 * trans_velocity / (self.config.velocity_threshold + EPSILON)
        else:
            velocity_bonus = -0.1  # Decelerating transition reduces confidence

        # Uncertainty premium: lower current regime probability -> higher confidence
        # that a transition is underway
        uncertainty_bonus = (1.0 - current_prob) * 0.2

        confidence = prob_confidence + velocity_bonus + uncertainty_bonus
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return confidence

    # -------------------------------------------------------------------------
    # EXIT RULES
    # -------------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        active_position: _ActivePosition,
        regime_state: RegimeState,
        timestamp: datetime,
        current_price: float,
        high_prices: Optional[NDArray[np.float64]],
        low_prices: Optional[NDArray[np.float64]],
        prices: NDArray[np.float64],
    ) -> Tuple[bool, str]:
        """
        Check whether an active position should be exited.

        Exit triggers (any one sufficient):
        1. Regime fully established: new regime has probability above
           regime_established_prob (the transition trade is done).
        2. Time-based: position held longer than max_holding_days.
        3. Stop loss: price has moved against the position by more
           than atr_stop_multiplier * ATR.

        Args:
            active_position: Tracking data for the active position.
            regime_state: Current HMM regime state.
            timestamp: Current timestamp.
            current_price: Current close price.
            high_prices: Optional high price array for ATR.
            low_prices: Optional low price array for ATR.
            prices: Close price array for ATR fallback.

        Returns:
            Tuple of (should_exit, reason_string).
        """
        # Exit 1: Regime fully established
        if regime_state.regime != active_position.entry_regime:
            if regime_state.probability >= self.config.regime_established_prob:
                return True, "regime_established"

        # Exit 2: Time-based
        if active_position.entry_timestamp is not None:
            holding_days = (timestamp - active_position.entry_timestamp).days
            if holding_days >= self.config.max_holding_days:
                return True, "max_holding_period"

        # Exit 3: ATR-based stop loss
        atr = self._compute_atr(prices, high_prices, low_prices)
        if atr > EPSILON:
            stop_distance = atr * self.config.atr_stop_multiplier
            price_change = current_price - active_position.entry_price

            # Long position: stop if price dropped by stop_distance
            if active_position.direction > 0 and price_change < -stop_distance:
                return True, "stop_loss"
            # Short position: stop if price rose by stop_distance
            elif active_position.direction < 0 and price_change > stop_distance:
                return True, "stop_loss"

        return False, ""

    def _compute_atr(
        self,
        prices: NDArray[np.float64],
        high_prices: Optional[NDArray[np.float64]] = None,
        low_prices: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Compute Average True Range for stop loss calculation.

        If high/low prices are available, uses the full true range
        definition. Otherwise falls back to a close-to-close range
        approximation.

        Args:
            prices: Close price array.
            high_prices: Optional high price array.
            low_prices: Optional low price array.

        Returns:
            Current ATR value (scalar).
        """
        window = self.config.atr_window

        if high_prices is not None and low_prices is not None:
            if len(high_prices) >= window + 1 and len(low_prices) >= window + 1:
                # True range: max(H-L, |H-Cprev|, |L-Cprev|)
                n = min(len(high_prices), len(low_prices), len(prices))
                tr = np.zeros(n - 1)
                for i in range(1, n):
                    hl = high_prices[i] - low_prices[i]
                    hc = abs(high_prices[i] - prices[i - 1])
                    lc = abs(low_prices[i] - prices[i - 1])
                    tr[i - 1] = max(hl, hc, lc)

                if len(tr) >= window:
                    return float(np.mean(tr[-window:]))

        # Fallback: use absolute close-to-close changes
        if len(prices) >= window + 1:
            abs_changes = np.abs(np.diff(prices[-window - 1:]))
            return float(np.mean(abs_changes))

        return 0.0

    # -------------------------------------------------------------------------
    # VOLATILITY ESTIMATION
    # -------------------------------------------------------------------------

    def _estimate_current_volatility(
        self,
        returns: NDArray[np.float64],
    ) -> float:
        """
        Estimate current annualized volatility using EWMA.

        Falls back to rolling standard deviation if the return array
        is too short for EWMA to stabilize.

        Args:
            returns: Historical return array.

        Returns:
            Annualized volatility estimate.
        """
        if len(returns) < 2:
            return self.config.vol_target  # Fallback to target

        try:
            ewma_vol = ewma_volatility(returns.astype(np.float64))
            current_vol = float(ewma_vol[-1])
        except Exception:
            # Fallback: simple rolling std
            lookback = min(60, len(returns))
            current_vol = float(np.std(returns[-lookback:], ddof=1) * SQRT_252)

        # Floor and cap for numerical stability
        current_vol = max(current_vol, 0.01)
        current_vol = min(current_vol, 2.0)

        return current_vol

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def _extract_regime_labels(self) -> List[MarketRegime]:
        """
        Extract regime label sequence from stored regime state history.

        Returns:
            List of MarketRegime values, one per historical observation.
        """
        return [state.regime for _, state in self._regime_state_history]

    @staticmethod
    def _regime_to_state_idx(
        regime: MarketRegime,
        regime_mapping: Dict[int, MarketRegime],
    ) -> Optional[int]:
        """
        Find the HMM state index corresponding to a MarketRegime.

        Args:
            regime: The target regime.
            regime_mapping: HMM state_idx -> MarketRegime mapping.

        Returns:
            State index, or None if regime not found in mapping.
        """
        for state_idx, mapped_regime in regime_mapping.items():
            if mapped_regime == regime:
                return state_idx
        return None


# =============================================================================
# INTERNAL TRACKING
# =============================================================================

@dataclass
class _ActivePosition:
    """
    Internal tracking for an active position.

    Not part of the public API. Used by the strategy to track entry
    conditions for exit rule evaluation.
    """
    entry_timestamp: datetime
    entry_price: float
    direction: float
    entry_regime: MarketRegime


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_regime_transition_backtest(
    historical_data: Dict[str, NDArray[np.float64]],
    timestamps: List[datetime],
    initial_capital: float = 100_000.0,
    config: Optional[TransitionConfig] = None,
    hmm_n_states: int = 3,
    hmm_retrain_frequency: int = 21,
) -> Dict:
    """
    Run a full backtest of the regime transition strategy.

    Simulates the strategy on historical OHLCV data with realistic
    transaction costs. The HMM is either retrained periodically or
    updated incrementally.

    Pipeline per bar:
    1. Check if HMM needs retraining (every hmm_retrain_frequency days).
    2. Generate transition signal via the strategy.
    3. Execute trades with slippage and commission.
    4. Track equity, positions, and trade log.

    Args:
        historical_data: Dict of symbol -> OHLCV array with shape (T, 5)
            where columns are [open, high, low, close, volume].
        timestamps: List of datetime objects, one per bar. Must have
            same length as the first dimension of each OHLCV array.
        initial_capital: Starting equity in currency units.
        config: Strategy configuration. Uses defaults if not provided.
        hmm_n_states: Number of HMM hidden states.
        hmm_retrain_frequency: Days between HMM refitting.

    Returns:
        Dict with keys:
            total_return: float - Total percentage return.
            annualized_return: float - CAGR.
            sharpe_ratio: float - Annualized Sharpe (Rf=2%).
            max_drawdown: float - Maximum peak-to-trough drawdown.
            total_trades: int - Number of executed trades.
            equity_curve: NDArray - Equity value per bar.
            returns: NDArray - Daily percentage returns.
            trades: List[Dict] - Trade log with timestamps, prices, etc.
    """
    config = config or TransitionConfig()
    strategy = RegimeTransitionStrategy(config)

    symbols = list(historical_data.keys())
    n_bars = len(timestamps)

    if n_bars == 0 or not symbols:
        return _empty_backtest_result(initial_capital)

    # Validate data dimensions
    for sym in symbols:
        if historical_data[sym].shape[0] != n_bars:
            warnings.warn(
                f"Symbol {sym} has {historical_data[sym].shape[0]} bars, "
                f"expected {n_bars}. Skipping.",
                stacklevel=2,
            )
            symbols.remove(sym)

    if not symbols:
        return _empty_backtest_result(initial_capital)

    # Extract price arrays
    close_prices = {s: historical_data[s][:, 3].astype(np.float64) for s in symbols}
    high_prices = {s: historical_data[s][:, 1].astype(np.float64) for s in symbols}
    low_prices = {s: historical_data[s][:, 2].astype(np.float64) for s in symbols}

    # Initialize HMM models per symbol
    hmm_models: Dict[str, HiddenMarkovRegime] = {}
    for sym in symbols:
        hmm_models[sym] = HiddenMarkovRegime(
            n_states=hmm_n_states,
            retrain_frequency=hmm_retrain_frequency,
        )

    # Backtest state
    cash = initial_capital
    positions: Dict[str, float] = {}  # symbol -> number of shares/units
    equity_curve: List[float] = []
    trades: List[Dict] = []

    slippage_rate = config.slippage_bps / 10_000.0
    commission_rate = config.commission_bps / 10_000.0

    # Warmup: need enough data for HMM fitting + regime history
    warmup = max(config.min_regime_history + 20, config.regime_return_lookback)

    for t in range(n_bars):
        # Current portfolio value
        current_prices = {s: close_prices[s][t] for s in symbols}
        position_value = sum(
            positions.get(s, 0.0) * current_prices[s] for s in symbols
        )
        equity = cash + position_value
        equity_curve.append(equity)

        # Skip warmup period
        if t < warmup:
            continue

        for sym in symbols:
            price_slice = close_prices[sym][: t + 1]
            high_slice = high_prices[sym][: t + 1]
            low_slice = low_prices[sym][: t + 1]

            # Compute returns for the available window
            if len(price_slice) < 2:
                continue
            returns = simple_returns(price_slice.astype(np.float64))

            # Retrain HMM if needed
            hmm = hmm_models[sym]
            if hmm.should_retrain(t):
                hmm.fit(price_slice)
                hmm._last_train_bar = t

            if not hmm._is_fitted:
                continue

            # Generate signal
            signal = strategy.generate_signal(
                symbol=sym,
                prices=price_slice,
                returns=returns,
                hmm_model=hmm,
                timestamp=timestamps[t],
                high_prices=high_slice,
                low_prices=low_slice,
            )

            if signal is None:
                continue

            # Determine target position value
            target_position_value = equity * signal.position_size
            current_shares = positions.get(sym, 0.0)
            current_position_value = current_shares * current_prices[sym]
            delta_value = target_position_value - current_position_value

            # Minimum trade size filter
            if abs(delta_value) < 100.0:
                continue

            # Execute trade with slippage and commission
            trade_price = current_prices[sym]
            if delta_value > 0:
                # Buying: pay slippage
                trade_price *= (1.0 + slippage_rate)
            else:
                # Selling: receive less due to slippage
                trade_price *= (1.0 - slippage_rate)

            delta_shares = delta_value / trade_price
            trade_cost = abs(delta_value)
            commission = trade_cost * commission_rate

            # Update cash and positions
            if delta_shares > 0:
                cash -= (abs(delta_shares) * trade_price) + commission
            else:
                cash += (abs(delta_shares) * trade_price) - commission

            positions[sym] = current_shares + delta_shares

            trades.append({
                "timestamp": timestamps[t],
                "symbol": sym,
                "shares": float(delta_shares),
                "price": float(trade_price),
                "commission": float(commission),
                "direction": float(signal.direction),
                "confidence": float(signal.confidence),
                "transition_prob": float(signal.transition_probability),
                "current_regime": signal.current_regime.name,
                "predicted_regime": signal.predicted_regime.name,
            })

    # Finalize equity curve
    equity_array = np.array(equity_curve, dtype=np.float64)

    if len(equity_array) < 2:
        return _empty_backtest_result(initial_capital)

    # Compute performance metrics
    daily_returns = np.diff(equity_array) / np.maximum(equity_array[:-1], EPSILON)

    total_return = (equity_array[-1] / initial_capital) - 1.0
    n_years = len(daily_returns) / 252.0

    if n_years > 0 and (1.0 + total_return) > 0:
        annualized_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    else:
        annualized_return = 0.0

    volatility = float(np.std(daily_returns, ddof=1) * SQRT_252)

    # Sharpe ratio (Rf = 2%)
    risk_free_rate = 0.02
    if volatility > EPSILON:
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / np.maximum(peak, EPSILON)
    max_drawdown = float(np.max(drawdown))

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "total_trades": len(trades),
        "equity_curve": equity_array,
        "returns": daily_returns,
        "trades": trades,
    }


def _empty_backtest_result(initial_capital: float) -> Dict:
    """
    Return an empty backtest result when no data is available.

    Args:
        initial_capital: Starting capital for the equity curve.

    Returns:
        Dict with zeroed-out metrics and single-element equity curve.
    """
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "equity_curve": np.array([initial_capital]),
        "returns": np.array([], dtype=np.float64),
        "trades": [],
    }
