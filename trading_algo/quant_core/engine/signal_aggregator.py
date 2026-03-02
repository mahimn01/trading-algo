"""
Signal Aggregator

Combines signals from multiple quantitative models into a unified
trading signal for each asset.

Signal Sources:
    1. Ornstein-Uhlenbeck: Mean reversion S-score
    2. Time-Series Momentum: Trend following
    3. Volatility-Managed Momentum: Vol-scaled momentum
    4. HMM Regime: Regime-based signal adjustment
    5. ML Combiner: Learned signal combination

Signal Combination Methods:
    - Equal weight averaging
    - Inverse-variance weighting
    - ML-based stacking
    - Regime-conditional weighting
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import logging

from trading_algo.quant_core.utils.constants import EPSILON, SQRT_252
from trading_algo.quant_core.models.ornstein_uhlenbeck import (
    OrnsteinUhlenbeck, OUSignal
)
from trading_algo.quant_core.models.tsmom import TimeSeriesMomentum
from trading_algo.quant_core.models.vol_managed_momentum import VolatilityManagedMomentum
from trading_algo.quant_core.models.hmm_regime import HiddenMarkovRegime, RegimeState, MarketRegime
from trading_algo.quant_core.ml.signal_combiner import SignalCombiner, CombinerConfig
from trading_algo.quant_core.ml.features import FeatureEngine


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    MEAN_REVERSION = auto()
    MOMENTUM = auto()
    VOL_MOMENTUM = auto()
    REGIME = auto()
    ML_COMBINED = auto()


@dataclass
class ModelSignal:
    """Signal from a single model."""
    signal_type: SignalType
    value: float              # Normalized signal [-1, 1]
    confidence: float         # Confidence in signal [0, 1]
    raw_value: float          # Raw model output
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedSignal:
    """
    Combined signal for a single asset.

    Contains the final trading signal along with breakdown
    by component and regime information.
    """
    symbol: str
    signal: float              # Combined signal [-1, 1]
    confidence: float          # Overall confidence [0, 1]
    market_regime: MarketRegime = MarketRegime.UNKNOWN  # Current market regime
    component_signals: Dict[SignalType, ModelSignal] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[Any] = None

    @property
    def direction(self) -> int:
        """Get signal direction: 1 (long), -1 (short), 0 (neutral)."""
        if self.signal > 0.1:
            return 1
        elif self.signal < -0.1:
            return -1
        return 0


@dataclass
class AggregatorConfig:
    """Configuration for signal aggregation."""
    # Model weights (will be normalized)
    ou_weight: float = 0.25
    tsmom_weight: float = 0.25
    vol_mom_weight: float = 0.25
    ml_weight: float = 0.25

    # Regime adjustments
    regime_scaling: bool = True
    crisis_exposure_mult: float = 0.25  # Reduce exposure in crisis
    regime_weight_tilt: float = 0.35    # Tilt model weights by detected regime

    # Signal thresholds
    min_signal_threshold: float = 0.1
    high_conviction_threshold: float = 0.5
    disagreement_penalty: float = 0.35  # Penalize conflicting model directions

    # OU parameters
    ou_lookback: int = 60
    ou_entry_threshold: float = 1.25

    # Momentum parameters
    momentum_lookback: int = 252
    vol_target: float = 0.15

    # ML combiner
    use_ml_combination: bool = True
    ml_lookback: int = 252


class SignalAggregator:
    """
    Aggregates signals from multiple quantitative models.

    Combines mean reversion, momentum, and regime signals
    into a unified trading signal with proper risk adjustment.

    Usage:
        aggregator = SignalAggregator(config)
        aggregator.initialize(symbols)

        # Update with new data
        signal = aggregator.generate_signal(symbol, price_history)
    """

    def __init__(self, config: Optional[AggregatorConfig] = None):
        """
        Initialize signal aggregator.

        Args:
            config: Aggregator configuration
        """
        self.config = config or AggregatorConfig()

        # Initialize models
        self.ou_models: Dict[str, OrnsteinUhlenbeck] = {}
        self.tsmom_models: Dict[str, TimeSeriesMomentum] = {}
        self.vol_mom_models: Dict[str, VolatilityManagedMomentum] = {}
        self.hmm_model = HiddenMarkovRegime()

        # ML combiner
        self.feature_engine = FeatureEngine()
        self.ml_combiner = SignalCombiner(CombinerConfig())
        self._ml_trained = False

        # State tracking
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_history: List[MarketRegime] = []

        # Normalize weights
        total_weight = (
            self.config.ou_weight +
            self.config.tsmom_weight +
            self.config.vol_mom_weight +
            self.config.ml_weight
        )
        self._weights = {
            SignalType.MEAN_REVERSION: self.config.ou_weight / total_weight,
            SignalType.MOMENTUM: self.config.tsmom_weight / total_weight,
            SignalType.VOL_MOMENTUM: self.config.vol_mom_weight / total_weight,
            SignalType.ML_COMBINED: self.config.ml_weight / total_weight,
        }

    def initialize(self, symbols: List[str]) -> None:
        """
        Initialize models for all symbols.

        Args:
            symbols: List of symbols to track
        """
        for symbol in symbols:
            self.ou_models[symbol] = OrnsteinUhlenbeck(
                lookback=self.config.ou_lookback
            )
            self.tsmom_models[symbol] = TimeSeriesMomentum(
                lookback=self.config.momentum_lookback,
                target_vol=self.config.vol_target,
            )
            self.vol_mom_models[symbol] = VolatilityManagedMomentum(
                target_vol=self.config.vol_target
            )

        logger.info(f"Initialized signal aggregator for {len(symbols)} symbols")

    def train_ml_combiner(
        self,
        historical_prices: Dict[str, NDArray[np.float64]],
        returns: NDArray[np.float64],
    ) -> None:
        """
        Train ML combiner on historical data.

        Args:
            historical_prices: Dict of symbol -> price history
            returns: Forward returns to predict
        """
        if not self.config.use_ml_combination:
            return

        # Generate features for each symbol
        all_features = []
        all_returns = []

        for symbol, prices in historical_prices.items():
            if len(prices) < self.config.ml_lookback:
                continue

            # Compute features
            feature_set = self.feature_engine.compute_features(prices)

            # Align with returns
            n_samples = min(len(feature_set.features), len(returns))
            all_features.append(feature_set.features[-n_samples:])
            all_returns.append(returns[-n_samples:])

        if not all_features:
            logger.warning("No data available for ML training")
            return

        # Stack and train
        X = np.vstack(all_features)
        y = np.concatenate(all_returns)

        # Remove NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            logger.warning("Insufficient data for ML training")
            return

        self.ml_combiner.fit(X, y)
        self._ml_trained = True
        logger.info(f"Trained ML combiner on {len(X)} samples")

    def update_regime(
        self,
        market_prices: NDArray[np.float64],
        market_returns: Optional[NDArray[np.float64]] = None,
        volumes: Optional[NDArray[np.float64]] = None,
    ) -> MarketRegime:
        """
        Update market regime detection.

        Args:
            market_prices: Market index prices (e.g., SPY)
            market_returns: Optional pre-computed returns
            volumes: Optional volume data

        Returns:
            Current regime state
        """
        try:
            # Fit HMM if needed
            if not self.hmm_model._is_fitted:
                self.hmm_model.fit(market_prices, market_returns, volumes)

            # Predict current regime
            regime_state = self.hmm_model.predict_regime(market_prices)
            self._current_regime = regime_state.regime
            self._regime_history.append(self._current_regime)

            # Keep limited history
            if len(self._regime_history) > 252:
                self._regime_history = self._regime_history[-252:]

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            self._current_regime = MarketRegime.UNKNOWN

        return self._current_regime

    def generate_signal(
        self,
        symbol: str,
        prices: NDArray[np.float64],
        volumes: Optional[NDArray[np.float64]] = None,
        high: Optional[NDArray[np.float64]] = None,
        low: Optional[NDArray[np.float64]] = None,
    ) -> AggregatedSignal:
        """
        Generate aggregated trading signal for a symbol.

        Args:
            symbol: Asset symbol
            prices: Historical prices (most recent last)
            volumes: Optional volume data
            high: Optional high prices
            low: Optional low prices

        Returns:
            AggregatedSignal with combined signal
        """
        component_signals = {}

        # 1. Ornstein-Uhlenbeck (Mean Reversion)
        ou_signal = self._generate_ou_signal(symbol, prices)
        if ou_signal:
            component_signals[SignalType.MEAN_REVERSION] = ou_signal

        # 2. Time-Series Momentum
        tsmom_signal = self._generate_tsmom_signal(symbol, prices)
        if tsmom_signal:
            component_signals[SignalType.MOMENTUM] = tsmom_signal

        # 3. Volatility-Managed Momentum
        vol_mom_signal = self._generate_vol_mom_signal(symbol, prices)
        if vol_mom_signal:
            component_signals[SignalType.VOL_MOMENTUM] = vol_mom_signal

        # 4. ML Combined Signal
        if self.config.use_ml_combination and self._ml_trained:
            ml_signal = self._generate_ml_signal(symbol, prices, volumes, high, low)
            if ml_signal:
                component_signals[SignalType.ML_COMBINED] = ml_signal

        # Combine signals
        combined_signal, confidence = self._combine_signals(component_signals)

        # Apply regime adjustment
        if self.config.regime_scaling:
            combined_signal = self._apply_regime_adjustment(combined_signal)

        return AggregatedSignal(
            symbol=symbol,
            signal=combined_signal,
            confidence=confidence,
            market_regime=self._current_regime,
            component_signals=component_signals,
        )

    def _generate_ou_signal(
        self,
        symbol: str,
        prices: NDArray[np.float64],
    ) -> Optional[ModelSignal]:
        """Generate Ornstein-Uhlenbeck mean reversion signal."""
        if symbol not in self.ou_models:
            return None

        try:
            ou = self.ou_models[symbol]

            # Fit model
            params = ou.fit(prices, symbol)

            if params.half_life <= 0 or params.half_life > 252:
                return None

            # Get signal
            ou_signal, s_score = ou.get_signal(prices[-1])

            # Convert to normalized signal
            if ou_signal == OUSignal.LONG:
                value = min(1.0, abs(s_score) / 2.0)  # Scale s-score
            elif ou_signal == OUSignal.SHORT:
                value = -min(1.0, abs(s_score) / 2.0)
            else:
                value = 0.0

            # Confidence based on model fit quality
            confidence = min(1.0, params.r_squared * 2)

            return ModelSignal(
                signal_type=SignalType.MEAN_REVERSION,
                value=value,
                confidence=confidence,
                raw_value=s_score,
                metadata={
                    'half_life': params.half_life,
                    'mean_level': params.theta,
                    'r_squared': params.r_squared,
                },
            )

        except Exception as e:
            logger.debug(f"OU signal generation failed for {symbol}: {e}")
            return None

    def _generate_tsmom_signal(
        self,
        symbol: str,
        prices: NDArray[np.float64],
    ) -> Optional[ModelSignal]:
        """Generate time-series momentum signal."""
        if symbol not in self.tsmom_models:
            return None

        try:
            tsmom = self.tsmom_models[symbol]
            signal = tsmom.generate_signal(prices)

            return ModelSignal(
                signal_type=SignalType.MOMENTUM,
                value=signal.scaled_position,
                confidence=abs(signal.scaled_position),  # Higher position = higher confidence
                raw_value=signal.momentum_return,
                metadata={
                    'raw_position': signal.raw_position,
                    'volatility': signal.volatility,
                },
            )

        except Exception as e:
            logger.debug(f"TSMOM signal generation failed for {symbol}: {e}")
            return None

    def _generate_vol_mom_signal(
        self,
        symbol: str,
        prices: NDArray[np.float64],
    ) -> Optional[ModelSignal]:
        """Generate volatility-managed momentum signal."""
        if symbol not in self.vol_mom_models:
            return None

        try:
            vol_mom = self.vol_mom_models[symbol]
            signal = vol_mom.generate_signal(prices)

            return ModelSignal(
                signal_type=SignalType.VOL_MOMENTUM,
                value=signal.position_size,
                confidence=min(1.0, signal.vol_scalar),  # Higher vol scaling = lower confidence
                raw_value=signal.raw_signal,
                metadata={
                    'current_vol': signal.current_vol,
                    'vol_scalar': signal.vol_scalar,
                },
            )

        except Exception as e:
            logger.debug(f"Vol-managed momentum failed for {symbol}: {e}")
            return None

    def _generate_ml_signal(
        self,
        symbol: str,
        prices: NDArray[np.float64],
        volumes: Optional[NDArray[np.float64]] = None,
        high: Optional[NDArray[np.float64]] = None,
        low: Optional[NDArray[np.float64]] = None,
    ) -> Optional[ModelSignal]:
        """Generate ML-combined signal."""
        try:
            # Compute features
            feature_set = self.feature_engine.compute_features(
                prices, volumes, high, low
            )

            # Get latest features
            if feature_set.n_samples == 0:
                return None

            latest_features = feature_set.features[-1:]

            # Get prediction
            result = self.ml_combiner.predict(latest_features)

            return ModelSignal(
                signal_type=SignalType.ML_COMBINED,
                value=result.signal,
                confidence=result.confidence,
                raw_value=result.signal,
                metadata={
                    'feature_importance': result.feature_importance,
                },
            )

        except Exception as e:
            logger.debug(f"ML signal generation failed for {symbol}: {e}")
            return None

    def _combine_signals(
        self,
        signals: Dict[SignalType, ModelSignal],
    ) -> Tuple[float, float]:
        """
        Combine component signals into final signal.

        Uses confidence-weighted averaging with regime adjustment.

        Returns:
            Tuple of (combined_signal, confidence)
        """
        if not signals:
            return 0.0, 0.0

        regime_weights = self._get_regime_adjusted_weights()
        total_weight = 0.0
        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_abs_signal = 0.0
        directional_signal = 0.0

        for signal_type, signal in signals.items():
            weight = regime_weights.get(signal_type, 0.0)
            adjusted_weight = weight * signal.confidence

            weighted_signal += signal.value * adjusted_weight
            weighted_confidence += signal.confidence * adjusted_weight
            total_weight += adjusted_weight
            total_abs_signal += abs(signal.value) * adjusted_weight
            directional_signal += signal.value * adjusted_weight

        if total_weight < EPSILON:
            return 0.0, 0.0

        combined = weighted_signal / total_weight
        confidence = weighted_confidence / total_weight

        # Penalize conflicted model directions to reduce churn in noisy regimes.
        if total_abs_signal > EPSILON:
            disagreement = 1.0 - abs(directional_signal) / total_abs_signal
            penalty = np.clip(self.config.disagreement_penalty, 0.0, 1.0) * disagreement
            combined *= (1.0 - penalty)
            confidence *= (1.0 - 0.5 * disagreement)

        # Debounce weak blended signals to avoid overtrading around zero.
        if abs(combined) < self.config.min_signal_threshold:
            combined = 0.0

        # Clip to valid range
        combined = np.clip(combined, -1.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        return float(combined), float(confidence)

    def _get_regime_adjusted_weights(self) -> Dict[SignalType, float]:
        """Adjust base model weights based on detected market regime."""
        weights = dict(self._weights)
        tilt = float(np.clip(self.config.regime_weight_tilt, 0.0, 0.75))

        if tilt < EPSILON:
            return weights

        if self._current_regime in (MarketRegime.BULL, MarketRegime.LOW_VOL):
            weights[SignalType.MOMENTUM] *= 1.0 + tilt
            weights[SignalType.VOL_MOMENTUM] *= 1.0 + (tilt * 0.75)
            weights[SignalType.MEAN_REVERSION] *= 1.0 - (tilt * 0.75)
        elif self._current_regime == MarketRegime.BEAR:
            weights[SignalType.MOMENTUM] *= 1.0 - (tilt * 0.75)
            weights[SignalType.VOL_MOMENTUM] *= 1.0 + (tilt * 0.5)
            weights[SignalType.MEAN_REVERSION] *= 1.0 + (tilt * 0.25)
        elif self._current_regime == MarketRegime.HIGH_VOL:
            weights[SignalType.MOMENTUM] *= 1.0 - tilt
            weights[SignalType.VOL_MOMENTUM] *= 1.0 - (tilt * 0.5)
            weights[SignalType.MEAN_REVERSION] *= 1.0 + tilt

        total = sum(max(0.0, w) for w in weights.values())
        if total < EPSILON:
            return dict(self._weights)

        return {k: max(0.0, w) / total for k, w in weights.items()}

    def _apply_regime_adjustment(self, signal: float) -> float:
        """
        Adjust signal based on current market regime.

        Reduces exposure in crisis regimes, maintains or increases
        in trending regimes.
        """
        regime = self._current_regime

        if regime == MarketRegime.HIGH_VOL:
            # Significantly reduce exposure
            adjusted = signal * self.config.crisis_exposure_mult

        elif regime == MarketRegime.BULL:
            # Favor momentum, slight boost
            adjusted = signal * 1.1

        elif regime == MarketRegime.BEAR:
            # Preserve directional conviction but reduce leverage.
            adjusted = signal * 0.75

        elif regime == MarketRegime.NEUTRAL:
            # Normal operation
            adjusted = signal

        elif regime == MarketRegime.LOW_VOL:
            adjusted = signal * 1.05

        else:
            # Unknown regime, be cautious
            adjusted = signal * 0.75

        return float(np.clip(adjusted, -1.0, 1.0))

    def get_signal_breakdown(
        self,
        aggregated: AggregatedSignal,
    ) -> Dict[str, Any]:
        """Get detailed breakdown of signal components."""
        regime_weights = self._get_regime_adjusted_weights()
        breakdown = {
            'symbol': aggregated.symbol,
            'final_signal': aggregated.signal,
            'confidence': aggregated.confidence,
            'regime': aggregated.market_regime.name,
            'direction': aggregated.direction,
            'components': {},
        }

        for sig_type, sig in aggregated.component_signals.items():
            breakdown['components'][sig_type.name] = {
                'value': sig.value,
                'confidence': sig.confidence,
                'raw_value': sig.raw_value,
                'weight': regime_weights.get(sig_type, 0.0),
                'metadata': sig.metadata,
            }

        return breakdown

    def reset(self) -> None:
        """Reset aggregator state (preserves model instances)."""
        # Only reset transient state, NOT the model instances
        self._regime_history.clear()
        self._current_regime = MarketRegime.UNKNOWN
        self._ml_trained = False
