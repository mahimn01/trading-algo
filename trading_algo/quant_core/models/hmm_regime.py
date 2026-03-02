"""
Hidden Markov Model Regime Detection

Probabilistic regime detection using Gaussian Hidden Markov Models.

Unlike rule-based regime detection, HMM:
    - Provides probability estimates for each regime
    - Learns regime characteristics from data
    - Captures regime transition dynamics
    - Adapts to changing market conditions

Model Structure:
    - N hidden states (typically 2-3: bull, bear, neutral)
    - Observable features: returns, volatility, volume
    - Emission: Gaussian distribution per state
    - Transitions: Markov chain between states

Key Applications:
    - Regime-dependent position sizing
    - Strategy selection based on regime
    - Risk management (reduce exposure in high-vol regimes)
    - Market timing signals

References:
    - https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/
    - https://blog.quantinsti.com/market-regime-detection-hidden-markov-model-project-fahim/
    - Hamilton, J.D. (1989): "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle"
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from collections import deque

from trading_algo.quant_core.utils.constants import (
    HMM_N_STATES,
    HMM_N_ITER,
    HMM_LOOKBACK,
    SQRT_252,
    EPSILON,
)
from trading_algo.quant_core.utils.math_utils import simple_returns


class MarketRegime(Enum):
    """Market regime classification."""
    UNKNOWN = auto()    # Unknown/uninitialized
    BULL = auto()       # High return, low volatility
    BEAR = auto()       # Low/negative return, high volatility
    NEUTRAL = auto()    # Mixed/transitional
    HIGH_VOL = auto()   # Crisis/high volatility
    LOW_VOL = auto()    # Calm market


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: MarketRegime
    probability: float              # Probability of current regime
    regime_probabilities: Dict[MarketRegime, float]  # All regime probs
    regime_duration: int            # Days in current regime
    transition_prob: float          # Probability of regime change
    features_used: Dict[str, float] # Features that drove classification


@dataclass
class RegimeStatistics:
    """Statistics for a detected regime."""
    regime: MarketRegime
    mean_return: float
    volatility: float
    avg_duration: float
    frequency: float  # % of time in this regime
    transition_probs: Dict[MarketRegime, float]


class HiddenMarkovRegime:
    """
    Hidden Markov Model for market regime detection.

    Uses Gaussian HMM to identify market regimes from observable
    features like returns, volatility, and range.

    Features:
    - Probabilistic regime assignment
    - Regime persistence (stickiness)
    - Sliding window retraining
    - Multiple regime support (2-4 states)

    Usage:
        hmm = HiddenMarkovRegime(n_states=3)

        # Fit model
        hmm.fit(returns, features)

        # Get current regime
        state = hmm.predict_regime(latest_features)

        # Get regime probabilities
        probs = state.regime_probabilities
    """

    def __init__(
        self,
        n_states: int = HMM_N_STATES,
        n_iter: int = HMM_N_ITER,
        lookback: int = HMM_LOOKBACK,
        features: Optional[List[str]] = None,
        covariance_type: str = "full",
        min_regime_duration: int = 5,
        retrain_frequency: int = 21,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (regimes)
            n_iter: EM algorithm iterations
            lookback: Training window size
            features: Features to use ['returns', 'volatility', 'range']
            covariance_type: 'full', 'diag', or 'spherical'
            min_regime_duration: Minimum days to stay in regime
            retrain_frequency: Days between model retraining
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.lookback = lookback
        self.features = features or ["returns", "volatility"]
        self.covariance_type = covariance_type
        self.min_regime_duration = min_regime_duration
        self.retrain_frequency = retrain_frequency

        # Model state
        self._model = None
        self._is_fitted = False
        self._regime_mapping: Dict[int, MarketRegime] = {}
        self._last_train_bar: int = 0

        # Tracking
        self._current_regime: MarketRegime = MarketRegime.NEUTRAL
        self._regime_duration: int = 0
        self._regime_history: deque = deque(maxlen=1000)
        self._feature_history: deque = deque(maxlen=lookback)

        # Statistics per regime
        self._regime_stats: Dict[int, Dict] = {}

    def _create_model(self):
        """Create HMM model instance."""
        try:
            from hmmlearn.hmm import GaussianHMM
            self._model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=42,
                min_covar=1e-4,
            )
        except ImportError:
            # Fallback to simple implementation
            self._model = SimpleGaussianHMM(
                n_states=self.n_states,
                n_iter=self.n_iter,
            )

    def _extract_features(
        self,
        prices: NDArray[np.float64],
        returns: Optional[NDArray[np.float64]] = None,
        volumes: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Extract features for HMM input.

        Returns:
            Feature matrix (T x n_features)
        """
        if returns is None:
            returns = simple_returns(prices)

        n = len(returns)
        feature_list = []

        for feature_name in self.features:
            if feature_name == "returns":
                feature_list.append(returns.reshape(-1, 1))

            elif feature_name == "volatility":
                # Rolling 20-day volatility
                vol = np.zeros(n)
                for i in range(20, n):
                    vol[i] = np.std(returns[i-20:i], ddof=1)
                # Fill early values
                vol[:20] = vol[20] if n > 20 else np.std(returns)
                feature_list.append(vol.reshape(-1, 1))

            elif feature_name == "range":
                # Daily range (high-low) / close
                if len(prices) >= n + 1:
                    # Approximate range from returns
                    range_proxy = np.abs(returns)
                    feature_list.append(range_proxy.reshape(-1, 1))

            elif feature_name == "volume" and volumes is not None:
                # Volume normalized by rolling mean
                vol_norm = np.zeros(n)
                for i in range(20, n):
                    mean_vol = np.mean(volumes[i-20:i])
                    vol_norm[i] = volumes[i] / mean_vol if mean_vol > 0 else 1.0
                vol_norm[:20] = 1.0
                feature_list.append(vol_norm.reshape(-1, 1))

        if not feature_list:
            return returns.reshape(-1, 1)

        return np.hstack(feature_list)

    def fit(
        self,
        prices: NDArray[np.float64],
        returns: Optional[NDArray[np.float64]] = None,
        volumes: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Fit HMM model to historical data.

        Args:
            prices: Price series
            returns: Optional return series
            volumes: Optional volume series
        """
        if self._model is None:
            self._create_model()

        # Extract features
        features = self._extract_features(prices, returns, volumes)

        if len(features) < 50:
            return

        # Use recent lookback window
        features = features[-self.lookback:]

        # Fit model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(features)
            self._is_fitted = True

            # Map states to regimes based on characteristics
            self._map_states_to_regimes(features)

            # Calculate regime statistics
            self._calculate_regime_stats(features)

        except Exception as e:
            print(f"HMM fitting failed: {e}")
            self._is_fitted = False

    def _map_states_to_regimes(self, features: NDArray) -> None:
        """
        Map HMM states to interpretable regimes.

        Uses mean return and volatility of each state.
        """
        if not self._is_fitted:
            return

        # Get state means
        means = self._model.means_
        n_features = means.shape[1]

        # First feature is typically returns
        return_means = means[:, 0]

        # Second feature (if exists) is typically volatility
        vol_means = means[:, 1] if n_features > 1 else np.ones(self.n_states)

        # Sort states by return (descending)
        sorted_indices = np.argsort(return_means)[::-1]

        # Map to regimes
        if self.n_states == 2:
            self._regime_mapping = {
                sorted_indices[0]: MarketRegime.BULL,
                sorted_indices[1]: MarketRegime.BEAR,
            }
        elif self.n_states == 3:
            self._regime_mapping = {
                sorted_indices[0]: MarketRegime.BULL,
                sorted_indices[1]: MarketRegime.NEUTRAL,
                sorted_indices[2]: MarketRegime.BEAR,
            }
        else:
            # For 4+ states, include volatility regimes
            self._regime_mapping = {}
            for i, idx in enumerate(sorted_indices):
                if i == 0:
                    self._regime_mapping[idx] = MarketRegime.BULL
                elif i == len(sorted_indices) - 1:
                    self._regime_mapping[idx] = MarketRegime.BEAR
                elif vol_means[idx] > np.mean(vol_means):
                    self._regime_mapping[idx] = MarketRegime.HIGH_VOL
                else:
                    self._regime_mapping[idx] = MarketRegime.NEUTRAL

    def _calculate_regime_stats(self, features: NDArray) -> None:
        """Calculate statistics for each regime."""
        if not self._is_fitted:
            return

        states = self._model.predict(features)

        for state in range(self.n_states):
            mask = states == state
            if np.sum(mask) < 5:
                continue

            state_features = features[mask]

            self._regime_stats[state] = {
                "mean_return": float(np.mean(state_features[:, 0])),
                "volatility": float(np.std(state_features[:, 0]) * SQRT_252),
                "frequency": float(np.mean(mask)),
                "observations": int(np.sum(mask)),
            }

    def predict_regime(
        self,
        prices: NDArray[np.float64],
        returns: Optional[NDArray[np.float64]] = None,
        volumes: Optional[NDArray[np.float64]] = None,
    ) -> RegimeState:
        """
        Predict current market regime.

        Args:
            prices: Recent price series (at least 20 points)
            returns: Optional return series
            volumes: Optional volume series

        Returns:
            RegimeState with regime and probabilities
        """
        if not self._is_fitted:
            return self._default_regime_state()

        # Extract features
        features = self._extract_features(prices, returns, volumes)

        if len(features) < 1:
            return self._default_regime_state()

        # Get state probabilities for last observation
        try:
            # Predict most likely state
            state = self._model.predict(features[-1:].reshape(1, -1))[0]

            # Get state probabilities
            if hasattr(self._model, 'predict_proba'):
                probs = self._model.predict_proba(features[-1:].reshape(1, -1))[0]
            else:
                probs = np.zeros(self.n_states)
                probs[state] = 1.0

            # Map to regime
            regime = self._regime_mapping.get(state, MarketRegime.NEUTRAL)
            probability = probs[state]

            # Build probability dict
            regime_probs = {}
            for s, r in self._regime_mapping.items():
                if r not in regime_probs:
                    regime_probs[r] = 0.0
                regime_probs[r] += probs[s]

            # Update tracking
            if regime == self._current_regime:
                self._regime_duration += 1
            else:
                # Apply regime persistence
                if self._regime_duration >= self.min_regime_duration:
                    self._current_regime = regime
                    self._regime_duration = 1
                else:
                    # Stay in current regime
                    regime = self._current_regime
                    self._regime_duration += 1

            self._regime_history.append(regime)

            # Calculate transition probability
            trans_prob = 1.0 - probability

            # Extract feature values for context
            feature_values = {
                self.features[i]: float(features[-1, i])
                for i in range(min(len(self.features), features.shape[1]))
            }

            return RegimeState(
                regime=regime,
                probability=float(probability),
                regime_probabilities=regime_probs,
                regime_duration=self._regime_duration,
                transition_prob=float(trans_prob),
                features_used=feature_values,
            )

        except Exception as e:
            print(f"Regime prediction failed: {e}")
            return self._default_regime_state()

    def get_transition_matrix(self) -> Optional[NDArray[np.float64]]:
        """Get regime transition probability matrix."""
        if not self._is_fitted or not hasattr(self._model, 'transmat_'):
            return None
        return self._model.transmat_

    def get_regime_statistics(self) -> Dict[MarketRegime, RegimeStatistics]:
        """Get statistics for each regime."""
        stats = {}

        for state, regime in self._regime_mapping.items():
            if state not in self._regime_stats:
                continue

            state_stats = self._regime_stats[state]

            # Get transition probabilities to this regime
            trans_probs = {}
            if hasattr(self._model, 'transmat_'):
                for other_state, other_regime in self._regime_mapping.items():
                    trans_probs[other_regime] = float(self._model.transmat_[other_state, state])

            stats[regime] = RegimeStatistics(
                regime=regime,
                mean_return=state_stats["mean_return"] * 252,  # Annualize
                volatility=state_stats["volatility"],
                avg_duration=1.0 / (1.0 - self._model.transmat_[state, state])
                            if hasattr(self._model, 'transmat_') else 10.0,
                frequency=state_stats["frequency"],
                transition_probs=trans_probs,
            )

        return stats

    def should_retrain(self, current_bar: int) -> bool:
        """Check if model should be retrained."""
        return (current_bar - self._last_train_bar) >= self.retrain_frequency

    def _default_regime_state(self) -> RegimeState:
        """Return default regime state when model not fitted."""
        return RegimeState(
            regime=MarketRegime.NEUTRAL,
            probability=0.5,
            regime_probabilities={
                MarketRegime.BULL: 0.33,
                MarketRegime.NEUTRAL: 0.34,
                MarketRegime.BEAR: 0.33,
            },
            regime_duration=0,
            transition_prob=0.1,
            features_used={},
        )


# =============================================================================
# SIMPLE HMM FALLBACK (No hmmlearn dependency)
# =============================================================================

class SimpleGaussianHMM:
    """
    Simple Gaussian HMM implementation as fallback.

    Uses EM algorithm for parameter estimation.
    Less efficient than hmmlearn but works without dependencies.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.means_ = None
        self.covars_ = None
        self._covars_inv = None  # Precomputed inverse covariances
        self.transmat_ = None
        self.startprob_ = None

    def fit(self, X: NDArray) -> "SimpleGaussianHMM":
        """Fit HMM using K-means initialisation + EM refinement."""
        n_samples, n_features = X.shape

        # Initialize parameters using K-means-like approach
        # Sort by first feature (returns) and divide into states
        sorted_indices = np.argsort(X[:, 0])
        chunk_size = n_samples // self.n_states

        self.means_ = np.zeros((self.n_states, n_features))
        self.covars_ = np.zeros((self.n_states, n_features, n_features))

        for i in range(self.n_states):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_states - 1 else n_samples
            chunk = X[sorted_indices[start:end]]

            self.means_[i] = np.mean(chunk, axis=0)
            cov = np.cov(chunk, rowvar=False)
            if n_features == 1:
                cov = cov.reshape(1, 1)
            self.covars_[i] = cov + np.eye(n_features) * 1e-6

        # Initialize transition matrix with mild self-transition bias
        self.transmat_ = np.full(
            (self.n_states, self.n_states), 0.05 / (self.n_states - 1)
        )
        np.fill_diagonal(self.transmat_, 0.95)
        self.startprob_ = np.ones(self.n_states) / self.n_states

        # EM refinement: re-estimate means, covariances, transitions
        for _ in range(min(self.n_iter, 20)):
            self._update_covars_inv(n_features)
            responsibilities = self.predict_proba(X)  # E-step soft assignments

            # M-step: update means and covariances
            for s in range(self.n_states):
                w = responsibilities[:, s]
                w_sum = w.sum()
                if w_sum < 1.0:
                    continue
                self.means_[s] = (w[:, None] * X).sum(axis=0) / w_sum
                diff = X - self.means_[s]
                self.covars_[s] = (
                    (diff * w[:, None]).T @ diff / w_sum
                    + np.eye(n_features) * 1e-6
                )

            # M-step: update transition matrix from hard assignments
            states = np.argmax(responsibilities, axis=1)
            trans_counts = np.ones((self.n_states, self.n_states)) * 0.1
            for t in range(1, len(states)):
                trans_counts[states[t - 1], states[t]] += 1.0
            self.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        # Final inverse covariance precompute
        self._update_covars_inv(n_features)

        # Final state assignments for start probabilities
        states = self.predict(X)
        counts = np.bincount(states, minlength=self.n_states).astype(float) + 0.1
        self.startprob_ = counts / counts.sum()

        return self

    def _update_covars_inv(self, n_features: int) -> None:
        """Precompute inverse covariance matrices."""
        self._covars_inv = np.zeros_like(self.covars_)
        for i in range(self.n_states):
            try:
                self._covars_inv[i] = np.linalg.inv(self.covars_[i])
            except np.linalg.LinAlgError:
                self._covars_inv[i] = np.eye(n_features)

    def predict(self, X: NDArray) -> NDArray:
        """Predict most likely states using precomputed inverse covariances."""
        if self.means_ is None:
            return np.zeros(len(X), dtype=int)

        n_samples = len(X)
        # Vectorized: compute Mahalanobis distance for all samples and states
        # likelihoods[t, s] = exp(-0.5 * (X[t]-mu_s) @ inv(cov_s) @ (X[t]-mu_s))
        log_likelihoods = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            diff = X - self.means_[s]  # (n_samples, n_features)
            cov_inv = self._covars_inv[s] if self._covars_inv is not None else np.eye(X.shape[1])
            # Batch Mahalanobis: (diff @ cov_inv) * diff, summed over features
            log_likelihoods[:, s] = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

        return np.argmax(log_likelihoods, axis=1).astype(int)

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict state probabilities using precomputed inverse covariances."""
        if self.means_ is None:
            return np.ones((len(X), self.n_states)) / self.n_states

        n_samples = len(X)
        log_probs = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            diff = X - self.means_[s]
            cov_inv = self._covars_inv[s] if self._covars_inv is not None else np.eye(X.shape[1])
            log_probs[:, s] = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

        # Softmax normalization (numerically stable)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        totals = np.sum(probs, axis=1, keepdims=True)
        probs = np.where(totals > 0, probs / totals, 1.0 / self.n_states)

        return probs
