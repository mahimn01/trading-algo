"""
Multi-Strategy Controller

Coordinates multiple independent trading strategies into a unified
portfolio with centralized risk management and capital allocation.

Responsibilities:
    1. Feed market data to all registered strategies
    2. Collect signals from every active strategy each bar
    3. Resolve conflicts (e.g. two strategies disagree on same symbol)
    4. Apply portfolio-level risk limits via the quant_core RiskController
    5. Produce a final list of StrategySignals the execution layer can act on

The controller does NOT execute orders itself — it only produces an
agreed-upon, risk-checked set of target signals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyAllocation:
    """Allocation configuration for a single strategy."""
    weight: float          # Fraction of capital allocated (e.g. 0.40)
    max_positions: int = 10  # Max concurrent positions for this strategy
    enabled: bool = True


@dataclass
class ControllerConfig:
    """
    Configuration for the MultiStrategyController.

    Default allocations:
      Orchestrator  40%  — core high-conviction ensemble
      ORB           15%  — short window, high edge-per-trade
      Pairs         15%  — market-neutral, uncorrelated
      Momentum      30%  — trend-following workhorse
    """

    allocations: Dict[str, StrategyAllocation] = field(default_factory=lambda: {
        "Orchestrator": StrategyAllocation(weight=0.20, max_positions=12),
        "ORB": StrategyAllocation(weight=0.10, max_positions=10),
        "PairsTrading": StrategyAllocation(weight=0.12, max_positions=10),
        "PureMomentum": StrategyAllocation(weight=0.15, max_positions=10),
        "RegimeTransition": StrategyAllocation(weight=0.10, max_positions=8),
        "CrossAssetDivergence": StrategyAllocation(weight=0.10, max_positions=8),
        "FlowPressure": StrategyAllocation(weight=0.13, max_positions=10),
        "LiquidityCycles": StrategyAllocation(weight=0.10, max_positions=10),
        "LeadLagArbitrage": StrategyAllocation(weight=0.08, max_positions=4),
        "HurstAdaptive": StrategyAllocation(weight=0.08, max_positions=6),
        "TimeAdaptive": StrategyAllocation(weight=0.06, max_positions=4),
    })

    # ── Portfolio-level limits ──────────────────────────────────────────
    max_gross_exposure: float = 1.5
    """Maximum gross exposure as fraction of equity."""

    max_net_exposure: float = 0.80
    """Maximum net (long - short) exposure as fraction of equity."""

    max_single_symbol_weight: float = 0.20
    """Maximum total weight in any one symbol across all strategies."""

    max_portfolio_positions: int = 40
    """Maximum total concurrent positions across all strategies."""

    # ── Conflict resolution ─────────────────────────────────────────────
    conflict_resolution: str = "weighted_confidence"
    """How to resolve conflicting signals on the same symbol.

    Options:
        - 'weighted_confidence': Pick direction with highest
          (confidence * allocation_weight).  Default.
        - 'veto': Any short signal blocks a long (conservative).
        - 'net': Sum directional weights and go with the net direction.
    """

    # ── Regime-adaptive allocation (Phase 5) ───────────────────────────
    enable_regime_adaptation: bool = False
    """When True, dynamically reweight strategy allocations based on
    the current HMM market regime.  Overweight strategies that
    historically perform well in the detected regime."""

    regime_blend_factor: float = 0.5
    """Blend factor between static and regime-adaptive weights.
    0.0 = fully static, 1.0 = fully regime-adaptive."""

    # ── Risk integration ────────────────────────────────────────────────
    enable_risk_controller: bool = True
    """When True, run quant_core RiskController before emitting signals."""

    max_drawdown: float = 0.40
    """Portfolio-level max drawdown (used by risk controller)."""

    daily_loss_limit: float = 0.05
    """Stop trading for the day if portfolio loses more than this %."""

    # ── Volatility management (Moreira & Muir 2017) ─────────────────────
    enable_vol_management: bool = True
    """Scale total exposure inversely with realized volatility.
    Based on Moreira & Muir (2017, JoF) 'Volatility-Managed Portfolios'.
    Improves Sharpe ratio by ~50% across asset classes."""

    vol_target: float = 0.15
    """Target annualized portfolio volatility (15%)."""

    vol_lookback: int = 20
    """Number of daily returns for realized vol estimate."""

    vol_scale_min: float = 0.25
    """Minimum vol scalar (floor). Prevents over-shrinking in high-vol regimes."""

    vol_scale_max: float = 2.0
    """Maximum vol scalar (cap). Prevents over-leveraging in low-vol regimes."""

    # ── Entropy filter ────────────────────────────────────────────────
    enable_entropy_filter: bool = False
    """When True, scale entry signals by market entropy.
    Low entropy (predictable) → full signal, high entropy (random) → near-block.
    Based on sample entropy of reference symbol returns."""


# ────────────────────────────────────────────────────────────────────────────
# Controller
# ────────────────────────────────────────────────────────────────────────────

class MultiStrategyController:
    """
    Central coordinator for multiple trading strategies.

    Usage::

        controller = MultiStrategyController(config)
        controller.register(orchestrator_adapter)
        controller.register(orb_adapter)
        controller.register(pairs_adapter)
        controller.register(momentum_adapter)

        # Each bar:
        for symbol in symbols:
            controller.update(symbol, ts, o, h, l, c, v)
        signals = controller.generate_signals(symbols, ts, equity)
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self._strategies: Dict[str, TradingStrategy] = {}

        # Portfolio state
        self._equity: float = 100_000.0
        self._peak_equity: float = 100_000.0
        self._current_positions: Dict[str, float] = {}  # symbol -> net weight
        self._daily_pnl: float = 0.0
        self._halted: bool = False
        self._dd_warned_today: bool = False
        self._daily_loss_warned_today: bool = False

        # Per-strategy position tracking
        self._strategy_positions: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Returns history for risk controller
        self._returns: List[float] = []

        # Regime detection (Phase 5)
        self._current_regime: str = "NEUTRAL"
        self._regime_weights: Dict[str, Dict[str, float]] = {}
        self._price_history_for_regime: List[float] = []  # SPY-like proxy
        self._hmm_model = None  # Lazy-loaded

        # Entropy filter (Phase 6)
        self._entropy_filter = None
        self._entropy_ref_symbol: str = "SPY"
        self._entropy_last_day: Optional[str] = None
        self._entropy_day_open: Optional[float] = None
        self._entropy_day_close: Optional[float] = None
        if self.config.enable_entropy_filter:
            try:
                from trading_algo.quant_core.strategies.entropy_regime_filter import (
                    EntropyRegimeFilter,
                )
                self._entropy_filter = EntropyRegimeFilter()
                self._entropy_ref_symbol = self._entropy_filter._config.reference_symbol
            except ImportError:
                logger.warning("Entropy filter requested but module not available")

    # ── Registration ────────────────────────────────────────────────────

    def register(self, strategy: TradingStrategy) -> None:
        """Register a strategy adapter with the controller."""
        name = strategy.name
        if name in self._strategies:
            logger.warning("Strategy %s already registered, replacing", name)
        self._strategies[name] = strategy
        logger.info("Registered strategy: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a strategy from the controller."""
        self._strategies.pop(name, None)
        self._strategy_positions.pop(name, None)

    @property
    def strategies(self) -> Dict[str, TradingStrategy]:
        return dict(self._strategies)

    @property
    def active_strategies(self) -> List[TradingStrategy]:
        """Return strategies that are currently active."""
        return [
            s for s in self._strategies.values()
            if s.state == StrategyState.ACTIVE
            and self._get_allocation(s.name).enabled
        ]

    # ── Data feed ───────────────────────────────────────────────────────

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
        """Feed one bar of data to all registered strategies."""
        for strategy in self._strategies.values():
            try:
                strategy.update(symbol, timestamp, open_price, high, low, close, volume)
            except Exception as e:
                logger.error("Error updating %s with %s: %s", strategy.name, symbol, e)

        # Feed entropy filter with reference symbol daily returns
        if self._entropy_filter is not None and symbol == self._entropy_ref_symbol:
            day_str = timestamp.strftime("%Y-%m-%d")
            if self._entropy_last_day != day_str:
                # New trading day — compute prior day's return and feed it
                if (
                    self._entropy_last_day is not None
                    and self._entropy_day_open is not None
                    and self._entropy_day_close is not None
                    and self._entropy_day_open > 0
                ):
                    daily_ret = (self._entropy_day_close / self._entropy_day_open) - 1.0
                    self._entropy_filter.update(daily_ret)
                self._entropy_last_day = day_str
                self._entropy_day_open = open_price
            self._entropy_day_close = close

    # ── Signal generation pipeline ──────────────────────────────────────

    def generate_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
        equity: Optional[float] = None,
    ) -> List[StrategySignal]:
        """
        Run the full signal generation pipeline.

        1. Collect raw signals from all active strategies.
        2. Scale weights by allocation.
        3. Resolve conflicts.
        4. Apply portfolio-level limits.
        5. Apply risk checks.
        6. Apply vol management (Moreira & Muir 2017).

        Returns:
            List of risk-checked, conflict-resolved StrategySignals
            ready for execution.
        """
        if equity is not None:
            self._equity = equity
            self._peak_equity = max(self._peak_equity, equity)

        # Daily loss circuit breaker
        if self._halted:
            return []

        # Step 1 — collect raw signals
        raw_signals = self._collect_signals(symbols, timestamp)
        if not raw_signals:
            return []

        # Step 1.5 — entropy filter (scale entries by market predictability)
        filtered = self._apply_entropy_filter(raw_signals)

        # Step 2 — scale by allocation weight
        scaled = self._apply_allocation_weights(filtered)

        # Step 3 — resolve conflicts (same symbol, different strategies)
        resolved = self._resolve_conflicts(scaled)

        # Step 4 — enforce portfolio-level limits
        limited = self._apply_portfolio_limits(resolved)

        # Step 5 — risk checks
        risk_checked = self._apply_risk_checks(limited)

        # Step 6 — vol management
        final = self._apply_vol_management(risk_checked)

        return final

    # ── Step 1: Collect ─────────────────────────────────────────────────

    def _collect_signals(
        self,
        symbols: List[str],
        timestamp: datetime,
    ) -> List[StrategySignal]:
        """Gather signals from all active strategies."""
        all_signals: List[StrategySignal] = []

        for strategy in self.active_strategies:
            alloc = self._get_allocation(strategy.name)

            # Check per-strategy position limit
            current_count = len(self._strategy_positions.get(strategy.name, {}))
            if current_count >= alloc.max_positions:
                # Only allow exit signals
                sigs = strategy.generate_signals(symbols, timestamp)
                sigs = [s for s in sigs if s.is_exit]
            else:
                sigs = strategy.generate_signals(symbols, timestamp)

            all_signals.extend(sigs)

        return all_signals

    # ── Step 1.5: Entropy filter ───────────────────────────────────────

    def _apply_entropy_filter(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """
        Scale entry signals by market entropy regime.

        Low entropy (predictable) → full passthrough.
        High entropy (random) → near-block on entries (10% scaling).
        Exit signals always pass through unscaled.
        """
        if self._entropy_filter is None:
            return signals

        scale = self._entropy_filter.get_scaling_factor()

        # No meaningful reduction — skip the copy
        if scale >= 0.99:
            return signals

        regime = self._entropy_filter.get_entropy_regime()
        logger.debug(
            "Entropy filter: regime=%s scale=%.2f (entropy=%.4f)",
            regime, scale, self._entropy_filter.get_current_entropy(),
        )

        return [
            StrategySignal(
                strategy_name=s.strategy_name,
                symbol=s.symbol,
                direction=s.direction,
                target_weight=s.target_weight * scale if s.is_entry else s.target_weight,
                confidence=s.confidence,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                entry_price=s.entry_price,
                trade_type=s.trade_type,
                metadata={**s.metadata, "entropy_scale": scale, "entropy_regime": regime},
            )
            for s in signals
        ]

    # ── Step 2: Allocation weighting ────────────────────────────────────

    def _apply_allocation_weights(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """Scale each signal's target_weight by the strategy's allocation.

        If regime adaptation is enabled, blends static allocations with
        regime-conditional weights based on the current detected regime.
        """
        scaled: List[StrategySignal] = []

        # Get regime-adaptive weight overrides (if enabled)
        regime_overrides = self._get_regime_adaptive_weights()

        for sig in signals:
            alloc = self._get_allocation(sig.strategy_name)
            static_weight = alloc.weight

            if regime_overrides and sig.strategy_name in regime_overrides:
                dynamic_weight = regime_overrides[sig.strategy_name]
                blend = self.config.regime_blend_factor
                effective_weight = (1 - blend) * static_weight + blend * dynamic_weight
            else:
                effective_weight = static_weight

            scaled_weight = sig.target_weight * effective_weight

            scaled.append(StrategySignal(
                strategy_name=sig.strategy_name,
                symbol=sig.symbol,
                direction=sig.direction,
                target_weight=scaled_weight,
                confidence=sig.confidence,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                entry_price=sig.entry_price,
                trade_type=sig.trade_type,
                metadata=sig.metadata,
            ))

        return scaled

    def _get_regime_adaptive_weights(self) -> Optional[Dict[str, float]]:
        """
        Compute regime-adaptive strategy weights.

        Uses HMM regime detection (when available) to look up
        pre-computed regime-conditional weight tables.
        """
        if not self.config.enable_regime_adaptation:
            return None

        # Check for pre-loaded regime weights
        if self._current_regime in self._regime_weights:
            return self._regime_weights[self._current_regime]

        # Fall back to default regime tilts from regime_analysis
        try:
            from trading_algo.multi_strategy.regime_analysis import RegimeAnalyzer
            defaults = RegimeAnalyzer.DEFAULT_REGIME_TILTS
            return defaults.get(self._current_regime, defaults.get("NEUTRAL"))
        except Exception:
            return None

    def set_regime(self, regime: str) -> None:
        """Manually set the current market regime."""
        self._current_regime = regime

    def set_regime_weights(self, weights: Dict[str, Dict[str, float]]) -> None:
        """Load regime-conditional weight tables (from RegimeAnalyzer output)."""
        self._regime_weights = weights

    def detect_regime(self) -> str:
        """
        Detect the current market regime from price history.

        Returns the regime label string.
        """
        if len(self._returns) < 60:
            return "NEUTRAL"

        try:
            from trading_algo.quant_core.models.hmm_regime import (
                HiddenMarkovRegime,
            )

            if self._hmm_model is None:
                self._hmm_model = HiddenMarkovRegime(n_states=3, lookback=252)

            returns_arr = np.array(self._returns[-252:], dtype=np.float64)
            prices = np.cumprod(1 + returns_arr) * 100  # Synthetic price

            self._hmm_model.fit(prices, returns_arr)
            state = self._hmm_model.predict_regime(prices, returns_arr)
            self._current_regime = state.regime.name
            return self._current_regime

        except Exception as e:
            logger.debug("Regime detection failed: %s", e)
            # Simple fallback: use return momentum
            recent = np.array(self._returns[-20:])
            if np.mean(recent) > 0.001:
                self._current_regime = "BULL"
            elif np.mean(recent) < -0.001:
                self._current_regime = "BEAR"
            elif np.std(recent) > 0.02:
                self._current_regime = "HIGH_VOL"
            else:
                self._current_regime = "NEUTRAL"
            return self._current_regime

    # ── Step 3: Conflict resolution ─────────────────────────────────────

    def _resolve_conflicts(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """
        When multiple strategies emit signals on the same symbol,
        combine them according to the configured resolution method.
        """
        # Group by symbol
        by_symbol: Dict[str, List[StrategySignal]] = defaultdict(list)
        for sig in signals:
            by_symbol[sig.symbol].append(sig)

        resolved: List[StrategySignal] = []

        for symbol, sigs in by_symbol.items():
            # Separate exit and entry signals
            exits = [s for s in sigs if s.is_exit]
            entries = [s for s in sigs if s.is_entry]

            # Always include exit signals (they don't conflict)
            resolved.extend(exits)

            if not entries:
                continue

            # Check for directional conflict
            longs = [s for s in entries if s.direction > 0]
            shorts = [s for s in entries if s.direction < 0]

            if longs and shorts:
                # Conflict!
                resolved.extend(
                    self._resolve_directional_conflict(symbol, longs, shorts)
                )
            else:
                # No conflict — merge signals in same direction
                resolved.extend(self._merge_same_direction(entries))

        return resolved

    def _resolve_directional_conflict(
        self,
        symbol: str,
        longs: List[StrategySignal],
        shorts: List[StrategySignal],
    ) -> List[StrategySignal]:
        """Resolve opposing long/short signals on the same symbol."""
        method = self.config.conflict_resolution

        if method == "veto":
            # Any short blocks all longs (conservative)
            logger.debug(
                "Conflict on %s: short veto blocks %d longs",
                symbol, len(longs),
            )
            return shorts  # Only shorts survive

        elif method == "net":
            # Sum directional weights
            long_weight = sum(s.target_weight * s.confidence for s in longs)
            short_weight = sum(s.target_weight * s.confidence for s in shorts)
            if long_weight >= short_weight:
                return self._merge_same_direction(longs)
            else:
                return self._merge_same_direction(shorts)

        else:  # weighted_confidence (default)
            long_score = sum(
                s.target_weight * s.confidence for s in longs
            )
            short_score = sum(
                s.target_weight * s.confidence for s in shorts
            )

            if long_score >= short_score:
                winner = longs
            else:
                winner = shorts

            logger.debug(
                "Conflict on %s: long=%.3f short=%.3f → %s wins",
                symbol, long_score, short_score,
                "long" if long_score >= short_score else "short",
            )
            return self._merge_same_direction(winner)

    def _merge_same_direction(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """
        Merge multiple same-direction signals on one symbol
        into a single combined signal.

        Uses the highest-confidence signal as the "primary" and
        sums the target weights.
        """
        if not signals:
            return []
        if len(signals) == 1:
            return signals

        # Sort by confidence descending — primary signal first
        signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
        primary = signals[0]

        merged_weight = sum(s.target_weight for s in signals)
        # Weighted average confidence
        total_wt = sum(s.target_weight for s in signals)
        if total_wt > 0:
            merged_confidence = sum(
                s.confidence * s.target_weight for s in signals
            ) / total_wt
        else:
            merged_confidence = primary.confidence

        contributors = [s.strategy_name for s in signals]

        return [StrategySignal(
            strategy_name="+".join(contributors),
            symbol=primary.symbol,
            direction=primary.direction,
            target_weight=merged_weight,
            confidence=min(1.0, merged_confidence),
            stop_loss=primary.stop_loss,
            take_profit=primary.take_profit,
            entry_price=primary.entry_price,
            trade_type=primary.trade_type,
            metadata={
                **primary.metadata,
                "contributors": contributors,
                "n_strategies": len(signals),
            },
        )]

    # ── Step 4: Portfolio limits ────────────────────────────────────────

    def _apply_portfolio_limits(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """Enforce portfolio-wide exposure and concentration limits."""
        # Group entry signals by symbol to check single-symbol limit
        entry_signals = [s for s in signals if s.is_entry]
        exit_signals = [s for s in signals if s.is_exit]

        # Cap single-symbol weight
        capped: List[StrategySignal] = []
        for sig in entry_signals:
            existing_weight = abs(self._current_positions.get(sig.symbol, 0.0))
            headroom = max(0.0, self.config.max_single_symbol_weight - existing_weight)

            if headroom <= 0:
                logger.debug(
                    "Skipping %s %s: already at max single-symbol weight",
                    sig.symbol, sig.strategy_name,
                )
                continue

            if sig.target_weight > headroom:
                sig = StrategySignal(
                    strategy_name=sig.strategy_name,
                    symbol=sig.symbol,
                    direction=sig.direction,
                    target_weight=headroom,
                    confidence=sig.confidence,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    entry_price=sig.entry_price,
                    trade_type=sig.trade_type,
                    metadata=sig.metadata,
                )
            capped.append(sig)

        # Check total position count
        current_count = len(self._current_positions)
        max_new = max(0, self.config.max_portfolio_positions - current_count)
        if len(capped) > max_new:
            # Keep highest-confidence signals
            capped = sorted(capped, key=lambda s: s.confidence, reverse=True)[:max_new]

        # Check gross exposure
        current_gross = sum(abs(w) for w in self._current_positions.values())
        new_gross = sum(s.target_weight for s in capped)
        total_gross = current_gross + new_gross

        if total_gross > self.config.max_gross_exposure:
            scale = max(0.0, (self.config.max_gross_exposure - current_gross) / new_gross) if new_gross > 0 else 0.0
            capped = [
                StrategySignal(
                    strategy_name=s.strategy_name,
                    symbol=s.symbol,
                    direction=s.direction,
                    target_weight=s.target_weight * scale,
                    confidence=s.confidence,
                    stop_loss=s.stop_loss,
                    take_profit=s.take_profit,
                    entry_price=s.entry_price,
                    trade_type=s.trade_type,
                    metadata=s.metadata,
                )
                for s in capped
                if s.target_weight * scale > 0.001  # Drop negligible signals
            ]

        return exit_signals + capped

    # ── Step 5: Risk checks ─────────────────────────────────────────────

    def _apply_risk_checks(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """
        Apply portfolio-level risk checks.

        Uses simplified checks inline (drawdown, daily loss).
        The full quant_core RiskController can be plugged in externally
        via the ``risk_evaluate()`` method.
        """
        if not self.config.enable_risk_controller:
            return signals

        # Drawdown check
        if self._equity < self._peak_equity:
            dd = 1.0 - self._equity / self._peak_equity
            if dd >= self.config.max_drawdown:
                if not self._dd_warned_today:
                    logger.warning(
                        "Max drawdown %.1f%% hit — blocking all entries",
                        dd * 100,
                    )
                    self._dd_warned_today = True
                return [s for s in signals if s.is_exit]

            # Scale down near drawdown limit
            if dd >= self.config.max_drawdown * 0.75:
                remaining = (self.config.max_drawdown - dd) / self.config.max_drawdown
                signals = [
                    StrategySignal(
                        strategy_name=s.strategy_name,
                        symbol=s.symbol,
                        direction=s.direction,
                        target_weight=s.target_weight * remaining if s.is_entry else s.target_weight,
                        confidence=s.confidence,
                        stop_loss=s.stop_loss,
                        take_profit=s.take_profit,
                        entry_price=s.entry_price,
                        trade_type=s.trade_type,
                        metadata=s.metadata,
                    )
                    for s in signals
                ]

        # Daily loss check
        if self._daily_pnl <= -self.config.daily_loss_limit:
            if not self._daily_loss_warned_today:
                logger.warning(
                    "Daily loss limit %.1f%% hit — halting entries",
                    self._daily_pnl * 100,
                )
                self._daily_loss_warned_today = True
            self._halted = True
            return [s for s in signals if s.is_exit]

        return signals

    # ── Step 6: Volatility management ──────────────────────────────────

    def _apply_vol_management(
        self,
        signals: List[StrategySignal],
    ) -> List[StrategySignal]:
        """
        Moreira & Muir (2017) volatility management.

        Scale all entry signal weights inversely with realized portfolio
        volatility.  When vol is high, reduce exposure; when vol is low,
        increase exposure.  Improves Sharpe by ~50%.
        """
        if not self.config.enable_vol_management:
            return signals

        if len(self._returns) < self.config.vol_lookback:
            return signals  # Not enough history

        recent = np.array(self._returns[-self.config.vol_lookback:])
        realized_vol = float(np.std(recent, ddof=1) * np.sqrt(252))

        if realized_vol <= 0:
            return signals

        vol_scalar = min(
            self.config.vol_scale_max,
            max(self.config.vol_scale_min, self.config.vol_target / realized_vol),
        )

        if abs(vol_scalar - 1.0) < 0.05:
            return signals  # No meaningful adjustment

        logger.debug(
            "Vol management: realized=%.1f%% target=%.1f%% scalar=%.2f",
            realized_vol * 100, self.config.vol_target * 100, vol_scalar,
        )

        return [
            StrategySignal(
                strategy_name=s.strategy_name,
                symbol=s.symbol,
                direction=s.direction,
                target_weight=s.target_weight * vol_scalar if s.is_entry else s.target_weight,
                confidence=s.confidence,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                entry_price=s.entry_price,
                trade_type=s.trade_type,
                metadata=s.metadata,
            )
            for s in signals
        ]

    def add_return(self, daily_return: float) -> None:
        """Record a daily portfolio return for vol management."""
        self._returns.append(daily_return)
        # Keep bounded
        if len(self._returns) > 504:
            self._returns = self._returns[-504:]

    # ── External risk integration ───────────────────────────────────────

    def risk_evaluate(
        self,
        signals: List[StrategySignal],
        equity: float,
        positions: Dict[str, float],
        returns: np.ndarray,
    ) -> List[StrategySignal]:
        """
        Optional: run the full quant_core RiskController.

        Call this externally if you want VaR/ES-based risk management
        on top of the controller's built-in checks.

        Returns filtered/scaled signals.
        """
        try:
            from trading_algo.quant_core.engine.risk_controller import (
                RiskController,
                RiskConfig,
            )

            rc = RiskController(RiskConfig(
                max_drawdown=self.config.max_drawdown,
                daily_loss_limit=self.config.daily_loss_limit,
                max_gross_exposure=self.config.max_gross_exposure,
            ))

            decision = rc.evaluate(equity, positions, returns)

            if not decision.can_trade:
                return [s for s in signals if s.is_exit]

            if decision.exposure_multiplier < 1.0:
                signals = [
                    StrategySignal(
                        strategy_name=s.strategy_name,
                        symbol=s.symbol,
                        direction=s.direction,
                        target_weight=(
                            s.target_weight * decision.exposure_multiplier
                            if s.is_entry else s.target_weight
                        ),
                        confidence=s.confidence,
                        stop_loss=s.stop_loss,
                        take_profit=s.take_profit,
                        entry_price=s.entry_price,
                        trade_type=s.trade_type,
                        metadata=s.metadata,
                    )
                    for s in signals
                ]

            return signals

        except Exception as e:
            logger.error("Risk evaluation failed: %s", e)
            return signals

    # ── State management ────────────────────────────────────────────────

    def update_portfolio_state(
        self,
        equity: float,
        positions: Dict[str, float],
        daily_pnl: float = 0.0,
    ) -> None:
        """
        Update the controller's view of the portfolio.

        Call this after execution to keep limits accurate.
        """
        self._equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        self._current_positions = dict(positions)
        self._daily_pnl = daily_pnl

    def new_trading_day(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self._daily_pnl = 0.0
        self._halted = False
        self._dd_warned_today = False
        self._daily_loss_warned_today = False

    def reset(self) -> None:
        """Full reset of controller and all strategies."""
        for strategy in self._strategies.values():
            strategy.reset()
        self._current_positions.clear()
        self._strategy_positions.clear()
        self._returns.clear()
        self._daily_pnl = 0.0
        self._halted = False
        self._dd_warned_today = False
        self._daily_loss_warned_today = False
        self._peak_equity = self._equity

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_allocation(self, strategy_name: str) -> StrategyAllocation:
        """Get allocation config for a strategy, with safe defaults."""
        return self.config.allocations.get(
            strategy_name,
            StrategyAllocation(weight=0.10, max_positions=5),
        )

    # ── Reporting ───────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return a summary dict for logging / dashboard."""
        strategy_status = {}
        for name, strat in self._strategies.items():
            alloc = self._get_allocation(name)
            strategy_status[name] = {
                "state": strat.state.name,
                "enabled": alloc.enabled,
                "allocation": alloc.weight,
                "exposure": strat.get_current_exposure(),
                "performance": strat.get_performance_stats(),
            }

        return {
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "drawdown": 1.0 - self._equity / self._peak_equity if self._peak_equity > 0 else 0.0,
            "daily_pnl": self._daily_pnl,
            "halted": self._halted,
            "n_positions": len(self._current_positions),
            "gross_exposure": sum(abs(w) for w in self._current_positions.values()),
            "strategies": strategy_status,
        }
