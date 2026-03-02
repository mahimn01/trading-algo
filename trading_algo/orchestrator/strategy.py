"""
The Orchestrator - Multi-Edge Ensemble Day Trading System

The main strategy that orchestrates all edge sources to make trading decisions.

Decision process:
1. Update all edge engines with latest data
2. Detect market regime
3. For each potential trade:
   a. Get votes from all edges (6 core + optional QuantEdge)
   b. Check for vetoes (any veto blocks the trade)
   c. Calculate consensus score
   d. Only trade if >= min_consensus_edges agree
4. Size position based on consensus strength
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Dict, Optional, Set

import numpy as np

from .config import OrchestratorConfig
from .types import (
    AssetState,
    EdgeSignal,
    EdgeVote,
    MarketRegime,
    OrchestratorSignal,
    TradeType,
)
from .edges import (
    MarketRegimeEngine,
    RelativeStrengthEngine,
    StatisticalExtremeDetector,
    VolumeProfileEngine,
    CrossAssetEngine,
    TimeOfDayEngine,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    The main strategy that orchestrates all edge sources.

    Accepts an optional ``OrchestratorConfig`` to override every tunable
    parameter.  When no config is supplied the defaults match the
    previous hard-coded values for full backward compatibility.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.cfg = config or OrchestratorConfig()

        # Core edge engines
        self.regime_engine = MarketRegimeEngine()
        self.relative_strength = RelativeStrengthEngine()
        self.statistics = StatisticalExtremeDetector()
        self.volume_profile = VolumeProfileEngine()
        self.cross_asset = CrossAssetEngine()
        self.time_of_day = TimeOfDayEngine()

        # Optional 7th edge (quant_core bridge)
        self._quant_edge = None
        if self.cfg.enable_quant_edge:
            try:
                from .edges.quant_edge import QuantEdge
                self._quant_edge = QuantEdge(min_bars=self.cfg.quant_edge_min_bars)
            except Exception:
                logger.info("QuantEdge disabled — quant_core not available")

        # Convenience aliases
        self.min_consensus_edges = self.cfg.min_consensus_edges
        self.min_consensus_score = self.cfg.min_consensus_score
        self.max_position_pct = self.cfg.max_position_pct
        self.min_reentry_bars = self.cfg.min_reentry_bars
        self.min_regime_confidence = self.cfg.min_regime_confidence
        self.min_atr_pct = self.cfg.min_atr_pct
        self.max_atr_pct = self.cfg.max_atr_pct
        self.max_opposition_score = self.cfg.max_opposition_score
        self.atr_stop_mult = self.cfg.atr_stop_mult
        self.atr_target_mult = self.cfg.atr_target_mult

        # Asset states
        self.asset_states: Dict[str, AssetState] = {}

        # Current positions
        self.positions: Dict[str, dict] = {}
        self.last_exit_bar_index: Dict[str, int] = {}

        # Reference assets we need to track
        self.reference_assets: Set[str] = {
            "SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE", "XLY", "XLV",
        }

        # ── Trade history for dynamic Kelly sizing ─────────────────────
        self._trade_returns: list = []  # P&L returns of completed trades
        self._kelly_calculator = None
        if self.cfg.sizing.use_kelly:
            try:
                from trading_algo.quant_core.portfolio.kelly import (
                    KellyCriterion,
                    KellyMode,
                )
                mode_map = {
                    0.125: KellyMode.EIGHTH,
                    0.25: KellyMode.QUARTER,
                    0.50: KellyMode.HALF,
                    0.75: KellyMode.THREE_QUARTER,
                    1.0: KellyMode.FULL,
                }
                frac = self.cfg.sizing.kelly_fraction
                mode = min(mode_map.items(), key=lambda kv: abs(kv[0] - frac))[1]
                self._kelly_calculator = KellyCriterion(
                    mode=mode,
                    max_position=self.cfg.max_position_pct,
                    min_samples=self.cfg.sizing.kelly_min_trades,
                )
                logger.info(
                    "Kelly sizing enabled (fraction=%.2f, min_trades=%d)",
                    frac,
                    self.cfg.sizing.kelly_min_trades,
                )
            except Exception:
                logger.info("Kelly sizing disabled — quant_core not available")

    # ── Data ingestion ────────────────────────────────────────────────

    def update_asset(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Update state for a single asset."""

        # Initialize state if needed
        if symbol not in self.asset_states:
            self.asset_states[symbol] = AssetState(symbol=symbol)

        state = self.asset_states[symbol]

        # Update price/volume history
        state.prices.append(close)
        state.highs.append(high)
        state.lows.append(low)
        state.volumes.append(volume)
        state.timestamps.append(timestamp)

        # Update day's metrics
        if len(state.prices) == 1 or timestamp.time() <= time(9, 35):
            state.day_open = open_price
            state.day_high = high
            state.day_low = low
            state.day_volume = volume
        else:
            state.day_high = max(state.day_high, high)
            state.day_low = min(state.day_low, low)
            state.day_volume += volume

        # --- VWAP (incremental running sum, periodic correction) ---
        tp = (high + low + close) / 3
        state._cum_tp_vol += tp * volume
        state._cum_vol += volume
        n_bars = len(state.prices)
        if n_bars >= state.prices.maxlen and n_bars % 50 == 0:
            # Periodic correction for evicted deque entries
            h_arr = np.asarray(state.highs)
            l_arr = np.asarray(state.lows)
            c_arr = np.asarray(state.prices)
            v_arr = np.asarray(state.volumes)
            state._cum_tp_vol = float(np.dot((h_arr + l_arr + c_arr) / 3.0, v_arr))
            state._cum_vol = float(np.sum(v_arr))
        state.vwap = state._cum_tp_vol / state._cum_vol if state._cum_vol > 0 else close

        # --- ATR (Wilder smoothing, O(1)) ---
        if n_bars > 1:
            prev_c = state.prices[-2]  # Previous close from deque
            true_range = max(high - low, abs(high - prev_c), abs(low - prev_c))
            if state._atr_count < 14:
                state._atr_count += 1
                state.atr = state.atr + (true_range - state.atr) / state._atr_count
            else:
                # Wilder smoothing: ATR = (prev_ATR * 13 + TR) / 14
                state.atr = (state.atr * 13 + true_range) / 14
            state.atr_pct = state.atr / close if close > 0 else 0

        # --- RSI (Wilder smoothing, O(1)) ---
        if n_bars > 1:
            change = close - state.prices[-2]
            gain = max(change, 0)
            loss = max(-change, 0)
            if state._rsi_count < 14:
                state._rsi_count += 1
                state._rsi_avg_gain += (gain - state._rsi_avg_gain) / state._rsi_count
                state._rsi_avg_loss += (loss - state._rsi_avg_loss) / state._rsi_count
            else:
                # Wilder smoothing: avg = (prev_avg * 13 + current) / 14
                state._rsi_avg_gain = (state._rsi_avg_gain * 13 + gain) / 14
                state._rsi_avg_loss = (state._rsi_avg_loss * 13 + loss) / 14
            if state._rsi_avg_loss > 0.0001:
                rs = state._rsi_avg_gain / state._rsi_avg_loss
                state.rsi = 100 - (100 / (1 + rs))
            elif state._rsi_avg_gain > 0:
                state.rsi = 100.0
            else:
                state.rsi = 50.0

        # Update edge engines
        self.regime_engine.update(symbol, state)
        self.relative_strength.update(symbol, state)
        self.cross_asset.update(symbol, state)

        if self._quant_edge is not None:
            self._quant_edge.update(symbol, state)
            # Feed benchmark data to the HMM regime model (once per day)
            if symbol == "SPY" and n_bars % 78 == 0:
                spy_prices = np.asarray(state.prices, dtype=np.float64)
                self._quant_edge.update_regime(spy_prices)

    # ── Signal generation ─────────────────────────────────────────────

    def generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> OrchestratorSignal:
        """
        Generate trading signal using ensemble of all edges.
        """
        if symbol not in self.asset_states:
            return self._hold_signal(symbol, timestamp, 0, "No data")

        state = self.asset_states[symbol]
        if len(state.prices) < self.cfg.warmup_bars:
            return self._hold_signal(symbol, timestamp, state.prices[-1] if state.prices else 0,
                                    "Warming up")

        price = state.prices[-1]
        current_bar = len(state.prices)

        # Check existing position
        if symbol in self.positions:
            return self._check_exit(symbol, timestamp, state)

        # Avoid immediate re-entries after exits.
        if symbol in self.last_exit_bar_index:
            bars_since_exit = current_bar - self.last_exit_bar_index[symbol]
            if bars_since_exit < self.min_reentry_bars:
                return self._hold_signal(
                    symbol,
                    timestamp,
                    price,
                    f"Cooldown active ({bars_since_exit}/{self.min_reentry_bars} bars)",
                )

        # Skip low-quality volatility regimes that tend to overtrade.
        if state.atr_pct < self.min_atr_pct:
            return self._hold_signal(
                symbol, timestamp, price,
                f"ATR too low ({state.atr_pct*100:.2f}%)",
            )
        if state.atr_pct > self.max_atr_pct:
            return self._hold_signal(
                symbol, timestamp, price,
                f"ATR too high ({state.atr_pct*100:.2f}%)",
            )

        # Step 1: Get market regime
        regime, regime_conf, regime_reason = self.regime_engine.detect_regime(timestamp)

        # Step 2: Skip if market is too uncertain or volatile
        if regime == MarketRegime.UNKNOWN:
            return self._hold_signal(symbol, timestamp, price, "Market regime unknown")
        if regime == MarketRegime.HIGH_VOLATILITY:
            return self._hold_signal(symbol, timestamp, price, "High volatility - sitting out")
        if regime_conf < self.min_regime_confidence:
            return self._hold_signal(
                symbol, timestamp, price,
                f"Weak regime confidence ({regime_conf:.2f})",
            )

        # Step 3: Determine potential trade direction based on regime
        mr_z = self.cfg.mean_reversion_zscore
        if regime in [MarketRegime.TREND_UP, MarketRegime.STRONG_TREND_UP, MarketRegime.REVERSAL_UP]:
            potential_direction = "long"
            trade_type = TradeType.MOMENTUM_CONTINUATION
        elif regime in [MarketRegime.TREND_DOWN, MarketRegime.STRONG_TREND_DOWN, MarketRegime.REVERSAL_DOWN]:
            potential_direction = "short"
            trade_type = TradeType.MOMENTUM_CONTINUATION
        else:  # Range bound
            stats = self.statistics.analyze(state)
            if stats["price_zscore"] > mr_z:
                potential_direction = "short"
                trade_type = TradeType.MEAN_REVERSION
            elif stats["price_zscore"] < -mr_z:
                potential_direction = "long"
                trade_type = TradeType.MEAN_REVERSION
            else:
                return self._hold_signal(symbol, timestamp, price,
                                        f"Range-bound, no extreme. z={stats['price_zscore']:.1f}")

        # Step 4: Collect votes from all edges
        votes: Dict[str, EdgeSignal] = {}

        # Edge 1: Relative Strength
        votes["RelativeStrength"] = self.relative_strength.get_vote(symbol)

        # Edge 2: Statistics
        votes["Statistics"] = self.statistics.get_vote(state, regime)

        # Edge 3: Volume Profile
        votes["VolumeProfile"] = self.volume_profile.get_vote(state, regime)

        # Edge 4: Cross-Asset
        votes["CrossAsset"] = self.cross_asset.get_vote(symbol, potential_direction)

        # Edge 5: Time of Day
        votes["TimeOfDay"] = self.time_of_day.get_vote(
            timestamp,
            trade_type,
            intended_direction=potential_direction,
        )

        # Edge 6: Regime (implicit vote based on regime strength)
        if regime_conf > 0.7:
            if potential_direction == "long":
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.STRONG_LONG, regime_conf, regime_reason)
            else:
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.STRONG_SHORT, regime_conf, regime_reason)
        elif regime_conf > 0.5:
            if potential_direction == "long":
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.LONG, regime_conf, regime_reason)
            else:
                votes["Regime"] = EdgeSignal("Regime", EdgeVote.SHORT, regime_conf, regime_reason)
        else:
            votes["Regime"] = EdgeSignal("Regime", EdgeVote.NEUTRAL, regime_conf, regime_reason)

        # Edge 7: Quant (optional)
        if self._quant_edge is not None:
            votes["Quant"] = self._quant_edge.get_vote(
                symbol, state, intended_direction=potential_direction,
            )

        # Step 5: Check for vetoes
        for edge_name, signal in votes.items():
            if potential_direction == "long" and signal.vote == EdgeVote.VETO_LONG:
                return self._hold_signal(symbol, timestamp, price,
                                        f"VETO from {edge_name}: {signal.reason}")
            if potential_direction == "short" and signal.vote == EdgeVote.VETO_SHORT:
                return self._hold_signal(symbol, timestamp, price,
                                        f"VETO from {edge_name}: {signal.reason}")

        # Step 6: Calculate consensus
        n_edges = len(votes)
        agreeing_edges = 0
        support_score = 0.0
        opposition_score = 0.0

        for signal in votes.values():
            vote_value = signal.vote.value

            if potential_direction == "long":
                if vote_value > 0:
                    agreeing_edges += 1
                    support_score += vote_value * signal.confidence
                elif vote_value < 0 and vote_value > EdgeVote.VETO_LONG.value:
                    opposition_score += abs(vote_value) * signal.confidence
            else:  # short
                if vote_value < 0:
                    agreeing_edges += 1
                    support_score += abs(vote_value) * signal.confidence
                elif vote_value > 0 and vote_value < EdgeVote.VETO_SHORT.value:
                    opposition_score += vote_value * signal.confidence

        # Normalize scores by number of edges
        consensus_score = support_score / n_edges if n_edges else 0
        opposition_score = opposition_score / n_edges if n_edges else 0
        agreement_ratio = (agreeing_edges / n_edges) if n_edges else 0.0
        directional_quality = (
            consensus_score / max(1e-9, consensus_score + opposition_score)
            if consensus_score > 0
            else 0.0
        )

        # Step 7: Check if we have enough agreement
        if agreeing_edges < self.min_consensus_edges:
            return self._hold_signal(symbol, timestamp, price,
                                    f"Insufficient consensus: {agreeing_edges}/{n_edges} edges agree")

        if consensus_score < self.min_consensus_score:
            return self._hold_signal(symbol, timestamp, price,
                                    f"Weak consensus score: {consensus_score:.2f}")

        if opposition_score > self.max_opposition_score:
            return self._hold_signal(
                symbol, timestamp, price,
                f"Opposition too strong: {opposition_score:.2f}",
            )

        if directional_quality < self.cfg.min_directional_quality:
            return self._hold_signal(
                symbol, timestamp, price,
                f"Low directional quality: {directional_quality:.2f}",
            )

        # Step 8: Calculate position size based on consensus strength
        sc = self.cfg.sizing
        base_size = sc.base_size
        quality_boost = (
            1.0
            + (consensus_score * sc.consensus_weight)
            + (agreement_ratio * sc.agreement_weight)
            + max(0.0, regime_conf - 0.5) * sc.regime_weight
        )
        volatility_scalar = max(
            sc.vol_scalar_min,
            min(sc.vol_scalar_max, sc.vol_target_atr_pct / max(state.atr_pct, 0.0005)),
        )
        size_multiplier = quality_boost * volatility_scalar
        position_size = min(self.max_position_pct, base_size * size_multiplier)

        # Step 8b: Kelly-criterion override (when enough trade history exists)
        if (
            self._kelly_calculator is not None
            and len(self._trade_returns) >= sc.kelly_min_trades
        ):
            kelly_est = self._kelly_calculator.calculate_from_trades(
                np.array(self._trade_returns, dtype=np.float64),
            )
            if kelly_est.position_size > 0 and kelly_est.confidence > 0.3:
                # Blend: use the larger of static and Kelly, capped at max
                position_size = min(
                    self.max_position_pct,
                    max(position_size, kelly_est.position_size),
                )

        # Step 8c: Intraday leverage scaling for high-conviction trades
        if self.cfg.intraday_leverage_mult > 1.0:
            if regime_conf >= self.cfg.leverage_min_regime_confidence:
                leverage = min(self.cfg.intraday_leverage_mult, 2.0)
                position_size = min(
                    self.max_position_pct * leverage,
                    position_size * leverage,
                )

        # Step 9: Calculate stops based on ATR
        atr = state.atr
        if atr <= 0:
            return self._hold_signal(symbol, timestamp, price, "ATR unavailable")
        if potential_direction == "long":
            stop_loss = price - (atr * self.atr_stop_mult)
            take_profit = price + (atr * self.atr_target_mult)
        else:
            stop_loss = price + (atr * self.atr_stop_mult)
            take_profit = price - (atr * self.atr_target_mult)

        # Step 10: Create position
        self.positions[symbol] = {
            "direction": 1 if potential_direction == "long" else -1,
            "entry_price": price,
            "entry_time": timestamp,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "best_price": price,
            "trailing_active": False,
            "regime_at_entry": regime,
            "entry_bar_index": current_bar,
        }

        # Build reason string
        vote_summary = ", ".join([f"{k}:{v.vote.name}" for k, v in votes.items()])
        reason = (
            f"{agreeing_edges}/{n_edges} edges agree, score={consensus_score:.2f}, "
            f"opp={opposition_score:.2f}, q={directional_quality:.2f}. {vote_summary}"
        )

        return OrchestratorSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="buy" if potential_direction == "long" else "short",
            trade_type=trade_type,
            size=position_size,
            confidence=consensus_score,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            edge_votes={k: v.vote for k, v in votes.items()},
            edge_reasons={k: v.reason for k, v in votes.items()},
            consensus_score=consensus_score,
            market_regime=regime,
            reason=reason,
        )

    # ── Exit logic ────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, timestamp: datetime, state: AssetState) -> OrchestratorSignal:
        """Check for exit signals on existing position."""
        position = self.positions[symbol]
        direction = position["direction"]
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        atr = position["atr"]
        best_price = position["best_price"]

        price = state.prices[-1]
        high = state.highs[-1]
        low = state.lows[-1]

        action = "hold"
        reason = ""

        ec = self.cfg.exit

        # Update best price for trailing
        if direction > 0 and price > best_price:
            position["best_price"] = price
            best_price = price
        elif direction < 0 and price < best_price:
            position["best_price"] = price
            best_price = price

        # Check trailing stop activation
        if not position["trailing_active"]:
            profit_distance = (best_price - entry_price) * direction
            if profit_distance >= atr * ec.trailing_activation_atr:
                position["trailing_active"] = True
                # Move stop to breakeven + offset
                if direction > 0:
                    new_stop = entry_price + atr * ec.trailing_breakeven_offset_atr
                    if new_stop > stop_loss:
                        position["stop_loss"] = new_stop
                        stop_loss = new_stop
                else:
                    new_stop = entry_price - atr * ec.trailing_breakeven_offset_atr
                    if new_stop < stop_loss:
                        position["stop_loss"] = new_stop
                        stop_loss = new_stop

        # Update trailing stop
        if position["trailing_active"]:
            if direction > 0:
                trailing_stop = best_price - atr * ec.trailing_distance_atr
                if trailing_stop > stop_loss:
                    position["stop_loss"] = trailing_stop
                    stop_loss = trailing_stop
            else:
                trailing_stop = best_price + atr * ec.trailing_distance_atr
                if trailing_stop < stop_loss:
                    position["stop_loss"] = trailing_stop
                    stop_loss = trailing_stop

        # Check exit conditions
        if direction > 0:  # Long
            if low <= stop_loss:
                action = "sell"
                pnl_pct = (stop_loss - entry_price) / entry_price * 100
                reason = f"Stop hit at ${stop_loss:.2f} | P&L: {pnl_pct:+.2f}%"
            elif high >= take_profit:
                action = "sell"
                pnl_pct = (take_profit - entry_price) / entry_price * 100
                reason = f"Target hit at ${take_profit:.2f} | P&L: {pnl_pct:+.2f}%"
        else:  # Short
            if high >= stop_loss:
                action = "cover"
                pnl_pct = (entry_price - stop_loss) / entry_price * 100
                reason = f"Stop hit at ${stop_loss:.2f} | P&L: {pnl_pct:+.2f}%"
            elif low <= take_profit:
                action = "cover"
                pnl_pct = (entry_price - take_profit) / entry_price * 100
                reason = f"Target hit at ${take_profit:.2f} | P&L: {pnl_pct:+.2f}%"

        # End of day close
        eod = time(ec.eod_exit_time_hour, ec.eod_exit_time_minute)
        if timestamp.time() >= eod:
            action = "sell" if direction > 0 else "cover"
            pnl_pct = (price - entry_price) * direction / entry_price * 100
            reason = f"End of day close | P&L: {pnl_pct:+.2f}%"

        if action in ("sell", "cover"):
            # Record trade return for Kelly-criterion sizing
            exit_price = price  # Use current price as fill estimate
            trade_return = (exit_price - entry_price) * direction / entry_price
            self._trade_returns.append(trade_return)
            # Keep a rolling window to prevent stale data from dominating
            if len(self._trade_returns) > 200:
                self._trade_returns = self._trade_returns[-200:]

            self.last_exit_bar_index[symbol] = len(state.prices)
            del self.positions[symbol]

        return OrchestratorSignal(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            trade_type=TradeType.MOMENTUM_CONTINUATION,
            size=0,
            confidence=0.8 if action != "hold" else 0,
            entry_price=price,
            reason=reason,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _hold_signal(self, symbol: str, timestamp: datetime, price: float, reason: str) -> OrchestratorSignal:
        """Generate a hold signal."""
        return OrchestratorSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="hold",
            trade_type=TradeType.MOMENTUM_CONTINUATION,
            size=0,
            confidence=0,
            entry_price=price,
            reason=reason,
        )

    def clear_positions(self):
        """Clear all positions (for warmup)."""
        self.positions.clear()
        self.last_exit_bar_index.clear()

    @property
    def trade_count(self) -> int:
        """Number of completed trades recorded for Kelly sizing."""
        return len(self._trade_returns)

    @property
    def trade_stats(self) -> Dict[str, float]:
        """Summary statistics of completed trades."""
        if not self._trade_returns:
            return {"count": 0, "win_rate": 0.0, "avg_return": 0.0}
        returns = self._trade_returns
        wins = [r for r in returns if r > 0]
        return {
            "count": len(returns),
            "win_rate": len(wins) / len(returns) if returns else 0.0,
            "avg_return": sum(returns) / len(returns) if returns else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(r for r in returns if r < 0) / max(1, len(returns) - len(wins)),
        }


def create_orchestrator(config: Optional[OrchestratorConfig] = None) -> Orchestrator:
    """Create an Orchestrator instance."""
    return Orchestrator(config)
