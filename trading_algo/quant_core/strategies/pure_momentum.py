"""
Pure Momentum Strategy

A simple, aggressive momentum strategy designed for high returns.
No mean reversion dampening, no conservative risk limits.

This is what's needed for 25-50% annual returns.

WARNING: This strategy can have 40-60% drawdowns.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum, auto


class TrendState(Enum):
    """Trend classification."""
    STRONG_UP = auto()
    UP = auto()
    NEUTRAL = auto()
    DOWN = auto()
    STRONG_DOWN = auto()


@dataclass
class MomentumSignal:
    """Momentum signal for a single asset."""
    symbol: str
    trend: TrendState
    momentum_score: float      # -1 to +1
    position_size: float       # Target position (0 to 1)
    entry_price: Optional[float] = None


@dataclass
class MomentumConfig:
    """Configuration for pure momentum strategy."""
    # Lookback periods
    fast_ma: int = 20          # Fast moving average
    slow_ma: int = 50          # Slow moving average
    trend_ma: int = 200        # Trend filter
    momentum_lookback: int = 60  # Momentum calculation period

    # Position sizing
    max_position: float = 0.30     # Max 30% per position
    min_position: float = 0.05     # Min 5% per position
    target_exposure: float = 1.5   # 150% target gross exposure

    # Entry/Exit
    trend_filter: bool = True      # Only long above 200 MA
    momentum_threshold: float = 0.0  # Min momentum to enter

    # Short selling
    allow_short: bool = False      # Enable short signals for bear markets
    short_momentum_threshold: float = -0.10  # Min negative momentum to short
    short_max_position: float = 0.15  # Max 15% per short position

    # Volatility scaling
    vol_scale: bool = True
    target_vol: float = 0.20       # 20% target volatility
    vol_lookback: int = 20


class PureMomentumStrategy:
    """
    Pure momentum strategy for aggressive returns.

    Logic:
    1. Calculate momentum score for each asset
    2. Filter by trend (price > 200 MA)
    3. Size positions by momentum strength
    4. Scale by inverse volatility

    No mean reversion. No risk dampening. Just momentum.
    """

    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()
        self._price_history: Dict[str, List[float]] = {}

    def update(self, symbol: str, price: float) -> None:
        """Update price history for a symbol."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)

        # Keep limited history
        max_len = max(
            self.config.trend_ma,
            self.config.momentum_lookback,
            self.config.vol_lookback
        ) + 10
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

    def generate_signals(
        self,
        symbols: List[str],
        prices: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, MomentumSignal]:
        """
        Generate momentum signals for all symbols.

        Args:
            symbols: List of symbols to analyze
            prices: Dict of symbol -> price array (close prices)

        Returns:
            Dict of symbol -> MomentumSignal
        """
        signals = {}
        momentum_scores = {}
        volatilities = {}

        for symbol in symbols:
            if symbol not in prices:
                continue

            price_array = prices[symbol]
            if len(price_array) < self.config.trend_ma:
                continue

            # Calculate indicators
            current_price = price_array[-1]

            # Moving averages
            fast_ma = np.mean(price_array[-self.config.fast_ma:])
            slow_ma = np.mean(price_array[-self.config.slow_ma:])
            trend_ma = np.mean(price_array[-self.config.trend_ma:])

            # Momentum (rate of change)
            lookback = min(self.config.momentum_lookback, len(price_array) - 1)
            momentum = (current_price / price_array[-lookback - 1]) - 1

            # Volatility
            returns = np.diff(price_array[-self.config.vol_lookback - 1:]) / price_array[-self.config.vol_lookback - 1:-1]
            volatility = np.std(returns) * np.sqrt(252)
            volatilities[symbol] = volatility

            # Determine trend state
            if current_price > trend_ma * 1.05 and fast_ma > slow_ma:
                trend = TrendState.STRONG_UP
            elif current_price > trend_ma and fast_ma > slow_ma:
                trend = TrendState.UP
            elif current_price < trend_ma * 0.95 and fast_ma < slow_ma:
                trend = TrendState.STRONG_DOWN
            elif current_price < trend_ma and fast_ma < slow_ma:
                trend = TrendState.DOWN
            else:
                trend = TrendState.NEUTRAL

            # Momentum score (-1 to +1)
            # Normalize momentum to approximate score
            momentum_score = np.clip(momentum * 5, -1, 1)  # Scale factor
            momentum_scores[symbol] = momentum_score

            # Apply trend filter
            if self.config.trend_filter and current_price < trend_ma:
                momentum_score = min(momentum_score, 0)  # No longs below trend

            signals[symbol] = MomentumSignal(
                symbol=symbol,
                trend=trend,
                momentum_score=momentum_score,
                position_size=0.0,  # Calculated below
                entry_price=current_price if momentum_score != 0 else None,
            )

        # --- Long allocation ---
        ranked = sorted(
            [(s, sig.momentum_score) for s, sig in signals.items() if sig.momentum_score > self.config.momentum_threshold],
            key=lambda x: x[1],
            reverse=True
        )

        total_score = sum(max(0, score) for _, score in ranked)

        if total_score > 0:
            for symbol, score in ranked:
                if score <= 0:
                    continue

                # Base position from momentum score
                base_position = (score / total_score) * self.config.target_exposure

                # Volatility scaling (inverse vol)
                if self.config.vol_scale and symbol in volatilities:
                    vol = volatilities[symbol]
                    if vol > 0:
                        vol_scalar = self.config.target_vol / vol
                        base_position *= min(vol_scalar, 2.0)  # Cap at 2x

                # Apply position limits
                position = np.clip(
                    base_position,
                    self.config.min_position,
                    self.config.max_position
                )

                signals[symbol].position_size = position

        # --- Short allocation (bear market alpha) ---
        if self.config.allow_short:
            short_candidates = sorted(
                [
                    (s, sig.momentum_score)
                    for s, sig in signals.items()
                    if sig.momentum_score < self.config.short_momentum_threshold
                    and sig.trend in (TrendState.DOWN, TrendState.STRONG_DOWN)
                ],
                key=lambda x: x[1],  # Most negative first
            )

            short_exposure = self.config.target_exposure * 0.3
            total_neg_score = sum(abs(score) for _, score in short_candidates)

            if total_neg_score > 0:
                for symbol, score in short_candidates:
                    base_short = (abs(score) / total_neg_score) * short_exposure

                    # Volatility scaling (inverse vol)
                    if self.config.vol_scale and symbol in volatilities:
                        vol = volatilities[symbol]
                        if vol > 0:
                            vol_scalar = self.config.target_vol / vol
                            base_short *= min(vol_scalar, 2.0)

                    # Cap at short_max_position
                    position = min(base_short, self.config.short_max_position)
                    position = max(position, self.config.min_position)

                    # Negative position_size indicates short
                    signals[symbol].position_size = -position

        return signals

    def get_target_weights(
        self,
        symbols: List[str],
        prices: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, float]:
        """
        Get target portfolio weights.

        Returns:
            Dict of symbol -> target weight (positive for long, negative for short)
        """
        signals = self.generate_signals(symbols, prices)

        weights = {}
        total_long = 0.0
        total_short = 0.0

        for symbol, signal in signals.items():
            if signal.position_size > 0:
                weights[symbol] = signal.position_size
                total_long += signal.position_size
            elif signal.position_size < 0:
                weights[symbol] = signal.position_size  # negative
                total_short += abs(signal.position_size)

        # Normalize longs if over target exposure
        if total_long > self.config.target_exposure:
            scale = self.config.target_exposure / total_long
            weights = {
                s: w * scale if w > 0 else w
                for s, w in weights.items()
            }

        # Normalize shorts if over half target exposure
        short_limit = self.config.target_exposure * 0.5
        if total_short > short_limit:
            scale = short_limit / total_short
            weights = {
                s: w * scale if w < 0 else w
                for s, w in weights.items()
            }

        return weights

    def reset(self) -> None:
        """Reset strategy state."""
        self._price_history.clear()


def run_pure_momentum_backtest(
    historical_data: Dict[str, NDArray[np.float64]],
    timestamps: List[datetime],
    initial_capital: float = 100000.0,
    config: Optional[MomentumConfig] = None,
) -> Dict:
    """
    Run pure momentum backtest.

    Args:
        historical_data: Dict of symbol -> OHLCV array (T, 5)
        timestamps: List of timestamps
        initial_capital: Starting capital
        config: Strategy configuration

    Returns:
        Dict with backtest results
    """
    config = config or MomentumConfig()
    strategy = PureMomentumStrategy(config)

    symbols = list(historical_data.keys())
    n_bars = len(timestamps)

    # Extract close prices
    close_prices = {s: historical_data[s][:, 3] for s in symbols}

    # State
    cash = initial_capital
    positions: Dict[str, float] = {}  # symbol -> shares
    equity_curve = [initial_capital]
    trades = []

    warmup = config.trend_ma + 10

    for t in range(warmup, n_bars):
        # Current prices
        current_prices = {s: close_prices[s][t] for s in symbols}

        # Get price history up to this point
        price_history = {s: close_prices[s][:t+1] for s in symbols}

        # Calculate current equity
        position_value = sum(
            positions.get(s, 0) * current_prices[s]
            for s in symbols
        )
        equity = cash + position_value

        # Get target weights
        target_weights = strategy.get_target_weights(symbols, price_history)

        # Rebalance
        target_values = {s: equity * w for s, w in target_weights.items()}

        for symbol in symbols:
            current_shares = positions.get(symbol, 0)
            current_value = current_shares * current_prices[symbol]
            target_value = target_values.get(symbol, 0)

            delta_value = target_value - current_value

            if abs(delta_value) > 100:  # Min trade size
                delta_shares = delta_value / current_prices[symbol]

                # Execute trade (with 5bp slippage)
                slippage = 0.0005
                if delta_shares > 0:
                    exec_price = current_prices[symbol] * (1 + slippage)
                else:
                    exec_price = current_prices[symbol] * (1 - slippage)

                cost = abs(delta_shares) * exec_price
                commission = cost * 0.001  # 10bp commission

                if delta_shares > 0:
                    cash -= cost + commission
                else:
                    cash += cost - commission

                positions[symbol] = current_shares + delta_shares

                trades.append({
                    'timestamp': timestamps[t],
                    'symbol': symbol,
                    'shares': delta_shares,
                    'price': exec_price,
                })

        # Update equity curve
        position_value = sum(
            positions.get(s, 0) * current_prices[s]
            for s in symbols
        )
        equity = cash + position_value
        equity_curve.append(equity)

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = (equity_curve[-1] / initial_capital) - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    volatility = np.std(returns) * np.sqrt(252)
    sharpe = (ann_return - 0.02) / volatility if volatility > 0 else 0  # Rf = 2%

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'total_trades': len(trades),
        'equity_curve': equity_curve,
        'returns': returns,
        'trades': trades,
        'final_value': equity_curve[-1],
    }
