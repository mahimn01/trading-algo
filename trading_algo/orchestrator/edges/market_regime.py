"""
Edge 1: Market Regime Engine

Determines what type of day the market is having.
This is CRITICAL - different regimes require different strategies.

Uses SPY as the market proxy, with sector ETF confirmation.
"""

from datetime import datetime
from typing import Optional, Tuple

from ..types import AssetState, MarketRegime


class MarketRegimeEngine:
    """
    Determines what type of day the market is having.

    This is CRITICAL - different regimes require different strategies:
    - Trend days: Trade with trend, don't fade
    - Range days: Fade extremes, mean revert
    - Reversal days: Be patient, wait for confirmation
    - High volatility: Reduce size, widen stops

    Uses SPY as the market proxy, with sector ETF confirmation.
    """

    def __init__(self):
        self.spy_state: Optional[AssetState] = None
        self.qqq_state: Optional[AssetState] = None
        self.iwm_state: Optional[AssetState] = None  # Small caps
        self.vix_proxy: float = 0.0  # We'll estimate from SPY volatility

        # Intraday tracking
        self.morning_direction: Optional[str] = None  # 'up', 'down', 'flat'
        self.midday_direction: Optional[str] = None
        self.trend_bars: int = 0  # Consecutive bars in same direction

    def update(self, symbol: str, state: AssetState):
        """Update with latest asset state."""
        if symbol == "SPY":
            self.spy_state = state
        elif symbol == "QQQ":
            self.qqq_state = state
        elif symbol == "IWM":
            self.iwm_state = state

    def detect_regime(self, current_time: datetime) -> Tuple[MarketRegime, float, str]:
        """
        Detect current market regime.

        Returns: (regime, confidence, reason)
        """
        if self.spy_state is None or len(self.spy_state.prices) < 20:
            return MarketRegime.UNKNOWN, 0.0, "Insufficient data"

        prices = list(self.spy_state.prices)
        highs = list(self.spy_state.highs)
        lows = list(self.spy_state.lows)

        current = prices[-1]

        # Calculate day's metrics
        if self.spy_state.day_open > 0:
            day_return = (current - self.spy_state.day_open) / self.spy_state.day_open
        else:
            day_return = 0

        # Short-term momentum (last 5 bars = 25 min)
        short_return = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # Medium-term momentum (last 20 bars = ~1.5 hours)
        medium_return = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0

        # Trend consistency: How many of last 20 bars closed in same direction?
        if len(prices) >= 20:
            up_bars = sum(1 for i in range(-20, 0) if prices[i] > prices[i-1])
            trend_consistency = abs(up_bars - 10) / 10  # 0 = balanced, 1 = all same direction
            trend_is_up = up_bars > 10
        else:
            trend_consistency = 0
            trend_is_up = day_return > 0

        # Range analysis: Is price making new highs/lows or stuck?
        if len(highs) >= 20:
            recent_high = max(highs[-20:])
            recent_low = min(lows[-20:])
            range_position = (current - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        else:
            range_position = 0.5

        # Volatility (ATR-based)
        volatility = self.spy_state.atr_pct if self.spy_state.atr_pct > 0 else 0.01

        # Breadth proxy: Compare SPY to QQQ and IWM
        breadth_score = 0
        if self.qqq_state and len(self.qqq_state.prices) >= 20:
            qqq_return = (self.qqq_state.prices[-1] - self.qqq_state.prices[-20]) / self.qqq_state.prices[-20]
            if (qqq_return > 0) == (medium_return > 0):
                breadth_score += 1
        if self.iwm_state and len(self.iwm_state.prices) >= 20:
            iwm_return = (self.iwm_state.prices[-1] - self.iwm_state.prices[-20]) / self.iwm_state.prices[-20]
            if (iwm_return > 0) == (medium_return > 0):
                breadth_score += 1

        # When day_return is near zero (e.g. market just opened), use
        # medium_return (last 20 bars spanning prior day) as a proxy for
        # the prevailing trend.  This prevents always falling through to
        # RANGE_BOUND at the open.
        effective_return = day_return
        is_early = abs(day_return) < 0.001
        if is_early and abs(medium_return) > abs(day_return):
            effective_return = medium_return

        # Now classify the regime

        # High volatility supersedes other regimes
        if volatility > 0.02:  # 2% ATR is huge
            return MarketRegime.HIGH_VOLATILITY, 0.8, f"Extreme volatility: ATR={volatility*100:.1f}%"

        # Strong trends
        if effective_return > 0.005 and trend_consistency > 0.6 and trend_is_up:
            conf = min(0.9, 0.5 + trend_consistency * 0.4)
            return MarketRegime.STRONG_TREND_UP, conf, f"Strong uptrend: +{effective_return*100:.1f}%, consistency={trend_consistency:.0%}"

        if effective_return < -0.005 and trend_consistency > 0.6 and not trend_is_up:
            conf = min(0.9, 0.5 + trend_consistency * 0.4)
            return MarketRegime.STRONG_TREND_DOWN, conf, f"Strong downtrend: {effective_return*100:.1f}%, consistency={trend_consistency:.0%}"

        # Moderate trends
        if effective_return > 0.002 and medium_return > 0:
            return MarketRegime.TREND_UP, 0.6, f"Upward bias: +{effective_return*100:.2f}%"

        if effective_return < -0.002 and medium_return < 0:
            return MarketRegime.TREND_DOWN, 0.6, f"Downward bias: {effective_return*100:.2f}%"

        # Reversals
        if effective_return < -0.003 and short_return > 0.001:
            return MarketRegime.REVERSAL_UP, 0.5, f"Potential reversal up from {effective_return*100:.2f}%"

        if effective_return > 0.003 and short_return < -0.001:
            return MarketRegime.REVERSAL_DOWN, 0.5, f"Potential reversal down from +{effective_return*100:.2f}%"

        # Range bound (default)
        return MarketRegime.RANGE_BOUND, 0.7, f"Range-bound: {effective_return*100:+.2f}%, low trend consistency"
