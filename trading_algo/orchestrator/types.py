"""
Core type definitions for the Orchestrator trading system.

This module contains all enums, dataclasses, and type definitions
used across the edge engines and main orchestrator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Deque, Dict, Optional


class MarketRegime(Enum):
    """What type of day is the market having?"""
    STRONG_TREND_UP = auto()     # Clear uptrend, buy dips
    TREND_UP = auto()            # Upward bias, be long
    RANGE_BOUND = auto()         # Chop, fade extremes
    TREND_DOWN = auto()          # Downward bias, be short
    STRONG_TREND_DOWN = auto()   # Clear downtrend, sell rallies
    REVERSAL_UP = auto()         # Was down, now reversing up
    REVERSAL_DOWN = auto()       # Was up, now reversing down
    HIGH_VOLATILITY = auto()     # Extreme moves, reduce size
    UNKNOWN = auto()             # Not enough data


class EdgeVote(Enum):
    """Vote from each edge source."""
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2
    VETO_LONG = -99    # Blocks long trades
    VETO_SHORT = 99    # Blocks short trades


class TradeType(Enum):
    """What kind of trade setup is this?"""
    MOMENTUM_CONTINUATION = auto()  # Ride existing trend
    MEAN_REVERSION = auto()          # Fade to the mean
    BREAKOUT = auto()                # New range expansion
    RELATIVE_VALUE = auto()          # Pairs/spread trade
    OPENING_DRIVE = auto()           # First 30 min momentum
    REVERSAL = auto()                # Trend change


@dataclass
class EdgeSignal:
    """Signal from a single edge source."""
    edge_name: str
    vote: EdgeVote
    confidence: float  # 0 to 1
    reason: str
    data: Dict = field(default_factory=dict)


@dataclass
class OrchestratorSignal:
    """Final signal from the ensemble."""
    symbol: str
    timestamp: datetime
    action: str  # 'buy', 'short', 'sell', 'cover', 'hold'
    trade_type: TradeType
    size: float  # Position size as fraction
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Edge breakdown
    edge_votes: Dict[str, EdgeVote] = field(default_factory=dict)
    edge_reasons: Dict[str, str] = field(default_factory=dict)
    consensus_score: float = 0.0  # -2 to +2

    # Context
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    relative_strength_rank: float = 0.0  # 0-100 percentile
    statistical_zscore: float = 0.0

    reason: str = ""


@dataclass
class AssetState:
    """Current state of a single asset."""
    symbol: str
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    highs: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    lows: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=500))

    # Derived metrics (updated on each bar)
    vwap: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    rsi: float = 50.0

    # Volume profile
    volume_by_price: Dict[float, float] = field(default_factory=dict)
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    point_of_control: float = 0.0

    # Statistical metrics
    price_zscore: float = 0.0
    volume_zscore: float = 0.0
    momentum_zscore: float = 0.0

    # Today's metrics
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    day_volume: float = 0.0

    # Running sums for incremental VWAP (O(1) updates)
    _cum_tp_vol: float = 0.0
    _cum_vol: float = 0.0

    # Running ATR state (Wilder smoothing)
    _atr_count: int = 0

    # Running RSI state (Wilder smoothing)
    _rsi_avg_gain: float = 0.0
    _rsi_avg_loss: float = 0.0
    _rsi_count: int = 0
