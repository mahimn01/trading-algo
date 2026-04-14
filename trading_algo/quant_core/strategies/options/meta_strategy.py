"""
Adaptive Meta-Strategy — Capstone Options Strategy

Automatically selects the optimal options strategy for the current market
regime and transitions between them as conditions evolve.

Sub-strategies:
    PMCC  — Strong uptrend with leverage via LEAPS diagonal
    WHEEL — Range-bound / mild uptrend premium harvesting
    WHEEL_WIDE — Downtrend wheel with far-OTM puts (0.20 delta)
    HOLD  — Buy stock when premium is too cheap to sell
    CASH  — Capital preservation, no positions

Regime detection uses 5 indicators computed daily:
    1. Trend direction (SMA-50 slope + price vs SMA)
    2. Trend strength (ADX-14)
    3. Volatility regime (realized vol percentile)
    4. IV environment (IV rank)
    5. Momentum (60-day rate of change)

The 5-day stability filter from hybrid_regime.py prevents whipsawing
during regime transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from trading_algo.quant_core.models.greeks import BlackScholesCalculator, OptionSpec
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_rank as compute_iv_rank,
    realized_volatility,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TrendDirection = Literal["STRONG_UP", "MILD_UP", "FLAT", "DOWN", "CRASH"]
VolRegime = Literal["LOW", "NORMAL", "HIGH", "EXTREME"]
IVEnvironment = Literal["RICH", "NORMAL", "CHEAP"]
MomentumBucket = Literal["STRONG_POS", "MOD_POS", "FLAT", "NEGATIVE"]
ActiveStrategy = Literal["PMCC", "WHEEL", "WHEEL_WIDE", "HOLD", "CASH"]


@dataclass
class RegimeSnapshot:
    trend: TrendDirection
    adx: float
    vol_regime: VolRegime
    iv_env: IVEnvironment
    momentum: MomentumBucket
    momentum_raw: float
    sma_slope: float
    rv_pctile: float
    iv_rank_val: float


@dataclass
class TradeEvent:
    date: datetime
    event_type: str
    details: dict = field(default_factory=dict)
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveMetaConfig:
    initial_capital: float = 12_500.0
    # Regime detection
    sma_period: int = 50
    sma_slope_window: int = 20
    adx_period: int = 14
    vol_window: int = 30
    vol_lookback: int = 252
    momentum_window: int = 60
    regime_stability_days: int = 5
    # Wheel strategy configs
    wheel_delta: float = 0.30
    wheel_wide_delta: float = 0.20
    wheel_dte: int = 45
    wheel_call_delta: float = 0.30
    wheel_profit_target: float = 0.50
    wheel_roll_dte: int = 14
    wheel_min_iv_rank: float = 25.0
    wheel_min_premium_pct: float = 0.005
    wheel_cash_reserve_pct: float = 0.20
    # PMCC configs
    pmcc_leaps_delta: float = 0.80
    pmcc_short_delta: float = 0.25
    pmcc_short_dte: int = 21
    pmcc_leaps_dte: int = 270
    pmcc_leaps_roll_dte: int = 90
    pmcc_leaps_max_pct: float = 0.65
    pmcc_short_profit_target: float = 0.50
    pmcc_short_roll_dte: int = 7
    pmcc_short_stop_loss: float = 2.0
    pmcc_min_short_premium_abs: float = 0.10
    # Risk
    max_drawdown_pct: float = 0.25
    risk_free_rate: float = 0.045
    commission_per_contract: float = 0.90
    commission_per_share: float = 0.005
    bid_ask_slip_per_share: float = 0.05
    pmcc_leaps_slip: float = 0.15
    pmcc_short_slip: float = 0.08
    skew_slope: float = 0.8
    dividend_yield: float = 0.0


# ---------------------------------------------------------------------------
# Helpers (BSM pricing + strike finding)
# ---------------------------------------------------------------------------

def _round_strike(raw: float, underlying: float) -> float:
    if underlying < 25:
        return round(raw * 2) / 2
    if underlying < 200:
        return round(raw)
    return round(raw / 5) * 5


def _find_strike(
    spot: float,
    target_abs_delta: float,
    tte_years: float,
    vol: float,
    rate: float,
    option_type: Literal["put", "call"],
    dividend_yield: float = 0.0,
) -> float:
    if tte_years <= 0:
        return spot

    def _err(K: float) -> float:
        spec = OptionSpec(
            spot=spot, strike=K, time_to_expiry=tte_years,
            volatility=vol, risk_free_rate=rate, option_type=option_type,
            dividend_yield=dividend_yield,
        )
        return abs(BlackScholesCalculator.delta(spec)) - target_abs_delta

    if option_type == "put":
        lo, hi = spot * 0.30, spot * 1.00
    else:
        lo, hi = spot * 0.30, spot * 2.00

    for _ in range(5):
        if _err(lo) * _err(hi) < 0:
            break
        lo *= 0.7
        hi *= 1.3
    try:
        raw = brentq(_err, lo, hi, xtol=0.01)
    except ValueError:
        z = norm.ppf(1 - target_abs_delta)
        if option_type == "put":
            raw = spot * np.exp(-z * vol * np.sqrt(tte_years))
        else:
            raw = spot * np.exp(z * vol * np.sqrt(tte_years))
    return _round_strike(raw, spot)


def _price_option(
    spot: float,
    strike: float,
    tte_years: float,
    vol: float,
    rate: float,
    option_type: Literal["put", "call"],
    dividend_yield: float = 0.0,
    skew_slope: float = 0.8,
) -> float:
    if tte_years <= 0:
        if option_type == "put":
            return max(strike - spot, 0.0)
        return max(spot - strike, 0.0)

    adjusted_vol = vol
    if spot > 0:
        if option_type == "put":
            otm_pct = max(0.0, (spot - strike) / spot)
            adjusted_vol = vol * (1.0 + skew_slope * otm_pct)
        else:
            otm_pct = max(0.0, (strike - spot) / spot)
            adjusted_vol = vol * (1.0 - skew_slope * 0.3 * otm_pct)

    spec = OptionSpec(
        spot=spot, strike=strike, time_to_expiry=tte_years,
        volatility=adjusted_vol, risk_free_rate=rate, option_type=option_type,
        dividend_yield=dividend_yield,
    )
    return max(BlackScholesCalculator.price(spec), 0.0)


def _delta_option(
    spot: float, strike: float, tte: float, vol: float, rate: float,
    otype: Literal["put", "call"],
) -> float:
    if tte <= 0:
        if otype == "call":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    spec = OptionSpec(
        spot=spot, strike=strike, time_to_expiry=tte,
        volatility=vol, risk_free_rate=rate, option_type=otype,
    )
    return BlackScholesCalculator.delta(spec)


# ---------------------------------------------------------------------------
# ADX computation (close-only proxy)
# ---------------------------------------------------------------------------

def _compute_adx_from_hlc(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14,
) -> float:
    n = len(closes)
    if n < period * 2 + 1:
        return 0.0

    tr = np.zeros(n - 1)
    plus_dm = np.zeros(n - 1)
    minus_dm = np.zeros(n - 1)

    for i in range(n - 1):
        h_diff = highs[i + 1] - highs[i]
        l_diff = lows[i] - lows[i + 1]
        tr[i] = max(
            highs[i + 1] - lows[i + 1],
            abs(highs[i + 1] - closes[i]),
            abs(lows[i + 1] - closes[i]),
        )
        plus_dm[i] = h_diff if h_diff > l_diff and h_diff > 0 else 0.0
        minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0.0

    atr = np.mean(tr[:period])
    atr_plus = np.mean(plus_dm[:period])
    atr_minus = np.mean(minus_dm[:period])

    dx_values: list[float] = []

    for i in range(period, len(tr)):
        atr = atr - atr / period + tr[i]
        atr_plus = atr_plus - atr_plus / period + plus_dm[i]
        atr_minus = atr_minus - atr_minus / period + minus_dm[i]

        di_plus = (atr_plus / atr * 100) if atr > 0 else 0.0
        di_minus = (atr_minus / atr * 100) if atr > 0 else 0.0
        di_sum = di_plus + di_minus
        dx = abs(di_plus - di_minus) / di_sum * 100 if di_sum > 0 else 0.0
        dx_values.append(dx)

    if len(dx_values) < period:
        return float(np.mean(dx_values)) if dx_values else 0.0

    adx = float(np.mean(dx_values[:period]))
    for i in range(period, len(dx_values)):
        adx = (adx * (period - 1) + dx_values[i]) / period
    return adx


def _compute_adx_close_only(closes: np.ndarray, period: int = 14) -> float:
    """Approximate ADX from close-only data using synthetic H/L."""
    if len(closes) < period * 3:
        return 0.0
    pct = 0.02
    highs = closes * (1 + pct / 2)
    lows = closes * (1 - pct / 2)
    return _compute_adx_from_hlc(highs, lows, closes, period)


# ---------------------------------------------------------------------------
# Internal position types
# ---------------------------------------------------------------------------

@dataclass
class ShortOptionLeg:
    option_type: Literal["put", "call"]
    strike: float
    expiry: datetime
    premium_per_share: float
    contracts: int
    entry_date: datetime
    entry_underlying: float

    def intrinsic(self, underlying: float) -> float:
        if self.option_type == "put":
            return max(self.strike - underlying, 0.0)
        return max(underlying - self.strike, 0.0)

    def dte(self, as_of: datetime) -> int:
        return max((self.expiry - as_of).days, 0)


@dataclass
class LongLeaps:
    strike: float
    expiry: datetime
    premium_per_share: float
    contracts: int
    entry_date: datetime
    entry_underlying: float

    def dte(self, as_of: datetime) -> int:
        return max((self.expiry - as_of).days, 0)


# ---------------------------------------------------------------------------
# Adaptive Meta-Strategy
# ---------------------------------------------------------------------------

class AdaptiveMetaStrategy:
    """
    Selects the optimal options strategy for the current market regime.

    Regime is classified daily using trend, ADX, volatility, IV rank,
    and momentum.  A 5-day stability filter prevents whipsawing.
    Position transitions are handled explicitly to minimize slippage.
    """

    def __init__(self, cfg: AdaptiveMetaConfig | None = None):
        self.cfg = cfg or AdaptiveMetaConfig()
        self.cash: float = self.cfg.initial_capital
        self._last_bar_date: datetime | None = None

        # Positions
        self.stock_qty: int = 0
        self.stock_avg_cost: float = 0.0
        self.short_option: ShortOptionLeg | None = None
        self.leaps: LongLeaps | None = None
        self.short_call: ShortOptionLeg | None = None  # PMCC short call

        # Current state
        self._active_strategy: ActiveStrategy = "CASH"
        self._wheel_phase: Literal["CSP", "CC"] = "CSP"

        # Regime detection state
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._rv_history: list[float] = []  # rolling 30d RV values

        self._current_regime: RegimeSnapshot | None = None
        self._candidate_strategy: ActiveStrategy = "CASH"
        self._candidate_days: int = 0

        # Circuit breaker
        self._peak_equity: float = self.cfg.initial_capital
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_date: datetime | None = None
        self._circuit_breaker_cooldown: int = 30

        # Tracking
        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.regime_history: list[tuple[datetime, ActiveStrategy, RegimeSnapshot | None]] = []
        self._strategy_day_counts: dict[ActiveStrategy, int] = {
            "PMCC": 0, "WHEEL": 0, "WHEEL_WIDE": 0, "HOLD": 0, "CASH": 0,
        }
        self._strategy_pnl: dict[ActiveStrategy, float] = {
            "PMCC": 0.0, "WHEEL": 0.0, "WHEEL_WIDE": 0.0, "HOLD": 0.0, "CASH": 0.0,
        }
        self._prev_equity: float = self.cfg.initial_capital
        self._transition_count: int = 0
        self._transition_cost: float = 0.0

        self.total_premium_collected: float = 0.0
        self.total_premium_paid: float = 0.0
        self.total_commissions: float = 0.0
        self.wins: int = 0
        self.losses: int = 0

    # ── Regime Detection ──────────────────────────────────────────────────

    def _detect_regime(
        self, price: float, iv: float, iv_rank_val: float,
        high: float | None, low: float | None,
    ) -> RegimeSnapshot:
        cfg = self.cfg
        n = len(self._prices)

        # --- Trend Direction ---
        trend: TrendDirection = "FLAT"
        sma_slope = 0.0
        if n >= cfg.sma_period:
            sma = float(np.mean(self._prices[-cfg.sma_period:]))
            pct_from_sma = (price - sma) / sma if sma > 0 else 0.0

            if n >= cfg.sma_period + cfg.sma_slope_window:
                sma_old = float(np.mean(
                    self._prices[-(cfg.sma_period + cfg.sma_slope_window):-cfg.sma_slope_window]
                ))
                sma_slope = (sma - sma_old) / sma_old / cfg.sma_slope_window if sma_old > 0 else 0.0
            else:
                sma_slope = 0.0

            # Check crash first
            if n >= 10:
                ret_10d = (price / self._prices[-10] - 1.0) if self._prices[-10] > 0 else 0.0
                if price < sma and ret_10d < -0.10:
                    trend = "CRASH"

            if trend != "CRASH":
                if price > sma and sma_slope > 0.0015:
                    trend = "STRONG_UP"
                elif price > sma:
                    trend = "MILD_UP"
                elif abs(pct_from_sma) < 0.03:
                    trend = "FLAT"
                elif price < sma and sma_slope < -0.0010:
                    trend = "DOWN"
                else:
                    trend = "FLAT"
        else:
            sma_slope = 0.0

        # --- ADX ---
        adx_val = 0.0
        needed = cfg.adx_period * 3
        if len(self._highs) >= needed and len(self._lows) >= needed:
            adx_val = _compute_adx_from_hlc(
                np.array(self._highs[-needed:]),
                np.array(self._lows[-needed:]),
                np.array(self._prices[-needed:]),
                cfg.adx_period,
            )
        elif n >= needed:
            adx_val = _compute_adx_close_only(np.array(self._prices[-needed:]), cfg.adx_period)

        # --- Volatility Regime ---
        vol_regime: VolRegime = "NORMAL"
        rv_pctile = 50.0
        if n >= cfg.vol_window + 1:
            log_ret = np.diff(np.log(self._prices[-(cfg.vol_window + 1):]))
            current_rv = float(np.std(log_ret, ddof=1) * np.sqrt(252))

            rv_vals = []
            lookback_end = n
            lookback_start = max(0, n - cfg.vol_lookback)
            for j in range(lookback_start + cfg.vol_window, lookback_end):
                seg = np.diff(np.log(self._prices[j - cfg.vol_window:j + 1]))
                rv_vals.append(float(np.std(seg, ddof=1) * np.sqrt(252)))

            if len(rv_vals) >= 20:
                rv_pctile = float(np.sum(np.array(rv_vals) < current_rv) / len(rv_vals) * 100)
            else:
                rv_pctile = 50.0

            if rv_pctile >= 90:
                vol_regime = "EXTREME"
            elif rv_pctile >= 75:
                vol_regime = "HIGH"
            elif rv_pctile <= 25:
                vol_regime = "LOW"
            else:
                vol_regime = "NORMAL"

        # --- IV Environment ---
        iv_env: IVEnvironment
        if iv_rank_val > 50:
            iv_env = "RICH"
        elif iv_rank_val >= 25:
            iv_env = "NORMAL"
        else:
            iv_env = "CHEAP"

        # --- Momentum ---
        momentum_bucket: MomentumBucket = "FLAT"
        momentum_raw = 0.0
        if n >= cfg.momentum_window and self._prices[-cfg.momentum_window] > 0:
            momentum_raw = (price / self._prices[-cfg.momentum_window] - 1.0) * 100
            if momentum_raw > 20:
                momentum_bucket = "STRONG_POS"
            elif momentum_raw > 5:
                momentum_bucket = "MOD_POS"
            elif momentum_raw > -5:
                momentum_bucket = "FLAT"
            else:
                momentum_bucket = "NEGATIVE"

        return RegimeSnapshot(
            trend=trend,
            adx=adx_val,
            vol_regime=vol_regime,
            iv_env=iv_env,
            momentum=momentum_bucket,
            momentum_raw=momentum_raw,
            sma_slope=sma_slope,
            rv_pctile=rv_pctile,
            iv_rank_val=iv_rank_val,
        )

    def _select_strategy(self, regime: RegimeSnapshot) -> ActiveStrategy:
        """Strategy selection matrix based on 5 regime indicators."""
        trend = regime.trend
        adx = regime.adx
        vol = regime.vol_regime
        iv = regime.iv_env
        mom = regime.momentum
        mom_raw = regime.momentum_raw

        # Crash: absolute capital preservation
        if trend == "CRASH":
            return "CASH"

        # Down + high/extreme vol: don't fight the trend
        if trend == "DOWN" and vol in ("HIGH", "EXTREME"):
            return "CASH"

        # Down + normal vol + rich IV: wheel with wide delta
        if trend == "DOWN" and iv == "RICH":
            return "WHEEL_WIDE"

        # Down + anything else: cash
        if trend == "DOWN":
            return "CASH"

        # Strong uptrend with ADX > 25 and positive momentum: PMCC
        if trend == "STRONG_UP" and adx > 25 and mom_raw > 5:
            return "PMCC"

        # Strong uptrend but weak ADX or weak momentum: HOLD
        if trend == "STRONG_UP":
            return "HOLD"

        # Mild up + low/normal vol + rich IV + positive momentum: Wheel
        if trend == "MILD_UP" and vol in ("LOW", "NORMAL") and iv in ("RICH", "NORMAL") and mom_raw > 0:
            return "WHEEL"

        # Mild up + cheap IV: HOLD (premium not worth selling)
        if trend == "MILD_UP" and iv == "CHEAP":
            return "HOLD"

        # Mild up + high vol: cash (too risky)
        if trend == "MILD_UP" and vol in ("HIGH", "EXTREME"):
            return "CASH"

        # Mild up fallthrough
        if trend == "MILD_UP":
            return "WHEEL"

        # Flat + rich IV: Wheel (premium income in range)
        if trend == "FLAT" and iv in ("RICH", "NORMAL"):
            return "WHEEL"

        # Flat + cheap IV: nothing to do
        if trend == "FLAT" and iv == "CHEAP":
            return "CASH"

        # Flat fallthrough
        if trend == "FLAT":
            return "CASH"

        return "CASH"

    def _apply_stability_filter(self, target: ActiveStrategy) -> ActiveStrategy:
        """5-day stability filter: only switch after N consecutive days signaling same strategy."""
        if target == self._active_strategy:
            self._candidate_strategy = target
            self._candidate_days = 0
            return self._active_strategy

        if target == self._candidate_strategy:
            self._candidate_days += 1
        else:
            self._candidate_strategy = target
            self._candidate_days = 1

        if self._candidate_days >= self.cfg.regime_stability_days:
            self._candidate_days = 0
            return target

        return self._active_strategy

    # ── Position Transitions ──────────────────────────────────────────────

    def _close_short_option(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        """Close Wheel short option (put or call)."""
        events: list[TradeEvent] = []
        if self.short_option is None:
            return events

        opt = self.short_option
        dte = opt.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        buyback = _price_option(
            price, opt.strike, tte, iv,
            self.cfg.risk_free_rate, opt.option_type,
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        buyback += self.cfg.bid_ask_slip_per_share
        cost = buyback * opt.contracts * 100
        commission = self.cfg.commission_per_contract * opt.contracts
        self.cash -= cost + commission
        self.total_premium_paid += cost
        self.total_commissions += commission

        pnl = (opt.premium_per_share - buyback) * opt.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        events.append(TradeEvent(
            date=date,
            event_type=f"transition_close_{opt.option_type}",
            details={
                "strike": opt.strike,
                "premium": round(opt.premium_per_share, 4),
                "buyback": round(buyback, 4),
                "underlying": round(price, 2),
            },
            pnl=pnl,
        ))
        self.short_option = None
        return events

    def _close_pmcc_short_call(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        """Close PMCC short call."""
        events: list[TradeEvent] = []
        if self.short_call is None:
            return events

        sc = self.short_call
        dte = sc.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        buyback = _price_option(
            price, sc.strike, tte, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        buyback += self.cfg.pmcc_short_slip
        cost = buyback * sc.contracts * 100
        commission = self.cfg.commission_per_contract * sc.contracts
        self.cash -= cost + commission
        self.total_premium_paid += cost
        self.total_commissions += commission

        pnl = (sc.premium_per_share - buyback) * sc.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        events.append(TradeEvent(
            date=date,
            event_type="transition_close_pmcc_short",
            details={
                "strike": sc.strike,
                "premium": round(sc.premium_per_share, 4),
                "buyback": round(buyback, 4),
            },
            pnl=pnl,
        ))
        self.short_call = None
        return events

    def _close_leaps(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        """Close PMCC LEAPS position."""
        events: list[TradeEvent] = []
        if self.leaps is None:
            return events

        lp = self.leaps
        tte = max(lp.dte(date) / 365.0, 1 / 365)
        close_price = _price_option(
            price, lp.strike, tte, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        close_price -= self.cfg.pmcc_leaps_slip
        close_price = max(close_price, 0.0)
        proceeds = close_price * lp.contracts * 100
        commission = self.cfg.commission_per_contract * lp.contracts
        self.cash += proceeds - commission
        self.total_commissions += commission

        pnl = (close_price - lp.premium_per_share) * lp.contracts * 100

        events.append(TradeEvent(
            date=date,
            event_type="transition_close_leaps",
            details={
                "strike": lp.strike,
                "entry_premium": round(lp.premium_per_share, 4),
                "close_price": round(close_price, 4),
            },
            pnl=pnl,
        ))
        self.leaps = None
        return events

    def _sell_stock(self, date: datetime, price: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        if self.stock_qty <= 0:
            return events

        shares = self.stock_qty
        proceeds = price * shares
        commission = self.cfg.commission_per_share * shares
        slip = self.cfg.bid_ask_slip_per_share * shares
        self.cash += proceeds - commission - slip
        self.total_commissions += commission

        pnl = (price - self.stock_avg_cost) * shares if self.stock_avg_cost > 0 else 0.0
        events.append(TradeEvent(
            date=date,
            event_type="transition_sell_stock",
            details={"shares": shares, "price": round(price, 2), "cost_basis": round(self.stock_avg_cost, 2)},
            pnl=pnl,
        ))
        self.stock_qty = 0
        self.stock_avg_cost = 0.0
        return events

    def _buy_stock(self, date: datetime, price: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        usable = self.cash * (1 - self.cfg.wheel_cash_reserve_pct)
        shares = int(usable / (price + self.cfg.bid_ask_slip_per_share + self.cfg.commission_per_share))
        shares = (shares // 100) * 100
        if shares <= 0:
            return events

        cost = price * shares
        commission = self.cfg.commission_per_share * shares
        slip = self.cfg.bid_ask_slip_per_share * shares
        self.cash -= cost + commission + slip
        self.total_commissions += commission
        self.stock_qty = shares
        self.stock_avg_cost = price

        events.append(TradeEvent(
            date=date,
            event_type="transition_buy_stock",
            details={"shares": shares, "price": round(price, 2)},
        ))
        return events

    def _handle_transition(
        self, date: datetime, price: float, iv: float,
        old_strategy: ActiveStrategy, new_strategy: ActiveStrategy,
    ) -> list[TradeEvent]:
        """Close positions from old strategy and prepare for new one."""
        events: list[TradeEvent] = []
        equity_before = self.get_equity(price, iv, as_of=date)

        events.append(TradeEvent(
            date=date,
            event_type="strategy_transition",
            details={"from": old_strategy, "to": new_strategy, "price": round(price, 2)},
        ))

        # Close old strategy positions
        if old_strategy == "PMCC":
            events.extend(self._close_pmcc_short_call(date, price, iv))
            events.extend(self._close_leaps(date, price, iv))

        elif old_strategy in ("WHEEL", "WHEEL_WIDE"):
            events.extend(self._close_short_option(date, price, iv))
            if self.stock_qty > 0:
                events.extend(self._sell_stock(date, price))
            self._wheel_phase = "CSP"

        elif old_strategy == "HOLD":
            events.extend(self._sell_stock(date, price))

        # Setup for new strategy (actual entry happens in next on_bar cycle)
        if new_strategy in ("WHEEL", "WHEEL_WIDE"):
            self._wheel_phase = "CSP"
        elif new_strategy == "HOLD":
            events.extend(self._buy_stock(date, price))

        equity_after = self.get_equity(price, iv, as_of=date)
        transition_cost = equity_before - equity_after
        self._transition_cost += max(transition_cost, 0.0)
        self._transition_count += 1

        return events

    # ── Wheel Logic ───────────────────────────────────────────────────────

    def _wheel_num_contracts(self, strike: float) -> int:
        if self._wheel_phase == "CSP":
            usable = self.cash * (1 - self.cfg.wheel_cash_reserve_pct)
            return max(int(usable / (strike * 100)), 0)
        return self.stock_qty // 100

    def _wheel_on_bar(
        self, date: datetime, price: float, iv: float, iv_rank_val: float, wide: bool = False,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        put_delta = self.cfg.wheel_wide_delta if wide else self.cfg.wheel_delta

        # Handle expiration
        if self.short_option and date >= self.short_option.expiry:
            events.extend(self._wheel_handle_expiration(date, price, iv))

        # Manage open position
        elif self.short_option:
            events.extend(self._wheel_manage_position(date, price, iv))

        # Open new position
        if self.short_option is None:
            events.extend(self._wheel_open(date, price, iv, iv_rank_val, put_delta))

        return events

    def _wheel_open(
        self, date: datetime, price: float, iv: float, iv_rank_val: float, put_delta: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        if iv_rank_val < self.cfg.wheel_min_iv_rank:
            return events

        tte = self.cfg.wheel_dte / 365.0

        if self._wheel_phase == "CSP":
            # Trend filter
            if len(self._prices) >= self.cfg.sma_period:
                sma = float(np.mean(self._prices[-self.cfg.sma_period:]))
                if price < sma:
                    return events

            strike = _find_strike(
                price, put_delta, tte, iv,
                self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
            )
            contracts = self._wheel_num_contracts(strike)
            if contracts <= 0:
                return events

            premium = _price_option(
                price, strike, tte, iv, self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            premium = max(premium - self.cfg.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.wheel_min_premium_pct:
                return events

            expiry = date + timedelta(days=self.cfg.wheel_dte)
            commission = self.cfg.commission_per_contract * contracts
            self.short_option = ShortOptionLeg(
                option_type="put", strike=strike, expiry=expiry,
                premium_per_share=premium, contracts=contracts,
                entry_date=date, entry_underlying=price,
            )
            self.cash += premium * contracts * 100 - commission
            self.total_premium_collected += premium * contracts * 100
            self.total_commissions += commission

            events.append(TradeEvent(
                date=date, event_type="wheel_sell_put",
                details={
                    "strike": strike, "dte": self.cfg.wheel_dte,
                    "premium": round(premium, 4), "contracts": contracts,
                    "underlying": round(price, 2), "delta": round(put_delta, 2),
                },
            ))

        elif self._wheel_phase == "CC":
            min_strike = self.stock_avg_cost
            strike = _find_strike(
                price, self.cfg.wheel_call_delta, tte, iv,
                self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
            )
            strike = max(strike, _round_strike(min_strike, price))
            contracts = self.stock_qty // 100
            if contracts <= 0:
                return events

            premium = _price_option(
                price, strike, tte, iv, self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            premium = max(premium - self.cfg.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.wheel_min_premium_pct:
                return events

            expiry = date + timedelta(days=self.cfg.wheel_dte)
            commission = self.cfg.commission_per_contract * contracts
            self.short_option = ShortOptionLeg(
                option_type="call", strike=strike, expiry=expiry,
                premium_per_share=premium, contracts=contracts,
                entry_date=date, entry_underlying=price,
            )
            self.cash += premium * contracts * 100 - commission
            self.total_premium_collected += premium * contracts * 100
            self.total_commissions += commission

            events.append(TradeEvent(
                date=date, event_type="wheel_sell_call",
                details={
                    "strike": strike, "dte": self.cfg.wheel_dte,
                    "premium": round(premium, 4), "contracts": contracts,
                    "underlying": round(price, 2),
                },
            ))

        return events

    def _wheel_handle_expiration(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        opt = self.short_option
        if opt is None:
            return events

        intrinsic = opt.intrinsic(price)

        if opt.option_type == "put":
            if intrinsic > 0:
                shares = opt.contracts * 100
                cost = opt.strike * shares
                commission = self.cfg.commission_per_share * shares
                self.cash -= cost + commission
                self.stock_qty += shares
                self.stock_avg_cost = opt.strike - opt.premium_per_share
                self.total_commissions += commission
                self._wheel_phase = "CC"
                events.append(TradeEvent(
                    date=date, event_type="wheel_put_assigned",
                    details={"strike": opt.strike, "shares": shares, "underlying": round(price, 2)},
                ))
            else:
                self.wins += 1
                pnl = opt.premium_per_share * opt.contracts * 100
                events.append(TradeEvent(
                    date=date, event_type="wheel_put_expired_otm",
                    details={"strike": opt.strike, "underlying": round(price, 2)},
                    pnl=pnl,
                ))
        else:
            if intrinsic > 0:
                shares = opt.contracts * 100
                proceeds = opt.strike * shares
                commission = self.cfg.commission_per_share * shares
                self.cash += proceeds - commission
                stock_pnl = (opt.strike - self.stock_avg_cost) * shares
                option_pnl = opt.premium_per_share * opt.contracts * 100
                total_pnl = stock_pnl + option_pnl
                self.stock_qty -= shares
                self.stock_avg_cost = 0.0
                self.total_commissions += commission
                self._wheel_phase = "CSP"
                if total_pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                events.append(TradeEvent(
                    date=date, event_type="wheel_called_away",
                    details={"strike": opt.strike, "shares": shares, "underlying": round(price, 2)},
                    pnl=total_pnl,
                ))
            else:
                self.wins += 1
                pnl = opt.premium_per_share * opt.contracts * 100
                events.append(TradeEvent(
                    date=date, event_type="wheel_call_expired_otm",
                    details={"strike": opt.strike, "underlying": round(price, 2)},
                    pnl=pnl,
                ))

        self.short_option = None
        return events

    def _wheel_manage_position(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        opt = self.short_option
        if opt is None:
            return events

        dte = opt.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        current = _price_option(
            price, opt.strike, tte, iv, self.cfg.risk_free_rate, opt.option_type,
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        pnl_per_share = opt.premium_per_share - current

        # Profit target
        if self.cfg.wheel_profit_target > 0 and opt.premium_per_share > 0:
            if pnl_per_share >= opt.premium_per_share * self.cfg.wheel_profit_target:
                return self._wheel_close_option(date, price, iv, current, "profit_target")

        # Roll
        if self.cfg.wheel_roll_dte > 0 and dte <= self.cfg.wheel_roll_dte:
            return self._wheel_close_option(date, price, iv, current, "roll")

        return events

    def _wheel_close_option(
        self, date: datetime, price: float, iv: float,
        buyback_price: float, reason: str,
    ) -> list[TradeEvent]:
        opt = self.short_option
        if opt is None:
            return []

        buyback_price += self.cfg.bid_ask_slip_per_share
        cost = buyback_price * opt.contracts * 100
        commission = self.cfg.commission_per_contract * opt.contracts
        self.cash -= cost + commission
        self.total_premium_paid += cost
        self.total_commissions += commission

        pnl = (opt.premium_per_share - buyback_price) * opt.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        event = TradeEvent(
            date=date,
            event_type=f"wheel_close_{opt.option_type}_{reason}",
            details={
                "strike": opt.strike,
                "premium": round(opt.premium_per_share, 4),
                "buyback": round(buyback_price, 4),
                "underlying": round(price, 2),
            },
            pnl=pnl,
        )
        self.short_option = None
        return [event]

    # ── PMCC Logic ────────────────────────────────────────────────────────

    def _pmcc_on_bar(
        self, date: datetime, price: float, iv: float, iv_rank_val: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        # Handle short call expiration
        if self.short_call and date >= self.short_call.expiry:
            events.extend(self._pmcc_handle_short_expiry(date, price, iv))

        # Handle LEAPS expiration
        if self.leaps and date >= self.leaps.expiry:
            events.extend(self._pmcc_handle_leaps_expiry(date, price, iv))

        # Roll LEAPS
        if self.leaps and self.cfg.pmcc_leaps_roll_dte > 0 and self.leaps.dte(date) <= self.cfg.pmcc_leaps_roll_dte:
            events.extend(self._pmcc_roll_leaps(date, price, iv))

        # Buy LEAPS if none
        if self.leaps is None:
            events.extend(self._pmcc_buy_leaps(date, price, iv))

        # Manage short call
        if self.short_call and date < self.short_call.expiry:
            events.extend(self._pmcc_manage_short(date, price, iv))

        # Sell short call if idle
        if self.leaps and self.short_call is None:
            events.extend(self._pmcc_sell_short(date, price, iv, iv_rank_val))

        return events

    def _pmcc_buy_leaps(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        # Trend filter
        if len(self._prices) >= self.cfg.sma_period:
            sma = float(np.mean(self._prices[-self.cfg.sma_period:]))
            if price < sma:
                return []

        tte = self.cfg.pmcc_leaps_dte / 365.0
        strike = _find_strike(
            price, self.cfg.pmcc_leaps_delta, tte, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
        )
        premium = _price_option(
            price, strike, tte, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        premium += self.cfg.pmcc_leaps_slip

        cost_per = premium * 100
        contracts = int(self.cash * self.cfg.pmcc_leaps_max_pct / cost_per)
        if contracts <= 0:
            return []

        # Notional exposure cap: delta-equiv <= 2x capital
        leaps_delta = _delta_option(price, strike, tte, iv, self.cfg.risk_free_rate, "call")
        max_notional = 2.0 * self.cfg.initial_capital
        while contracts > 0:
            notional = abs(leaps_delta) * contracts * 100 * price
            if notional <= max_notional:
                break
            contracts -= 1
        if contracts <= 0:
            return []

        total_cost = premium * contracts * 100
        commission = self.cfg.commission_per_contract * contracts
        self.cash -= total_cost + commission
        self.total_commissions += commission

        self.leaps = LongLeaps(
            strike=strike, expiry=date + timedelta(days=self.cfg.pmcc_leaps_dte),
            premium_per_share=premium, contracts=contracts,
            entry_date=date, entry_underlying=price,
        )

        return [TradeEvent(
            date=date, event_type="pmcc_buy_leaps",
            details={
                "strike": strike, "dte": self.cfg.pmcc_leaps_dte,
                "premium": round(premium, 4), "contracts": contracts,
                "underlying": round(price, 2),
            },
        )]

    def _pmcc_roll_leaps(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        old = self.leaps
        if old is None:
            return events

        tte_old = max(old.dte(date) / 365.0, 1 / 365)
        close_price = _price_option(
            price, old.strike, tte_old, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        close_price -= self.cfg.pmcc_leaps_slip
        close_price = max(close_price, 0.0)
        proceeds = close_price * old.contracts * 100
        commission = self.cfg.commission_per_contract * old.contracts

        # Check affordability before closing
        cash_after = self.cash + proceeds - commission
        new_tte = self.cfg.pmcc_leaps_dte / 365.0
        new_strike = _find_strike(
            price, self.cfg.pmcc_leaps_delta, new_tte, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
        )
        new_prem = _price_option(
            price, new_strike, new_tte, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        new_prem += self.cfg.pmcc_leaps_slip
        affordable = int(cash_after * self.cfg.pmcc_leaps_max_pct / (new_prem * 100))

        if affordable <= 0:
            # Can't afford roll; close everything
            self.cash += proceeds - commission
            self.total_commissions += commission
            pnl = (close_price - old.premium_per_share) * old.contracts * 100
            events.append(TradeEvent(
                date=date, event_type="pmcc_close_leaps_unaffordable",
                details={"strike": old.strike, "close_price": round(close_price, 4)},
                pnl=pnl,
            ))
            self.leaps = None
            if self.short_call:
                events.extend(self._close_pmcc_short_call(date, price, iv))
            return events

        # Normal roll
        self.cash += proceeds - commission
        self.total_commissions += commission
        pnl = (close_price - old.premium_per_share) * old.contracts * 100
        events.append(TradeEvent(
            date=date, event_type="pmcc_roll_leaps",
            details={"old_strike": old.strike, "close_price": round(close_price, 4)},
            pnl=pnl,
        ))
        self.leaps = None
        events.extend(self._pmcc_buy_leaps(date, price, iv))
        return events

    def _pmcc_handle_leaps_expiry(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        old = self.leaps
        if old is None:
            return []

        intrinsic = max(price - old.strike, 0.0)
        proceeds = intrinsic * old.contracts * 100
        commission = self.cfg.commission_per_contract * old.contracts
        self.cash += proceeds - commission
        self.total_commissions += commission
        pnl = (intrinsic - old.premium_per_share) * old.contracts * 100
        self.leaps = None
        return [TradeEvent(date=date, event_type="pmcc_leaps_expired", details={"strike": old.strike}, pnl=pnl)]

    def _pmcc_sell_short(
        self, date: datetime, price: float, iv: float, iv_rank_val: float,
    ) -> list[TradeEvent]:
        if self.leaps is None:
            return []
        if iv_rank_val < self.cfg.wheel_min_iv_rank:
            return []

        tte = self.cfg.pmcc_short_dte / 365.0
        strike = _find_strike(
            price, self.cfg.pmcc_short_delta, tte, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
        )
        min_strike = self.leaps.strike + self.leaps.premium_per_share
        strike = max(strike, _round_strike(min_strike, price))

        premium = _price_option(
            price, strike, tte, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        premium = max(premium - self.cfg.pmcc_short_slip, 0.01)

        if premium < self.cfg.pmcc_min_short_premium_abs:
            return []
        if premium / price < self.cfg.wheel_min_premium_pct:
            return []

        contracts = self.leaps.contracts
        commission = self.cfg.commission_per_contract * contracts
        self.cash += premium * contracts * 100 - commission
        self.total_premium_collected += premium * contracts * 100
        self.total_commissions += commission

        self.short_call = ShortOptionLeg(
            option_type="call", strike=strike,
            expiry=date + timedelta(days=self.cfg.pmcc_short_dte),
            premium_per_share=premium, contracts=contracts,
            entry_date=date, entry_underlying=price,
        )

        return [TradeEvent(
            date=date, event_type="pmcc_sell_short",
            details={
                "strike": strike, "dte": self.cfg.pmcc_short_dte,
                "premium": round(premium, 4), "contracts": contracts,
                "underlying": round(price, 2),
            },
        )]

    def _pmcc_handle_short_expiry(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        sc = self.short_call
        if sc is None:
            return []

        intrinsic = sc.intrinsic(price)
        if intrinsic > 0:
            if self.leaps:
                spread_profit = (sc.strike - self.leaps.strike) * sc.contracts * 100
                net_pnl = spread_profit - (self.leaps.premium_per_share * self.leaps.contracts * 100) + (sc.premium_per_share * sc.contracts * 100)
                commission = self.cfg.commission_per_contract * sc.contracts * 2
                self.cash += spread_profit - commission
                self.total_commissions += commission
                self.leaps = None
                self.short_call = None
                if net_pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                return [TradeEvent(
                    date=date, event_type="pmcc_short_assigned_spread",
                    details={"short_strike": sc.strike, "underlying": round(price, 2)},
                    pnl=net_pnl,
                )]
            loss = intrinsic * sc.contracts * 100
            self.cash -= loss
            self.short_call = None
            self.losses += 1
            return [TradeEvent(date=date, event_type="pmcc_short_assigned_naked", pnl=-loss)]

        pnl = sc.premium_per_share * sc.contracts * 100
        self.wins += 1
        self.short_call = None
        return [TradeEvent(
            date=date, event_type="pmcc_short_expired_otm",
            details={"strike": sc.strike, "underlying": round(price, 2)},
            pnl=pnl,
        )]

    def _pmcc_manage_short(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        sc = self.short_call
        if sc is None:
            return []

        dte = sc.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        current = _price_option(
            price, sc.strike, tte, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        pnl_per_share = sc.premium_per_share - current

        # Profit target
        if self.cfg.pmcc_short_profit_target > 0 and sc.premium_per_share > 0:
            if pnl_per_share >= sc.premium_per_share * self.cfg.pmcc_short_profit_target:
                return self._pmcc_close_short(date, price, iv, current, "profit_target")

        # Roll
        if self.cfg.pmcc_short_roll_dte > 0 and dte <= self.cfg.pmcc_short_roll_dte:
            return self._pmcc_close_short(date, price, iv, current, "roll")

        # Stop loss
        if self.cfg.pmcc_short_stop_loss > 0 and sc.premium_per_share > 0:
            if pnl_per_share < -sc.premium_per_share * self.cfg.pmcc_short_stop_loss:
                return self._pmcc_close_short(date, price, iv, current, "stop_loss")

        return []

    def _pmcc_close_short(
        self, date: datetime, price: float, iv: float,
        buyback: float, reason: str,
    ) -> list[TradeEvent]:
        sc = self.short_call
        if sc is None:
            return []

        buyback += self.cfg.pmcc_short_slip
        cost = buyback * sc.contracts * 100
        commission = self.cfg.commission_per_contract * sc.contracts
        self.cash -= cost + commission
        self.total_premium_paid += cost
        self.total_commissions += commission

        pnl = (sc.premium_per_share - buyback) * sc.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        self.short_call = None
        return [TradeEvent(
            date=date, event_type=f"pmcc_close_short_{reason}",
            details={
                "strike": sc.strike,
                "premium": round(sc.premium_per_share, 4),
                "buyback": round(buyback, 4),
                "underlying": round(price, 2),
            },
            pnl=pnl,
        )]

    # ── Public Interface ──────────────────────────────────────────────────

    def on_bar(
        self,
        date: datetime,
        price: float,
        iv: float,
        iv_rank_val: float,
        high: float | None = None,
        low: float | None = None,
    ) -> list[TradeEvent]:
        self._last_bar_date = date
        self._prices.append(price)
        self._highs.append(high if high is not None else price)
        self._lows.append(low if low is not None else price)
        bar_events: list[TradeEvent] = []

        # Circuit breaker cooldown
        if self._circuit_breaker_active:
            if self._circuit_breaker_date is not None:
                days_since = (date - self._circuit_breaker_date).days
                if days_since >= self._circuit_breaker_cooldown:
                    self._circuit_breaker_active = False
                    self._circuit_breaker_date = None
                    self._peak_equity = self.cash
                    bar_events.append(TradeEvent(
                        date=date, event_type="circuit_breaker_cooldown_ended",
                    ))

        # Max drawdown circuit breaker
        if self.cfg.max_drawdown_pct > 0 and not self._circuit_breaker_active:
            current_eq = self.get_equity(price, iv, as_of=date)
            self._peak_equity = max(self._peak_equity, current_eq)
            if self._peak_equity > 0:
                dd = (self._peak_equity - current_eq) / self._peak_equity
                if dd >= self.cfg.max_drawdown_pct:
                    bar_events.extend(self._trip_circuit_breaker(date, price, iv, dd))
                    eq = self.get_equity(price, iv, as_of=date)
                    self.equity_curve.append((date, eq))
                    self._track_daily_pnl(eq)
                    self.events.extend(bar_events)
                    return bar_events

        if self._circuit_breaker_active:
            eq = self.get_equity(price, iv, as_of=date)
            self.equity_curve.append((date, eq))
            self._strategy_day_counts["CASH"] += 1
            self._track_daily_pnl(eq)
            self.regime_history.append((date, "CASH", None))
            self.events.extend(bar_events)
            return bar_events

        # Detect regime
        regime = self._detect_regime(price, iv, iv_rank_val, high, low)
        self._current_regime = regime
        target_strategy = self._select_strategy(regime)
        new_strategy = self._apply_stability_filter(target_strategy)

        # Handle transition
        if new_strategy != self._active_strategy:
            old = self._active_strategy
            bar_events.extend(self._handle_transition(date, price, iv, old, new_strategy))
            self._active_strategy = new_strategy

        # Execute current strategy
        strat = self._active_strategy
        self._strategy_day_counts[strat] += 1

        if strat == "PMCC":
            bar_events.extend(self._pmcc_on_bar(date, price, iv, iv_rank_val))
        elif strat == "WHEEL":
            bar_events.extend(self._wheel_on_bar(date, price, iv, iv_rank_val, wide=False))
        elif strat == "WHEEL_WIDE":
            bar_events.extend(self._wheel_on_bar(date, price, iv, iv_rank_val, wide=True))
        elif strat == "HOLD":
            if self.stock_qty == 0:
                bar_events.extend(self._buy_stock(date, price))

        # Record equity
        eq = self.get_equity(price, iv, as_of=date)
        self.equity_curve.append((date, eq))
        self.regime_history.append((date, strat, regime))

        # Track P&L per strategy
        self._track_daily_pnl(eq)

        # Update peak for circuit breaker
        if self.cfg.max_drawdown_pct > 0:
            self._peak_equity = max(self._peak_equity, eq)

        self.events.extend(bar_events)
        return bar_events

    def _track_daily_pnl(self, equity: float) -> None:
        daily_pnl = equity - self._prev_equity
        strat = self._active_strategy
        if self._circuit_breaker_active:
            strat = "CASH"
        self._strategy_pnl[strat] += daily_pnl
        self._prev_equity = equity

    def _trip_circuit_breaker(
        self, date: datetime, price: float, iv: float, drawdown: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        # Close everything
        if self.short_call:
            events.extend(self._close_pmcc_short_call(date, price, iv))
        if self.leaps:
            events.extend(self._close_leaps(date, price, iv))
        if self.short_option:
            events.extend(self._close_short_option(date, price, iv))
        if self.stock_qty > 0:
            events.extend(self._sell_stock(date, price))

        self._circuit_breaker_active = True
        self._circuit_breaker_date = date
        self._active_strategy = "CASH"

        events.append(TradeEvent(
            date=date, event_type="circuit_breaker_tripped",
            details={
                "drawdown_pct": round(drawdown * 100, 2),
                "peak_equity": round(self._peak_equity, 2),
                "cash": round(self.cash, 2),
            },
        ))
        return events

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
        ref_date = as_of or self._last_bar_date or datetime.now()
        equity = self.cash + self.stock_qty * price

        # Short option (Wheel)
        if self.short_option:
            dte = max((self.short_option.expiry - ref_date).days, 0)
            tte = max(dte / 365.0, 1 / 365)
            mtm = _price_option(
                price, self.short_option.strike, tte, iv,
                self.cfg.risk_free_rate, self.short_option.option_type,
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            equity -= mtm * self.short_option.contracts * 100

        # LEAPS (long)
        if self.leaps:
            dte = max((self.leaps.expiry - ref_date).days, 0)
            tte = max(dte / 365.0, 1 / 365)
            leaps_val = _price_option(
                price, self.leaps.strike, tte, iv,
                self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            equity += leaps_val * self.leaps.contracts * 100

        # Short call (PMCC)
        if self.short_call:
            dte = max((self.short_call.expiry - ref_date).days, 0)
            tte = max(dte / 365.0, 1 / 365)
            short_val = _price_option(
                price, self.short_call.strike, tte, iv,
                self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            equity -= short_val * self.short_call.contracts * 100

        return equity

    def summary(self) -> dict:
        dates = [d for d, _ in self.equity_curve]
        equities = np.array([e for _, e in self.equity_curve])
        total_trades = self.wins + self.losses

        returns = np.diff(equities) / equities[:-1] if len(equities) > 1 else np.array([])
        valid_returns = returns[np.isfinite(returns)]

        daily_rf = self.cfg.risk_free_rate / 252
        sharpe = 0.0
        if len(valid_returns) > 1 and np.std(valid_returns) > 0:
            excess = valid_returns - daily_rf
            sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

        max_dd = 0.0
        if len(equities) > 1:
            peak = np.maximum.accumulate(equities)
            dd = (peak - equities) / peak
            max_dd = float(np.max(dd) * 100)

        total_days = sum(self._strategy_day_counts.values()) or 1
        strategy_pct = {s: round(c / total_days * 100, 1) for s, c in self._strategy_day_counts.items()}

        strategy_pnl_summary = {
            f"{s}_pnl": round(p, 2) for s, p in self._strategy_pnl.items()
        }

        return {
            "initial_capital": self.cfg.initial_capital,
            "final_equity": round(equities[-1], 2) if len(equities) else self.cfg.initial_capital,
            "total_return_pct": round(
                (equities[-1] / self.cfg.initial_capital - 1) * 100, 2
            ) if len(equities) else 0.0,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / total_trades * 100, 1) if total_trades else 0.0,
            "total_premium_collected": round(self.total_premium_collected, 2),
            "total_premium_paid": round(self.total_premium_paid, 2),
            "net_premium": round(self.total_premium_collected - self.total_premium_paid, 2),
            "total_commissions": round(self.total_commissions, 2),
            "active_strategy": self._active_strategy,
            "start_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "end_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "days": len(dates),
            "strategy_pct": strategy_pct,
            "strategy_pnl": strategy_pnl_summary,
            "transition_count": self._transition_count,
            "transition_cost": round(self._transition_cost, 2),
        }
