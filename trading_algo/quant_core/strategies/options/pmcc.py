"""
Poor Man's Covered Call  (PMCC / Diagonal Debit Spread)

Capital-efficient covered-call substitute using LEAPS.

Structure:
    Long leg :  Deep ITM LEAPS call  (delta >= 0.80, 6-12 months out)
    Short leg:  OTM call             (delta 0.20-0.35, 2-4 weeks out)

Honesty notes:
    - LEAPS use max 50% of capital (margin buffer).
    - Fixed dollar slippage, not percentage (wider for LEAPS).
    - Exchange fees included.
    - Short call stop-loss enforced.
    - get_equity uses simulation date, not wall-clock.
    - Max drawdown circuit breaker with cooldown.
    - Dual trend filter (SMA-50 + SMA-200) on LEAPS entry.
    - Max notional exposure cap.
    - Min short premium filter ($0.10/share).
    - Graceful LEAPS roll (affordability check).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from trading_algo.quant_core.models.greeks import BlackScholesCalculator, OptionSpec


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PMCCConfig:
    initial_capital: float = 5_000.0
    leaps_delta: float = 0.80
    leaps_dte: int = 270
    leaps_roll_dte: int = 90
    leaps_max_capital_pct: float = 0.50
    short_delta: float = 0.25
    short_dte: int = 28
    short_profit_target: float = 0.50  # 0 = disabled (let expire)
    short_roll_dte: int = 7  # 0 = let expire (no early roll)
    short_stop_loss: float = 2.0
    min_iv_rank: float = 25.0
    min_short_premium_pct: float = 0.008
    min_short_premium_abs: float = 0.10  # min $0.10/share to bother selling
    trend_sma_period: int = 50  # 0 = disabled
    risk_free_rate: float = 0.045
    commission_per_contract: float = 0.90
    leaps_slip_per_share: float = 0.15
    short_slip_per_share: float = 0.08
    # Circuit breaker
    max_drawdown_pct: float = 0.40  # 0 = disabled
    cooldown_days: int = 30
    # Notional exposure cap
    max_notional_exposure: float = 2.0  # LEAPS delta-equiv notional <= 2x capital


# ---------------------------------------------------------------------------
# Internal position types
# ---------------------------------------------------------------------------

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


@dataclass
class ShortCall:
    strike: float
    expiry: datetime
    premium_per_share: float
    contracts: int
    entry_date: datetime
    entry_underlying: float

    def intrinsic(self, underlying: float) -> float:
        return max(underlying - self.strike, 0.0)

    def dte(self, as_of: datetime) -> int:
        return max((self.expiry - as_of).days, 0)


@dataclass
class TradeEvent:
    date: datetime
    event_type: str
    details: dict = field(default_factory=dict)
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
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
) -> float:
    if tte_years <= 0:
        return spot

    def _err(K: float) -> float:
        spec = OptionSpec(
            spot=spot, strike=K, time_to_expiry=tte_years,
            volatility=vol, risk_free_rate=rate, option_type=option_type,
        )
        return abs(BlackScholesCalculator.delta(spec)) - target_abs_delta

    lo = spot * 0.30 if option_type == "call" else spot * 0.50
    hi = spot * 2.00 if option_type == "call" else spot * 1.00

    for _ in range(5):
        if _err(lo) * _err(hi) < 0:
            break
        lo *= 0.7
        hi *= 1.3
    try:
        raw = brentq(_err, lo, hi, xtol=0.01)
    except ValueError:
        z = norm.ppf(1 - target_abs_delta)
        raw = spot * np.exp(z * vol * np.sqrt(tte_years))
    return _round_strike(raw, spot)


def _price_opt(
    spot: float, strike: float, tte: float, vol: float, rate: float,
    otype: Literal["put", "call"],
) -> float:
    if tte <= 0:
        return max(spot - strike, 0.0) if otype == "call" else max(strike - spot, 0.0)
    spec = OptionSpec(
        spot=spot, strike=strike, time_to_expiry=tte,
        volatility=vol, risk_free_rate=rate, option_type=otype,
    )
    return max(BlackScholesCalculator.price(spec), 0.0)


def _delta_opt(
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


def _sma(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))


# ---------------------------------------------------------------------------
# PMCC Strategy
# ---------------------------------------------------------------------------

class PMCCStrategy:

    def __init__(self, cfg: PMCCConfig | None = None):
        self.cfg = cfg or PMCCConfig()
        self.cash: float = self.cfg.initial_capital
        self.leaps: LongLeaps | None = None
        self.short_call: ShortCall | None = None
        self._last_bar_date: datetime | None = None

        # Tracking
        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.price_history: list[float] = []
        self.total_leaps_cost: float = 0.0
        self.total_short_premium: float = 0.0
        self.total_commissions: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.leaps_rolls: int = 0
        self.short_cycles: int = 0

        # Circuit breaker state
        self._peak_equity: float = self.cfg.initial_capital
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_date: datetime | None = None

    # ---- public interface --------------------------------------------------

    def on_bar(
        self,
        date: datetime,
        price: float,
        iv: float,
        iv_rank: float,
    ) -> list[TradeEvent]:
        self._last_bar_date = date
        bar_events: list[TradeEvent] = []
        self.price_history.append(price)

        # Check circuit breaker cooldown
        if self._circuit_breaker_active:
            if self._circuit_breaker_date is not None:
                days_since = (date - self._circuit_breaker_date).days
                if days_since >= self.cfg.cooldown_days:
                    self._circuit_breaker_active = False
                    self._circuit_breaker_date = None
                    self._peak_equity = self.cash  # reset peak to current cash
                    bar_events.append(TradeEvent(
                        date=date, event_type="circuit_breaker_cooldown_ended",
                        details={"days_cooled": days_since, "cash": round(self.cash, 2)},
                    ))

        # Check max drawdown circuit breaker (if enabled and not already active)
        if self.cfg.max_drawdown_pct > 0 and not self._circuit_breaker_active:
            current_eq = self.get_equity(price, iv, as_of=date)
            self._peak_equity = max(self._peak_equity, current_eq)
            if self._peak_equity > 0:
                drawdown = (self._peak_equity - current_eq) / self._peak_equity
                if drawdown >= self.cfg.max_drawdown_pct:
                    bar_events.extend(self._trip_circuit_breaker(date, price, iv, drawdown))
                    eq = self.get_equity(price, iv, as_of=date)
                    self.equity_curve.append((date, eq))
                    self.events.extend(bar_events)
                    return bar_events

        # If circuit breaker is active, just track equity and skip trading
        if self._circuit_breaker_active:
            eq = self.get_equity(price, iv, as_of=date)
            self.equity_curve.append((date, eq))
            self.events.extend(bar_events)
            return bar_events

        # 1. Handle short call expiration
        if self.short_call and date >= self.short_call.expiry:
            bar_events.extend(self._handle_short_expiry(date, price, iv))

        # 2. Handle LEAPS expiration
        if self.leaps and date >= self.leaps.expiry:
            bar_events.extend(self._handle_leaps_expiry(date, price, iv))

        # 3. Roll LEAPS if approaching expiry
        if self.leaps and self.cfg.leaps_roll_dte > 0 and self.leaps.dte(date) <= self.cfg.leaps_roll_dte:
            bar_events.extend(self._roll_leaps(date, price, iv))

        # 4. Buy LEAPS if none
        if self.leaps is None:
            bar_events.extend(self._buy_leaps(date, price, iv))

        # 5. Manage short call
        if self.short_call and date < self.short_call.expiry:
            bar_events.extend(self._manage_short_call(date, price, iv))

        # 6. Sell short call if idle
        if self.leaps and self.short_call is None:
            bar_events.extend(self._sell_short_call(date, price, iv, iv_rank))

        eq = self.get_equity(price, iv, as_of=date)
        self.equity_curve.append((date, eq))

        # Update peak equity for circuit breaker tracking
        if self.cfg.max_drawdown_pct > 0:
            self._peak_equity = max(self._peak_equity, eq)

        self.events.extend(bar_events)
        return bar_events

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
        ref_date = as_of or self._last_bar_date or datetime.now()
        equity = self.cash
        if self.leaps:
            dte_days = max((self.leaps.expiry - ref_date).days, 0)
            tte = max(dte_days / 365.0, 1 / 365)
            leaps_val = _price_opt(price, self.leaps.strike, tte, iv, self.cfg.risk_free_rate, "call")
            equity += leaps_val * self.leaps.contracts * 100
        if self.short_call:
            dte_days = max((self.short_call.expiry - ref_date).days, 0)
            tte = max(dte_days / 365.0, 1 / 365)
            short_val = _price_opt(price, self.short_call.strike, tte, iv, self.cfg.risk_free_rate, "call")
            equity -= short_val * self.short_call.contracts * 100
        return equity

    # ---- circuit breaker ---------------------------------------------------

    def _trip_circuit_breaker(
        self, date: datetime, price: float, iv: float, drawdown: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        # Close short call first
        if self.short_call:
            dte = self.short_call.dte(date)
            tte = max(dte / 365.0, 1 / 365)
            current = _price_opt(price, self.short_call.strike, tte, iv, self.cfg.risk_free_rate, "call")
            events.extend(self._close_short_call(date, price, iv, current, "circuit_breaker"))

        # Close LEAPS
        if self.leaps:
            tte_old = max(self.leaps.dte(date) / 365.0, 1 / 365)
            close_price = _price_opt(price, self.leaps.strike, tte_old, iv, self.cfg.risk_free_rate, "call")
            close_price -= self.cfg.leaps_slip_per_share
            close_price = max(close_price, 0.0)
            proceeds = close_price * self.leaps.contracts * 100
            commission = self.cfg.commission_per_contract * self.leaps.contracts
            pnl = (close_price - self.leaps.premium_per_share) * self.leaps.contracts * 100
            self.cash += proceeds - commission
            self.total_commissions += commission
            events.append(TradeEvent(
                date=date, event_type="close_leaps_circuit_breaker",
                details={
                    "strike": self.leaps.strike,
                    "close_price": round(close_price, 4),
                    "drawdown_pct": round(drawdown * 100, 2),
                },
                pnl=pnl,
            ))
            self.leaps = None

        self._circuit_breaker_active = True
        self._circuit_breaker_date = date

        events.append(TradeEvent(
            date=date, event_type="circuit_breaker_tripped",
            details={
                "drawdown_pct": round(drawdown * 100, 2),
                "peak_equity": round(self._peak_equity, 2),
                "cash_after": round(self.cash, 2),
                "cooldown_days": self.cfg.cooldown_days,
            },
        ))
        return events

    # ---- trend filters -----------------------------------------------------

    def _passes_trend_filter(self, price: float) -> bool:
        # Primary trend filter (configurable SMA period)
        if self.cfg.trend_sma_period > 0:
            sma_primary = _sma(self.price_history, self.cfg.trend_sma_period)
            if sma_primary is not None and price < sma_primary:
                return False

        # Secondary trend filter: 200-SMA (only if enough history)
        sma_200 = _sma(self.price_history, 200)
        if sma_200 is not None and price < sma_200:
            return False

        return True

    # ---- notional exposure check -------------------------------------------

    def _check_notional_exposure(
        self, price: float, iv: float, contracts: int, strike: float, tte: float,
    ) -> int:
        if self.cfg.max_notional_exposure <= 0:
            return contracts
        leaps_delta = _delta_opt(price, strike, tte, iv, self.cfg.risk_free_rate, "call")
        max_notional = self.cfg.max_notional_exposure * self.cfg.initial_capital
        # delta-equivalent notional = delta * contracts * 100 * price
        while contracts > 0:
            notional = abs(leaps_delta) * contracts * 100 * price
            if notional <= max_notional:
                break
            contracts -= 1
        return contracts

    # ---- LEAPS management --------------------------------------------------

    def _buy_leaps(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        # Trend filter applies to LEAPS entry
        if not self._passes_trend_filter(price):
            return []

        tte = self.cfg.leaps_dte / 365.0
        strike = _find_strike(price, self.cfg.leaps_delta, tte, iv, self.cfg.risk_free_rate, "call")
        premium = _price_opt(price, strike, tte, iv, self.cfg.risk_free_rate, "call")
        premium += self.cfg.leaps_slip_per_share

        cost_per = premium * 100
        contracts = int(self.cash * self.cfg.leaps_max_capital_pct / cost_per)
        if contracts <= 0:
            return []

        # Notional exposure cap
        contracts = self._check_notional_exposure(price, iv, contracts, strike, tte)
        if contracts <= 0:
            return []

        total_cost = premium * contracts * 100
        commission = self.cfg.commission_per_contract * contracts
        self.cash -= total_cost + commission
        self.total_leaps_cost += total_cost
        self.total_commissions += commission

        self.leaps = LongLeaps(
            strike=strike, expiry=date + timedelta(days=self.cfg.leaps_dte),
            premium_per_share=premium, contracts=contracts,
            entry_date=date, entry_underlying=price,
        )

        return [TradeEvent(
            date=date, event_type="buy_leaps",
            details={
                "strike": strike, "dte": self.cfg.leaps_dte,
                "premium": round(premium, 4), "contracts": contracts,
                "debit": round(total_cost, 2), "underlying": round(price, 2),
            },
        )]

    def _roll_leaps(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        old = self.leaps
        assert old is not None

        tte_old = max(old.dte(date) / 365.0, 1 / 365)
        close_price = _price_opt(price, old.strike, tte_old, iv, self.cfg.risk_free_rate, "call")
        close_price -= self.cfg.leaps_slip_per_share
        close_price = max(close_price, 0.0)
        proceeds = close_price * old.contracts * 100
        commission = self.cfg.commission_per_contract * old.contracts
        cash_after_close = self.cash + proceeds - commission

        # Check if we can afford a new LEAPS before closing old one
        new_tte = self.cfg.leaps_dte / 365.0
        new_strike = _find_strike(price, self.cfg.leaps_delta, new_tte, iv, self.cfg.risk_free_rate, "call")
        new_premium = _price_opt(price, new_strike, new_tte, iv, self.cfg.risk_free_rate, "call")
        new_premium += self.cfg.leaps_slip_per_share
        new_cost_per = new_premium * 100
        affordable_contracts = int(cash_after_close * self.cfg.leaps_max_capital_pct / new_cost_per)

        if affordable_contracts <= 0:
            # Can't afford new LEAPS — close everything gracefully
            self.cash += proceeds - commission
            self.total_commissions += commission
            old_pnl = (close_price - old.premium_per_share) * old.contracts * 100
            events.append(TradeEvent(
                date=date, event_type="close_leaps_unaffordable_roll",
                details={
                    "old_strike": old.strike,
                    "close_price": round(close_price, 4),
                    "pnl": round(old_pnl, 2),
                    "cash_available": round(cash_after_close, 2),
                    "new_cost_per_contract": round(new_cost_per, 2),
                },
                pnl=old_pnl,
            ))
            self.leaps = None

            # Also close short call if open
            if self.short_call:
                dte = self.short_call.dte(date)
                tte = max(dte / 365.0, 1 / 365)
                current = _price_opt(price, self.short_call.strike, tte, iv, self.cfg.risk_free_rate, "call")
                events.extend(self._close_short_call(date, price, iv, current, "unaffordable_roll"))

            return events

        # Proceed with normal roll
        self.cash += proceeds - commission
        self.total_commissions += commission

        old_pnl = (close_price - old.premium_per_share) * old.contracts * 100
        events.append(TradeEvent(
            date=date, event_type="close_leaps_roll",
            details={"old_strike": old.strike, "close_price": round(close_price, 4), "pnl": round(old_pnl, 2)},
            pnl=old_pnl,
        ))

        self.leaps = None
        self.leaps_rolls += 1
        events.extend(self._buy_leaps(date, price, iv))
        return events

    def _handle_leaps_expiry(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        old = self.leaps
        assert old is not None
        intrinsic = max(price - old.strike, 0.0)
        proceeds = intrinsic * old.contracts * 100
        commission = self.cfg.commission_per_contract * old.contracts
        self.cash += proceeds - commission
        self.total_commissions += commission
        pnl = (intrinsic - old.premium_per_share) * old.contracts * 100
        self.leaps = None
        return [TradeEvent(date=date, event_type="leaps_expired", details={"strike": old.strike}, pnl=pnl)]

    # ---- Short call management ---------------------------------------------

    def _sell_short_call(
        self, date: datetime, price: float, iv: float, iv_rank: float,
    ) -> list[TradeEvent]:
        if self.leaps is None:
            return []
        if iv_rank < self.cfg.min_iv_rank:
            return []

        # Trend filter for short call entry (primary SMA only)
        if self.cfg.trend_sma_period > 0:
            sma = _sma(self.price_history, self.cfg.trend_sma_period)
            if sma is not None and price < sma:
                return []

        tte = self.cfg.short_dte / 365.0
        strike = _find_strike(price, self.cfg.short_delta, tte, iv, self.cfg.risk_free_rate, "call")

        min_strike = self.leaps.strike + self.leaps.premium_per_share
        strike = max(strike, _round_strike(min_strike, price))

        premium = _price_opt(price, strike, tte, iv, self.cfg.risk_free_rate, "call")
        premium = max(premium - self.cfg.short_slip_per_share, 0.01)

        # Minimum absolute premium filter
        if premium < self.cfg.min_short_premium_abs:
            return []

        if premium / price < self.cfg.min_short_premium_pct:
            return []

        contracts = self.leaps.contracts
        commission = self.cfg.commission_per_contract * contracts
        self.cash += premium * contracts * 100 - commission
        self.total_short_premium += premium * contracts * 100
        self.total_commissions += commission

        self.short_call = ShortCall(
            strike=strike, expiry=date + timedelta(days=self.cfg.short_dte),
            premium_per_share=premium, contracts=contracts,
            entry_date=date, entry_underlying=price,
        )
        self.short_cycles += 1

        return [TradeEvent(
            date=date, event_type="sell_short_call",
            details={
                "strike": strike, "dte": self.cfg.short_dte,
                "premium": round(premium, 4), "contracts": contracts,
                "underlying": round(price, 2), "leaps_strike": self.leaps.strike,
            },
        )]

    def _handle_short_expiry(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        sc = self.short_call
        assert sc is not None

        intrinsic = sc.intrinsic(price)
        if intrinsic > 0:
            leaps = self.leaps
            if leaps:
                spread_profit = (sc.strike - leaps.strike) * sc.contracts * 100
                net_pnl = spread_profit - (leaps.premium_per_share * leaps.contracts * 100) + (sc.premium_per_share * sc.contracts * 100)
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
                    date=date, event_type="short_call_assigned_close_spread",
                    details={"short_strike": sc.strike, "leaps_strike": leaps.strike, "underlying": round(price, 2)},
                    pnl=net_pnl,
                )]
            loss = intrinsic * sc.contracts * 100
            self.cash -= loss
            self.short_call = None
            self.losses += 1
            return [TradeEvent(date=date, event_type="short_call_assigned_naked", pnl=-loss)]

        pnl = sc.premium_per_share * sc.contracts * 100
        self.wins += 1
        self.short_call = None
        return [TradeEvent(
            date=date, event_type="short_call_expired_otm",
            details={"strike": sc.strike, "underlying": round(price, 2)},
            pnl=pnl,
        )]

    def _manage_short_call(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        sc = self.short_call
        assert sc is not None

        dte = sc.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        current = _price_opt(price, sc.strike, tte, iv, self.cfg.risk_free_rate, "call")
        pnl_per_share = sc.premium_per_share - current

        # Profit target (0 = disabled)
        if self.cfg.short_profit_target > 0:
            if sc.premium_per_share > 0 and pnl_per_share >= sc.premium_per_share * self.cfg.short_profit_target:
                return self._close_short_call(date, price, iv, current, "profit_target")

        # Roll at DTE threshold (0 = disabled, let expire)
        if self.cfg.short_roll_dte > 0 and dte <= self.cfg.short_roll_dte:
            return self._close_short_call(date, price, iv, current, "roll")

        # Stop loss
        if sc.premium_per_share > 0 and pnl_per_share < -sc.premium_per_share * self.cfg.short_stop_loss:
            return self._close_short_call(date, price, iv, current, "stop_loss")

        return []

    def _close_short_call(
        self, date: datetime, price: float, iv: float,
        buyback: float, reason: str,
    ) -> list[TradeEvent]:
        sc = self.short_call
        assert sc is not None

        buyback += self.cfg.short_slip_per_share
        cost = buyback * sc.contracts * 100
        commission = self.cfg.commission_per_contract * sc.contracts
        self.cash -= cost + commission
        self.total_commissions += commission

        pnl = (sc.premium_per_share - buyback) * sc.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        self.short_call = None
        return [TradeEvent(
            date=date, event_type=f"close_short_call_{reason}",
            details={"strike": sc.strike, "entry_premium": round(sc.premium_per_share, 4),
                     "buyback": round(buyback, 4), "underlying": round(price, 2)},
            pnl=pnl,
        )]

    # ---- reporting ---------------------------------------------------------

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
            "short_cycles": self.short_cycles,
            "leaps_rolls": self.leaps_rolls,
            "total_leaps_cost": round(self.total_leaps_cost, 2),
            "total_short_premium": round(self.total_short_premium, 2),
            "total_commissions": round(self.total_commissions, 2),
            "start_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "end_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "days": len(dates),
        }
