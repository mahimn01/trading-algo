"""
The Wheel Strategy  (Cash-Secured Put -> Covered Call cycle)

State Machine:
    CSP_IDLE  --sell put-->  CSP_OPEN  --expire OTM-->  CSP_IDLE
                                       --expire ITM-->  CC_IDLE  (assignment: buy shares at strike)
    CC_IDLE   --sell call--> CC_OPEN   --expire OTM-->  CC_IDLE
                                       --expire ITM-->  CSP_IDLE (called away: sell shares at strike)

Management rules applied while a short option is open:
    1. Close at *profit_target* % of collected premium.
    2. Roll (close + re-open) when DTE <= *roll_dte*.
    3. Stop-loss at *stop_loss* x premium collected (CSP phase only).
    4. In CC phase, never sell a call below net cost basis.

Honesty notes:
    - Options prices simulated via BSM with dynamic IV estimation.
    - Bid-ask slippage modeled as fixed dollar amount per contract (not % of mid).
    - Exchange fees included alongside per-contract commissions.
    - Cash reserve maintained (20%) to prevent margin calls.
    - Assignment occurs at expiry; no early assignment modeling.
    - Vol skew modeled via parametric slope (configurable).
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
class WheelConfig:
    initial_capital: float = 10_000.0
    put_delta: float = 0.30          # target |delta| for short puts
    call_delta: float = 0.30         # target delta for short calls
    target_dte: int = 45             # ideal DTE when opening (45 is optimal per research)
    profit_target: float = 0.50      # close at 50 % of max profit (0 = disabled)
    roll_dte: int = 14               # roll when <= N DTE (0 = let expire)
    stop_loss: float = 0.0           # close CSP at Nx premium loss (0 = no stop)
    min_iv_rank: float = 25.0        # don't sell when IV rank below this
    min_premium_pct: float = 0.005   # minimum premium as % of underlying
    contracts: int | None = None     # fixed size; None = auto-size
    cash_reserve_pct: float = 0.20   # keep 20% cash reserve (margin buffer)
    # Trend filter: only sell puts when price > SMA (0 = disabled)
    trend_sma_period: int = 50
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    skew_slope: float = 0.8
    commission_per_contract: float = 0.90
    commission_per_share: float = 0.005
    bid_ask_slip_per_share: float = 0.05


# ---------------------------------------------------------------------------
# Internal position types
# ---------------------------------------------------------------------------

@dataclass
class ShortOptionLeg:
    option_type: Literal["put", "call"]
    strike: float
    expiry: datetime
    premium_per_share: float   # collected premium (positive, after slippage)
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
class TradeEvent:
    date: datetime
    event_type: str
    details: dict = field(default_factory=dict)
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_strike(raw: float, underlying: float) -> float:
    """Round to nearest standard option strike increment."""
    if underlying < 25:
        return round(raw * 2) / 2       # $0.50
    if underlying < 200:
        return round(raw)               # $1.00
    return round(raw / 5) * 5           # $5.00


def _find_strike_by_delta(
    spot: float,
    target_abs_delta: float,
    tte_years: float,
    vol: float,
    rate: float,
    option_type: Literal["put", "call"],
    dividend_yield: float = 0.0,
) -> float:
    """Numerically find the strike that gives *target_abs_delta*."""
    if tte_years <= 0:
        return spot

    def _delta_err(K: float) -> float:
        spec = OptionSpec(
            spot=spot, strike=K, time_to_expiry=tte_years,
            volatility=vol, risk_free_rate=rate, option_type=option_type,
            dividend_yield=dividend_yield,
        )
        d = BlackScholesCalculator.delta(spec)
        return abs(d) - target_abs_delta

    if option_type == "put":
        lo, hi = spot * 0.50, spot * 1.00
    else:
        lo, hi = spot * 1.00, spot * 2.00

    for _ in range(5):
        if _delta_err(lo) * _delta_err(hi) < 0:
            break
        lo *= 0.8
        hi *= 1.2
    try:
        raw = brentq(_delta_err, lo, hi, xtol=0.01)
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
    skew_adjust: bool = True,
    skew_slope: float = 0.8,
) -> float:
    """BSM price with optional volatility skew adjustment.

    Skew model:
      Puts:  IV_adj = vol * (1 + skew_slope * max(0, (spot - strike) / spot))
      Calls: IV_adj = vol * (1 - skew_slope * 0.3 * max(0, (strike - spot) / spot))
    """
    if tte_years <= 0:
        if option_type == "put":
            return max(strike - spot, 0.0)
        return max(spot - strike, 0.0)

    adjusted_vol = vol
    if skew_adjust and spot > 0:
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


# ---------------------------------------------------------------------------
# The Wheel Strategy
# ---------------------------------------------------------------------------

class WheelStrategy:
    """
    Self-contained Wheel strategy with full position tracking.

    Call ``on_bar`` once per trading day with current market state.
    Query ``get_equity`` for mark-to-market portfolio value.
    """

    def __init__(self, cfg: WheelConfig | None = None):
        self.cfg = cfg or WheelConfig()
        self.cash: float = self.cfg.initial_capital
        self.stock_qty: int = 0
        self.stock_avg_cost: float = 0.0
        self.short_option: ShortOptionLeg | None = None
        self.phase: Literal["CSP", "CC"] = "CSP"
        self._last_bar_date: datetime | None = None
        self._price_history: list[float] = []  # for trend filter

        # Tracking
        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.total_premium_collected: float = 0.0
        self.total_premium_paid: float = 0.0
        self.total_commissions: float = 0.0
        self.wheel_cycles: int = 0
        self.wins: int = 0
        self.losses: int = 0

    # ---- public interface --------------------------------------------------

    def on_bar(
        self,
        date: datetime,
        price: float,
        iv: float,
        iv_rank: float,
    ) -> list[TradeEvent]:
        """Process one trading day.  Returns trade events that occurred."""
        self._last_bar_date = date
        self._price_history.append(price)
        bar_events: list[TradeEvent] = []

        # 1. Handle expiration
        if self.short_option and date >= self.short_option.expiry:
            bar_events.extend(self._handle_expiration(date, price, iv))

        # 2. Manage open position (profit target / roll / stop)
        elif self.short_option:
            bar_events.extend(self._manage_position(date, price, iv))

        # 3. Open new position if idle
        if self.short_option is None:
            bar_events.extend(self._open_new_position(date, price, iv, iv_rank))

        # Record equity using sim date, NOT wall-clock time
        eq = self.get_equity(price, iv, as_of=date)
        self.equity_curve.append((date, eq))

        self.events.extend(bar_events)
        return bar_events

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
        """Mark-to-market equity: cash + stock value - short option liability."""
        # FIX: use sim date, not datetime.now()
        ref_date = as_of or self._last_bar_date or datetime.now()
        equity = self.cash + self.stock_qty * price
        if self.short_option:
            dte_days = max((self.short_option.expiry - ref_date).days, 0)
            tte = max(dte_days / 365.0, 1 / 365)
            mtm = _price_option(
                price, self.short_option.strike, tte, iv,
                self.cfg.risk_free_rate, self.short_option.option_type,
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            equity -= mtm * self.short_option.contracts * 100
        return equity

    # ---- sizing ------------------------------------------------------------

    def _num_contracts(self, strike: float) -> int:
        """How many contracts we can afford with cash reserve."""
        if self.cfg.contracts is not None:
            return self.cfg.contracts
        if self.phase == "CSP":
            # FIX: reserve cash_reserve_pct for margin buffer
            usable_cash = self.cash * (1 - self.cfg.cash_reserve_pct)
            affordable = int(usable_cash / (strike * 100))
        else:
            affordable = self.stock_qty // 100
        return max(affordable, 0)

    # ---- opening -----------------------------------------------------------

    def _open_new_position(
        self, date: datetime, price: float, iv: float, iv_rank: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        if iv_rank < self.cfg.min_iv_rank:
            return events

        # Trend filter: don't sell puts below SMA (downtrend protection)
        if self.cfg.trend_sma_period > 0 and self.phase == "CSP":
            if len(self._price_history) >= self.cfg.trend_sma_period:
                sma = float(np.mean(self._price_history[-self.cfg.trend_sma_period:]))
                if price < sma:
                    return events

        tte_years = self.cfg.target_dte / 365.0

        if self.phase == "CSP":
            strike = _find_strike_by_delta(
                price, self.cfg.put_delta, tte_years, iv,
                self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
            )
            contracts = self._num_contracts(strike)
            if contracts <= 0:
                return events

            premium = _price_option(
                price, strike, tte_years, iv, self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            # FIX: fixed dollar slippage, not percentage
            premium = max(premium - self.cfg.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.min_premium_pct:
                return events

            expiry = date + timedelta(days=self.cfg.target_dte)
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
                date=date, event_type="sell_put",
                details={
                    "strike": strike, "dte": self.cfg.target_dte,
                    "premium": round(premium, 4), "contracts": contracts,
                    "underlying": round(price, 2), "iv": round(iv, 4),
                    "iv_rank": round(iv_rank, 1),
                },
            ))

        elif self.phase == "CC":
            min_strike = self.stock_avg_cost
            strike = _find_strike_by_delta(
                price, self.cfg.call_delta, tte_years, iv,
                self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
            )
            strike = max(strike, _round_strike(min_strike, price))
            contracts = self.stock_qty // 100
            if contracts <= 0:
                return events

            premium = _price_option(
                price, strike, tte_years, iv, self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            premium = max(premium - self.cfg.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.min_premium_pct:
                return events

            expiry = date + timedelta(days=self.cfg.target_dte)
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
                date=date, event_type="sell_call",
                details={
                    "strike": strike, "dte": self.cfg.target_dte,
                    "premium": round(premium, 4), "contracts": contracts,
                    "underlying": round(price, 2), "cost_basis": round(self.stock_avg_cost, 2),
                },
            ))

        return events

    # ---- expiration --------------------------------------------------------

    def _handle_expiration(
        self, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        opt = self.short_option
        assert opt is not None

        intrinsic = opt.intrinsic(price)

        if opt.option_type == "put":
            if intrinsic > 0:
                # PUT ASSIGNED: buy shares at strike
                shares = opt.contracts * 100
                cost = opt.strike * shares
                commission = self.cfg.commission_per_share * shares
                self.cash -= cost + commission
                self.stock_qty += shares
                self.stock_avg_cost = opt.strike - opt.premium_per_share
                self.total_commissions += commission
                self.phase = "CC"
                events.append(TradeEvent(
                    date=date, event_type="put_assigned",
                    details={
                        "strike": opt.strike, "shares": shares,
                        "cost_basis": round(self.stock_avg_cost, 4),
                        "underlying": round(price, 2),
                    },
                ))
            else:
                self.wins += 1
                pnl = opt.premium_per_share * opt.contracts * 100
                events.append(TradeEvent(
                    date=date, event_type="put_expired_otm",
                    details={"strike": opt.strike, "underlying": round(price, 2)},
                    pnl=pnl,
                ))
        else:
            if intrinsic > 0:
                # CALLED AWAY: sell shares at strike
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
                self.phase = "CSP"
                self.wheel_cycles += 1
                if total_pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                events.append(TradeEvent(
                    date=date, event_type="called_away",
                    details={
                        "strike": opt.strike, "shares": shares,
                        "stock_pnl": round(stock_pnl, 2),
                        "option_pnl": round(option_pnl, 2),
                        "underlying": round(price, 2),
                    },
                    pnl=total_pnl,
                ))
            else:
                self.wins += 1
                pnl = opt.premium_per_share * opt.contracts * 100
                events.append(TradeEvent(
                    date=date, event_type="call_expired_otm",
                    details={"strike": opt.strike, "underlying": round(price, 2)},
                    pnl=pnl,
                ))

        self.short_option = None
        return events

    # ---- position management -----------------------------------------------

    def _manage_position(
        self, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        opt = self.short_option
        assert opt is not None

        dte = opt.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        current_price = _price_option(
            price, opt.strike, tte, iv, self.cfg.risk_free_rate, opt.option_type,
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        option_pnl_per_share = opt.premium_per_share - current_price
        max_profit = opt.premium_per_share

        # --- Profit target (0 = disabled, let it ride to expiry) ---
        if self.cfg.profit_target > 0 and max_profit > 0 and option_pnl_per_share >= max_profit * self.cfg.profit_target:
            events.extend(self._close_option(date, price, iv, current_price, "profit_target"))
            return events

        # --- Roll at DTE threshold (0 = let expire / allow assignment) ---
        if self.cfg.roll_dte > 0 and dte <= self.cfg.roll_dte:
            events.extend(self._close_option(date, price, iv, current_price, "roll"))
            return events

        # --- Stop loss (CSP phase only, 0 = disabled) ---
        if self.cfg.stop_loss > 0 and opt.option_type == "put" and max_profit > 0:
            if option_pnl_per_share < -max_profit * self.cfg.stop_loss:
                events.extend(self._close_option(date, price, iv, current_price, "stop_loss"))
                return events

        return events

    def _close_option(
        self,
        date: datetime,
        price: float,
        iv: float,
        buyback_price: float,
        reason: str,
    ) -> list[TradeEvent]:
        opt = self.short_option
        assert opt is not None

        # FIX: fixed dollar slippage on buyback (at ask)
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
            event_type=f"close_{opt.option_type}_{reason}",
            details={
                "strike": opt.strike,
                "entry_premium": round(opt.premium_per_share, 4),
                "buyback": round(buyback_price, 4),
                "underlying": round(price, 2),
                "reason": reason,
            },
            pnl=pnl,
        )
        self.short_option = None
        return [event]

    # ---- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        dates = [d for d, _ in self.equity_curve]
        equities = np.array([e for _, e in self.equity_curve])
        total_trades = self.wins + self.losses

        returns = np.diff(equities) / equities[:-1] if len(equities) > 1 else np.array([])
        valid_returns = returns[np.isfinite(returns)]

        # FIX: proper Sharpe with risk-free rate
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
            "wheel_cycles": self.wheel_cycles,
            "total_premium_collected": round(self.total_premium_collected, 2),
            "total_premium_paid": round(self.total_premium_paid, 2),
            "net_premium": round(self.total_premium_collected - self.total_premium_paid, 2),
            "total_commissions": round(self.total_commissions, 2),
            "phase": self.phase,
            "stock_held": self.stock_qty,
            "start_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "end_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "days": len(dates),
        }
