"""
Jade Lizard Strategy (Short Put + Short Call Spread)

Structure:
    SELL OTM put         (bearish insurance income)
    SELL OTM call        (bullish premium)
    BUY  further OTM call (cap upside risk)

P&L profile:
    Max profit   = total credit (put premium + call spread credit)
    Max loss up  = call spread width - call credit  (can be ZERO if credit > width)
    Max loss down= put strike × 100 - total credit  (like CSP but offset by call income)

The key insight: if total_credit >= call_spread_width, there is ZERO upside risk.
The downside risk is the same as a cash-secured put but reduced by the extra call
spread premium. This makes it strictly better than a naked put when IV is elevated
and the underlying has low probability of a large move in either direction.

Best on:  high IV rank (>35), range-bound or mildly bullish underlyings.
Avoid on: strong directional trends, low IV environments.

State machine:
    IDLE  --sell jade lizard-->  OPEN  --profit target-->  IDLE
                                       --stop loss-->      IDLE
                                       --roll DTE-->       IDLE
                                       --expiry-->         IDLE (settle all 3 legs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from trading_algo.quant_core.strategies.options.wheel import (
    ShortOptionLeg,
    TradeEvent,
    _find_strike_by_delta,
    _price_option,
    _round_strike,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JadeLizardConfig:
    initial_capital: float = 10_000.0
    put_delta: float = 0.25
    call_short_delta: float = 0.25
    call_spread_width: float = 5.0     # $ width of call spread
    target_dte: int = 45
    profit_target: float = 0.50
    stop_loss: float = 2.0             # close at 2x credit (0 = disabled)
    roll_dte: int = 14
    min_iv_rank: float = 35.0          # needs higher IV to work
    max_risk_per_trade_pct: float = 0.20  # higher than spreads since it's a CSP-like structure
    trend_sma_period: int = 50
    risk_free_rate: float = 0.045
    commission_per_contract: float = 0.90
    bid_ask_slip_per_share: float = 0.05
    dividend_yield: float = 0.0
    skew_slope: float = 0.8
    cash_reserve_pct: float = 0.20


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class JadeLizardLeg:
    put_strike: float
    call_short_strike: float
    call_long_strike: float
    expiry: datetime
    put_premium: float         # per-share credit from put (after slippage)
    call_credit: float         # per-share net credit from call spread (after slippage)
    total_credit: float        # put_premium + call_credit
    contracts: int
    entry_date: datetime
    entry_underlying: float

    @property
    def call_spread_width(self) -> float:
        return self.call_long_strike - self.call_short_strike

    def max_loss_upside_per_share(self) -> float:
        return max(self.call_spread_width - self.call_credit, 0.0)

    def max_loss_downside_per_share(self) -> float:
        return self.put_strike - self.total_credit

    def max_loss_per_share(self) -> float:
        return max(self.max_loss_upside_per_share(), self.max_loss_downside_per_share())

    def dte(self, as_of: datetime) -> int:
        return max((self.expiry - as_of).days, 0)

    def settlement_pnl_per_share(self, underlying: float) -> float:
        # Put leg
        put_intrinsic = max(self.put_strike - underlying, 0.0)
        # Call spread: short call intrinsic - long call intrinsic
        short_call_intrinsic = max(underlying - self.call_short_strike, 0.0)
        long_call_intrinsic = max(underlying - self.call_long_strike, 0.0)
        call_spread_intrinsic = short_call_intrinsic - long_call_intrinsic

        total_payout = put_intrinsic + call_spread_intrinsic
        return self.total_credit - total_payout


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class JadeLizardStrategy:
    """
    Jade Lizard: short put + short call spread, same expiry.

    Collects premium from both sides. If structured so total credit >= call
    spread width, there is zero upside risk.
    """

    def __init__(self, cfg: JadeLizardConfig | None = None):
        self.cfg = cfg or JadeLizardConfig()
        self.cash: float = self.cfg.initial_capital
        self.position: JadeLizardLeg | None = None
        self._last_bar_date: datetime | None = None
        self._price_history: list[float] = []

        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.total_credit_collected: float = 0.0
        self.total_debit_paid: float = 0.0
        self.total_commissions: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.zero_upside_risk_trades: int = 0

    # ---- public interface --------------------------------------------------

    def on_bar(
        self,
        date: datetime,
        price: float,
        iv: float,
        iv_rank: float,
    ) -> list[TradeEvent]:
        self._last_bar_date = date
        self._price_history.append(price)
        bar_events: list[TradeEvent] = []

        if self.position and date >= self.position.expiry:
            bar_events.extend(self._handle_expiration(date, price, iv))
        elif self.position:
            bar_events.extend(self._manage_position(date, price, iv))

        if self.position is None:
            bar_events.extend(self._open_new_position(date, price, iv, iv_rank))

        eq = self.get_equity(price, iv, as_of=date)
        self.equity_curve.append((date, eq))
        self.events.extend(bar_events)
        return bar_events

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
        ref_date = as_of or self._last_bar_date or datetime.now()
        equity = self.cash
        if self.position:
            dte_days = max((self.position.expiry - ref_date).days, 0)
            tte = max(dte_days / 365.0, 1 / 365)

            # Put liability
            put_mtm = _price_option(
                price, self.position.put_strike, tte, iv,
                self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            # Short call liability
            short_call_mtm = _price_option(
                price, self.position.call_short_strike, tte, iv,
                self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            # Long call asset
            long_call_mtm = _price_option(
                price, self.position.call_long_strike, tte, iv,
                self.cfg.risk_free_rate, "call",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )

            net_liability = (put_mtm + short_call_mtm - long_call_mtm) * self.position.contracts * 100
            equity -= net_liability
        return equity

    # ---- sizing ------------------------------------------------------------

    def _size_position(self, total_credit: float, put_strike: float) -> int:
        if put_strike <= 0:
            return 0
        # Broker margin for jade lizard = put_strike - total_credit (per share)
        # This is the actual capital at risk per contract
        margin_per_share = put_strike - total_credit
        if margin_per_share <= 0:
            margin_per_share = 0.01  # free trade (credit exceeds all risk)
        # Size by risk budget using margin requirement
        risk_budget = self.cash * self.cfg.max_risk_per_trade_pct
        by_risk = int(risk_budget / (margin_per_share * 100))
        # Also cap by collateral with reserve
        usable_cash = self.cash * (1 - self.cfg.cash_reserve_pct)
        by_collateral = int(usable_cash / (margin_per_share * 100))
        return max(min(by_risk, by_collateral), 0)

    # ---- opening -----------------------------------------------------------

    def _open_new_position(
        self, date: datetime, price: float, iv: float, iv_rank: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        if iv_rank < self.cfg.min_iv_rank:
            return events

        # Trend filter: don't sell puts below SMA
        if self.cfg.trend_sma_period > 0 and len(self._price_history) >= self.cfg.trend_sma_period:
            sma = float(np.mean(self._price_history[-self.cfg.trend_sma_period:]))
            if price < sma:
                return events

        tte_years = self.cfg.target_dte / 365.0

        # --- Put leg ---
        put_strike = _find_strike_by_delta(
            price, self.cfg.put_delta, tte_years, iv,
            self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
        )
        put_premium = _price_option(
            price, put_strike, tte_years, iv, self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        put_premium = max(put_premium - self.cfg.bid_ask_slip_per_share, 0.01)

        # --- Call spread leg ---
        call_short_strike = _find_strike_by_delta(
            price, self.cfg.call_short_delta, tte_years, iv,
            self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
        )
        call_long_strike = _round_strike(
            call_short_strike + self.cfg.call_spread_width, price
        )

        short_call_premium = _price_option(
            price, call_short_strike, tte_years, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        long_call_premium = _price_option(
            price, call_long_strike, tte_years, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        # Slippage: sell short call at bid, buy long call at ask
        short_call_premium = max(short_call_premium - self.cfg.bid_ask_slip_per_share, 0.01)
        long_call_premium = long_call_premium + self.cfg.bid_ask_slip_per_share

        call_credit = short_call_premium - long_call_premium
        if call_credit < 0:
            call_credit = 0.0  # call spread might not yield credit in low IV

        total_credit = put_premium + call_credit
        if total_credit <= 0:
            return events

        call_spread_width = call_long_strike - call_short_strike
        upside_risk = max(call_spread_width - call_credit, 0.0)
        downside_risk = put_strike - total_credit
        max_loss_per_share = max(upside_risk, downside_risk)

        contracts = self._size_position(total_credit, put_strike)
        if contracts <= 0:
            return events

        # Commission: 3 legs
        commission = 3 * self.cfg.commission_per_contract * contracts
        expiry = date + timedelta(days=self.cfg.target_dte)

        zero_upside = total_credit >= call_spread_width
        if zero_upside:
            self.zero_upside_risk_trades += 1

        self.position = JadeLizardLeg(
            put_strike=put_strike,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            expiry=expiry,
            put_premium=put_premium,
            call_credit=call_credit,
            total_credit=total_credit,
            contracts=contracts,
            entry_date=date,
            entry_underlying=price,
        )
        self.cash += total_credit * contracts * 100 - commission
        self.total_credit_collected += total_credit * contracts * 100
        self.total_commissions += commission

        events.append(TradeEvent(
            date=date, event_type="sell_jade_lizard",
            details={
                "put_strike": put_strike,
                "call_short_strike": call_short_strike,
                "call_long_strike": call_long_strike,
                "put_premium": round(put_premium, 4),
                "call_credit": round(call_credit, 4),
                "total_credit": round(total_credit, 4),
                "zero_upside_risk": zero_upside,
                "max_loss_down": round(downside_risk * contracts * 100, 2),
                "max_loss_up": round(upside_risk * contracts * 100, 2),
                "contracts": contracts,
                "dte": self.cfg.target_dte,
                "underlying": round(price, 2),
                "iv": round(iv, 4),
                "iv_rank": round(iv_rank, 1),
            },
        ))
        return events

    # ---- expiration --------------------------------------------------------

    def _handle_expiration(
        self, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        pos = self.position
        assert pos is not None

        pnl_per_share = pos.settlement_pnl_per_share(price)
        pnl = pnl_per_share * pos.contracts * 100

        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        # Cash: credit already received at entry, now pay out settlement
        payout = pos.total_credit - pnl_per_share  # = intrinsic payout
        self.cash -= payout * pos.contracts * 100

        events.append(TradeEvent(
            date=date,
            event_type="jade_lizard_expired",
            details={
                "put_strike": pos.put_strike,
                "call_short_strike": pos.call_short_strike,
                "call_long_strike": pos.call_long_strike,
                "underlying": round(price, 2),
                "total_credit": round(pos.total_credit, 4),
                "settlement_pnl": round(pnl_per_share, 4),
            },
            pnl=pnl,
        ))
        self.position = None
        return events

    # ---- position management -----------------------------------------------

    def _manage_position(
        self, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        pos = self.position
        assert pos is not None

        dte = pos.dte(date)
        tte = max(dte / 365.0, 1 / 365)

        # Current cost to close all 3 legs
        put_price = _price_option(
            price, pos.put_strike, tte, iv, self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        short_call_price = _price_option(
            price, pos.call_short_strike, tte, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        long_call_price = _price_option(
            price, pos.call_long_strike, tte, iv, self.cfg.risk_free_rate, "call",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )

        # Net cost to close = buy put + buy short call - sell long call
        close_cost = put_price + short_call_price - long_call_price
        pnl_per_share = pos.total_credit - close_cost

        # Profit target
        if self.cfg.profit_target > 0 and pnl_per_share >= pos.total_credit * self.cfg.profit_target:
            events.extend(self._close_position(date, price, iv, close_cost, "profit_target"))
            return events

        # Roll
        if self.cfg.roll_dte > 0 and dte <= self.cfg.roll_dte:
            events.extend(self._close_position(date, price, iv, close_cost, "roll"))
            return events

        # Stop loss
        if self.cfg.stop_loss > 0 and pos.total_credit > 0:
            if pnl_per_share < -pos.total_credit * self.cfg.stop_loss:
                events.extend(self._close_position(date, price, iv, close_cost, "stop_loss"))
                return events

        return events

    def _close_position(
        self,
        date: datetime,
        price: float,
        iv: float,
        close_cost_per_share: float,
        reason: str,
    ) -> list[TradeEvent]:
        pos = self.position
        assert pos is not None

        # Slippage: 3 legs each with bid-ask slippage
        close_cost_per_share += 3 * self.cfg.bid_ask_slip_per_share
        close_cost = close_cost_per_share * pos.contracts * 100
        commission = 3 * self.cfg.commission_per_contract * pos.contracts

        self.cash -= close_cost + commission
        self.total_debit_paid += close_cost
        self.total_commissions += commission

        pnl = (pos.total_credit - close_cost_per_share) * pos.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        event = TradeEvent(
            date=date,
            event_type=f"close_jade_lizard_{reason}",
            details={
                "put_strike": pos.put_strike,
                "call_short_strike": pos.call_short_strike,
                "call_long_strike": pos.call_long_strike,
                "entry_credit": round(pos.total_credit, 4),
                "close_cost": round(close_cost_per_share, 4),
                "underlying": round(price, 2),
                "reason": reason,
            },
            pnl=pnl,
        )
        self.position = None
        return [event]

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

        net_premium = self.total_credit_collected - self.total_debit_paid

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
            "zero_upside_risk_trades": self.zero_upside_risk_trades,
            "total_credit_collected": round(self.total_credit_collected, 2),
            "total_debit_paid": round(self.total_debit_paid, 2),
            "net_premium": round(net_premium, 2),
            "total_commissions": round(self.total_commissions, 2),
            "start_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "end_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "days": len(dates),
        }
