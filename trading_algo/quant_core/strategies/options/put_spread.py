"""
Bull Put Spread Strategy (Vertical Credit Spread)

Structure:
    SELL higher-strike put (e.g., 30-delta)
    BUY  lower-strike put  (e.g., 15-delta or fixed $ width below)

P&L profile:
    Max profit  = net credit received
    Max loss    = spread width - net credit  (DEFINED, known at entry)
    Breakeven   = short strike - net credit

Margin requirement = max loss only (not full assignment cost).
For a $5K account selling a $2-wide spread, max loss ≈ $150-180 per spread.
That's 3-4% of capital vs 15-30% for a naked put on the same underlying.

State machine:
    IDLE  --sell spread-->  OPEN  --profit target-->  IDLE (close both legs)
                                  --stop loss-->      IDLE (close both legs)
                                  --roll DTE-->       IDLE (close, re-open)
                                  --expiry OTM-->     IDLE (both expire worthless)
                                  --expiry ITM-->     IDLE (max loss realized)
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
class PutSpreadConfig:
    initial_capital: float = 5_000.0
    short_delta: float = 0.30
    spread_width: float = 0.0          # 0 = auto (half of short_delta distance from spot)
    spread_width_dollars: float = 2.0  # fixed $ width between strikes (used if > 0)
    target_dte: int = 45
    profit_target: float = 0.50        # close at 50% of max profit
    stop_loss: float = 2.0             # close at 2x credit received (0 = disabled)
    roll_dte: int = 14
    min_iv_rank: float = 30.0
    max_risk_per_trade_pct: float = 0.05  # risk max 5% of capital per spread
    trend_sma_period: int = 50
    risk_free_rate: float = 0.045
    commission_per_contract: float = 0.90
    bid_ask_slip_per_share: float = 0.05
    dividend_yield: float = 0.0
    skew_slope: float = 0.8


# ---------------------------------------------------------------------------
# Spread position
# ---------------------------------------------------------------------------

@dataclass
class SpreadLeg:
    short_strike: float
    long_strike: float
    expiry: datetime
    net_credit: float          # per-share credit after slippage
    contracts: int
    entry_date: datetime
    entry_underlying: float

    @property
    def width(self) -> float:
        return self.short_strike - self.long_strike

    def max_loss_per_share(self) -> float:
        return self.width - self.net_credit

    def max_loss_total(self) -> float:
        return self.max_loss_per_share() * self.contracts * 100

    def short_intrinsic(self, underlying: float) -> float:
        return max(self.short_strike - underlying, 0.0)

    def long_intrinsic(self, underlying: float) -> float:
        return max(self.long_strike - underlying, 0.0)

    def spread_intrinsic(self, underlying: float) -> float:
        return self.short_intrinsic(underlying) - self.long_intrinsic(underlying)

    def dte(self, as_of: datetime) -> int:
        return max((self.expiry - as_of).days, 0)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class PutSpreadStrategy:
    """
    Bull Put Spread: sell higher put, buy lower put, same expiry.

    Call ``on_bar`` once per trading day. Query ``get_equity`` for M2M value.
    """

    def __init__(self, cfg: PutSpreadConfig | None = None):
        self.cfg = cfg or PutSpreadConfig()
        self.cash: float = self.cfg.initial_capital
        self.spread: SpreadLeg | None = None
        self._last_bar_date: datetime | None = None
        self._price_history: list[float] = []

        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.total_credit_collected: float = 0.0
        self.total_debit_paid: float = 0.0
        self.total_commissions: float = 0.0
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
        self._last_bar_date = date
        self._price_history.append(price)
        bar_events: list[TradeEvent] = []

        if self.spread and date >= self.spread.expiry:
            bar_events.extend(self._handle_expiration(date, price, iv))
        elif self.spread:
            bar_events.extend(self._manage_position(date, price, iv))

        if self.spread is None:
            bar_events.extend(self._open_new_position(date, price, iv, iv_rank))

        eq = self.get_equity(price, iv, as_of=date)
        self.equity_curve.append((date, eq))
        self.events.extend(bar_events)
        return bar_events

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
        ref_date = as_of or self._last_bar_date or datetime.now()
        equity = self.cash
        if self.spread:
            dte_days = max((self.spread.expiry - ref_date).days, 0)
            tte = max(dte_days / 365.0, 1 / 365)
            short_mtm = _price_option(
                price, self.spread.short_strike, tte, iv,
                self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            long_mtm = _price_option(
                price, self.spread.long_strike, tte, iv,
                self.cfg.risk_free_rate, "put",
                dividend_yield=self.cfg.dividend_yield,
                skew_slope=self.cfg.skew_slope,
            )
            # We are short the higher put, long the lower put
            # Liability = short_mtm - long_mtm (net option value we'd pay to close)
            net_liability = (short_mtm - long_mtm) * self.spread.contracts * 100
            equity -= net_liability
        return equity

    # ---- sizing ------------------------------------------------------------

    def _size_spread(self, max_loss_per_contract: float) -> int:
        if max_loss_per_contract <= 0:
            return 0
        max_risk = self.cash * self.cfg.max_risk_per_trade_pct
        return max(int(max_risk / (max_loss_per_contract * 100)), 0)

    # ---- opening -----------------------------------------------------------

    def _open_new_position(
        self, date: datetime, price: float, iv: float, iv_rank: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        if iv_rank < self.cfg.min_iv_rank:
            return events

        # Trend filter
        if self.cfg.trend_sma_period > 0 and len(self._price_history) >= self.cfg.trend_sma_period:
            sma = float(np.mean(self._price_history[-self.cfg.trend_sma_period:]))
            if price < sma:
                return events

        tte_years = self.cfg.target_dte / 365.0

        # Short leg: sell put at target delta
        short_strike = _find_strike_by_delta(
            price, self.cfg.short_delta, tte_years, iv,
            self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
        )

        # Long leg: buy put further OTM
        if self.cfg.spread_width_dollars > 0:
            long_strike = _round_strike(short_strike - self.cfg.spread_width_dollars, price)
        elif self.cfg.spread_width > 0:
            long_strike = _round_strike(short_strike - self.cfg.spread_width, price)
        else:
            # Auto: half the distance from spot to short strike
            auto_width = (price - short_strike) * 0.5
            auto_width = max(auto_width, 1.0)
            long_strike = _round_strike(short_strike - auto_width, price)

        if long_strike <= 0 or long_strike >= short_strike:
            return events

        spread_width = short_strike - long_strike

        # Price both legs
        short_premium = _price_option(
            price, short_strike, tte_years, iv, self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        long_premium = _price_option(
            price, long_strike, tte_years, iv, self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )

        # Slippage: sell short at bid (lower), buy long at ask (higher)
        short_premium = max(short_premium - self.cfg.bid_ask_slip_per_share, 0.01)
        long_premium = long_premium + self.cfg.bid_ask_slip_per_share

        net_credit = short_premium - long_premium
        if net_credit <= 0:
            return events

        max_loss_per_share = spread_width - net_credit
        if max_loss_per_share <= 0:
            return events

        contracts = self._size_spread(max_loss_per_share)
        if contracts <= 0:
            return events

        # Commission: 2 legs per spread
        commission = 2 * self.cfg.commission_per_contract * contracts
        expiry = date + timedelta(days=self.cfg.target_dte)

        self.spread = SpreadLeg(
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=expiry,
            net_credit=net_credit,
            contracts=contracts,
            entry_date=date,
            entry_underlying=price,
        )
        self.cash += net_credit * contracts * 100 - commission
        self.total_credit_collected += net_credit * contracts * 100
        self.total_commissions += commission

        events.append(TradeEvent(
            date=date, event_type="sell_put_spread",
            details={
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": round(spread_width, 2),
                "net_credit": round(net_credit, 4),
                "max_loss": round(max_loss_per_share * contracts * 100, 2),
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
        spread = self.spread
        assert spread is not None

        # At expiry: net settlement = short intrinsic - long intrinsic
        net_intrinsic = spread.spread_intrinsic(price)

        # PnL = credit received - net intrinsic paid out
        pnl_per_share = spread.net_credit - net_intrinsic
        pnl = pnl_per_share * spread.contracts * 100

        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        # Cash adjustment: we already received credit at entry.
        # At expiry, we pay out net intrinsic.
        self.cash -= net_intrinsic * spread.contracts * 100

        events.append(TradeEvent(
            date=date,
            event_type="spread_expired",
            details={
                "short_strike": spread.short_strike,
                "long_strike": spread.long_strike,
                "underlying": round(price, 2),
                "net_intrinsic": round(net_intrinsic, 4),
                "credit_received": round(spread.net_credit, 4),
            },
            pnl=pnl,
        ))
        self.spread = None
        return events

    # ---- position management -----------------------------------------------

    def _manage_position(
        self, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        spread = self.spread
        assert spread is not None

        dte = spread.dte(date)
        tte = max(dte / 365.0, 1 / 365)

        # Current spread value (what it costs to close)
        short_price = _price_option(
            price, spread.short_strike, tte, iv, self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        long_price = _price_option(
            price, spread.long_strike, tte, iv, self.cfg.risk_free_rate, "put",
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
        current_spread_value = short_price - long_price  # cost to close (buy short, sell long)

        # PnL per share = credit - current_spread_value
        pnl_per_share = spread.net_credit - current_spread_value

        # Profit target
        if self.cfg.profit_target > 0 and pnl_per_share >= spread.net_credit * self.cfg.profit_target:
            events.extend(self._close_spread(date, price, iv, current_spread_value, "profit_target"))
            return events

        # Roll at DTE threshold
        if self.cfg.roll_dte > 0 and dte <= self.cfg.roll_dte:
            events.extend(self._close_spread(date, price, iv, current_spread_value, "roll"))
            return events

        # Stop loss
        if self.cfg.stop_loss > 0 and spread.net_credit > 0:
            max_acceptable_loss = spread.net_credit * self.cfg.stop_loss
            if pnl_per_share < -max_acceptable_loss:
                events.extend(self._close_spread(date, price, iv, current_spread_value, "stop_loss"))
                return events

        return events

    def _close_spread(
        self,
        date: datetime,
        price: float,
        iv: float,
        current_spread_value: float,
        reason: str,
    ) -> list[TradeEvent]:
        spread = self.spread
        assert spread is not None

        # Close: buy back short (at ask), sell long (at bid)
        close_cost_per_share = current_spread_value + 2 * self.cfg.bid_ask_slip_per_share
        close_cost = close_cost_per_share * spread.contracts * 100
        commission = 2 * self.cfg.commission_per_contract * spread.contracts

        self.cash -= close_cost + commission
        self.total_debit_paid += close_cost
        self.total_commissions += commission

        pnl = (spread.net_credit - close_cost_per_share) * spread.contracts * 100
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

        event = TradeEvent(
            date=date,
            event_type=f"close_spread_{reason}",
            details={
                "short_strike": spread.short_strike,
                "long_strike": spread.long_strike,
                "entry_credit": round(spread.net_credit, 4),
                "close_cost": round(close_cost_per_share, 4),
                "underlying": round(price, 2),
                "reason": reason,
            },
            pnl=pnl,
        )
        self.spread = None
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
            "total_credit_collected": round(self.total_credit_collected, 2),
            "total_debit_paid": round(self.total_debit_paid, 2),
            "net_premium": round(net_premium, 2),
            "total_commissions": round(self.total_commissions, 2),
            "start_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "end_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "days": len(dates),
        }
