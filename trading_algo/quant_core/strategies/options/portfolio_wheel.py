"""
Portfolio-Level Wheel Strategy Backtest

Runs the Wheel on N symbols simultaneously with SHARED capital, realistic
margin requirements, and correlation tracking.

Key differences from the single-symbol WheelStrategy:
  - ONE cash pool across all positions
  - Portfolio-level delta cap
  - Margin model: short put margin = max(prem + 0.20*S - OTM, prem + 0.10*K)
  - Symbol rotation by IV rank (highest = best premium)
  - Tracks simultaneous assignments, margin utilization, cross-symbol correlation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from trading_algo.quant_core.models.greeks import BlackScholesCalculator, OptionSpec
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
class PortfolioWheelConfig:
    initial_capital: float = 50_000.0
    max_symbols: int = 5
    max_allocation_per_symbol: float = 0.25
    max_portfolio_delta: float = 200.0
    max_margin_utilization: float = 0.70
    put_delta: float = 0.30
    call_delta: float = 0.30
    target_dte: int = 45
    profit_target: float = 0.50
    roll_dte: int = 14
    stop_loss: float = 0.0
    min_iv_rank: float = 25.0
    min_premium_pct: float = 0.005
    trend_sma_period: int = 50
    risk_free_rate: float = 0.045
    commission_per_contract: float = 0.90
    commission_per_share: float = 0.005
    bid_ask_slip_per_share: float = 0.05


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------

@dataclass
class SymbolState:
    phase: Literal["CSP", "CC"] = "CSP"
    stock_qty: int = 0
    stock_avg_cost: float = 0.0
    short_option: ShortOptionLeg | None = None
    price_history: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    assignment_dates: list[datetime] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Margin calculation
# ---------------------------------------------------------------------------

def _short_put_margin(
    premium: float, underlying: float, strike: float,
) -> float:
    """Reg-T short put margin per share."""
    otm_amount = max(strike - underlying, 0.0)
    if underlying >= strike:
        otm_amount = underlying - strike
    else:
        otm_amount = 0.0
    broad = premium + 0.20 * underlying - otm_amount
    narrow = premium + 0.10 * strike
    return max(broad, narrow)


# ---------------------------------------------------------------------------
# Portfolio Wheel
# ---------------------------------------------------------------------------

class PortfolioWheel:
    def __init__(self, cfg: PortfolioWheelConfig | None = None):
        self.cfg = cfg or PortfolioWheelConfig()
        self.cash: float = self.cfg.initial_capital
        self.symbols: dict[str, SymbolState] = {}

        # Tracking
        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.margin_history: list[tuple[datetime, float]] = []
        self.delta_history: list[tuple[datetime, float]] = []
        self.simultaneous_assignments: list[tuple[datetime, int]] = []
        self.margin_breaches: int = 0
        self.peak_assignments: int = 0

        self.total_premium_collected: float = 0.0
        self.total_premium_paid: float = 0.0
        self.total_commissions: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.wheel_cycles: int = 0

    # ---- portfolio queries -------------------------------------------------

    def _active_positions(self) -> int:
        return sum(
            1 for s in self.symbols.values()
            if s.short_option is not None or s.stock_qty > 0
        )

    def _total_margin_used(self, prices: dict[str, float], ivs: dict[str, float]) -> float:
        total = 0.0
        for sym, st in self.symbols.items():
            if st.short_option is None:
                continue
            price = prices.get(sym, 0.0)
            opt = st.short_option
            if opt.option_type == "put":
                per_share = _short_put_margin(opt.premium_per_share, price, opt.strike)
                total += per_share * opt.contracts * 100
            else:
                # Covered call: margin = stock cost (already paid in cash)
                pass
        return total

    def _portfolio_delta(
        self, prices: dict[str, float], ivs: dict[str, float], date: datetime,
    ) -> float:
        total_delta = 0.0
        for sym, st in self.symbols.items():
            price = prices.get(sym, 0.0)
            iv = ivs.get(sym, 0.25)
            # Stock delta
            total_delta += st.stock_qty
            # Option delta
            if st.short_option is not None:
                opt = st.short_option
                dte_days = max((opt.expiry - date).days, 0)
                tte = max(dte_days / 365.0, 1 / 365)
                spec = OptionSpec(
                    spot=price, strike=opt.strike, time_to_expiry=tte,
                    volatility=iv, risk_free_rate=self.cfg.risk_free_rate,
                    option_type=opt.option_type,
                )
                d = BlackScholesCalculator.delta(spec)
                # We are SHORT the option -> negate the delta
                total_delta -= d * opt.contracts * 100
        return total_delta

    def get_equity(
        self, prices: dict[str, float], ivs: dict[str, float], date: datetime,
    ) -> float:
        equity = self.cash
        for sym, st in self.symbols.items():
            price = prices.get(sym, 0.0)
            iv = ivs.get(sym, 0.25)
            equity += st.stock_qty * price
            if st.short_option is not None:
                opt = st.short_option
                dte_days = max((opt.expiry - date).days, 0)
                tte = max(dte_days / 365.0, 1 / 365)
                mtm = _price_option(
                    price, opt.strike, tte, iv,
                    self.cfg.risk_free_rate, opt.option_type,
                )
                equity -= mtm * opt.contracts * 100
        return equity

    # ---- main entry --------------------------------------------------------

    def on_day(
        self,
        date: datetime,
        prices: dict[str, float],
        ivs: dict[str, float],
        iv_ranks: dict[str, float],
    ) -> list[TradeEvent]:
        day_events: list[TradeEvent] = []

        # Ensure all symbols have state
        for sym in prices:
            if sym not in self.symbols:
                self.symbols[sym] = SymbolState()

        # Update price history + daily returns
        for sym, price in prices.items():
            st = self.symbols[sym]
            if st.price_history:
                prev = st.price_history[-1]
                if prev > 0:
                    st.daily_returns.append((price - prev) / prev)
            st.price_history.append(price)

        # 1. Handle expirations across all symbols
        assignments_today = 0
        for sym in list(self.symbols):
            st = self.symbols[sym]
            if st.short_option and date >= st.short_option.expiry:
                evts = self._handle_expiration(sym, st, date, prices[sym], ivs.get(sym, 0.25))
                for e in evts:
                    if "assigned" in e.event_type:
                        assignments_today += 1
                day_events.extend(evts)

        if assignments_today > 0:
            self.simultaneous_assignments.append((date, assignments_today))
            self.peak_assignments = max(self.peak_assignments, assignments_today)

        # 2. Manage open positions (profit target / roll / stop)
        for sym in list(self.symbols):
            st = self.symbols[sym]
            if st.short_option and date < st.short_option.expiry:
                evts = self._manage_position(sym, st, date, prices[sym], ivs.get(sym, 0.25))
                day_events.extend(evts)

        # 3. Open new positions — prioritize by IV rank descending
        candidates = []
        for sym in prices:
            st = self.symbols[sym]
            if st.short_option is not None:
                continue
            rank = iv_ranks.get(sym, 0.0)
            if rank < self.cfg.min_iv_rank:
                continue
            # Trend filter for CSP phase
            if st.phase == "CSP" and self.cfg.trend_sma_period > 0:
                if len(st.price_history) >= self.cfg.trend_sma_period:
                    sma = float(np.mean(st.price_history[-self.cfg.trend_sma_period:]))
                    if prices[sym] < sma:
                        continue
            # CC phase: must have stock
            if st.phase == "CC" and st.stock_qty < 100:
                continue
            candidates.append((sym, rank))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for sym, rank in candidates:
            if self._active_positions() >= self.cfg.max_symbols:
                break
            evts = self._open_position(
                sym, self.symbols[sym], date,
                prices[sym], ivs.get(sym, 0.25), rank,
                prices, ivs,
            )
            day_events.extend(evts)

        # Record portfolio state
        equity = self.get_equity(prices, ivs, date)
        self.equity_curve.append((date, equity))

        margin_used = self._total_margin_used(prices, ivs)
        margin_util = margin_used / equity if equity > 0 else 0.0
        self.margin_history.append((date, margin_util))

        port_delta = self._portfolio_delta(prices, ivs, date)
        self.delta_history.append((date, port_delta))

        self.events.extend(day_events)
        return day_events

    # ---- opening -----------------------------------------------------------

    def _open_position(
        self,
        sym: str,
        st: SymbolState,
        date: datetime,
        price: float,
        iv: float,
        iv_rank_val: float,
        all_prices: dict[str, float],
        all_ivs: dict[str, float],
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        equity = self.get_equity(all_prices, all_ivs, date)
        max_alloc = equity * self.cfg.max_allocation_per_symbol

        tte_years = self.cfg.target_dte / 365.0

        if st.phase == "CSP":
            strike = _find_strike_by_delta(
                price, self.cfg.put_delta, tte_years, iv,
                self.cfg.risk_free_rate, "put",
            )
            # Size: max contracts within allocation limit
            max_contracts_by_alloc = int(max_alloc / (strike * 100))
            # Also limited by available cash (with margin)
            max_contracts_by_cash = int(self.cash * 0.90 / (strike * 100))
            contracts = min(max_contracts_by_alloc, max_contracts_by_cash)
            contracts = max(contracts, 0)
            if contracts <= 0:
                return events

            premium = _price_option(price, strike, tte_years, iv, self.cfg.risk_free_rate, "put")
            premium = max(premium - self.cfg.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.min_premium_pct:
                return events

            # Margin check
            per_share_margin = _short_put_margin(premium, price, strike)
            new_margin = per_share_margin * contracts * 100
            current_margin = self._total_margin_used(all_prices, all_ivs)
            if (current_margin + new_margin) / equity > self.cfg.max_margin_utilization:
                self.margin_breaches += 1
                return events

            # Delta check
            port_delta = self._portfolio_delta(all_prices, all_ivs, date)
            # Short put delta contribution: short a put with negative delta -> we are short negative delta = positive delta exposure
            spec = OptionSpec(
                spot=price, strike=strike, time_to_expiry=tte_years,
                volatility=iv, risk_free_rate=self.cfg.risk_free_rate,
                option_type="put",
            )
            new_delta = -BlackScholesCalculator.delta(spec) * contracts * 100
            if abs(port_delta + new_delta) > self.cfg.max_portfolio_delta:
                return events

            expiry = date + timedelta(days=self.cfg.target_dte)
            commission = self.cfg.commission_per_contract * contracts

            st.short_option = ShortOptionLeg(
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
                    "symbol": sym, "strike": strike,
                    "dte": self.cfg.target_dte,
                    "premium": round(premium, 4),
                    "contracts": contracts,
                    "underlying": round(price, 2),
                    "iv_rank": round(iv_rank_val, 1),
                },
            ))

        elif st.phase == "CC":
            min_strike = st.stock_avg_cost
            strike = _find_strike_by_delta(
                price, self.cfg.call_delta, tte_years, iv,
                self.cfg.risk_free_rate, "call",
            )
            strike = max(strike, _round_strike(min_strike, price))
            contracts = st.stock_qty // 100
            if contracts <= 0:
                return events

            premium = _price_option(price, strike, tte_years, iv, self.cfg.risk_free_rate, "call")
            premium = max(premium - self.cfg.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.min_premium_pct:
                return events

            expiry = date + timedelta(days=self.cfg.target_dte)
            commission = self.cfg.commission_per_contract * contracts

            st.short_option = ShortOptionLeg(
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
                    "symbol": sym, "strike": strike,
                    "dte": self.cfg.target_dte,
                    "premium": round(premium, 4),
                    "contracts": contracts,
                    "underlying": round(price, 2),
                    "cost_basis": round(st.stock_avg_cost, 2),
                },
            ))

        return events

    # ---- expiration --------------------------------------------------------

    def _handle_expiration(
        self, sym: str, st: SymbolState, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        opt = st.short_option
        if opt is None:
            return events

        intrinsic = opt.intrinsic(price)

        if opt.option_type == "put":
            if intrinsic > 0:
                shares = opt.contracts * 100
                cost = opt.strike * shares
                commission = self.cfg.commission_per_share * shares
                self.cash -= cost + commission
                st.stock_qty += shares
                st.stock_avg_cost = opt.strike - opt.premium_per_share
                self.total_commissions += commission
                st.phase = "CC"
                st.assignment_dates.append(date)
                events.append(TradeEvent(
                    date=date, event_type="put_assigned",
                    details={
                        "symbol": sym, "strike": opt.strike,
                        "shares": shares,
                        "cost_basis": round(st.stock_avg_cost, 4),
                        "underlying": round(price, 2),
                    },
                ))
            else:
                self.wins += 1
                pnl = opt.premium_per_share * opt.contracts * 100
                events.append(TradeEvent(
                    date=date, event_type="put_expired_otm",
                    details={"symbol": sym, "strike": opt.strike, "underlying": round(price, 2)},
                    pnl=pnl,
                ))
        else:  # call
            if intrinsic > 0:
                shares = opt.contracts * 100
                proceeds = opt.strike * shares
                commission = self.cfg.commission_per_share * shares
                self.cash += proceeds - commission
                stock_pnl = (opt.strike - st.stock_avg_cost) * shares
                option_pnl = opt.premium_per_share * opt.contracts * 100
                total_pnl = stock_pnl + option_pnl
                st.stock_qty -= shares
                st.stock_avg_cost = 0.0
                self.total_commissions += commission
                st.phase = "CSP"
                self.wheel_cycles += 1
                if total_pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                events.append(TradeEvent(
                    date=date, event_type="called_away",
                    details={
                        "symbol": sym, "strike": opt.strike,
                        "shares": shares,
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
                    details={"symbol": sym, "strike": opt.strike, "underlying": round(price, 2)},
                    pnl=pnl,
                ))

        st.short_option = None
        return events

    # ---- management --------------------------------------------------------

    def _manage_position(
        self, sym: str, st: SymbolState, date: datetime, price: float, iv: float,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        opt = st.short_option
        if opt is None:
            return events

        dte = opt.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        current_price = _price_option(
            price, opt.strike, tte, iv, self.cfg.risk_free_rate, opt.option_type,
        )
        option_pnl_per_share = opt.premium_per_share - current_price
        max_profit = opt.premium_per_share

        # Profit target
        if self.cfg.profit_target > 0 and max_profit > 0:
            if option_pnl_per_share >= max_profit * self.cfg.profit_target:
                events.extend(self._close_option(sym, st, date, price, iv, current_price, "profit_target"))
                return events

        # Roll at DTE
        if self.cfg.roll_dte > 0 and dte <= self.cfg.roll_dte:
            events.extend(self._close_option(sym, st, date, price, iv, current_price, "roll"))
            return events

        # Stop loss (CSP only)
        if self.cfg.stop_loss > 0 and opt.option_type == "put" and max_profit > 0:
            if option_pnl_per_share < -max_profit * self.cfg.stop_loss:
                events.extend(self._close_option(sym, st, date, price, iv, current_price, "stop_loss"))
                return events

        return events

    def _close_option(
        self,
        sym: str,
        st: SymbolState,
        date: datetime,
        price: float,
        iv: float,
        buyback_price: float,
        reason: str,
    ) -> list[TradeEvent]:
        opt = st.short_option
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
            event_type=f"close_{opt.option_type}_{reason}",
            details={
                "symbol": sym,
                "strike": opt.strike,
                "entry_premium": round(opt.premium_per_share, 4),
                "buyback": round(buyback_price, 4),
                "underlying": round(price, 2),
                "reason": reason,
            },
            pnl=pnl,
        )
        st.short_option = None
        return [event]

    # ---- reporting ---------------------------------------------------------

    def correlation_matrix(self) -> tuple[list[str], np.ndarray]:
        """Per-symbol daily return correlation matrix."""
        syms = [s for s, st in self.symbols.items() if len(st.daily_returns) > 30]
        if len(syms) < 2:
            return syms, np.array([])
        min_len = min(len(self.symbols[s].daily_returns) for s in syms)
        mat = np.column_stack([
            np.array(self.symbols[s].daily_returns[-min_len:]) for s in syms
        ])
        corr = np.corrcoef(mat.T)
        return syms, corr

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

        total_ret = 0.0
        if len(equities) > 0:
            total_ret = (equities[-1] / self.cfg.initial_capital - 1) * 100

        margin_utils = [m for _, m in self.margin_history]
        avg_margin = float(np.mean(margin_utils)) if margin_utils else 0.0
        max_margin = float(np.max(margin_utils)) if margin_utils else 0.0

        return {
            "initial_capital": self.cfg.initial_capital,
            "final_equity": round(equities[-1], 2) if len(equities) else self.cfg.initial_capital,
            "total_return_pct": round(total_ret, 2),
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
            "margin_breaches": self.margin_breaches,
            "peak_simultaneous_assignments": self.peak_assignments,
            "avg_margin_utilization": round(avg_margin * 100, 1),
            "max_margin_utilization": round(max_margin * 100, 1),
            "start_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "end_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "days": len(dates),
        }
