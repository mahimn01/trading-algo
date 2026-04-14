"""
Hybrid Regime Strategy — Dynamic switching between Buy-and-Hold and Wheel

Regime detection via ADX + SMA slope + IV rank determines whether to:
  - STRONG_UPTREND: Hold stock (don't cap upside with covered calls)
  - WEAK_UPTREND: Run the Wheel normally
  - RANGE_BOUND: Run the Wheel aggressively (tighter delta, more premium)
  - DOWNTREND: Cash — close everything, wait for regime change

Stability filter: regime only changes after N consecutive days in new regime
to prevent whipsawing during transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from trading_algo.quant_core.strategies.options.wheel import (
    WheelConfig,
    WheelStrategy,
    ShortOptionLeg,
    TradeEvent,
    _price_option,
    _find_strike_by_delta,
    _round_strike,
)

Regime = Literal["STRONG_UPTREND", "WEAK_UPTREND", "RANGE_BOUND", "DOWNTREND"]


@dataclass(frozen=True)
class HybridRegimeConfig:
    initial_capital: float = 10_000.0
    put_delta: float = 0.30
    call_delta: float = 0.30
    target_dte: int = 45
    profit_target: float = 0.50
    roll_dte: int = 14
    stop_loss: float = 0.0
    min_iv_rank: float = 25.0
    min_premium_pct: float = 0.005
    contracts: int | None = None
    cash_reserve_pct: float = 0.20
    trend_sma_period: int = 50
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    skew_slope: float = 0.8
    commission_per_contract: float = 0.90
    commission_per_share: float = 0.005
    bid_ask_slip_per_share: float = 0.05

    # Regime detection
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    sma_period: int = 50
    sma_slope_window: int = 20
    regime_stability_days: int = 5

    # Regime-specific deltas
    uptrend_delta: float = 0.25
    range_delta: float = 0.35

    # Allow buying stock in strong uptrend
    allow_stock_purchase: bool = True

    def to_wheel_config(self, put_delta: float | None = None, call_delta: float | None = None) -> WheelConfig:
        return WheelConfig(
            initial_capital=self.initial_capital,
            put_delta=put_delta or self.put_delta,
            call_delta=call_delta or self.call_delta,
            target_dte=self.target_dte,
            profit_target=self.profit_target,
            roll_dte=self.roll_dte,
            stop_loss=self.stop_loss,
            min_iv_rank=self.min_iv_rank,
            min_premium_pct=self.min_premium_pct,
            contracts=self.contracts,
            cash_reserve_pct=self.cash_reserve_pct,
            trend_sma_period=0,  # we handle trend ourselves
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            skew_slope=self.skew_slope,
            commission_per_contract=self.commission_per_contract,
            commission_per_share=self.commission_per_share,
            bid_ask_slip_per_share=self.bid_ask_slip_per_share,
        )


def _compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
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

    # Wilder smoothing
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


class HybridRegimeStrategy:
    def __init__(self, cfg: HybridRegimeConfig | None = None):
        self.cfg = cfg or HybridRegimeConfig()
        self.cash: float = self.cfg.initial_capital
        self.stock_qty: int = 0
        self.stock_avg_cost: float = 0.0
        self.short_option: ShortOptionLeg | None = None
        self.phase: Literal["CSP", "CC", "HOLD", "CASH"] = "CASH"
        self._last_bar_date: datetime | None = None

        # Price history for regime detection
        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []

        # Regime tracking
        self._current_regime: Regime = "RANGE_BOUND"
        self._candidate_regime: Regime = "RANGE_BOUND"
        self._candidate_days: int = 0
        self.regime_history: list[tuple[datetime, Regime]] = []
        self._regime_returns: dict[Regime, list[float]] = {
            "STRONG_UPTREND": [], "WEAK_UPTREND": [], "RANGE_BOUND": [], "DOWNTREND": [],
        }
        self._regime_day_counts: dict[Regime, int] = {
            "STRONG_UPTREND": 0, "WEAK_UPTREND": 0, "RANGE_BOUND": 0, "DOWNTREND": 0,
        }
        self.transition_events: list[tuple[datetime, Regime, Regime]] = []

        # Internal Wheel for CSP/CC phases — created on demand
        self._wheel: WheelStrategy | None = None

        # Tracking
        self.events: list[TradeEvent] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.total_premium_collected: float = 0.0
        self.total_premium_paid: float = 0.0
        self.total_commissions: float = 0.0
        self.wheel_cycles: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self._prev_equity: float = self.cfg.initial_capital

    def _detect_regime(self, price: float, iv_rank: float) -> Regime:
        n = len(self._prices)
        sma_period = self.cfg.sma_period
        adx_period = self.cfg.adx_period
        slope_window = self.cfg.sma_slope_window

        # Need enough data
        if n < max(sma_period, adx_period * 3):
            return "RANGE_BOUND"

        # SMA
        sma = float(np.mean(self._prices[-sma_period:]))

        # SMA slope: compare current SMA to SMA from slope_window days ago
        if n >= sma_period + slope_window:
            sma_old = float(np.mean(self._prices[-(sma_period + slope_window):-slope_window]))
            sma_slope = (sma - sma_old) / sma_old
        else:
            sma_slope = 0.0

        # ADX
        adx = _compute_adx(
            np.array(self._highs[-adx_period * 3:]),
            np.array(self._lows[-adx_period * 3:]),
            np.array(self._prices[-adx_period * 3:]),
            adx_period,
        )

        # Price distance from SMA
        pct_from_sma = (price - sma) / sma if sma > 0 else 0.0

        # Classification
        if price > sma and sma_slope > 0 and adx > self.cfg.adx_trend_threshold:
            return "STRONG_UPTREND"
        elif price < sma and sma_slope < 0:
            return "DOWNTREND"
        elif adx < self.cfg.adx_range_threshold or abs(pct_from_sma) < 0.03:
            return "RANGE_BOUND"
        elif price > sma:
            return "WEAK_UPTREND"
        else:
            return "RANGE_BOUND"

    def _update_regime(self, date: datetime, price: float, iv_rank: float) -> bool:
        raw_regime = self._detect_regime(price, iv_rank)

        if raw_regime == self._current_regime:
            self._candidate_regime = raw_regime
            self._candidate_days = 0
            return False

        if raw_regime == self._candidate_regime:
            self._candidate_days += 1
        else:
            self._candidate_regime = raw_regime
            self._candidate_days = 1

        if self._candidate_days >= self.cfg.regime_stability_days:
            old = self._current_regime
            self._current_regime = raw_regime
            self._candidate_days = 0
            self.transition_events.append((date, old, raw_regime))
            return True

        return False

    def _sync_from_wheel(self) -> None:
        if self._wheel is None:
            return
        self.cash = self._wheel.cash
        self.stock_qty = self._wheel.stock_qty
        self.stock_avg_cost = self._wheel.stock_avg_cost
        self.short_option = self._wheel.short_option
        self.total_premium_collected = self._wheel.total_premium_collected
        self.total_premium_paid = self._wheel.total_premium_paid
        self.total_commissions = self._wheel.total_commissions
        self.wheel_cycles = self._wheel.wheel_cycles
        self.wins = self._wheel.wins
        self.losses = self._wheel.losses

    def _sync_to_wheel(self) -> None:
        if self._wheel is None:
            return
        self._wheel.cash = self.cash
        self._wheel.stock_qty = self.stock_qty
        self._wheel.stock_avg_cost = self.stock_avg_cost
        self._wheel.short_option = self.short_option
        self._wheel.total_premium_collected = self.total_premium_collected
        self._wheel.total_premium_paid = self.total_premium_paid
        self._wheel.total_commissions = self.total_commissions
        self._wheel.wheel_cycles = self.wheel_cycles
        self._wheel.wins = self.wins
        self._wheel.losses = self.losses

    def _ensure_wheel(self, put_delta: float | None = None, call_delta: float | None = None) -> WheelStrategy:
        wcfg = self.cfg.to_wheel_config(put_delta=put_delta, call_delta=call_delta)
        if self._wheel is None:
            self._wheel = WheelStrategy(wcfg)
            self._wheel._price_history = list(self._prices)
        else:
            self._wheel.cfg = wcfg
            self._wheel._price_history = list(self._prices)
        self._sync_to_wheel()
        return self._wheel

    def _close_short_option(self, date: datetime, price: float, iv: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        if self.short_option is None:
            return events

        opt = self.short_option
        dte = opt.dte(date)
        tte = max(dte / 365.0, 1 / 365)
        buyback_price = _price_option(
            price, opt.strike, tte, iv,
            self.cfg.risk_free_rate, opt.option_type,
            dividend_yield=self.cfg.dividend_yield,
            skew_slope=self.cfg.skew_slope,
        )
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

        events.append(TradeEvent(
            date=date,
            event_type=f"regime_close_{opt.option_type}",
            details={
                "strike": opt.strike,
                "entry_premium": round(opt.premium_per_share, 4),
                "buyback": round(buyback_price, 4),
                "underlying": round(price, 2),
                "reason": "regime_transition",
            },
            pnl=pnl,
        ))
        self.short_option = None
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

        pnl = (price - self.stock_avg_cost) * shares
        events.append(TradeEvent(
            date=date,
            event_type="regime_sell_stock",
            details={
                "shares": shares,
                "price": round(price, 2),
                "cost_basis": round(self.stock_avg_cost, 2),
            },
            pnl=pnl,
        ))
        self.stock_qty = 0
        self.stock_avg_cost = 0.0
        return events

    def _buy_stock(self, date: datetime, price: float) -> list[TradeEvent]:
        events: list[TradeEvent] = []
        if not self.cfg.allow_stock_purchase:
            return events
        if self.stock_qty > 0:
            return events

        usable = self.cash * (1 - self.cfg.cash_reserve_pct)
        shares = int(usable / (price + self.cfg.bid_ask_slip_per_share + self.cfg.commission_per_share))
        # Round down to nearest 100 for option compatibility
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
            event_type="regime_buy_stock",
            details={
                "shares": shares,
                "price": round(price, 2),
            },
        ))
        return events

    def _handle_transition(
        self, date: datetime, price: float, iv: float, old_regime: Regime, new_regime: Regime,
    ) -> list[TradeEvent]:
        events: list[TradeEvent] = []

        events.append(TradeEvent(
            date=date,
            event_type="regime_change",
            details={"from": old_regime, "to": new_regime, "price": round(price, 2)},
        ))

        # ANY -> DOWNTREND: close everything
        if new_regime == "DOWNTREND":
            events.extend(self._close_short_option(date, price, iv))
            events.extend(self._sell_stock(date, price))
            self.phase = "CASH"
            return events

        # STRONG_UPTREND -> WEAK_UPTREND: start selling calls if holding stock
        if old_regime == "STRONG_UPTREND" and new_regime in ("WEAK_UPTREND", "RANGE_BOUND"):
            if self.stock_qty > 0 and self.short_option is None:
                self.phase = "CC"
            return events

        # WEAK_UPTREND/RANGE_BOUND -> STRONG_UPTREND: buy back short call, hold
        if new_regime == "STRONG_UPTREND":
            if self.short_option and self.short_option.option_type == "call":
                events.extend(self._close_short_option(date, price, iv))
            if self.stock_qty > 0:
                self.phase = "HOLD"
            else:
                events.extend(self._buy_stock(date, price))
                self.phase = "HOLD" if self.stock_qty > 0 else "CSP"
            return events

        # DOWNTREND -> WEAK_UPTREND/RANGE_BOUND: re-enter via CSP
        if old_regime == "DOWNTREND":
            self.phase = "CSP"
            return events

        # WEAK_UPTREND <-> RANGE_BOUND: just adjust deltas, handled in on_bar
        return events

    def on_bar(
        self,
        date: datetime,
        price: float,
        iv: float,
        iv_rank: float,
        high: float | None = None,
        low: float | None = None,
    ) -> list[TradeEvent]:
        self._last_bar_date = date
        self._prices.append(price)
        self._highs.append(high if high is not None else price)
        self._lows.append(low if low is not None else price)
        bar_events: list[TradeEvent] = []

        # Record daily return for regime attribution
        if len(self._prices) >= 2:
            daily_ret = (self._prices[-1] / self._prices[-2]) - 1.0
        else:
            daily_ret = 0.0

        # Check regime change (with stability filter)
        old_regime = self._current_regime
        changed = self._update_regime(date, price, iv_rank)

        self.regime_history.append((date, self._current_regime))
        self._regime_day_counts[self._current_regime] += 1
        self._regime_returns[self._current_regime].append(daily_ret)

        if changed:
            bar_events.extend(self._handle_transition(date, price, iv, old_regime, self._current_regime))

        regime = self._current_regime

        # --- DOWNTREND: stay in cash ---
        if regime == "DOWNTREND":
            eq = self.get_equity(price, iv, as_of=date)
            self.equity_curve.append((date, eq))
            self._prev_equity = eq
            self.events.extend(bar_events)
            return bar_events

        # --- STRONG_UPTREND: hold stock, no options ---
        if regime == "STRONG_UPTREND":
            if self.stock_qty == 0 and self.short_option is None:
                bar_events.extend(self._buy_stock(date, price))
                if self.stock_qty > 0:
                    self.phase = "HOLD"
            eq = self.get_equity(price, iv, as_of=date)
            self.equity_curve.append((date, eq))
            self._prev_equity = eq
            self.events.extend(bar_events)
            return bar_events

        # --- WEAK_UPTREND / RANGE_BOUND: run the Wheel ---
        if regime == "WEAK_UPTREND":
            pd = self.cfg.uptrend_delta
            cd = self.cfg.uptrend_delta
        else:  # RANGE_BOUND
            pd = self.cfg.range_delta
            cd = self.cfg.range_delta

        wheel = self._ensure_wheel(put_delta=pd, call_delta=cd)

        # Set wheel phase based on our state
        if self.stock_qty > 0:
            wheel.phase = "CC"
        elif self.phase == "CASH" or self.phase == "HOLD":
            wheel.phase = "CSP"
        else:
            wheel.phase = self.phase  # type: ignore

        wheel_events = wheel.on_bar(date, price, iv, iv_rank)
        self._sync_from_wheel()

        # Update our phase based on wheel state
        if self.stock_qty > 0:
            self.phase = "CC"
        else:
            self.phase = "CSP"

        bar_events.extend(wheel_events)

        eq = self.get_equity(price, iv, as_of=date)
        self.equity_curve.append((date, eq))
        self._prev_equity = eq
        self.events.extend(bar_events)
        return bar_events

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
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

        total_days = sum(self._regime_day_counts.values()) or 1
        regime_pct = {r: round(c / total_days * 100, 1) for r, c in self._regime_day_counts.items()}

        regime_ret_summary: dict[str, float] = {}
        for r, rets in self._regime_returns.items():
            if rets:
                regime_ret_summary[f"{r}_ann_ret"] = round(float(np.mean(rets)) * 252 * 100, 2)
            else:
                regime_ret_summary[f"{r}_ann_ret"] = 0.0

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
            "regime_pct": regime_pct,
            "regime_transitions": len(self.transition_events),
            **regime_ret_summary,
        }
