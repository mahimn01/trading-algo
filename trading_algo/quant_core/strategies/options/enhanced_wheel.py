"""
Enhanced Wheel Strategy with 8 configurable alpha signals.

Wraps the base WheelStrategy and intercepts trade decisions with
additional filters and strike/size adjustments. Each signal can be
independently toggled so marginal value is measurable.

Signals:
    1. VIX Regime Overlay      — vol-percentile-based size & delta adjustment
    2. Earnings Avoidance      — skip trades when earnings fall within DTE
    3. Mean Reversion Entry    — RSI filter for put-selling timing
    4. Momentum Delta Adjust   — trend-aware strike selection
    5. Term Structure Signal   — short vs long realized vol ratio
    6. Volume Confirmation     — only trade when volume confirms participation
    7. Adaptive DTE            — IV-rank-based DTE selection
    8. Sector / Correlation    — limit correlated positions (portfolio-level)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from trading_algo.quant_core.strategies.options.wheel import (
    WheelConfig,
    WheelStrategy,
    ShortOptionLeg,
    TradeEvent,
    _find_strike_by_delta,
    _price_option,
    _round_strike,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnhancedWheelConfig:
    base: WheelConfig = field(default_factory=WheelConfig)

    # Signal 1 — VIX Regime Overlay
    vix_overlay: bool = True
    vix_lookback: int = 252
    vix_high_pctile: float = 80.0   # top 20th percentile = "high fear"
    vix_low_pctile: float = 20.0    # bottom 20th percentile = "low fear"
    vix_high_size_mult: float = 0.50
    vix_low_size_mult: float = 1.25
    vix_high_delta_override: float = 0.20
    vix_low_delta_override: float = 0.35

    # Signal 2 — Earnings Avoidance
    earnings_avoidance: bool = True
    earnings_interval_days: int = 63  # ~quarterly in trading days

    # Signal 3 — Mean Reversion (RSI)
    rsi_filter: bool = True
    rsi_period: int = 14
    rsi_oversold: float = 40.0
    rsi_overbought: float = 70.0

    # Signal 4 — Momentum Delta Adjustment
    momentum_delta_adjust: bool = True
    momentum_lookback: int = 20
    momentum_strong_up: float = 0.10    # +10%
    momentum_strong_down: float = -0.10 # -10%
    momentum_up_delta_add: float = 0.05
    momentum_down_delta_sub: float = 0.05

    # Signal 5 — Vol Term Structure
    term_structure_filter: bool = True
    ts_short_window: int = 10
    ts_long_window: int = 60
    ts_contango_threshold: float = 1.20  # short/long > 1.2 = contango (good)
    ts_backwardation_threshold: float = 0.80  # short/long < 0.8 = wait

    # Signal 6 — Volume Confirmation
    volume_filter: bool = True
    vol_short_window: int = 5
    vol_long_window: int = 20

    # Signal 7 — Adaptive DTE
    adaptive_dte: bool = True
    dte_high_iv: int = 30       # when IV rank > 70
    dte_mid_iv: int = 45        # when 30 <= IV rank <= 70
    dte_low_iv: int = 60        # when IV rank < 30
    dte_skip_low_iv: bool = False  # skip entirely when IV rank < 30

    # Signal 8 — Sector / Correlation Limit (portfolio-level)
    sector_limit: int = 2       # 0 = disabled


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI. Returns array same length as prices, NaN-padded."""
    out = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return out

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


def _realized_vol(prices: np.ndarray, window: int) -> np.ndarray:
    """Annualized realized vol (same as iv_rank module but standalone)."""
    out = np.full(len(prices), np.nan)
    if len(prices) < window + 1:
        return out
    log_ret = np.diff(np.log(prices))
    for i in range(window, len(log_ret) + 1):
        out[i] = float(np.std(log_ret[i - window: i], ddof=1) * np.sqrt(252))
    return out


def _vol_percentile(rv_series: np.ndarray, idx: int, lookback: int = 252) -> float:
    """What percentile is the current RV within the trailing lookback?"""
    start = max(0, idx - lookback)
    window = rv_series[start: idx + 1]
    window = window[~np.isnan(window)]
    if len(window) < 20:
        return 50.0
    current = window[-1]
    below = np.sum(window[:-1] < current)
    return float(below / (len(window) - 1) * 100.0)


# ---------------------------------------------------------------------------
# Enhanced Wheel
# ---------------------------------------------------------------------------

class EnhancedWheel:
    """
    Wraps WheelStrategy with 8 configurable alpha signals.

    Usage is identical to WheelStrategy — call on_bar() each day.
    The class intercepts the "should we open?" decision with signal
    filters and adjusts delta/size/DTE before delegating to the
    base strategy's mechanics.
    """

    def __init__(self, cfg: EnhancedWheelConfig | None = None, symbol: str = ""):
        self.cfg = cfg or EnhancedWheelConfig()
        self.symbol = symbol

        # Build the underlying strategy with base config
        self._base = WheelStrategy(self.cfg.base)

        # Price / volume history (fed each bar)
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._dates: list[datetime] = []

        # Pre-computed indicator caches (updated lazily)
        self._rsi_cache: np.ndarray = np.array([])
        self._rv30_cache: np.ndarray = np.array([])
        self._rv10_cache: np.ndarray = np.array([])
        self._rv60_cache: np.ndarray = np.array([])

        # Signal state log — populated on each bar where a decision is made
        self.signal_log: list[dict] = []

        # For sector limit (portfolio-level, set externally)
        self._correlation_matrix: dict[tuple[str, str], float] = {}
        self._active_symbols: set[str] = set()

    # ---- proxy properties to match OptionsStrategy protocol ----------------

    @property
    def events(self) -> list[TradeEvent]:
        return self._base.events

    @property
    def equity_curve(self) -> list[tuple[datetime, float]]:
        return self._base.equity_curve

    @property
    def phase(self) -> str:
        return self._base.phase

    @property
    def short_option(self) -> ShortOptionLeg | None:
        return self._base.short_option

    @property
    def cash(self) -> float:
        return self._base.cash

    @property
    def stock_qty(self) -> int:
        return self._base.stock_qty

    # ---- indicator computation ---------------------------------------------

    def _update_indicators(self) -> None:
        prices = np.array(self._prices, dtype=float)
        n = len(prices)

        if self.cfg.rsi_filter and n >= self.cfg.rsi_period + 2:
            self._rsi_cache = _rsi(prices, self.cfg.rsi_period)

        if self.cfg.vix_overlay and n >= self.cfg.vix_lookback + 2:
            self._rv30_cache = _realized_vol(prices, 30)

        if self.cfg.term_structure_filter:
            if n >= self.cfg.ts_long_window + 2:
                self._rv10_cache = _realized_vol(prices, self.cfg.ts_short_window)
                self._rv60_cache = _realized_vol(prices, self.cfg.ts_long_window)

    def _current_rsi(self) -> float | None:
        if len(self._rsi_cache) == 0 or len(self._rsi_cache) <= len(self._prices) - 1:
            return None
        val = self._rsi_cache[len(self._prices) - 1]
        return None if np.isnan(val) else float(val)

    def _current_rv30(self) -> float | None:
        if len(self._rv30_cache) == 0:
            return None
        val = self._rv30_cache[len(self._prices) - 1]
        return None if np.isnan(val) else float(val)

    def _current_vol_percentile(self) -> float | None:
        if len(self._rv30_cache) == 0:
            return None
        idx = len(self._prices) - 1
        val = self._rv30_cache[idx]
        if np.isnan(val):
            return None
        return _vol_percentile(self._rv30_cache, idx, self.cfg.vix_lookback)

    def _current_term_structure_ratio(self) -> float | None:
        idx = len(self._prices) - 1
        if len(self._rv10_cache) == 0 or len(self._rv60_cache) == 0:
            return None
        short = self._rv10_cache[idx]
        long_ = self._rv60_cache[idx]
        if np.isnan(short) or np.isnan(long_) or long_ < 0.01:
            return None
        return float(short / long_)

    def _current_momentum(self) -> float | None:
        n = len(self._prices)
        lb = self.cfg.momentum_lookback
        if n < lb + 1:
            return None
        return (self._prices[-1] / self._prices[-lb - 1]) - 1.0

    def _current_volume_ratio(self) -> float | None:
        n = len(self._volumes)
        if n < self.cfg.vol_long_window:
            return None
        short_avg = float(np.mean(self._volumes[-self.cfg.vol_short_window:]))
        long_avg = float(np.mean(self._volumes[-self.cfg.vol_long_window:]))
        if long_avg <= 0:
            return None
        return short_avg / long_avg

    def _is_near_earnings(self, dte: int) -> bool:
        """Approximate whether earnings fall within the DTE window."""
        n_bars = len(self._prices)
        if n_bars < self.cfg.earnings_interval_days:
            return False
        # Estimate last earnings as most recent multiple of interval
        last_earnings_bar = n_bars - (n_bars % self.cfg.earnings_interval_days)
        next_earnings_bar = last_earnings_bar + self.cfg.earnings_interval_days
        bars_until_earnings = next_earnings_bar - n_bars
        # Convert DTE (calendar days) to approx trading days
        dte_trading = int(dte * 5 / 7)
        return bars_until_earnings <= dte_trading

    # ---- signal evaluation -------------------------------------------------

    def _evaluate_signals(
        self, price: float, iv: float, iv_rank: float, dte: int,
    ) -> tuple[bool, dict]:
        """
        Evaluate all enabled signals. Returns (should_trade, signal_state).
        signal_state contains all computed values + per-signal pass/fail.
        """
        state: dict = {"date": self._dates[-1].strftime("%Y-%m-%d") if self._dates else ""}
        should_trade = True

        # --- Signal 1: VIX Regime Overlay ---
        vol_pctile = self._current_vol_percentile()
        state["vol_percentile"] = round(vol_pctile, 1) if vol_pctile is not None else None
        state["vix_regime"] = "normal"
        if self.cfg.vix_overlay and vol_pctile is not None:
            if vol_pctile >= self.cfg.vix_high_pctile:
                state["vix_regime"] = "high_fear"
            elif vol_pctile <= self.cfg.vix_low_pctile:
                state["vix_regime"] = "low_fear"
        state["signal_1_pass"] = True  # VIX overlay adjusts but doesn't block

        # --- Signal 2: Earnings Avoidance ---
        near_earnings = self._is_near_earnings(dte)
        state["near_earnings"] = near_earnings
        if self.cfg.earnings_avoidance and near_earnings:
            state["signal_2_pass"] = False
            should_trade = False
        else:
            state["signal_2_pass"] = True

        # --- Signal 3: RSI Filter ---
        rsi_val = self._current_rsi()
        state["rsi"] = round(rsi_val, 1) if rsi_val is not None else None
        if self.cfg.rsi_filter and rsi_val is not None:
            if rsi_val > self.cfg.rsi_overbought:
                state["signal_3_pass"] = False
                should_trade = False
            elif rsi_val > self.cfg.rsi_oversold:
                # Between oversold and overbought — neutral, allow but not ideal
                state["signal_3_pass"] = True
            else:
                # RSI < oversold threshold — ideal entry
                state["signal_3_pass"] = True
        else:
            state["signal_3_pass"] = True

        # --- Signal 4: Momentum Delta Adjust ---
        momentum = self._current_momentum()
        state["momentum_20d"] = round(momentum, 4) if momentum is not None else None
        state["signal_4_pass"] = True  # adjusts delta, doesn't block

        # --- Signal 5: Term Structure ---
        ts_ratio = self._current_term_structure_ratio()
        state["term_structure_ratio"] = round(ts_ratio, 3) if ts_ratio is not None else None
        if self.cfg.term_structure_filter and ts_ratio is not None:
            if ts_ratio < self.cfg.ts_backwardation_threshold:
                state["signal_5_pass"] = False
                should_trade = False
            else:
                state["signal_5_pass"] = True
        else:
            state["signal_5_pass"] = True

        # --- Signal 6: Volume Confirmation ---
        vol_ratio = self._current_volume_ratio()
        state["volume_ratio"] = round(vol_ratio, 3) if vol_ratio is not None else None
        if self.cfg.volume_filter and vol_ratio is not None:
            if vol_ratio < 1.0:
                state["signal_6_pass"] = False
                should_trade = False
            else:
                state["signal_6_pass"] = True
        else:
            state["signal_6_pass"] = True

        # --- Signal 7: Adaptive DTE ---
        state["iv_rank"] = round(iv_rank, 1)
        if self.cfg.adaptive_dte:
            if iv_rank < 30 and self.cfg.dte_skip_low_iv:
                state["signal_7_pass"] = False
                should_trade = False
            else:
                state["signal_7_pass"] = True
        else:
            state["signal_7_pass"] = True

        # --- Signal 8: Sector Limit (checked externally) ---
        state["signal_8_pass"] = True  # placeholder; portfolio-level check below

        return should_trade, state

    def _adjusted_delta(self, base_delta: float, state: dict) -> float:
        """Apply VIX and momentum adjustments to the target delta."""
        delta = base_delta

        # VIX regime adjustment
        if self.cfg.vix_overlay:
            regime = state.get("vix_regime", "normal")
            if regime == "high_fear":
                delta = self.cfg.vix_high_delta_override
            elif regime == "low_fear":
                delta = self.cfg.vix_low_delta_override

        # Momentum adjustment
        if self.cfg.momentum_delta_adjust:
            mom = state.get("momentum_20d")
            if mom is not None:
                if mom > self.cfg.momentum_strong_up:
                    delta = min(delta + self.cfg.momentum_up_delta_add, 0.50)
                elif mom < self.cfg.momentum_strong_down:
                    delta = max(delta - self.cfg.momentum_down_delta_sub, 0.10)

        return delta

    def _adjusted_dte(self, iv_rank: float) -> int:
        """Return DTE based on IV rank when adaptive DTE is enabled."""
        if not self.cfg.adaptive_dte:
            return self.cfg.base.target_dte
        if iv_rank > 70:
            return self.cfg.dte_high_iv
        if iv_rank >= 30:
            return self.cfg.dte_mid_iv
        return self.cfg.dte_low_iv

    def _adjusted_contracts(self, base_contracts: int, state: dict) -> int:
        """Apply VIX-based position sizing."""
        if not self.cfg.vix_overlay:
            return base_contracts
        regime = state.get("vix_regime", "normal")
        if regime == "high_fear":
            return max(int(base_contracts * self.cfg.vix_high_size_mult), 0)
        if regime == "low_fear":
            return max(int(base_contracts * self.cfg.vix_low_size_mult), base_contracts)
        return base_contracts

    # ---- main entry point --------------------------------------------------

    def on_bar(
        self,
        date: datetime,
        price: float,
        iv: float,
        iv_rank: float,
        volume: float = 0.0,
    ) -> list[TradeEvent]:
        """
        Process one trading day. Drop-in replacement for WheelStrategy.on_bar
        with an extra optional `volume` parameter.
        """
        self._prices.append(price)
        self._volumes.append(volume)
        self._dates.append(date)

        # Update indicators every bar (cheap — numpy vectorized)
        self._update_indicators()

        # --- If we have an open position, delegate management to base ---
        # We intercept only the OPEN decision, not management.
        # Sync the base's price history for its trend filter
        self._base._price_history.append(price)
        self._base._last_bar_date = date

        bar_events: list[TradeEvent] = []

        # 1. Handle expiration
        if self._base.short_option and date >= self._base.short_option.expiry:
            bar_events.extend(self._base._handle_expiration(date, price, iv))

        # 2. Manage open position
        elif self._base.short_option:
            bar_events.extend(self._base._manage_position(date, price, iv))

        # 3. Open new position with enhanced signals
        if self._base.short_option is None:
            bar_events.extend(self._enhanced_open(date, price, iv, iv_rank))

        # Record equity
        eq = self._base.get_equity(price, iv, as_of=date)
        self._base.equity_curve.append((date, eq))
        self._base.events.extend(bar_events)

        return bar_events

    def _enhanced_open(
        self, date: datetime, price: float, iv: float, iv_rank: float,
    ) -> list[TradeEvent]:
        """Enhanced position opening with all signal filters."""
        events: list[TradeEvent] = []

        # Base IV rank filter still applies
        if iv_rank < self.cfg.base.min_iv_rank:
            return events

        # Base trend filter still applies
        if self.cfg.base.trend_sma_period > 0 and self._base.phase == "CSP":
            if len(self._base._price_history) >= self.cfg.base.trend_sma_period:
                sma = float(np.mean(
                    self._base._price_history[-self.cfg.base.trend_sma_period:]
                ))
                if price < sma:
                    return events

        # Determine effective DTE
        dte = self._adjusted_dte(iv_rank)

        # Evaluate all enhanced signals
        should_trade, state = self._evaluate_signals(price, iv, iv_rank, dte)

        if not should_trade:
            triggered = [k for k in state if k.startswith("signal_") and k.endswith("_pass") and not state[k]]
            log.debug(
                "Skipping trade on %s: signals blocked: %s",
                date.strftime("%Y-%m-%d"),
                triggered,
            )
            self.signal_log.append(state)
            return events

        tte_years = dte / 365.0

        if self._base.phase == "CSP":
            # Adjusted delta
            put_delta = self._adjusted_delta(self.cfg.base.put_delta, state)
            state["effective_delta"] = round(put_delta, 3)
            state["effective_dte"] = dte

            strike = _find_strike_by_delta(
                price, put_delta, tte_years, iv,
                self.cfg.base.risk_free_rate, "put",
            )

            # Base sizing
            usable_cash = self._base.cash * (1 - self.cfg.base.cash_reserve_pct)
            base_contracts = (
                self.cfg.base.contracts
                if self.cfg.base.contracts is not None
                else max(int(usable_cash / (strike * 100)), 0)
            )
            contracts = self._adjusted_contracts(base_contracts, state)
            state["base_contracts"] = base_contracts
            state["adjusted_contracts"] = contracts

            if contracts <= 0:
                self.signal_log.append(state)
                return events

            premium = _price_option(
                price, strike, tte_years, iv,
                self.cfg.base.risk_free_rate, "put",
            )
            premium = max(premium - self.cfg.base.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.base.min_premium_pct:
                self.signal_log.append(state)
                return events

            expiry = date + timedelta(days=dte)
            commission = self.cfg.base.commission_per_contract * contracts

            self._base.short_option = ShortOptionLeg(
                option_type="put", strike=strike, expiry=expiry,
                premium_per_share=premium, contracts=contracts,
                entry_date=date, entry_underlying=price,
            )
            self._base.cash += premium * contracts * 100 - commission
            self._base.total_premium_collected += premium * contracts * 100
            self._base.total_commissions += commission

            state["action"] = "sell_put"
            self.signal_log.append(state)

            events.append(TradeEvent(
                date=date, event_type="sell_put",
                details={
                    "strike": strike,
                    "dte": dte,
                    "premium": round(premium, 4),
                    "contracts": contracts,
                    "underlying": round(price, 2),
                    "iv": round(iv, 4),
                    "iv_rank": round(iv_rank, 1),
                    "effective_delta": round(put_delta, 3),
                    "signals": {k: v for k, v in state.items() if k.startswith("signal_")},
                },
            ))

        elif self._base.phase == "CC":
            # Covered call — less signal filtering (we already own shares)
            call_delta = self._adjusted_delta(self.cfg.base.call_delta, state)
            min_strike = self._base.stock_avg_cost
            strike = _find_strike_by_delta(
                price, call_delta, tte_years, iv,
                self.cfg.base.risk_free_rate, "call",
            )
            strike = max(strike, _round_strike(min_strike, price))
            contracts = self._base.stock_qty // 100
            if contracts <= 0:
                self.signal_log.append(state)
                return events

            premium = _price_option(
                price, strike, tte_years, iv,
                self.cfg.base.risk_free_rate, "call",
            )
            premium = max(premium - self.cfg.base.bid_ask_slip_per_share, 0.01)

            if premium / price < self.cfg.base.min_premium_pct:
                self.signal_log.append(state)
                return events

            expiry = date + timedelta(days=dte)
            commission = self.cfg.base.commission_per_contract * contracts

            self._base.short_option = ShortOptionLeg(
                option_type="call", strike=strike, expiry=expiry,
                premium_per_share=premium, contracts=contracts,
                entry_date=date, entry_underlying=price,
            )
            self._base.cash += premium * contracts * 100 - commission
            self._base.total_premium_collected += premium * contracts * 100
            self._base.total_commissions += commission

            state["action"] = "sell_call"
            self.signal_log.append(state)

            events.append(TradeEvent(
                date=date, event_type="sell_call",
                details={
                    "strike": strike,
                    "dte": dte,
                    "premium": round(premium, 4),
                    "contracts": contracts,
                    "underlying": round(price, 2),
                    "cost_basis": round(self._base.stock_avg_cost, 2),
                    "effective_delta": round(call_delta, 3),
                    "signals": {k: v for k, v in state.items() if k.startswith("signal_")},
                },
            ))

        return events

    # ---- delegated reporting -----------------------------------------------

    def get_equity(self, price: float, iv: float, as_of: datetime | None = None) -> float:
        return self._base.get_equity(price, iv, as_of=as_of)

    def summary(self) -> dict:
        s = self._base.summary()
        # Augment with signal stats
        total_decisions = len(self.signal_log)
        if total_decisions > 0:
            for sig_num in range(1, 9):
                key = f"signal_{sig_num}_pass"
                passed = sum(1 for entry in self.signal_log if entry.get(key, True))
                s[f"signal_{sig_num}_pass_rate"] = round(passed / total_decisions * 100, 1)
        s["signal_decisions"] = total_decisions
        s["enhanced"] = True
        return s


# ---------------------------------------------------------------------------
# Portfolio-level sector correlation tracker
# ---------------------------------------------------------------------------

class PortfolioCorrelationTracker:
    """
    Tracks rolling correlations between symbols for the sector limit signal.

    Usage:
        tracker = PortfolioCorrelationTracker(window=60)
        tracker.update("AAPL", price)
        tracker.update("MSFT", price)
        ...
        if tracker.can_open("GOOG", active_symbols, limit=2):
            # open position
    """

    def __init__(self, window: int = 60, threshold: float = 0.70):
        self.window = window
        self.threshold = threshold
        self._histories: dict[str, list[float]] = {}

    def update(self, symbol: str, price: float) -> None:
        if symbol not in self._histories:
            self._histories[symbol] = []
        self._histories[symbol].append(price)

    def correlation(self, sym_a: str, sym_b: str) -> float | None:
        ha = self._histories.get(sym_a, [])
        hb = self._histories.get(sym_b, [])
        n = min(len(ha), len(hb), self.window)
        if n < 20:
            return None
        a = np.array(ha[-n:])
        b = np.array(hb[-n:])
        ret_a = np.diff(np.log(a))
        ret_b = np.diff(np.log(b))
        if len(ret_a) < 10:
            return None
        corr = float(np.corrcoef(ret_a, ret_b)[0, 1])
        return corr if np.isfinite(corr) else None

    def can_open(self, symbol: str, active_symbols: set[str], limit: int = 2) -> bool:
        if limit <= 0:
            return True
        correlated_active = 0
        for active in active_symbols:
            if active == symbol:
                continue
            corr = self.correlation(symbol, active)
            if corr is not None and corr > self.threshold:
                correlated_active += 1
        return correlated_active < limit
