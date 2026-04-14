"""Phase 2: PPO RL fine-tuning for ATLAS with real BSM options mechanics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.features import ATLASFeatureComputer, RollingNormalizer
from trading_algo.quant_core.strategies.options.iv_rank import iv_series_from_prices, iv_rank
from trading_algo.quant_core.strategies.options.wheel import (
    _find_strike_by_delta,
    _price_option,
    _round_strike,
)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    features: list[Tensor] = field(default_factory=list)
    timestamps: list[Tensor] = field(default_factory=list)
    dow: list[Tensor] = field(default_factory=list)
    month: list[Tensor] = field(default_factory=list)
    is_opex: list[Tensor] = field(default_factory=list)
    is_qtr: list[Tensor] = field(default_factory=list)
    pre_mu: list[Tensor] = field(default_factory=list)
    pre_sigma: list[Tensor] = field(default_factory=list)
    rtg: list[Tensor] = field(default_factory=list)
    actions: list[Tensor] = field(default_factory=list)
    log_probs: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Options environment with real BSM pricing
# ---------------------------------------------------------------------------

_RISK_FREE_RATE: float = 0.045
_SKEW_SLOPE: float = 0.8
_SLIPPAGE_PER_SHARE: float = 0.05
_COMMISSION_PER_CONTRACT: float = 0.90
_COMMISSION_PER_SHARE: float = 0.005
_INITIAL_CAPITAL: float = 100_000.0
_MIN_EQUITY_FRACTION: float = 0.50  # episode ends at 50% loss


class OptionsEnvironment:
    """Simulates single-symbol options trading with real BSM pricing for PPO rollouts.

    Action vector (5D, all in [0,1] from model, rescaled here):
        delta in [0, 0.50]:     strike selection (< 0.05 = no trade)
        direction in [-1, +1]:  < -0.3 sell put, > 0.3 buy call, else hold
        leverage in [0, 1]:     fraction of capital to deploy
        dte in [14, 90]:        days to expiration
        profit_target in [0,1]: 0 = let expire, 0.5 = close at 50% profit, etc.
    """

    def __init__(
        self,
        all_data: dict[str, dict],
        config: ATLASConfig,
        regime_filter: str = "all",
        reward_shaping: str = "none",
    ) -> None:
        """
        Args:
            all_data: sym -> {
                "closes": np.ndarray (float64),
                "ivs": np.ndarray (float64),
                "iv_ranks": np.ndarray (float64),
                "normed": np.ndarray (float32, T x 16),
                "mu": np.ndarray (float32, T x 16),
                "sigma": np.ndarray (float32, T x 16),
                "timestamps": np.ndarray (float64),
                "dow": np.ndarray (int32),
                "month": np.ndarray (int32),
            }
            config: ATLASConfig
            regime_filter: "all", "high_iv", "uptrend", "crash"
            reward_shaping: "none", "high_iv", "uptrend", "crash"
        """
        self.all_data = all_data
        self.config = config
        self.regime_filter = regime_filter
        self.reward_shaping = reward_shaping
        self.symbols = list(all_data.keys())
        self._episode_len = 500
        self._eligible_episodes: list[tuple[str, int]] = []
        self._build_eligible_episodes()
        self.reset()

    def _build_eligible_episodes(self) -> None:
        """Pre-compute list of (sym, start_idx) pairs that match the regime filter."""
        if self.regime_filter == "all":
            return

        L = self.config.context_len
        for sym, data in self.all_data.items():
            closes = data["closes"]
            iv_ranks = data["iv_ranks"]
            ivs = data["ivs"]
            T = len(closes)

            max_ep_len = min(self._episode_len, T - L - 60)
            if max_ep_len < 100:
                continue

            for start in range(L, max(L + 1, T - max_ep_len), 50):
                end = min(start + max_ep_len, T - 1)
                window_closes = closes[start:end]
                window_iv_ranks = iv_ranks[start:end]
                window_ivs = ivs[start:end]

                if len(window_closes) < 30:
                    continue

                if self.regime_filter == "high_iv":
                    mean_ivr = float(np.nanmean(window_iv_ranks))
                    if mean_ivr > 60:
                        self._eligible_episodes.append((sym, start))

                elif self.regime_filter == "uptrend":
                    if len(window_closes) >= 60:
                        ret_60 = (window_closes[59] - window_closes[0]) / max(window_closes[0], 1e-8)
                        sma50 = float(np.mean(window_closes[:50]))
                        if ret_60 > 0.15 and window_closes[59] > sma50:
                            self._eligible_episodes.append((sym, start))

                elif self.regime_filter == "crash":
                    log_rets = np.diff(np.log(np.maximum(window_closes[:30], 1e-8)))
                    rv_30 = float(np.std(log_rets) * np.sqrt(252)) if len(log_rets) > 1 else 0.0
                    ret_20 = (window_closes[min(19, len(window_closes) - 1)] - window_closes[0]) / max(window_closes[0], 1e-8) if len(window_closes) >= 20 else 0.0
                    if rv_30 > 0.30 or ret_20 < -0.08:  # broader: was (0.40, -0.10)
                        self._eligible_episodes.append((sym, start))

        if not self._eligible_episodes:
            for sym in self.symbols:
                data = self.all_data[sym]
                T = len(data["closes"])
                max_ep_len = min(self._episode_len, T - L - 60)
                if max_ep_len >= 100:
                    mid = L + (T - L - max_ep_len) // 2
                    self._eligible_episodes.append((sym, mid))

    def reset(self) -> dict[str, Tensor]:
        if self.regime_filter != "all" and self._eligible_episodes:
            sym, start = random.choice(self._eligible_episodes)
            data = self.all_data[sym]
            T = len(data["closes"])
            max_ep_len = min(self._episode_len, T - start - 1)
            if max_ep_len < 100:
                max_ep_len = 100
        else:
            sym = random.choice(self.symbols)
            data = self.all_data[sym]
            T = len(data["closes"])
            max_ep_len = min(self._episode_len, T - self.config.context_len - 60)
            if max_ep_len < 100:
                max_ep_len = 100
            start = random.randint(
                self.config.context_len,
                max(self.config.context_len, T - max_ep_len - 1),
            )
        self._data = data
        self._sym = sym
        self._t = start
        self._start = start
        self._end = min(start + max_ep_len, T - 1)

        # Underlying price arrays (float64 for financial math)
        self._closes: np.ndarray = data["closes"]
        self._ivs: np.ndarray = data["ivs"]
        self._iv_ranks: np.ndarray = data["iv_ranks"]

        # Position state (all float64)
        self.position_type: str = "none"  # "none", "short_put", "short_call", "long_call", "stock"
        self.position_strike: float = 0.0
        self.position_expiry_days_left: int = 0
        self.position_entry_premium: float = 0.0
        self.position_contracts: int = 0
        self.position_entry_dte: int = 0
        self.profit_target: float = 0.0

        self.stock_qty: int = 0
        self.stock_avg_cost: float = 0.0
        self.cash: float = _INITIAL_CAPITAL

        # Equity tracking
        price = float(self._closes[self._t])
        iv = float(self._ivs[self._t])
        self._prev_equity = self._compute_equity(price, iv)
        self._peak = self._prev_equity
        self._target_sharpe = random.uniform(0.3, 2.0)

        return self._get_obs()

    # ---- observation -------------------------------------------------------

    def _get_obs(self) -> dict[str, Tensor]:
        L = self.config.context_len
        t = self._t

        normed = self._data["normed"]
        mu = self._data["mu"]
        sigma = self._data["sigma"]
        timestamps = self._data["timestamps"]
        dow = self._data["dow"]
        month = self._data["month"]

        window_normed = normed[t - L + 1: t + 1].copy()  # (L, 16)
        window_mu = mu[t - L + 1: t + 1].copy()
        window_sigma = sigma[t - L + 1: t + 1].copy()
        window_ts = timestamps[t - L + 1: t + 1].copy()
        window_dow = dow[t - L + 1: t + 1].copy()
        window_month = month[t - L + 1: t + 1].copy()

        # Inject position state into features 12-15 of the last timestep
        price = float(self._closes[t])
        iv = float(self._ivs[t])
        equity = self._compute_equity(price, iv)

        # pos_state: -1 (short option), 0 (none), +1 (long option/stock)
        if self.position_type in ("short_put", "short_call"):
            pos_state = -1.0
        elif self.position_type in ("long_call", "stock"):
            pos_state = 1.0
        else:
            pos_state = 0.0

        # pos_pnl: unrealized P&L as fraction of capital
        pos_pnl = (equity - _INITIAL_CAPITAL) / _INITIAL_CAPITAL

        # days_in: days remaining / entry DTE (0 if no position)
        if self.position_type != "none" and self.position_entry_dte > 0:
            days_in = float(self.position_expiry_days_left) / float(self.position_entry_dte)
        elif self.position_type == "stock":
            days_in = 1.0
        else:
            days_in = 0.0

        # cash_pct: cash / total equity
        cash_pct = self.cash / max(equity, 1.0)

        # Write position state into the last row of the window (features 12-15)
        window_normed[-1, 12] = np.float32(pos_state)
        window_normed[-1, 13] = np.float32(np.clip(pos_pnl, -2.0, 2.0))
        window_normed[-1, 14] = np.float32(np.clip(days_in, 0.0, 1.0))
        window_normed[-1, 15] = np.float32(np.clip(cash_pct, 0.0, 1.0))

        # Replace NaN with 0
        window_normed = np.nan_to_num(window_normed, nan=0.0)
        window_mu = np.nan_to_num(window_mu, nan=0.0)
        window_sigma = np.nan_to_num(window_sigma, nan=1.0)

        features = torch.tensor(window_normed, dtype=torch.float32).unsqueeze(0)
        ts = torch.tensor(window_ts, dtype=torch.float32).unsqueeze(0)
        dow_t = torch.tensor(window_dow, dtype=torch.long).unsqueeze(0)
        mo_t = torch.tensor(window_month, dtype=torch.long).unsqueeze(0)
        opex = torch.zeros(1, L)
        qtr = torch.zeros(1, L)
        mu_t = torch.tensor(window_mu, dtype=torch.float32).unsqueeze(0)
        sigma_t = torch.tensor(window_sigma, dtype=torch.float32).unsqueeze(0)
        rtg = torch.tensor([self._target_sharpe])

        return {
            "features": features, "timestamps": ts, "dow": dow_t, "month": mo_t,
            "is_opex": opex, "is_qtr": qtr, "pre_mu": mu_t, "pre_sigma": sigma_t, "rtg": rtg,
        }

    # ---- crash detection for adaptive stop-loss -----------------------------

    def _is_crash_active(self) -> bool:
        """Detect crash: extreme realized vol OR sharp drawdown.

        Thresholds are intentionally aggressive to avoid false positives
        during normal high-IV environments (where positions should hold).
        RV > 0.55 = ~2 sigma event, ret10 < -0.12 = genuine crash.
        """
        t = self._t
        closes = self._closes
        # 10-day realized vol (shorter = faster reaction)
        rv_win = min(10, t)
        if rv_win >= 5:
            seg = closes[max(0, t - rv_win): t + 1]
            lr = np.diff(np.log(np.maximum(seg, 1e-8)))
            if len(lr) > 1:
                rv = float(np.std(lr) * np.sqrt(252))
                if rv > 0.55:
                    return True
        # 10-day return (shorter window, stricter threshold)
        if t >= 10:
            ret10 = (closes[t] - closes[t - 10]) / max(closes[t - 10], 1e-8)
            if ret10 < -0.12:
                return True
        return False

    # ---- equity computation ------------------------------------------------

    def _compute_equity(self, price: float, iv: float) -> float:
        equity = self.cash + self.stock_qty * price

        if self.position_type == "short_put":
            tte = max(self.position_expiry_days_left / 365.0, 1.0 / 365.0)
            current_val = _price_option(
                price, self.position_strike, tte, iv,
                _RISK_FREE_RATE, "put",
                skew_slope=_SKEW_SLOPE,
            )
            equity -= current_val * self.position_contracts * 100

        elif self.position_type == "short_call":
            tte = max(self.position_expiry_days_left / 365.0, 1.0 / 365.0)
            current_val = _price_option(
                price, self.position_strike, tte, iv,
                _RISK_FREE_RATE, "call",
                skew_slope=_SKEW_SLOPE,
            )
            equity -= current_val * self.position_contracts * 100

        elif self.position_type == "long_call":
            tte = max(self.position_expiry_days_left / 365.0, 1.0 / 365.0)
            current_val = _price_option(
                price, self.position_strike, tte, iv,
                _RISK_FREE_RATE, "call",
                skew_slope=_SKEW_SLOPE,
            )
            equity += current_val * self.position_contracts * 100

        return equity

    # ---- opening positions -------------------------------------------------

    def _open_position(self, action: np.ndarray, price: float, iv: float) -> None:
        delta_raw = float(action[0])
        direction = float(action[1])
        leverage = float(action[2])
        dte_raw = float(action[3])
        profit_target = float(action[4])

        # Rescale from model output [0,1] to actual ranges
        delta = np.clip(delta_raw, 0.0, 0.50)
        dte = int(np.clip(dte_raw * (self.config.dte_max - self.config.dte_min) + self.config.dte_min, self.config.dte_min, self.config.dte_max))
        leverage = np.clip(leverage, 0.0, 1.0)
        profit_target = np.clip(profit_target, 0.0, 1.0)

        if delta < 0.05:
            return  # no trade

        tte_years = dte / 365.0

        if direction < -0.3:
            # SELL PUT
            if iv <= 0.01:
                return
            strike = _find_strike_by_delta(
                price, delta, tte_years, iv,
                _RISK_FREE_RATE, "put",
            )
            if strike <= 0:
                return

            collateral_per_contract = strike * 100.0
            deployable = self.cash * leverage * 0.50
            contracts = int(deployable / collateral_per_contract)
            if contracts <= 0:
                return

            premium_per_share = _price_option(
                price, strike, tte_years, iv,
                _RISK_FREE_RATE, "put",
                skew_slope=_SKEW_SLOPE,
            )
            premium_per_share = max(premium_per_share - _SLIPPAGE_PER_SHARE, 0.01)
            commission = _COMMISSION_PER_CONTRACT * contracts

            # Collect premium
            self.cash += premium_per_share * contracts * 100 - commission

            self.position_type = "short_put"
            self.position_strike = strike
            self.position_expiry_days_left = dte
            self.position_entry_premium = premium_per_share
            self.position_contracts = contracts
            self.position_entry_dte = dte
            self.profit_target = profit_target

        elif direction > 0.3:
            # BUY CALL (LEAPS-like directional)
            if iv <= 0.01:
                return
            strike = _find_strike_by_delta(
                price, delta, tte_years, iv,
                _RISK_FREE_RATE, "call",
            )
            if strike <= 0:
                return

            premium_per_share = _price_option(
                price, strike, tte_years, iv,
                _RISK_FREE_RATE, "call",
                skew_slope=_SKEW_SLOPE,
            )
            premium_per_share += _SLIPPAGE_PER_SHARE  # pay at the ask
            if premium_per_share <= 0:
                return

            premium_per_contract = premium_per_share * 100.0
            deployable = self.cash * leverage * 0.50
            contracts = int(deployable / premium_per_contract)
            if contracts <= 0:
                return

            cost = premium_per_contract * contracts
            commission = _COMMISSION_PER_CONTRACT * contracts
            self.cash -= cost + commission

            self.position_type = "long_call"
            self.position_strike = strike
            self.position_expiry_days_left = dte
            self.position_entry_premium = premium_per_share
            self.position_contracts = contracts
            self.position_entry_dte = dte
            self.profit_target = profit_target

        elif self.position_type == "stock":
            # If holding stock from assignment, can sell covered call
            if self.stock_qty >= 100:
                if iv <= 0.01:
                    return
                cc_contracts = self.stock_qty // 100
                cc_strike = _find_strike_by_delta(
                    price, max(delta, 0.20), tte_years, iv,
                    _RISK_FREE_RATE, "call",
                )
                # Never sell call below cost basis
                cc_strike = max(cc_strike, _round_strike(self.stock_avg_cost, price))
                if cc_strike <= 0:
                    return

                premium_per_share = _price_option(
                    price, cc_strike, tte_years, iv,
                    _RISK_FREE_RATE, "call",
                    skew_slope=_SKEW_SLOPE,
                )
                premium_per_share = max(premium_per_share - _SLIPPAGE_PER_SHARE, 0.01)
                commission = _COMMISSION_PER_CONTRACT * cc_contracts

                self.cash += premium_per_share * cc_contracts * 100 - commission

                self.position_type = "short_call"
                self.position_strike = cc_strike
                self.position_expiry_days_left = dte
                self.position_entry_premium = premium_per_share
                self.position_contracts = cc_contracts
                self.position_entry_dte = dte
                self.profit_target = profit_target

    # ---- daily step --------------------------------------------------------

    def step(self, action: np.ndarray) -> tuple[dict[str, Tensor], float, bool]:
        self._t += 1
        if self._t >= len(self._closes) or self._t >= self._end:
            return self._get_obs(), 0.0, True

        price = float(self._closes[self._t])
        iv = float(self._ivs[self._t])
        just_closed = False

        # --- Manage existing option positions ---
        if self.position_type == "short_put":
            self.position_expiry_days_left -= 1

            if self.position_expiry_days_left <= 0:
                # EXPIRY
                intrinsic = max(self.position_strike - price, 0.0)
                if intrinsic > 0:
                    # Assignment: buy shares at strike
                    shares = self.position_contracts * 100
                    cost = self.position_strike * shares
                    commission = _COMMISSION_PER_SHARE * shares
                    self.cash -= cost + commission
                    self.stock_qty += shares
                    self.stock_avg_cost = self.position_strike - self.position_entry_premium
                    self.position_type = "stock"
                else:
                    # Expired worthless: keep full premium
                    self.position_type = "none"
                self.position_strike = 0.0
                self.position_contracts = 0
                self.position_entry_premium = 0.0
                just_closed = True
            else:
                # Check profit target
                tte = max(self.position_expiry_days_left / 365.0, 1.0 / 365.0)
                current_value = _price_option(
                    price, self.position_strike, tte, iv,
                    _RISK_FREE_RATE, "put",
                    skew_slope=_SKEW_SLOPE,
                )
                if self.position_entry_premium > 0:
                    pnl_pct = (self.position_entry_premium - current_value) / self.position_entry_premium
                else:
                    pnl_pct = 0.0

                if self.profit_target > 0.05 and pnl_pct >= self.profit_target:
                    # Close: buy back the put at profit
                    buyback = current_value + _SLIPPAGE_PER_SHARE
                    self.cash -= buyback * self.position_contracts * 100
                    commission = _COMMISSION_PER_CONTRACT * self.position_contracts
                    self.cash -= commission
                    self.position_type = "none"
                    self.position_strike = 0.0
                    self.position_contracts = 0
                    self.position_entry_premium = 0.0
                    just_closed = True
                elif (self.position_entry_premium > 0.01
                        and current_value >= 3.0 * self.position_entry_premium
                        and self._is_crash_active()):
                    # Adaptive stop-loss: only fires during crash conditions (3× rule)
                    buyback = current_value + _SLIPPAGE_PER_SHARE
                    self.cash -= buyback * self.position_contracts * 100
                    commission = _COMMISSION_PER_CONTRACT * self.position_contracts
                    self.cash -= commission
                    self.position_type = "none"
                    self.position_strike = 0.0
                    self.position_contracts = 0
                    self.position_entry_premium = 0.0
                    just_closed = True

        elif self.position_type == "short_call":
            self.position_expiry_days_left -= 1

            if self.position_expiry_days_left <= 0:
                intrinsic = max(price - self.position_strike, 0.0)
                if intrinsic > 0:
                    # Called away: sell shares at strike
                    shares = self.position_contracts * 100
                    proceeds = self.position_strike * shares
                    commission = _COMMISSION_PER_SHARE * shares
                    self.cash += proceeds - commission
                    self.stock_qty -= shares
                    if self.stock_qty <= 0:
                        self.stock_qty = 0
                        self.stock_avg_cost = 0.0
                    self.position_type = "none" if self.stock_qty == 0 else "stock"
                else:
                    # Expired worthless: keep premium, still hold stock
                    self.position_type = "stock"
                self.position_strike = 0.0
                self.position_contracts = 0
                self.position_entry_premium = 0.0
                just_closed = True
            else:
                tte = max(self.position_expiry_days_left / 365.0, 1.0 / 365.0)
                current_value = _price_option(
                    price, self.position_strike, tte, iv,
                    _RISK_FREE_RATE, "call",
                    skew_slope=_SKEW_SLOPE,
                )
                if self.position_entry_premium > 0:
                    pnl_pct = (self.position_entry_premium - current_value) / self.position_entry_premium
                else:
                    pnl_pct = 0.0

                if self.profit_target > 0.05 and pnl_pct >= self.profit_target:
                    buyback = current_value + _SLIPPAGE_PER_SHARE
                    self.cash -= buyback * self.position_contracts * 100
                    commission = _COMMISSION_PER_CONTRACT * self.position_contracts
                    self.cash -= commission
                    self.position_type = "stock"
                    self.position_strike = 0.0
                    self.position_contracts = 0
                    self.position_entry_premium = 0.0
                    just_closed = True
                elif (self.position_entry_premium > 0.01
                        and current_value >= 3.0 * self.position_entry_premium
                        and self._is_crash_active()):
                    # Adaptive stop-loss on short call: only during crashes (3× rule)
                    buyback = current_value + _SLIPPAGE_PER_SHARE
                    self.cash -= buyback * self.position_contracts * 100
                    commission = _COMMISSION_PER_CONTRACT * self.position_contracts
                    self.cash -= commission
                    self.position_type = "stock"
                    self.position_strike = 0.0
                    self.position_contracts = 0
                    self.position_entry_premium = 0.0
                    just_closed = True

        elif self.position_type == "long_call":
            self.position_expiry_days_left -= 1

            if self.position_expiry_days_left <= 0:
                # Expiry
                intrinsic = max(price - self.position_strike, 0.0)
                if intrinsic > 0:
                    # Exercise: collect intrinsic value
                    self.cash += intrinsic * self.position_contracts * 100
                    commission = _COMMISSION_PER_SHARE * self.position_contracts * 100
                    self.cash -= commission
                # If OTM, premium was already paid — no action needed
                self.position_type = "none"
                self.position_strike = 0.0
                self.position_contracts = 0
                self.position_entry_premium = 0.0
                just_closed = True
            else:
                tte = max(self.position_expiry_days_left / 365.0, 1.0 / 365.0)
                current_value = _price_option(
                    price, self.position_strike, tte, iv,
                    _RISK_FREE_RATE, "call",
                    skew_slope=_SKEW_SLOPE,
                )
                if self.position_entry_premium > 0:
                    pnl_pct = (current_value - self.position_entry_premium) / self.position_entry_premium
                else:
                    pnl_pct = 0.0

                if self.profit_target > 0.05 and pnl_pct >= self.profit_target:
                    # Close: sell the call
                    sell_price = max(current_value - _SLIPPAGE_PER_SHARE, 0.0)
                    self.cash += sell_price * self.position_contracts * 100
                    commission = _COMMISSION_PER_CONTRACT * self.position_contracts
                    self.cash -= commission
                    self.position_type = "none"
                    self.position_strike = 0.0
                    self.position_contracts = 0
                    self.position_entry_premium = 0.0
                    just_closed = True

        # --- Open new position if flat and not just closed ---
        if self.position_type in ("none", "stock") and not just_closed:
            self._open_position(action, price, iv)

        # --- Compute reward ---
        new_equity = self._compute_equity(price, iv)
        if self._prev_equity > 0:
            daily_return = (new_equity - self._prev_equity) / self._prev_equity
        else:
            daily_return = 0.0
        reward = daily_return * 100.0  # scale to percentage points

        # Mild drawdown penalty
        self._peak = max(self._peak, new_equity)
        dd = (self._peak - new_equity) / max(self._peak, 1.0)
        if dd > 0.15:
            reward -= 0.5 * (dd - 0.15) ** 2

        # --- Curriculum reward shaping bonuses ---
        if self.reward_shaping != "none":
            delta_raw = float(action[0])
            direction_raw = float(action[1])
            iv_rank_now = float(self._iv_ranks[self._t])

            if self.reward_shaping == "high_iv":
                if iv_rank_now > 60 and delta_raw > 0.25:
                    reward += 0.5
            elif self.reward_shaping == "uptrend":
                if direction_raw > 0:
                    reward += 0.3
            elif self.reward_shaping == "crash":
                # Real-time crash detection: RV30 and 20-day return
                _t = self._t
                _rv = 0.0
                _rv_win = min(30, _t)
                if _rv_win >= 5:
                    _seg = self._closes[max(0, _t - _rv_win): _t + 1]
                    _lr = np.diff(np.log(np.maximum(_seg, 1e-8)))
                    if len(_lr) > 1:
                        _rv = float(np.std(_lr) * np.sqrt(252))
                _ret20 = 0.0
                if _t >= 20:
                    _ret20 = (self._closes[_t] - self._closes[_t - 20]) / max(self._closes[_t - 20], 1e-8)
                _crash_active = (_rv > 0.30) or (_ret20 < -0.10)

                if _crash_active:
                    if self.position_type == "none":
                        reward += 0.5   # flat in crash: positive signal
                    elif self.position_type in ("short_put", "short_call"):
                        reward -= 0.5   # holding short premium in crash: penalty
                    elif self.position_type == "long_call":
                        reward -= 0.3   # long calls get IV-crushed in crashes
                else:
                    if delta_raw < 0.05:
                        reward += 0.2   # mild bonus for caution when calm

        # Clip reward to prevent explosions
        reward = float(np.clip(reward, -5.0, 5.0))

        self._prev_equity = new_equity

        # Episode termination
        done = self._t >= self._end
        if new_equity < _INITIAL_CAPITAL * _MIN_EQUITY_FRACTION:
            done = True  # 50% loss = game over

        return self._get_obs(), reward, done


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    T = len(rewards)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val = float(values[t + 1]) if t + 1 < T and not dones[t] else 0.0
        delta = float(rewards[t]) + gamma * next_val - float(values[t])
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        advantages[t] = float(last_gae)
        returns[t] = float(advantages[t]) + float(values[t])

    return advantages, returns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data_v2(
    feature_dir: str = "data/atlas_features_v2",
    min_len: int = 400,
) -> dict[str, dict]:
    """Load pre-computed v2 features (which include closes/ivs/iv_ranks in the npz)."""
    feature_path = Path(feature_dir)
    all_data: dict[str, dict] = {}

    for npz_file in sorted(feature_path.glob("*_features.npz")):
        sym = npz_file.stem.replace("_features", "")
        npz = np.load(npz_file)

        closes = npz["closes"].astype(np.float64)
        T = len(closes)
        if T < min_len:
            continue

        all_data[sym] = {
            "closes": closes,
            "ivs": npz["ivs"].astype(np.float64),
            "iv_ranks": npz["iv_ranks"].astype(np.float64),
            "normed": npz["normed"],
            "mu": npz["mu"],
            "sigma": npz["sigma"],
            "timestamps": npz["timestamps"],
            "dow": npz["dow"],
            "month": npz["month"],
        }

    return all_data


def _load_training_data(
    feature_dir: str = "data/atlas_features",
    cache_dir: str = "data/atlas_cache",
) -> dict[str, dict]:
    """Load pre-computed features and raw price data for each symbol."""
    feature_path = Path(feature_dir)
    cache_path = Path(cache_dir)
    all_data: dict[str, dict] = {}

    for npz_file in sorted(feature_path.glob("*_features.npz")):
        sym = npz_file.stem.replace("_features", "")
        parquet_file = cache_path / f"{sym}.parquet"
        if not parquet_file.exists():
            continue

        npz = np.load(npz_file)
        df = pd.read_parquet(parquet_file)

        closes = df["Close"].values.astype(np.float64)
        T = len(closes)

        # Compute IV series from prices
        ivs = iv_series_from_prices(closes, rv_window=30)
        ivs = np.nan_to_num(ivs, nan=0.25).astype(np.float64)

        # Compute IV rank series
        iv_est = iv_series_from_prices(closes, rv_window=30)
        iv_ranks = np.full(T, 50.0, dtype=np.float64)
        for t in range(T):
            if not np.isnan(iv_est[t]):
                iv_ranks[t] = iv_rank(iv_est, t, lookback=252)

        normed = npz["normed"]  # (T, 16) float32
        mu = npz["mu"]          # (T, 16) float32
        sigma = npz["sigma"]    # (T, 16) float32
        timestamps = npz["timestamps"]  # (T,) float64
        dow = npz["dow"]        # (T,) int32
        month = npz["month"]    # (T,) int32

        # Verify lengths match
        min_len = min(T, len(normed))
        all_data[sym] = {
            "closes": closes[:min_len],
            "ivs": ivs[:min_len],
            "iv_ranks": iv_ranks[:min_len],
            "normed": normed[:min_len],
            "mu": mu[:min_len],
            "sigma": sigma[:min_len],
            "timestamps": timestamps[:min_len],
            "dow": dow[:min_len],
            "month": month[:min_len],
        }

    return all_data


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------

def train_ppo(
    model: nn.Module,
    train_features: dict[str, np.ndarray] | None = None,
    config: ATLASConfig | None = None,
    checkpoint_dir: str = "checkpoints/atlas",
    device: str = "auto",
    n_iterations: int = 500,
    rollout_steps: int = 2048,
    feature_dir: str = "data/atlas_features",
    cache_dir: str = "data/atlas_cache",
    regime_filter: str = "all",
    reward_shaping: str = "none",
    all_data_preloaded: dict[str, dict] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Phase 2: PPO RL fine-tuning with real BSM options mechanics.

    Args:
        model: ATLASModel (pre-trained via BC).
        train_features: Legacy param, ignored. Data loaded from feature_dir/cache_dir.
        config: ATLASConfig.
        n_iterations: Number of PPO iterations.
        rollout_steps: Steps per rollout collection.
        feature_dir: Directory containing *_features.npz files.
        cache_dir: Directory containing *.parquet price files.
        regime_filter: "all", "high_iv", "uptrend", "crash" — filter episodes by regime.
        reward_shaping: "none", "high_iv", "uptrend", "crash" — apply bonus rewards.
        all_data_preloaded: Pre-loaded data dict to avoid reloading across stages.
        optimizer: Existing optimizer to reuse across curriculum stages.

    Returns:
        Training history dict.
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config is None:
        config = ATLASConfig()

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).float()

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

    if all_data_preloaded is not None:
        all_data = all_data_preloaded
        print(f"Using pre-loaded data: {len(all_data)} symbols.", flush=True)
    else:
        print("Loading training data with BSM pricing support...", flush=True)
        all_data = _load_training_data(feature_dir, cache_dir)
        print(f"Loaded {len(all_data)} symbols for options RL training.", flush=True)

    env = OptionsEnvironment(all_data, config, regime_filter=regime_filter, reward_shaping=reward_shaping)
    if regime_filter != "all":
        print(f"  Regime filter: {regime_filter} ({len(env._eligible_episodes)} eligible episodes)", flush=True)
    history: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "mean_reward": [],
    }

    for iteration in range(n_iterations):
        rollout = Rollout()
        obs = env.reset()
        ep_rewards: list[float] = []

        model.eval()
        for _ in range(rollout_steps):
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in obs.items()}
                action_mean, value = model.forward_with_value(
                    inputs["features"], inputs["timestamps"], inputs["dow"],
                    inputs["month"], inputs["is_opex"], inputs["is_qtr"],
                    inputs["pre_mu"], inputs["pre_sigma"], inputs["rtg"],
                )
                if torch.isnan(action_mean).any():
                    action_mean = torch.nan_to_num(action_mean, nan=0.0)
                if torch.isnan(value).any():
                    value = torch.nan_to_num(value, nan=0.0)
                value = value.clamp(-100, 100)

                dist = model.get_action_distribution(action_mean)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done = env.step(action_np)

            for k in obs:
                getattr(rollout, {
                    "features": "features", "timestamps": "timestamps",
                    "dow": "dow", "month": "month", "is_opex": "is_opex",
                    "is_qtr": "is_qtr", "pre_mu": "pre_mu",
                    "pre_sigma": "pre_sigma", "rtg": "rtg",
                }.get(k, k)).append(obs[k].cpu())

            rollout.actions.append(action.cpu())
            rollout.log_probs.append(log_prob.cpu())
            rollout.rewards.append(reward)
            rollout.values.append(value.item())
            rollout.dones.append(done)
            ep_rewards.append(reward)

            obs = next_obs
            if done:
                obs = env.reset()

        # GAE
        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones,
            config.gamma, config.gae_lambda,
        )

        # PPO update
        model.train()
        n_steps = len(rollout.rewards)
        indices = np.arange(n_steps)
        mini_batch_size = min(512, n_steps)

        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, mini_batch_size):
                end = min(start + mini_batch_size, n_steps)
                mb_idx = indices[start:end]

                mb_features = torch.cat([rollout.features[i] for i in mb_idx]).to(device)
                mb_ts = torch.cat([rollout.timestamps[i] for i in mb_idx]).to(device)
                mb_dow = torch.cat([rollout.dow[i] for i in mb_idx]).to(device)
                mb_month = torch.cat([rollout.month[i] for i in mb_idx]).to(device)
                mb_opex = torch.cat([rollout.is_opex[i] for i in mb_idx]).to(device)
                mb_qtr = torch.cat([rollout.is_qtr[i] for i in mb_idx]).to(device)
                mb_mu = torch.cat([rollout.pre_mu[i] for i in mb_idx]).to(device)
                mb_sigma = torch.cat([rollout.pre_sigma[i] for i in mb_idx]).to(device)
                mb_rtg = torch.cat([rollout.rtg[i] for i in mb_idx]).to(device)
                mb_actions = torch.cat([rollout.actions[i] for i in mb_idx]).to(device)
                mb_old_lp = torch.tensor([rollout.log_probs[i].item() for i in mb_idx], device=device)
                mb_adv = advantages[mb_idx].to(device)
                mb_ret = returns[mb_idx].to(device)

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_mean, new_val = model.forward_with_value(
                    mb_features, mb_ts, mb_dow, mb_month, mb_opex, mb_qtr,
                    mb_mu, mb_sigma, mb_rtg,
                )
                if torch.isnan(new_mean).any() or torch.isnan(new_val).any():
                    optimizer.zero_grad()
                    continue
                new_val = new_val.clamp(-100, 100)

                dist = model.get_action_distribution(new_mean)
                new_lp = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                if torch.isnan(new_lp).any() or torch.isnan(entropy):
                    optimizer.zero_grad()
                    continue

                ratio = (new_lp - mb_old_lp).exp()
                clipped = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
                value_loss = nn.functional.mse_loss(new_val, mb_ret)

                loss = policy_loss + 0.5 * value_loss - config.entropy_coeff * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        avg_pl = total_pl / max(n_updates, 1)
        avg_vl = total_vl / max(n_updates, 1)
        avg_ent = total_ent / max(n_updates, 1)
        avg_rew = float(np.mean(ep_rewards))

        history["policy_loss"].append(avg_pl)
        history["value_loss"].append(avg_vl)
        history["entropy"].append(avg_ent)
        history["mean_reward"].append(avg_rew)

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  PPO iter {iteration+1}/{n_iterations}: "
                  f"policy={avg_pl:.4f} value={avg_vl:.4f} "
                  f"entropy={avg_ent:.4f} reward={avg_rew:.4f}",
                  flush=True)
        if (iteration + 1) % 25 == 0 or iteration == 0:
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"{checkpoint_dir}/atlas_ppo_iter{iteration+1}.pt")

    return history
