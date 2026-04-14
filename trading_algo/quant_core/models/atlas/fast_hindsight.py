"""Numba-compiled hindsight action computation for ATLAS.

~100x faster than the pure-Python version by:
  1. Inlining BSM math (no Python object overhead)
  2. JIT-compiling all inner loops
  3. Vectorizing the grid search across days
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np


RISK_FREE_RATE = 0.045
SKEW_SLOPE = 0.8

DELTA_GRID = np.array([0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
DIRECTION_GRID = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
LEVERAGE_GRID = np.array([0.25, 0.50, 0.75, 1.0])
DTE_GRID = np.array([21, 30, 45, 60], dtype=np.int64)
PROFIT_TARGET_GRID = np.array([0.0, 0.25, 0.50, 0.75])

# Deltas eligible for sell-put (skip delta=0)
SELL_PUT_DELTAS = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])


@nb.njit(cache=True, fastmath=True)
def _norm_cdf(x: float) -> float:
    """Fast normal CDF approximation (Abramowitz & Stegun)."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


@nb.njit(cache=True, fastmath=True)
def _bsm_price(spot: float, strike: float, tte: float, vol: float,
               rate: float, is_put: bool) -> float:
    """BSM option price with skew adjustment."""
    if tte <= 0.0:
        if is_put:
            return max(strike - spot, 0.0)
        return max(spot - strike, 0.0)

    # Skew adjustment
    adj_vol = vol
    if spot > 0:
        if is_put:
            otm_pct = max(0.0, (spot - strike) / spot)
            adj_vol = vol * (1.0 + SKEW_SLOPE * otm_pct)
        else:
            otm_pct = max(0.0, (strike - spot) / spot)
            adj_vol = vol * (1.0 - SKEW_SLOPE * 0.3 * otm_pct)

    if adj_vol <= 0.0:
        adj_vol = 0.001

    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + (rate + 0.5 * adj_vol * adj_vol) * tte) / (adj_vol * sqrt_t)
    d2 = d1 - adj_vol * sqrt_t

    if is_put:
        price = strike * math.exp(-rate * tte) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    else:
        price = spot * _norm_cdf(d1) - strike * math.exp(-rate * tte) * _norm_cdf(d2)

    return max(price, 0.0)


@nb.njit(cache=True, fastmath=True)
def _find_strike_fast(spot: float, target_delta: float, tte: float,
                      vol: float, rate: float, is_put: bool) -> float:
    """Find strike by delta using bisection (numba-compatible)."""
    if tte <= 0.0:
        return spot

    # Initial bounds
    if is_put:
        lo = spot * 0.50
        hi = spot * 1.00
    else:
        lo = spot * 1.00
        hi = spot * 2.00

    if vol <= 0.0:
        vol = 0.001

    sqrt_t = math.sqrt(tte)

    # Bisection: 30 iterations gives precision ~1e-9 of range
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        d1 = (math.log(spot / mid) + (rate + 0.5 * vol * vol) * tte) / (vol * sqrt_t)
        if is_put:
            delta_abs = _norm_cdf(-d1)
        else:
            delta_abs = _norm_cdf(d1)

        if delta_abs > target_delta:
            if is_put:
                hi = mid  # Move strike down to reduce put delta
            else:
                lo = mid  # Move strike up to reduce call delta
        else:
            if is_put:
                lo = mid
            else:
                hi = mid

    # Round to nearest 0.50
    raw = 0.5 * (lo + hi)
    return round(raw * 2.0) / 2.0


@nb.njit(cache=True, fastmath=True)
def _sim_sell_put(closes: np.ndarray, ivs: np.ndarray, t: int,
                  delta_val: float, dte: int) -> np.ndarray:
    """Simulate sell-put, return array of returns for each profit target."""
    n = len(closes)
    n_pt = 4  # len(PROFIT_TARGET_GRID)
    results = np.full(n_pt, 0.0)
    triggered = np.zeros(n_pt, dtype=nb.boolean)

    price_t = closes[t]
    iv_t = ivs[t]
    tte_years = dte / 365.0

    strike = _find_strike_fast(price_t, delta_val, tte_years, iv_t, RISK_FREE_RATE, True)
    premium = _bsm_price(price_t, strike, tte_years, iv_t, RISK_FREE_RATE, True)
    premium -= 0.05  # Commission

    if premium <= 0.0:
        for i in range(n_pt):
            results[i] = -1.0
        return results

    capital_at_risk = strike * 100.0
    pt_grid = np.array([0.0, 0.25, 0.50, 0.75])

    for d in range(1, min(dte + 1, n - t)):
        future_price = closes[t + d]
        future_iv = ivs[min(t + d, n - 1)]
        days_left = dte - d

        if days_left > 0:
            current_val = _bsm_price(future_price, strike, days_left / 365.0, future_iv, RISK_FREE_RATE, True)
        else:
            current_val = max(strike - future_price, 0.0)

        pnl_per_contract = (premium - current_val) * 100.0

        # Check profit targets
        for i in range(n_pt):
            if triggered[i]:
                continue
            pt = pt_grid[i]
            if pt > 0.0 and premium > 0.0:
                if (premium - current_val) / premium >= pt:
                    final_pnl = pnl_per_contract - 5.0
                    results[i] = final_pnl / capital_at_risk
                    triggered[i] = True

        # At expiry
        if days_left <= 0:
            if current_val > 0.0:
                assignment_loss = (strike - future_price) * 100.0
                final_pnl = premium * 100.0 - assignment_loss
            else:
                final_pnl = premium * 100.0
            expiry_ret = final_pnl / capital_at_risk
            for i in range(n_pt):
                if not triggered[i]:
                    results[i] = expiry_ret
                    triggered[i] = True
            break

    return results


@nb.njit(cache=True, fastmath=True)
def _sim_buy_call(closes: np.ndarray, ivs: np.ndarray, t: int,
                  dte: int) -> np.ndarray:
    """Simulate buy-call (delta=0.80), return array of returns for each profit target."""
    n = len(closes)
    n_pt = 4
    results = np.full(n_pt, 0.0)
    triggered = np.zeros(n_pt, dtype=nb.boolean)

    price_t = closes[t]
    iv_t = ivs[t]
    tte_years = dte / 365.0

    strike = _find_strike_fast(price_t, 0.80, tte_years, iv_t, RISK_FREE_RATE, False)
    premium = _bsm_price(price_t, strike, tte_years, iv_t, RISK_FREE_RATE, False)
    premium += 0.15  # Commission

    cost = premium * 100.0
    if cost <= 0.0:
        for i in range(n_pt):
            results[i] = -1.0
        return results

    pt_grid = np.array([0.0, 0.25, 0.50, 0.75])

    for d in range(1, min(dte + 1, n - t)):
        future_price = closes[t + d]
        future_iv = ivs[min(t + d, n - 1)]
        days_left = dte - d

        if days_left > 0:
            current_val = _bsm_price(future_price, strike, days_left / 365.0, future_iv, RISK_FREE_RATE, False)
        else:
            current_val = max(future_price - strike, 0.0)

        pnl = (current_val - premium) * 100.0

        for i in range(n_pt):
            if triggered[i]:
                continue
            pt = pt_grid[i]
            if pt > 0.0 and premium > 0.0:
                if current_val / premium - 1.0 >= pt:
                    results[i] = pnl / cost
                    triggered[i] = True

        if days_left <= 0:
            expiry_ret = pnl / cost
            for i in range(n_pt):
                if not triggered[i]:
                    results[i] = expiry_ret
                    triggered[i] = True
            break

    return results


@nb.njit(cache=True, fastmath=True)
def _process_day_fast(t: int, closes: np.ndarray, ivs: np.ndarray) -> np.ndarray:
    """Find best action combo for day t (fully JIT-compiled)."""
    n = len(closes)
    best_score = -1e30
    best_combo = np.array([0.0, 0.0, 0.25, 21.0, 0.0])

    sell_put_deltas = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    leverage_grid = np.array([0.25, 0.50, 0.75, 1.0])
    dte_grid = np.array([21, 30, 45, 60], dtype=np.int64)
    pt_grid = np.array([0.0, 0.25, 0.50, 0.75])

    # Sell-put combos
    for di in range(len(sell_put_deltas)):
        delta_val = sell_put_deltas[di]
        for dti in range(len(dte_grid)):
            dte = dte_grid[dti]
            if t + dte >= n:
                continue
            pt_returns = _sim_sell_put(closes, ivs, t, delta_val, dte)
            for direction in (-1.0, -0.5):
                for li in range(len(leverage_grid)):
                    lev = leverage_grid[li]
                    for pi in range(len(pt_grid)):
                        score = lev * pt_returns[pi]
                        if score > best_score:
                            best_score = score
                            best_combo[0] = delta_val
                            best_combo[1] = direction
                            best_combo[2] = lev
                            best_combo[3] = float(dte)
                            best_combo[4] = pt_grid[pi]

    # Buy-call combos
    for dti in range(len(dte_grid)):
        dte = dte_grid[dti]
        if t + dte >= n:
            continue
        pt_returns = _sim_buy_call(closes, ivs, t, dte)
        for di in range(len(sell_put_deltas)):
            delta_val = sell_put_deltas[di]
            for direction in (0.5, 1.0):
                for li in range(len(leverage_grid)):
                    lev = leverage_grid[li]
                    for pi in range(len(pt_grid)):
                        score = lev * pt_returns[pi]
                        if score > best_score:
                            best_score = score
                            best_combo[0] = delta_val
                            best_combo[1] = direction
                            best_combo[2] = lev
                            best_combo[3] = float(dte)
                            best_combo[4] = pt_grid[pi]

    # Cash / neutral
    for dti in range(len(dte_grid)):
        dte = dte_grid[dti]
        cash_score = 0.045 * float(dte) / 365.0
        if cash_score > best_score:
            best_score = cash_score
            best_combo[0] = 0.0
            best_combo[1] = 0.0
            best_combo[2] = 0.25
            best_combo[3] = float(dte)
            best_combo[4] = 0.0

    return best_combo


@nb.njit(cache=True, fastmath=True, parallel=True)
def compute_hindsight_fast(closes: np.ndarray, ivs: np.ndarray) -> np.ndarray:
    """Compute hindsight-optimal actions for all days (parallel numba)."""
    T = len(closes)
    max_dte = 60
    valid_end = T - max_dte

    if valid_end <= 0:
        return np.zeros((T, 5))

    optimal = np.empty((valid_end, 5))

    for t in nb.prange(valid_end):
        optimal[t] = _process_day_fast(t, closes, ivs)

    # Pad tail
    result = np.empty((T, 5))
    result[:valid_end] = optimal
    for t in range(valid_end, T):
        result[t] = optimal[valid_end - 1]

    return result


def warmup() -> None:
    """Force JIT compilation with dummy data."""
    dummy_closes = np.linspace(100, 110, 200).astype(np.float64)
    dummy_ivs = np.full(200, 0.25)
    compute_hindsight_fast(dummy_closes, dummy_ivs)
