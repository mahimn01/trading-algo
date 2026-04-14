"""
ATLAS Hindsight Action Computation (v2)

Scores actions by simulating actual options outcomes using BSM pricing,
not simplified daily returns. Delta, direction, leverage, DTE, and profit
target all affect the score through realistic option payoff simulation.

Optimized: caches strike/premium per unique (delta, dte, direction_class)
and simulates the option value path once, then evaluates all profit_targets
and leverages against the cached path.
"""

from __future__ import annotations

import itertools

import numpy as np

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.strategies.options.wheel import (
    _find_strike_by_delta,
    _price_option,
)

RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0
SKEW_SLOPE = 0.8

DELTA_GRID = np.array([0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
DIRECTION_GRID = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
LEVERAGE_GRID = np.array([0.25, 0.50, 0.75, 1.0])
DTE_GRID = np.array([21, 30, 45, 60])
PROFIT_TARGET_GRID = np.array([0.0, 0.25, 0.50, 0.75])

ALL_COMBOS = np.array(
    list(itertools.product(DELTA_GRID, DIRECTION_GRID, LEVERAGE_GRID, DTE_GRID, PROFIT_TARGET_GRID)),
    dtype=np.float64,
)  # (2560, 5)

# Pre-compute unique trade setups: (delta, direction_class, dte) combos
# direction_class: "sell_put" (dir < -0.3), "buy_call" (dir > 0.3), "cash" (else)
_SELL_PUT_DELTAS = DELTA_GRID[DELTA_GRID >= 0.05]
_BUY_CALL_DELTAS = DELTA_GRID[DELTA_GRID >= 0.05]  # delta used for calls is fixed at 0.80


def _price_opt(
    spot: float,
    strike: float,
    tte_years: float,
    vol: float,
    option_type: str,
) -> float:
    return _price_option(
        spot, strike, tte_years, vol, RISK_FREE_RATE, option_type,
        dividend_yield=DIVIDEND_YIELD, skew_adjust=True, skew_slope=SKEW_SLOPE,
    )


def _find_strike(
    spot: float,
    target_delta: float,
    tte_years: float,
    vol: float,
    option_type: str,
) -> float:
    return _find_strike_by_delta(
        spot, target_delta, tte_years, vol, RISK_FREE_RATE, option_type,
        dividend_yield=DIVIDEND_YIELD,
    )


def _simulate_sell_put(
    closes: np.ndarray,
    ivs: np.ndarray,
    t: int,
    delta_val: float,
    dte: int,
) -> dict[float, float]:
    """
    Simulate a sell-put trade and return {profit_target: return_on_risk}
    for all profit targets. profit_target=0 means hold to expiry.
    """
    n = len(closes)
    price_t = closes[t]
    iv_t = ivs[t]
    tte_years = dte / 365.0

    strike = _find_strike(price_t, delta_val, tte_years, iv_t, "put")
    premium = _price_opt(price_t, strike, tte_years, iv_t, "put")
    premium -= 0.05
    if premium <= 0:
        return {pt: -1.0 for pt in PROFIT_TARGET_GRID}

    capital_at_risk = strike * 100
    results: dict[float, float | None] = {pt: None for pt in PROFIT_TARGET_GRID}

    for d in range(1, min(dte + 1, n - t)):
        future_price = closes[t + d]
        future_iv = ivs[t + d] if t + d < len(ivs) else iv_t
        days_left = dte - d

        if days_left > 0:
            current_val = _price_opt(future_price, strike, days_left / 365.0, future_iv, "put")
        else:
            current_val = max(strike - future_price, 0.0)

        pnl_per_contract = (premium - current_val) * 100

        # Check profit targets (only for those not yet triggered)
        for pt in PROFIT_TARGET_GRID:
            if results[pt] is not None:
                continue
            if pt > 0 and premium > 0:
                if (premium - current_val) / premium >= pt:
                    final_pnl = pnl_per_contract - 5
                    results[pt] = final_pnl / capital_at_risk

        # At expiry, fill in anything not yet triggered
        if days_left <= 0:
            if current_val > 0:
                assignment_loss = (strike - future_price) * 100
                final_pnl = premium * 100 - assignment_loss
            else:
                final_pnl = premium * 100
            expiry_ret = final_pnl / capital_at_risk
            for pt in PROFIT_TARGET_GRID:
                if results[pt] is None:
                    results[pt] = expiry_ret
            break

    # Fill any remaining None (shouldn't happen but safety)
    for pt in PROFIT_TARGET_GRID:
        if results[pt] is None:
            results[pt] = 0.0

    return results  # type: ignore[return-value]


def _simulate_buy_call(
    closes: np.ndarray,
    ivs: np.ndarray,
    t: int,
    dte: int,
) -> dict[float, float]:
    """
    Simulate a buy-call trade (deep ITM, delta=0.80) and return
    {profit_target: return_on_cost} for all profit targets.
    """
    n = len(closes)
    price_t = closes[t]
    iv_t = ivs[t]
    tte_years = dte / 365.0

    strike = _find_strike(price_t, 0.80, tte_years, iv_t, "call")
    premium = _price_opt(price_t, strike, tte_years, iv_t, "call")
    premium += 0.15
    cost = premium * 100

    if cost <= 0:
        return {pt: -1.0 for pt in PROFIT_TARGET_GRID}

    results: dict[float, float | None] = {pt: None for pt in PROFIT_TARGET_GRID}

    for d in range(1, min(dte + 1, n - t)):
        future_price = closes[t + d]
        future_iv = ivs[t + d] if t + d < len(ivs) else iv_t
        days_left = dte - d

        if days_left > 0:
            current_val = _price_opt(future_price, strike, days_left / 365.0, future_iv, "call")
        else:
            current_val = max(future_price - strike, 0.0)

        pnl = (current_val - premium) * 100

        for pt in PROFIT_TARGET_GRID:
            if results[pt] is not None:
                continue
            if pt > 0 and premium > 0:
                if current_val / premium - 1 >= pt:
                    results[pt] = pnl / cost

        if days_left <= 0:
            expiry_ret = pnl / cost
            for pt in PROFIT_TARGET_GRID:
                if results[pt] is None:
                    results[pt] = expiry_ret
            break

    for pt in PROFIT_TARGET_GRID:
        if results[pt] is None:
            results[pt] = 0.0

    return results  # type: ignore[return-value]


def _process_day(
    t: int,
    closes: np.ndarray,
    ivs: np.ndarray,
) -> np.ndarray:
    """Find best combo for day t using cached trade simulations."""
    n = len(closes)
    max_dte = int(DTE_GRID.max())
    if t + max_dte >= n:
        return ALL_COMBOS[0].copy()

    best_score = -np.inf
    best_combo = ALL_COMBOS[0].copy()

    # --- Sell put simulations ---
    # Cache: (delta, dte) -> {profit_target: base_return}
    sell_put_cache: dict[tuple[float, int], dict[float, float]] = {}
    for delta_val in _SELL_PUT_DELTAS:
        for dte in DTE_GRID:
            dte_int = int(dte)
            if t + dte_int >= n:
                continue
            sell_put_cache[(delta_val, dte_int)] = _simulate_sell_put(closes, ivs, t, delta_val, dte_int)

    # Score sell-put combos (direction < -0.3: -1.0 and -0.5)
    for delta_val in _SELL_PUT_DELTAS:
        for dte in DTE_GRID:
            dte_int = int(dte)
            key = (delta_val, dte_int)
            if key not in sell_put_cache:
                continue
            pt_returns = sell_put_cache[key]
            for direction in [-1.0, -0.5]:
                for leverage in LEVERAGE_GRID:
                    for pt in PROFIT_TARGET_GRID:
                        score = leverage * pt_returns[pt]
                        if score > best_score:
                            best_score = score
                            best_combo = np.array([delta_val, direction, leverage, dte, pt])

    # --- Buy call simulations ---
    # Cache: dte -> {profit_target: base_return}
    # (strike is always at delta=0.80, independent of combo delta)
    buy_call_cache: dict[int, dict[float, float]] = {}
    for dte in DTE_GRID:
        dte_int = int(dte)
        if t + dte_int >= n:
            continue
        buy_call_cache[dte_int] = _simulate_buy_call(closes, ivs, t, dte_int)

    # Score buy-call combos (direction > 0.3: 0.5 and 1.0)
    for delta_val in _BUY_CALL_DELTAS:
        for dte in DTE_GRID:
            dte_int = int(dte)
            if dte_int not in buy_call_cache:
                continue
            pt_returns = buy_call_cache[dte_int]
            for direction in [0.5, 1.0]:
                for leverage in LEVERAGE_GRID:
                    for pt in PROFIT_TARGET_GRID:
                        score = leverage * pt_returns[pt]
                        if score > best_score:
                            best_score = score
                            best_combo = np.array([delta_val, direction, leverage, dte, pt])

    # --- Cash / neutral combos (direction ~ 0) ---
    for dte in DTE_GRID:
        cash_score = 0.045 * dte / 365.0
        if cash_score > best_score:
            best_score = cash_score
            # For cash, pick delta=0 (no trade)
            best_combo = np.array([0.0, 0.0, LEVERAGE_GRID[0], dte, 0.0])

    return best_combo


def compute_hindsight_actions(
    closes: np.ndarray,
    ivs: np.ndarray,
    iv_ranks: np.ndarray,
    config: ATLASConfig,
) -> np.ndarray:
    """
    For each valid day t, find the action that would have maximized
    risk-adjusted return using actual BSM options simulation.

    Returns: (T, 5) array of optimal [delta, direction, leverage, dte, profit_target]
    """
    T = len(closes)
    max_dte = int(DTE_GRID.max())
    valid_end = T - max_dte

    if valid_end <= 0:
        return np.zeros((T, 5), dtype=np.float64)

    optimal = np.empty((valid_end, 5), dtype=np.float64)

    for t in range(valid_end):
        optimal[t] = _process_day(t, closes, ivs)

    # Pad tail days with the last valid action
    if valid_end < T:
        pad = np.tile(optimal[-1], (T - valid_end, 1))
        optimal = np.concatenate([optimal, pad], axis=0)

    return optimal
