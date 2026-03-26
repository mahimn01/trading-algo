"""
IV Rank and IV Percentile Calculator

IV Rank = (Current IV - 52wk Low IV) / (52wk High IV - 52wk Low IV)
IV Percentile = % of days in past year where IV was below current IV

Key thresholds for premium selling:
  - IV Rank > 50: favorable to sell premium
  - IV Rank 30-50: acceptable
  - IV Rank < 30: avoid selling, consider buying
"""

from __future__ import annotations

import numpy as np


def realized_volatility(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """
    Rolling realized volatility (annualized) from close prices.

    Args:
        prices: Array of close prices.
        window: Lookback window in trading days.

    Returns:
        Array of annualized vol values (same length as prices, NaN-padded).
    """
    prices = np.asarray(prices, dtype=float)
    log_returns = np.diff(np.log(prices))
    out = np.full(len(prices), np.nan)
    for i in range(window, len(log_returns) + 1):
        out[i] = np.std(log_returns[i - window : i], ddof=1) * np.sqrt(252)
    return out


def iv_series_from_prices(
    prices: np.ndarray,
    rv_window: int = 30,
    iv_premium: float = 1.20,
    dynamic: bool = True,
) -> np.ndarray:
    """
    Estimate an implied-volatility series from underlying prices.

    When dynamic=True (default), the IV/RV premium ratio varies with market
    conditions rather than being constant.  In calm markets (low RV), IV
    trades at 1.3-1.5x RV.  During selloffs (high RV), IV/RV compresses
    toward 1.0 or even below.  This models the well-known VRP collapse
    during crises.

    Args:
        prices: Close price array.
        rv_window: Window for realized vol calculation.
        iv_premium: Base IV/RV premium (used when dynamic=False).
        dynamic: Use dynamic IV/RV ratio that varies with market regime.

    Returns:
        Estimated IV series (same length, NaN-padded at start).
    """
    rv = realized_volatility(prices, rv_window)

    if not dynamic:
        return rv * iv_premium

    # Dynamic IV/RV ratio:
    # Low RV (<15%): IV/RV ~ 1.40 (calm markets, high insurance premium)
    # Medium RV (15-30%): IV/RV ~ 1.15 (normal)
    # High RV (>30%): IV/RV ~ 1.00-1.05 (panic, RV catches up to IV)
    # Very high RV (>50%): IV/RV ~ 0.95 (RV overshoots IV during crashes)
    iv = np.full_like(rv, np.nan)
    for i in range(len(rv)):
        if np.isnan(rv[i]):
            continue
        r = rv[i]
        if r < 0.15:
            ratio = 1.40
        elif r < 0.30:
            # Linear interpolation: 1.40 at 15% -> 1.10 at 30%
            ratio = 1.40 - (r - 0.15) / 0.15 * 0.30
        elif r < 0.50:
            # 1.10 at 30% -> 0.95 at 50%
            ratio = 1.10 - (r - 0.30) / 0.20 * 0.15
        else:
            ratio = 0.95
        iv[i] = r * ratio
    return iv


def iv_rank(iv_series: np.ndarray, current_idx: int, lookback: int = 252) -> float:
    """
    IV Rank at a given index.

    IV Rank = (current - low) / (high - low) * 100
    """
    start = max(0, current_idx - lookback)
    window = iv_series[start : current_idx + 1]
    window = window[~np.isnan(window)]
    if len(window) < 20:
        return 50.0
    current = window[-1]
    lo, hi = float(np.min(window)), float(np.max(window))
    if hi == lo:
        return 50.0
    return float((current - lo) / (hi - lo) * 100.0)


def iv_percentile(iv_series: np.ndarray, current_idx: int, lookback: int = 252) -> float:
    """
    IV Percentile at a given index.

    IV Percentile = (# days IV was below current) / total days * 100
    """
    start = max(0, current_idx - lookback)
    window = iv_series[start : current_idx + 1]
    window = window[~np.isnan(window)]
    if len(window) < 20:
        return 50.0
    current = window[-1]
    below = np.sum(window[:-1] < current)
    return float(below / (len(window) - 1) * 100.0)
