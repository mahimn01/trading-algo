from __future__ import annotations

import itertools
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from trading_algo.quant_core.models.atlas.config import ATLASConfig


DELTA_GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
DIRECTION_GRID = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
LEVERAGE_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
DTE_GRID = np.array([21, 35, 45, 60])
PROFIT_TARGET_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

ALL_COMBOS = np.array(
    list(itertools.product(DELTA_GRID, DIRECTION_GRID, LEVERAGE_GRID, DTE_GRID, PROFIT_TARGET_GRID)),
    dtype=np.float64,
)  # (3000, 5)


def _score_combo(
    combo: np.ndarray,
    daily_returns: np.ndarray,
) -> float:
    direction = combo[1]
    leverage = combo[2]
    dte = int(combo[3])

    if len(daily_returns) < dte:
        return -np.inf

    window_returns = daily_returns[:dte]
    trade_returns = direction * leverage * window_returns

    if trade_returns.std() < 1e-12:
        return 0.0

    return float(trade_returns.mean() / trade_returns.std())


def _process_day(
    t: int,
    closes: np.ndarray,
    combos: np.ndarray,
) -> np.ndarray:
    max_dte = int(combos[:, 3].max())
    if t + max_dte >= len(closes):
        return combos[0]

    future_prices = closes[t : t + max_dte + 1]
    daily_returns = np.diff(future_prices) / future_prices[:-1]

    best_score = -np.inf
    best_combo = combos[0]

    for combo in combos:
        score = _score_combo(combo, daily_returns)
        if score > best_score:
            best_score = score
            best_combo = combo

    return best_combo


def _worker(args: tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
    t, closes, combos = args
    return _process_day(t, closes, combos)


def compute_hindsight_actions(
    closes: np.ndarray,
    ivs: np.ndarray,
    iv_ranks: np.ndarray,
    config: ATLASConfig,
) -> np.ndarray:
    """
    For each valid day t, find the action that would have maximized
    risk-adjusted return over the next `dte` days.

    Returns: (T, 5) array of optimal [delta, direction, leverage, dte, profit_target]
    """
    T = len(closes)
    max_dte = int(DTE_GRID.max())
    valid_end = T - max_dte

    if valid_end <= 0:
        return np.zeros((T, 5), dtype=np.float64)

    tasks = [(t, closes, ALL_COMBOS) for t in range(valid_end)]

    n_workers = min(8, cpu_count() or 1)
    if n_workers > 1 and valid_end > 100:
        with Pool(n_workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(task) for task in tasks]

    optimal = np.array(results, dtype=np.float64)  # (valid_end, 5)

    # Pad tail days with the last valid action
    if valid_end < T:
        pad = np.tile(optimal[-1], (T - valid_end, 1))
        optimal = np.concatenate([optimal, pad], axis=0)

    return optimal
