from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.edge_validation.types import (
    PatternOccurrence,
    WalkForwardEdgeResult,
    WalkForwardFold,
)


def _compute_fold_metrics(
    returns: NDArray[np.float64],
) -> tuple[float, float, float]:
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    wr = float(np.sum(returns > 0) / len(returns))
    mean_ret = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    sr = mean_ret / std if std > 1e-10 else 0.0
    return wr, mean_ret, sr


def walk_forward_edge_test(
    occurrences: list[PatternOccurrence],
    returns: NDArray[np.float64],
    train_days: int = 126,
    test_days: int = 42,
    step_days: int = 42,
) -> WalkForwardEdgeResult:
    """Test if edge persists across rolling time windows.

    Split occurrences chronologically into train/test folds.
    For each fold: compute win rate, mean return, Sharpe on IS and OOS.
    WF efficiency = mean(OOS_sharpe) / mean(IS_sharpe) (clamped to [-2, 2]).
    is_stable = WF_efficiency > 0.5 AND >50% of folds have positive OOS mean return.

    Folds: slide a [train_days | test_days] window by step_days.
    Occurrences are assigned to folds by their day field.
    Minimum 10 occurrences per fold required, otherwise skip.
    """
    if len(occurrences) != len(returns):
        raise ValueError(f"occurrences ({len(occurrences)}) and returns ({len(returns)}) must have equal length")

    if len(occurrences) == 0:
        return WalkForwardEdgeResult(folds=(), wf_efficiency=0.0, pct_folds_positive=0.0, is_stable=False)

    days = np.array([occ.day for occ in occurrences])
    sort_idx = np.argsort(days)
    days = days[sort_idx]
    returns = returns[sort_idx]

    min_day = days[0]
    max_day = days[-1]
    total_span = (max_day - min_day).days

    window_total = train_days + test_days
    if total_span < window_total:
        return WalkForwardEdgeResult(folds=(), wf_efficiency=0.0, pct_folds_positive=0.0, is_stable=False)

    folds: list[WalkForwardFold] = []
    fold_idx = 0
    offset = 0

    while True:
        train_start = min_day + timedelta(days=offset)
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        if test_end > max_day:
            break

        train_mask = (days >= train_start) & (days <= train_end)
        test_mask = (days >= test_start) & (days <= test_end)

        train_returns = returns[train_mask]
        test_returns = returns[test_mask]

        if len(train_returns) >= 10 and len(test_returns) >= 10:
            is_wr, is_mean, is_sr = _compute_fold_metrics(train_returns)
            oos_wr, oos_mean, oos_sr = _compute_fold_metrics(test_returns)

            folds.append(WalkForwardFold(
                fold_index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_win_rate=is_wr,
                oos_win_rate=oos_wr,
                is_mean_return=is_mean,
                oos_mean_return=oos_mean,
                is_sharpe=is_sr,
                oos_sharpe=oos_sr,
            ))
            fold_idx += 1

        offset += step_days

    if not folds:
        return WalkForwardEdgeResult(folds=(), wf_efficiency=0.0, pct_folds_positive=0.0, is_stable=False)

    is_sharpes = np.array([f.is_sharpe for f in folds])
    oos_sharpes = np.array([f.oos_sharpe for f in folds])
    oos_means = np.array([f.oos_mean_return for f in folds])

    mean_is_sr = float(np.mean(is_sharpes))
    mean_oos_sr = float(np.mean(oos_sharpes))

    if abs(mean_is_sr) < 1e-10:
        wf_eff = 0.0
    else:
        wf_eff = float(np.clip(mean_oos_sr / mean_is_sr, -2.0, 2.0))

    pct_positive = float(np.mean(oos_means > 0))
    is_stable = wf_eff > 0.5 and pct_positive > 0.5

    return WalkForwardEdgeResult(
        folds=tuple(folds),
        wf_efficiency=wf_eff,
        pct_folds_positive=pct_positive,
        is_stable=is_stable,
    )
