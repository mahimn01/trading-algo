from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from trading_algo.quant_core.edge_validation.types import (
    BootstrapCI,
    PermutationResult,
    WhitesRCResult,
)
from trading_algo.quant_core.utils.statistics import profit_factor, sharpe_ratio


def bootstrap_confidence_intervals(
    returns: NDArray[np.float64],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[BootstrapCI, BootstrapCI, BootstrapCI]:
    """Returns (sharpe_ci, profit_factor_ci, mean_return_ci)"""
    n = len(returns)
    if n < 2:
        default = BootstrapCI(metric_name="", point_estimate=0.0, ci_lower=0.0, ci_upper=0.0, n_resamples=0)
        return (
            BootstrapCI(metric_name="sharpe_ratio", point_estimate=0.0, ci_lower=0.0, ci_upper=0.0, n_resamples=0),
            BootstrapCI(metric_name="profit_factor", point_estimate=0.0, ci_lower=0.0, ci_upper=0.0, n_resamples=0),
            BootstrapCI(metric_name="mean_return", point_estimate=0.0, ci_lower=0.0, ci_upper=0.0, n_resamples=0),
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))

    resampled = returns[indices]

    means = np.mean(resampled, axis=1)
    stds = np.std(resampled, axis=1, ddof=1)
    safe_stds = np.where(stds < 1e-10, np.nan, stds)
    sharpe_samples = means / safe_stds
    sharpe_samples = np.nan_to_num(sharpe_samples, nan=0.0)

    pf_samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        pf_samples[i] = profit_factor(resampled[i])

    alpha = (1 - confidence) / 2
    lo, hi = alpha * 100, (1 - alpha) * 100

    obs_sharpe = sharpe_ratio(returns, annualize=False)
    obs_pf = profit_factor(returns)
    obs_mean = float(np.mean(returns))

    sharpe_ci = BootstrapCI(
        metric_name="sharpe_ratio",
        point_estimate=obs_sharpe,
        ci_lower=float(np.percentile(sharpe_samples, lo)),
        ci_upper=float(np.percentile(sharpe_samples, hi)),
        n_resamples=n_resamples,
    )
    pf_ci = BootstrapCI(
        metric_name="profit_factor",
        point_estimate=obs_pf,
        ci_lower=float(np.percentile(pf_samples, lo)),
        ci_upper=float(np.percentile(pf_samples, hi)),
        n_resamples=n_resamples,
    )
    mean_ci = BootstrapCI(
        metric_name="mean_return",
        point_estimate=obs_mean,
        ci_lower=float(np.percentile(means, lo)),
        ci_upper=float(np.percentile(means, hi)),
        n_resamples=n_resamples,
    )

    return sharpe_ci, pf_ci, mean_ci


def random_entry_permutation_test(
    pattern_returns: NDArray[np.float64],
    all_bar_returns: NDArray[np.float64],
    n_permutations: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> PermutationResult:
    """Test if pattern returns are significantly better than random entry returns.

    all_bar_returns: returns from ALL possible entry points (not just pattern entries).
    For each permutation, sample len(pattern_returns) from all_bar_returns, compute Sharpe.
    p-value = (count where random_sharpe >= observed_sharpe + 1) / (n_permutations + 1)
    """
    k = len(pattern_returns)
    if k < 2:
        return PermutationResult(observed_metric=0.0, p_value=1.0, n_permutations=n_permutations, significant=False)

    std_obs = np.std(pattern_returns, ddof=1)
    if std_obs < 1e-10:
        observed_sharpe = 0.0
    else:
        observed_sharpe = float(np.mean(pattern_returns) / std_obs)

    rng = np.random.default_rng(seed)
    n_all = len(all_bar_returns)
    indices = rng.integers(0, n_all, size=(n_permutations, k))
    resampled = all_bar_returns[indices]

    means = np.mean(resampled, axis=1)
    stds = np.std(resampled, axis=1, ddof=1)
    safe_stds = np.where(stds < 1e-10, np.nan, stds)
    random_sharpes = means / safe_stds
    random_sharpes = np.nan_to_num(random_sharpes, nan=0.0)

    count_ge = int(np.sum(random_sharpes >= observed_sharpe)) + 1
    p_value = count_ge / (n_permutations + 1)

    return PermutationResult(
        observed_metric=observed_sharpe,
        p_value=p_value,
        n_permutations=n_permutations,
        significant=p_value < alpha,
    )


def whites_reality_check(
    strategy_returns: dict[str, NDArray[np.float64]],
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> WhitesRCResult:
    """White's Reality Check for multiple strategy comparison.

    strategy_returns: dict mapping strategy name to its returns array.
    Tests whether the best strategy's performance is genuine vs data snooping.

    1. Compute mean return for each strategy -> find best
    2. Center returns (subtract mean) to create null hypothesis
    3. Bootstrap: resample centered returns, find max mean across strategies
    4. p-value = fraction of bootstrap max means exceeding observed best mean
    """
    if not strategy_returns:
        return WhitesRCResult(
            best_strategy="", observed_best_metric=0.0, data_mining_bias=0.0,
            adjusted_metric=0.0, p_value=1.0, significant=False,
        )

    names = list(strategy_returns.keys())
    arrays = [strategy_returns[n] for n in names]

    T = min(len(a) for a in arrays)
    if T < 2:
        return WhitesRCResult(
            best_strategy=names[0], observed_best_metric=0.0, data_mining_bias=0.0,
            adjusted_metric=0.0, p_value=1.0, significant=False,
        )

    trimmed = np.column_stack([a[:T] for a in arrays])
    means = np.mean(trimmed, axis=0)
    best_idx = int(np.argmax(means))
    best_name = names[best_idx]
    observed_best = float(means[best_idx])

    centered = trimmed - means[np.newaxis, :]

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, T, size=(n_bootstrap, T))

    boot_max_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        boot_sample = centered[indices[i]]
        boot_means = np.mean(boot_sample, axis=0)
        boot_max_means[i] = np.max(boot_means)

    p_value = float(np.mean(boot_max_means >= observed_best))
    data_mining_bias = float(np.mean(boot_max_means))
    adjusted = observed_best - data_mining_bias

    return WhitesRCResult(
        best_strategy=best_name,
        observed_best_metric=observed_best,
        data_mining_bias=data_mining_bias,
        adjusted_metric=adjusted,
        p_value=p_value,
        significant=p_value < alpha,
    )
