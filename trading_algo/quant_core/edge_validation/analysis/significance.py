from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from trading_algo.quant_core.edge_validation.types import (
    BinomialResult,
    DSRResult,
    MinTRLResult,
    PSRResult,
    SignificanceResult,
    TTestResult,
)

EULER_MASCHERONI = 0.5772156649


def binomial_test(n_wins: int, n_total: int, p0: float = 0.5, alpha: float = 0.05) -> BinomialResult:
    if n_total == 0:
        return BinomialResult(n_wins=0, n_total=0, win_rate=0.0, p_value=1.0, significant=False)
    result = stats.binomtest(n_wins, n_total, p0, alternative="greater")
    pval = float(result.pvalue)
    return BinomialResult(
        n_wins=n_wins,
        n_total=n_total,
        win_rate=n_wins / n_total,
        p_value=pval,
        significant=pval < alpha,
    )


def ttest_mean_positive(returns: NDArray[np.float64], alpha: float = 0.05) -> TTestResult:
    if len(returns) < 2:
        return TTestResult(mean=0.0, t_statistic=0.0, p_value=1.0, significant=False)
    t_stat, p_two = stats.ttest_1samp(returns, 0.0)
    p_one = float(p_two) / 2.0 if t_stat > 0 else 1.0 - float(p_two) / 2.0
    return TTestResult(
        mean=float(np.mean(returns)),
        t_statistic=float(t_stat),
        p_value=p_one,
        significant=p_one < alpha,
    )


def probabilistic_sharpe_ratio(
    returns: NDArray[np.float64],
    sr_benchmark: float = 0.0,
) -> PSRResult:
    T = len(returns)
    if T < 3:
        return PSRResult(sharpe_ratio=0.0, se_sharpe=0.0, psr=0.0, significant=False)

    std = np.std(returns, ddof=1)
    if std < 1e-10:
        return PSRResult(sharpe_ratio=0.0, se_sharpe=0.0, psr=0.0, significant=False)

    sr = float(np.mean(returns) / std)
    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

    se_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (T - 1))
    if se_sr < 1e-10:
        return PSRResult(sharpe_ratio=sr, se_sharpe=0.0, psr=0.0, significant=False)

    psr_val = float(stats.norm.cdf((sr - sr_benchmark) / se_sr))
    return PSRResult(
        sharpe_ratio=sr,
        se_sharpe=float(se_sr),
        psr=psr_val,
        significant=psr_val > 0.95,
    )


def deflated_sharpe_ratio(
    returns: NDArray[np.float64],
    n_trials: int,
    alpha: float = 0.05,
) -> DSRResult:
    T = len(returns)
    if T < 3 or n_trials < 1:
        return DSRResult(observed_sr=0.0, sr_zero=0.0, dsr=0.0, n_trials=n_trials, significant=False)

    std = np.std(returns, ddof=1)
    if std < 1e-10:
        return DSRResult(observed_sr=0.0, sr_zero=0.0, dsr=0.0, n_trials=n_trials, significant=False)

    sr = float(np.mean(returns) / std)
    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

    v_sr = (1 + sr ** 2 / 2) / T
    N = max(n_trials, 1)
    gamma = EULER_MASCHERONI

    if N <= 1:
        sr_zero = 0.0
    else:
        sr_zero = float(
            np.sqrt(v_sr) * (
                (1 - gamma) * stats.norm.ppf(1 - 1.0 / N)
                + gamma * stats.norm.ppf(1 - 1.0 / (N * np.e))
            )
        )

    se_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (T - 1))
    if se_sr < 1e-10:
        return DSRResult(observed_sr=sr, sr_zero=sr_zero, dsr=0.0, n_trials=n_trials, significant=False)

    dsr_val = float(stats.norm.cdf((sr - sr_zero) / se_sr))
    return DSRResult(
        observed_sr=sr,
        sr_zero=sr_zero,
        dsr=dsr_val,
        n_trials=n_trials,
        significant=dsr_val > (1 - alpha),
    )


def minimum_track_record_length(
    returns: NDArray[np.float64],
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
) -> MinTRLResult:
    T = len(returns)
    if T < 3:
        return MinTRLResult(min_trl=999999, n_observations=T, sufficient=False)

    std = np.std(returns, ddof=1)
    if std < 1e-10:
        return MinTRLResult(min_trl=999999, n_observations=T, sufficient=False)

    sr = float(np.mean(returns) / std)
    if abs(sr - sr_benchmark) < 1e-10:
        return MinTRLResult(min_trl=999999, n_observations=T, sufficient=False)

    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))
    z_alpha = float(stats.norm.ppf(confidence))

    numerator = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    denominator = (sr - sr_benchmark) ** 2
    min_trl = int(np.ceil(numerator * (z_alpha ** 2) / denominator))
    min_trl = max(min_trl, 2)

    return MinTRLResult(
        min_trl=min_trl,
        n_observations=T,
        sufficient=T >= min_trl,
    )


def run_all_significance(
    returns: NDArray[np.float64],
    n_trials: int = 1,
    alpha: float = 0.05,
) -> SignificanceResult:
    n_wins = int(np.sum(returns > 0))
    n_total = len(returns)

    return SignificanceResult(
        binomial=binomial_test(n_wins, n_total, 0.5, alpha),
        ttest=ttest_mean_positive(returns, alpha),
        psr=probabilistic_sharpe_ratio(returns, 0.0),
        dsr=deflated_sharpe_ratio(returns, n_trials, alpha),
        min_trl=minimum_track_record_length(returns, 0.0, 1 - alpha),
    )
