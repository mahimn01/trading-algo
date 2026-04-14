from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from trading_algo.quant_core.edge_validation.types import (
    PatternOccurrence,
    RegimeBreakdown,
    RegimeEdgeResult,
    VarianceRatioResult,
)


def classify_trading_days(
    bars: pd.DataFrame,
) -> dict[date, str]:
    """Classify each trading day into a regime.

    Regimes based on daily stats computed from 1-min bars:
    - "high_vol_up": ATR > 75th percentile AND close > open
    - "high_vol_down": ATR > 75th percentile AND close < open
    - "low_vol_up": ATR <= 75th percentile AND close > open
    - "low_vol_down": ATR <= 75th percentile AND close < open

    ATR percentile computed over rolling 20-day window.
    """
    bars = bars.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index)

    high_col = "high" if "high" in bars.columns else "High"
    low_col = "low" if "low" in bars.columns else "Low"
    open_col = "open" if "open" in bars.columns else "Open"
    close_col = "close" if "close" in bars.columns else "Close"

    bars["_date"] = bars.index.date
    daily = bars.groupby("_date").agg(
        day_high=(high_col, "max"),
        day_low=(low_col, "min"),
        day_open=(open_col, "first"),
        day_close=(close_col, "last"),
    )
    daily["atr"] = daily["day_high"] - daily["day_low"]
    daily["atr_p75"] = daily["atr"].rolling(window=20, min_periods=1).quantile(0.75)

    result: dict[date, str] = {}
    for d, row in daily.iterrows():
        high_vol = row["atr"] > row["atr_p75"]
        up = row["day_close"] > row["day_open"]

        if high_vol and up:
            regime = "high_vol_up"
        elif high_vol and not up:
            regime = "high_vol_down"
        elif not high_vol and up:
            regime = "low_vol_up"
        else:
            regime = "low_vol_down"
        result[d] = regime

    return result


def regime_conditional_test(
    occurrences: list[PatternOccurrence],
    returns: NDArray[np.float64],
    regime_labels: dict[date, str],
    alpha: float = 0.05,
) -> RegimeEdgeResult:
    """Test edge within each regime.

    For each regime: compute win rate, mean return, Sharpe.
    Run t-test for significance within each regime.
    """
    if len(occurrences) != len(returns):
        raise ValueError("occurrences and returns must have equal length")

    regime_returns: dict[str, list[float]] = {}
    for occ, ret in zip(occurrences, returns):
        regime = regime_labels.get(occ.day)
        if regime is None:
            continue
        regime_returns.setdefault(regime, []).append(float(ret))

    breakdowns: list[RegimeBreakdown] = []
    for regime_name, rets_list in sorted(regime_returns.items()):
        rets = np.array(rets_list, dtype=np.float64)
        n = len(rets)
        if n < 2:
            breakdowns.append(RegimeBreakdown(
                regime_name=regime_name, n_occurrences=n,
                win_rate=float(np.mean(rets > 0)) if n > 0 else 0.0,
                mean_return=float(np.mean(rets)) if n > 0 else 0.0,
                sharpe=0.0, p_value=1.0,
            ))
            continue

        wr = float(np.mean(rets > 0))
        mean_ret = float(np.mean(rets))
        std_ret = float(np.std(rets, ddof=1))
        sr = mean_ret / std_ret if std_ret > 1e-10 else 0.0

        _, p_two = stats.ttest_1samp(rets, 0.0)
        p_val = float(p_two) / 2.0 if mean_ret > 0 else 1.0 - float(p_two) / 2.0

        breakdowns.append(RegimeBreakdown(
            regime_name=regime_name,
            n_occurrences=n,
            win_rate=wr,
            mean_return=mean_ret,
            sharpe=sr,
            p_value=p_val,
        ))

    return RegimeEdgeResult(
        breakdowns=tuple(breakdowns),
        hurst_exponent=None,
        hurst_supports_mr=None,
        variance_ratios=None,
    )


def hurst_exponent_test(series: NDArray[np.float64]) -> float:
    """Compute Hurst exponent using R/S method. H < 0.5 = mean reverting."""
    try:
        from trading_algo.quant_core.ml.fractal_analysis import hurst_exponent_rs
        h = hurst_exponent_rs(series)
        return h if not np.isnan(h) else 0.5
    except ImportError:
        pass

    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < 20:
        return 0.5

    max_k = n // 2
    min_k = 10
    if max_k < min_k:
        return 0.5

    sizes = np.unique(np.geomspace(min_k, max_k, num=20).astype(int))
    sizes = sizes[(sizes >= min_k) & (n // sizes >= 2)]

    if len(sizes) < 2:
        return 0.5

    log_ns: list[float] = []
    log_rs: list[float] = []

    for w in sizes:
        w = int(w)
        n_blocks = n // w
        rs_vals: list[float] = []
        for b in range(n_blocks):
            block = x[b * w: (b + 1) * w]
            m = np.mean(block)
            cumdev = np.cumsum(block - m)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)
            if s > 1e-10:
                rs_vals.append(r / s)
        if rs_vals:
            avg_rs = np.mean(rs_vals)
            if avg_rs > 1e-10:
                log_ns.append(np.log(w))
                log_rs.append(np.log(avg_rs))

    if len(log_ns) < 2:
        return 0.5

    log_ns_arr = np.array(log_ns)
    log_rs_arr = np.array(log_rs)
    dx = log_ns_arr - np.mean(log_ns_arr)
    dy = log_rs_arr - np.mean(log_rs_arr)
    ss_xx = np.dot(dx, dx)
    if ss_xx < 1e-10:
        return 0.5
    return float(np.dot(dx, dy) / ss_xx)


def variance_ratio_test(
    series: NDArray[np.float64],
    periods: tuple[int, ...] = (2, 5, 10, 20),
) -> tuple[VarianceRatioResult, ...]:
    """Lo-MacKinlay variance ratio test.

    VR(k) = Var(k-period returns) / (k * Var(1-period returns))
    Under random walk: VR = 1
    Z(k) = (VR(k) - 1) / sqrt(phi(k)) where phi(k) = 2(2k-1)(k-1)/(3kT)
    Reject RW if |Z| > 1.96 at 95% confidence.
    """
    series = np.asarray(series, dtype=np.float64)
    T = len(series)
    if T < 3:
        return tuple(
            VarianceRatioResult(period=k, vr=1.0, z_stat=0.0, p_value=1.0, rejects_random_walk=False)
            for k in periods
        )

    one_period = np.diff(series)
    var_1 = np.var(one_period, ddof=1)

    results: list[VarianceRatioResult] = []
    for k in periods:
        if T <= k:
            results.append(VarianceRatioResult(period=k, vr=1.0, z_stat=0.0, p_value=1.0, rejects_random_walk=False))
            continue

        k_period = series[k:] - series[:-k]
        var_k = np.var(k_period, ddof=1)

        if var_1 < 1e-10:
            results.append(VarianceRatioResult(period=k, vr=1.0, z_stat=0.0, p_value=1.0, rejects_random_walk=False))
            continue

        vr = var_k / (k * var_1)
        n = len(one_period)
        phi = 2.0 * (2.0 * k - 1) * (k - 1) / (3.0 * k * n)

        if phi < 1e-10:
            results.append(VarianceRatioResult(period=k, vr=vr, z_stat=0.0, p_value=1.0, rejects_random_walk=False))
            continue

        z = (vr - 1.0) / np.sqrt(phi)
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        results.append(VarianceRatioResult(
            period=k,
            vr=float(vr),
            z_stat=float(z),
            p_value=float(p_value),
            rejects_random_walk=float(p_value) < 0.05,
        ))

    return tuple(results)
