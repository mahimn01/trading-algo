from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Verdict = Literal["PASS", "WEAK", "FAIL"]


@dataclass(frozen=True)
class PatternOccurrence:
    timestamp: float
    pattern: str
    direction: Literal["long", "short"]
    entry_price: float
    day: date
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExcursionResult:
    mfe_points: NDArray[np.float64]
    mae_points: NDArray[np.float64]
    realized_pnl: NDArray[np.float64]
    holding_bars: NDArray[np.int64]
    n_occurrences: int


@dataclass(frozen=True)
class KSTestResult:
    statistic: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class ExcursionComparison:
    pattern: ExcursionResult
    random: ExcursionResult
    ks_mfe: KSTestResult
    ks_mae: KSTestResult
    ks_pnl: KSTestResult


@dataclass(frozen=True)
class BinomialResult:
    n_wins: int
    n_total: int
    win_rate: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class TTestResult:
    mean: float
    t_statistic: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class PSRResult:
    sharpe_ratio: float
    se_sharpe: float
    psr: float
    significant: bool


@dataclass(frozen=True)
class DSRResult:
    observed_sr: float
    sr_zero: float
    dsr: float
    n_trials: int
    significant: bool


@dataclass(frozen=True)
class MinTRLResult:
    min_trl: int
    n_observations: int
    sufficient: bool


@dataclass(frozen=True)
class SignificanceResult:
    binomial: BinomialResult
    ttest: TTestResult
    psr: PSRResult
    dsr: DSRResult
    min_trl: MinTRLResult


@dataclass(frozen=True)
class BootstrapCI:
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_resamples: int


@dataclass(frozen=True)
class PermutationResult:
    observed_metric: float
    p_value: float
    n_permutations: int
    significant: bool


@dataclass(frozen=True)
class WhitesRCResult:
    best_strategy: str
    observed_best_metric: float
    data_mining_bias: float
    adjusted_metric: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class MonteCarloResult:
    bootstrap_sharpe: BootstrapCI
    bootstrap_profit_factor: BootstrapCI
    bootstrap_mean_return: BootstrapCI
    permutation: PermutationResult


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    is_win_rate: float
    oos_win_rate: float
    is_mean_return: float
    oos_mean_return: float
    is_sharpe: float
    oos_sharpe: float


@dataclass(frozen=True)
class WalkForwardEdgeResult:
    folds: tuple[WalkForwardFold, ...]
    wf_efficiency: float
    pct_folds_positive: float
    is_stable: bool


@dataclass(frozen=True)
class VarianceRatioResult:
    period: int
    vr: float
    z_stat: float
    p_value: float
    rejects_random_walk: bool


@dataclass(frozen=True)
class RegimeBreakdown:
    regime_name: str
    n_occurrences: int
    win_rate: float
    mean_return: float
    sharpe: float
    p_value: float


@dataclass(frozen=True)
class RegimeEdgeResult:
    breakdowns: tuple[RegimeBreakdown, ...]
    hurst_exponent: float | None
    hurst_supports_mr: bool | None
    variance_ratios: tuple[VarianceRatioResult, ...] | None


@dataclass(frozen=True)
class PatternEdgeReport:
    pattern_name: str
    symbol: str
    n_occurrences: int
    win_rate: float
    mean_return_points: float
    median_return_points: float
    profit_factor: float
    excursion: ExcursionComparison
    significance: SignificanceResult
    monte_carlo: MonteCarloResult
    walk_forward: WalkForwardEdgeResult
    regime: RegimeEdgeResult
    verdict: Verdict
    conditions_met: int
    conditions_total: int


@dataclass
class EdgeValidationReport:
    symbol_reports: dict[str, list[PatternEdgeReport]] = field(default_factory=dict)
    whites_rc: WhitesRCResult | None = None
    total_patterns_tested: int = 0
    patterns_passed: int = 0
    patterns_weak: int = 0
    patterns_failed: int = 0
