from __future__ import annotations

import numpy as np
import pytest

from trading_algo.quant_core.edge_validation.analysis.monte_carlo import (
    bootstrap_confidence_intervals,
    random_entry_permutation_test,
    whites_reality_check,
)


class TestBootstrap:
    def test_ci_contains_true_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 2.0, 500)
        cis = bootstrap_confidence_intervals(returns, n_resamples=5000, seed=42)

        sharpe_ci = cis[0]
        assert sharpe_ci.ci_lower < sharpe_ci.point_estimate < sharpe_ci.ci_upper

        mean_ci = cis[2]
        assert mean_ci.ci_lower < 0.5 < mean_ci.ci_upper

    def test_wider_ci_with_fewer_samples(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 2.0, 50)
        cis_small = bootstrap_confidence_intervals(returns, n_resamples=5000, seed=42)

        returns2 = rng.normal(0.5, 2.0, 500)
        cis_large = bootstrap_confidence_intervals(returns2, n_resamples=5000, seed=42)

        width_small = cis_small[2].ci_upper - cis_small[2].ci_lower
        width_large = cis_large[2].ci_upper - cis_large[2].ci_lower
        assert width_small > width_large

    def test_returns_three_cis(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 2.0, 100)
        cis = bootstrap_confidence_intervals(returns, n_resamples=1000, seed=42)
        assert len(cis) == 3
        assert "sharpe" in cis[0].metric_name.lower()
        assert "profit" in cis[1].metric_name.lower()
        assert "mean" in cis[2].metric_name.lower() or "return" in cis[2].metric_name.lower()


class TestPermutationTest:
    def test_detects_genuine_edge(self) -> None:
        rng = np.random.default_rng(42)
        pattern_returns = rng.normal(1.0, 2.0, 200)
        all_returns = rng.normal(0.0, 2.0, 5000)

        result = random_entry_permutation_test(
            pattern_returns, all_returns, n_permutations=5000, seed=42,
        )
        assert result.p_value < 0.05
        assert result.significant is True

    def test_no_edge_detected_for_random(self) -> None:
        rng = np.random.default_rng(42)
        pattern_returns = rng.normal(0.0, 2.0, 200)
        all_returns = rng.normal(0.0, 2.0, 5000)

        result = random_entry_permutation_test(
            pattern_returns, all_returns, n_permutations=5000, seed=42,
        )
        assert result.p_value > 0.01

    def test_deterministic_with_seed(self) -> None:
        rng = np.random.default_rng(42)
        pattern_returns = rng.normal(0.5, 2.0, 100)
        all_returns = rng.normal(0.0, 2.0, 2000)

        r1 = random_entry_permutation_test(pattern_returns, all_returns, seed=123)
        r2 = random_entry_permutation_test(pattern_returns, all_returns, seed=123)
        assert r1.p_value == r2.p_value


class TestWhitesRC:
    def test_best_strategy_identified(self) -> None:
        rng = np.random.default_rng(42)
        strategies = {
            "good": rng.normal(0.8, 2.0, 300),
            "mediocre": rng.normal(0.1, 2.0, 300),
            "bad": rng.normal(-0.3, 2.0, 300),
        }
        result = whites_reality_check(strategies, n_bootstrap=5000, seed=42)
        assert result.best_strategy == "good"
        assert result.data_mining_bias >= 0
        assert result.adjusted_metric <= result.observed_best_metric

    def test_significant_when_edge_exists(self) -> None:
        rng = np.random.default_rng(42)
        strategies = {
            "strong": rng.normal(1.0, 1.0, 500),
            "noise1": rng.normal(0.0, 1.0, 500),
            "noise2": rng.normal(0.0, 1.0, 500),
        }
        result = whites_reality_check(strategies, n_bootstrap=5000, seed=42)
        assert result.significant is True
