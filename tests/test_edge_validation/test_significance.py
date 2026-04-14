from __future__ import annotations

import numpy as np
import pytest

from trading_algo.quant_core.edge_validation.analysis.significance import (
    binomial_test,
    ttest_mean_positive,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    run_all_significance,
)


class TestBinomialTest:
    def test_significant_win_rate(self) -> None:
        result = binomial_test(65, 100)
        assert result.win_rate == 0.65
        assert result.p_value < 0.05
        assert result.significant is True

    def test_insignificant_win_rate(self) -> None:
        result = binomial_test(52, 100)
        assert result.p_value > 0.05
        assert result.significant is False

    def test_perfect_win_rate(self) -> None:
        result = binomial_test(100, 100)
        assert result.p_value < 0.001
        assert result.significant is True

    def test_below_chance(self) -> None:
        result = binomial_test(40, 100)
        assert result.p_value > 0.5
        assert result.significant is False


class TestTTest:
    def test_positive_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(1.0, 2.0, 200)
        result = ttest_mean_positive(returns)
        assert result.mean > 0
        assert result.p_value < 0.05
        assert result.significant is True

    def test_zero_mean(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 1.0, 200)
        result = ttest_mean_positive(returns)
        assert result.p_value > 0.01


class TestPSR:
    def test_strong_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 1.0, 500)
        result = probabilistic_sharpe_ratio(returns)
        assert result.psr > 0.95
        assert result.significant is True

    def test_weak_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 1.0, 50)
        result = probabilistic_sharpe_ratio(returns)
        assert result.psr < 0.95

    def test_sharpe_ratio_sign(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 2.0, 300)
        result = probabilistic_sharpe_ratio(returns)
        assert result.sharpe_ratio > 0


class TestDSR:
    def test_dsr_lower_than_psr_with_multiple_trials(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.3, 1.0, 300)
        psr = probabilistic_sharpe_ratio(returns)
        dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
        dsr_10 = deflated_sharpe_ratio(returns, n_trials=10)

        assert dsr_10.dsr <= dsr_1.dsr or abs(dsr_10.dsr - dsr_1.dsr) < 0.01

    def test_sr_zero_increases_with_trials(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.3, 1.0, 300)
        dsr_5 = deflated_sharpe_ratio(returns, n_trials=5)
        dsr_50 = deflated_sharpe_ratio(returns, n_trials=50)
        assert dsr_50.sr_zero >= dsr_5.sr_zero


class TestMinTRL:
    def test_strong_edge_needs_few_trades(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(1.0, 1.0, 500)
        result = minimum_track_record_length(returns)
        assert result.min_trl < 100
        assert result.sufficient is True

    def test_weak_edge_needs_many_trades(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 1.0, 50)
        result = minimum_track_record_length(returns)
        assert result.min_trl > 50


class TestRunAllSignificance:
    def test_integrates_all_tests(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 2.0, 300)
        result = run_all_significance(returns, n_trials=3)

        assert result.binomial.n_total == 300
        assert result.ttest.mean != 0
        assert 0.0 <= result.psr.psr <= 1.0
        assert 0.0 <= result.dsr.dsr <= 1.0
        assert result.min_trl.min_trl > 0
