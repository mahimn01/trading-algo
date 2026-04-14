from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_algo.quant_core.edge_validation.types import PatternOccurrence
from trading_algo.quant_core.edge_validation.analysis.walk_forward import (
    walk_forward_edge_test,
)


def _make_occurrences(n: int, start_date: str = "2024-01-01") -> list[PatternOccurrence]:
    days = pd.bdate_range(start_date, periods=n)
    return [
        PatternOccurrence(
            timestamp=0.0,
            pattern="test",
            direction="long",
            entry_price=5000,
            day=d.date(),
        )
        for d in days
    ]


class TestWalkForwardEdgeTest:
    def test_stable_edge(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 3.0, 300)
        occs = _make_occurrences(300)

        result = walk_forward_edge_test(occs, returns, train_days=60, test_days=30, step_days=30)

        assert len(result.folds) >= 3
        assert result.pct_folds_positive > 0.4
        assert result.wf_efficiency > 0

    def test_no_edge(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 3.0, 300)
        occs = _make_occurrences(300)

        result = walk_forward_edge_test(occs, returns, train_days=60, test_days=30, step_days=30)
        assert result.pct_folds_positive < 0.8

    def test_decaying_edge(self) -> None:
        rng = np.random.default_rng(42)
        returns = np.concatenate([
            rng.normal(2.0, 1.0, 150),
            rng.normal(-1.0, 1.0, 150),
        ])
        occs = _make_occurrences(300)

        result = walk_forward_edge_test(occs, returns, train_days=60, test_days=30, step_days=30)
        assert len(result.folds) >= 3

    def test_too_few_occurrences_returns_empty(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 1.0, 5)
        occs = _make_occurrences(5)

        result = walk_forward_edge_test(occs, returns, train_days=60, test_days=30, step_days=30)
        assert len(result.folds) == 0
        assert result.is_stable is False

    def test_folds_have_correct_fields(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 2.0, 300)
        occs = _make_occurrences(300)

        result = walk_forward_edge_test(occs, returns, train_days=60, test_days=30, step_days=30)

        for fold in result.folds:
            assert fold.train_start < fold.train_end
            assert fold.test_start < fold.test_end
            assert fold.test_start >= fold.train_end
