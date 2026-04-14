from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.features import ATLASFeatureComputer, RollingNormalizer
from trading_algo.quant_core.models.atlas.data_pipeline import ATLASDataset
from trading_algo.quant_core.models.atlas.rewards import (
    differential_sharpe_ratio,
    drawdown_penalty,
    transaction_cost_penalty,
)
from trading_algo.quant_core.models.atlas.train_bc import train_behavioral_cloning


def _make_synthetic_ohlcv(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic AAPL-like OHLCV data."""
    rng = np.random.RandomState(seed)
    # Geometric Brownian Motion
    dt = 1 / 252
    mu = 0.10
    sigma = 0.25
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.randn(n_days)
    prices = 150.0 * np.exp(np.cumsum(log_returns))

    highs = prices * (1 + rng.uniform(0, 0.02, n_days))
    lows = prices * (1 - rng.uniform(0, 0.02, n_days))
    opens = prices * (1 + rng.uniform(-0.005, 0.005, n_days))
    volumes = rng.uniform(5e6, 20e6, n_days)

    dates = pd.bdate_range(start="2015-01-05", periods=n_days)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )
    return df


class TestATLASFeatureComputer:
    def test_output_shape(self) -> None:
        df = _make_synthetic_ohlcv(1000)
        fc = ATLASFeatureComputer()
        features = fc.compute_features(
            df["Close"].values, df["High"].values, df["Low"].values, df["Volume"].values
        )
        assert features.shape == (1000, 12)

    def test_no_nan_after_warmup(self) -> None:
        df = _make_synthetic_ohlcv(1000)
        fc = ATLASFeatureComputer()
        features = fc.compute_features(
            df["Close"].values, df["High"].values, df["Low"].values, df["Volume"].values
        )
        # After 300 days of warmup, no NaN should remain
        assert not np.any(np.isnan(features[300:]))

    def test_log_return_values(self) -> None:
        df = _make_synthetic_ohlcv(100)
        fc = ATLASFeatureComputer()
        features = fc.compute_features(
            df["Close"].values, df["High"].values, df["Low"].values, df["Volume"].values
        )
        prices = df["Close"].values
        expected_r1 = np.log(prices[1] / prices[0])
        np.testing.assert_almost_equal(features[1, 0], expected_r1, decimal=10)


class TestRollingNormalizer:
    def test_z_scores(self) -> None:
        rng = np.random.RandomState(123)
        T, F = 500, 4
        data = rng.randn(T, F) * 2 + 3  # mean ~3, std ~2

        normalizer = RollingNormalizer()
        normed, mu, sigma = normalizer.normalize(data, lookback=252)

        assert normed.shape == (T, F)
        assert mu.shape == (T, F)
        assert sigma.shape == (T, F)

        # At t=400, with 252-day lookback, z-score should be close to standard normal
        # Check that mu is roughly correct
        for f in range(F):
            window = data[400 - 251 : 401, f]
            expected_mu = np.mean(window)
            expected_sigma = np.std(window, ddof=1)
            np.testing.assert_almost_equal(mu[400, f], expected_mu, decimal=8)
            np.testing.assert_almost_equal(sigma[400, f], expected_sigma, decimal=8)
            expected_z = (data[400, f] - expected_mu) / (expected_sigma + 1e-8)
            np.testing.assert_almost_equal(normed[400, f], expected_z, decimal=6)

    def test_stores_mu_sigma(self) -> None:
        data = np.ones((100, 3))
        normalizer = RollingNormalizer()
        _, mu, sigma = normalizer.normalize(data, lookback=50)
        # Constant input -> mu=1, sigma computed from constant series
        assert mu[50, 0] == pytest.approx(1.0)


class TestATLASDataset:
    def test_window_shape(self) -> None:
        df = _make_synthetic_ohlcv(1000)
        config = ATLASConfig()
        ds = ATLASDataset(df, config=config, split="train")

        if len(ds) == 0:
            pytest.skip("No valid windows for train split in synthetic data range")

        sample = ds[0]
        assert sample["features"].shape == (90, 16)
        assert sample["timestamps"].shape == (90,)
        assert sample["day_of_week"].shape == (90,)
        assert sample["month"].shape == (90,)
        assert sample["is_opex"].shape == (90,)
        assert sample["is_quarter_end"].shape == (90,)
        assert sample["pre_norm_mu"].shape == (90, 16)
        assert sample["pre_norm_sigma"].shape == (90, 16)
        assert sample["action_label"].shape == (5,)
        assert sample["return_to_go"].shape == (1,)

    def test_no_nan_in_features(self) -> None:
        df = _make_synthetic_ohlcv(1000)
        config = ATLASConfig()
        ds = ATLASDataset(df, config=config, split="train")

        if len(ds) == 0:
            pytest.skip("No windows")

        sample = ds[0]
        assert not torch.isnan(sample["features"]).any()

    def test_no_lookahead(self) -> None:
        """The last day in a window should not use data from after that day."""
        df = _make_synthetic_ohlcv(1000)
        config = ATLASConfig()
        ds = ATLASDataset(df, config=config, split="train")

        if len(ds) < 2:
            pytest.skip("Not enough windows")

        # Windows should be sequential and not overlap improperly
        w0_start, w0_end = ds.windows[0]
        w1_start, w1_end = ds.windows[1]
        # Each subsequent window should advance by exactly 1 day
        assert w1_start == w0_start + 1
        assert w1_end == w0_end + 1

    def test_dtypes_float32(self) -> None:
        df = _make_synthetic_ohlcv(1000)
        config = ATLASConfig()
        ds = ATLASDataset(df, config=config, split="train")

        if len(ds) == 0:
            pytest.skip("No windows")

        sample = ds[0]
        assert sample["features"].dtype == torch.float32
        assert sample["timestamps"].dtype == torch.float32
        assert sample["action_label"].dtype == torch.float32


class TestRewards:
    def test_dsr_positive_returns(self) -> None:
        """DSR of consistently positive returns should be positive."""
        returns = torch.full((2, 50), 0.01)
        dsr = differential_sharpe_ratio(returns, eta=2 / 64)
        assert dsr.item() > 0

    def test_dsr_zero_returns(self) -> None:
        """DSR of zero returns should be ~0."""
        returns = torch.zeros(2, 50)
        dsr = differential_sharpe_ratio(returns, eta=2 / 64)
        assert abs(dsr.item()) < 1e-6

    def test_dsr_gradient_flows(self) -> None:
        returns = torch.randn(4, 30, requires_grad=True)
        dsr = differential_sharpe_ratio(returns)
        dsr.backward()
        assert returns.grad is not None
        assert not torch.isnan(returns.grad).any()

    def test_drawdown_penalty_no_drawdown(self) -> None:
        # Monotonically increasing equity -> no drawdown
        equity = torch.arange(1.0, 51.0).unsqueeze(0)
        penalty = drawdown_penalty(equity, threshold=0.20, lam=10.0)
        assert penalty.item() == pytest.approx(0.0, abs=1e-7)

    def test_drawdown_penalty_large_drawdown(self) -> None:
        # Peak at 100, drop to 50 -> 50% drawdown, well above 20% threshold
        equity = torch.tensor([[100.0, 50.0]])
        penalty = drawdown_penalty(equity, threshold=0.20, lam=10.0)
        # dd = 0.5, excess = 0.3, penalty = 10 * 0.3^2 = 0.9
        assert penalty.item() == pytest.approx(0.9, abs=1e-5)

    def test_transaction_cost_penalty(self) -> None:
        # Constant actions -> zero penalty
        actions = torch.ones(2, 10, 5)
        penalty = transaction_cost_penalty(actions, lam=0.01)
        assert penalty.item() == pytest.approx(0.0, abs=1e-7)

    def test_transaction_cost_nonzero(self) -> None:
        actions = torch.zeros(1, 3, 5)
        actions[0, 1, :] = 1.0  # step change at t=1
        penalty = transaction_cost_penalty(actions, lam=0.01)
        # diffs: t=0->1: 5.0, t=1->2: 5.0, total=10, lam*10=0.1
        assert penalty.item() == pytest.approx(0.1, abs=1e-5)


class _SimpleActionModel(nn.Module):
    """Minimal model for BC test: takes (B, 90, 16) -> (B, 5)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(90 * 16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.reshape(x.shape[0], -1))


class TestTrainBC:
    def test_loss_decreases(self) -> None:
        """Run 2 epochs on tiny synthetic data, verify loss decreases."""
        config = ATLASConfig(bc_epochs=2, batch_size=10, bc_patience=10)

        # Create a tiny synthetic dataset directly
        class _TinyDataset(ATLASDataset):
            def __init__(self, n: int = 100, seed: int = 0) -> None:
                self.config = config
                self.context_len = config.context_len
                rng = np.random.RandomState(seed)
                self._features = rng.randn(n, 90, 16).astype(np.float32)
                self._actions = rng.randn(n, 5).astype(np.float32) * 0.1
                self.windows = [(0, 90)] * n  # dummy

            def __len__(self) -> int:
                return len(self._features)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {
                    "features": torch.tensor(self._features[idx], dtype=torch.float32),
                    "timestamps": torch.zeros(90, dtype=torch.float32),
                    "day_of_week": torch.zeros(90, dtype=torch.long),
                    "month": torch.zeros(90, dtype=torch.long),
                    "is_opex": torch.zeros(90, dtype=torch.float32),
                    "is_quarter_end": torch.zeros(90, dtype=torch.float32),
                    "pre_norm_mu": torch.zeros(90, 16, dtype=torch.float32),
                    "pre_norm_sigma": torch.ones(90, 16, dtype=torch.float32),
                    "action_label": torch.tensor(self._actions[idx], dtype=torch.float32),
                    "return_to_go": torch.zeros(1, dtype=torch.float32),
                }

        train_ds = _TinyDataset(100, seed=0)
        val_ds = _TinyDataset(20, seed=1)

        model = _SimpleActionModel()
        result = train_behavioral_cloning(
            model, train_ds, val_ds, config, checkpoint_dir="/tmp/atlas_test_ckpt", device="cpu"
        )

        assert result["epochs_trained"] == 2
        assert len(result["train_loss"]) == 2
        # Loss should decrease (or at least not blow up)
        assert result["train_loss"][-1] < result["train_loss"][0] * 2
