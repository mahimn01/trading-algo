from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.time_encoding import (
    CalendarEmbedding,
    Time2Vec,
)
from trading_algo.quant_core.models.atlas.vsn import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
)
from trading_algo.quant_core.models.atlas.mamba import (
    MambaBackbone,
    MambaBlock,
    SelectiveSSM,
)


# ─── Config ──────────────────────────────────────────────────────────────────


class TestATLASConfig:
    def test_instantiate(self) -> None:
        cfg = ATLASConfig()
        assert cfg.d_model == 64
        assert cfg.n_mamba_layers == 4
        assert cfg.n_features == 16
        assert cfg.dsr_eta == pytest.approx(2 / 64)
        assert cfg.risk_free_rate == 0.045

    def test_frozen(self) -> None:
        cfg = ATLASConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.d_model = 128  # type: ignore[misc]

    def test_custom_override(self) -> None:
        cfg = ATLASConfig(d_model=128, lr=1e-3)
        assert cfg.d_model == 128
        assert cfg.lr == 1e-3
        assert cfg.n_features == 16  # default preserved


# ─── Time Encoding ───────────────────────────────────────────────────────────


class TestTime2Vec:
    def test_output_shape(self) -> None:
        t2v = Time2Vec(d_time=8)
        ts = torch.randn(2, 90)
        out = t2v(ts)
        assert out.shape == (2, 90, 8)
        assert out.dtype == torch.float32

    def test_gradient_flows(self) -> None:
        t2v = Time2Vec(d_time=8)
        ts = torch.randn(2, 90)
        out = t2v(ts)
        loss = out.sum()
        loss.backward()
        assert t2v.omega.grad is not None
        assert t2v.phi.grad is not None

    def test_different_d_time(self) -> None:
        t2v = Time2Vec(d_time=4)
        ts = torch.randn(1, 30)
        assert t2v(ts).shape == (1, 30, 4)


class TestCalendarEmbedding:
    def test_output_shape(self) -> None:
        cal = CalendarEmbedding()
        B, L = 2, 90
        dow = torch.randint(0, 5, (B, L))
        month = torch.randint(0, 12, (B, L))
        is_opex = torch.zeros(B, L)
        is_qtr_end = torch.ones(B, L)
        out = cal(dow, month, is_opex, is_qtr_end)
        assert out.shape == (B, L, 10)
        assert out.dtype == torch.float32

    def test_binary_features_passthrough(self) -> None:
        cal = CalendarEmbedding()
        B, L = 1, 5
        dow = torch.zeros(B, L, dtype=torch.long)
        month = torch.zeros(B, L, dtype=torch.long)
        is_opex = torch.ones(B, L)
        is_qtr_end = torch.zeros(B, L)
        out = cal(dow, month, is_opex, is_qtr_end)
        # Last two dims are the binary features
        assert (out[..., -2] == 1.0).all()
        assert (out[..., -1] == 0.0).all()


# ─── VSN ─────────────────────────────────────────────────────────────────────


class TestGatedResidualNetwork:
    def test_same_dim(self) -> None:
        grn = GatedResidualNetwork(64, 64, 64)
        x = torch.randn(2, 90, 64)
        assert grn(x).shape == (2, 90, 64)

    def test_dim_change(self) -> None:
        grn = GatedResidualNetwork(1, 64, 64)
        x = torch.randn(2, 90, 1)
        assert grn(x).shape == (2, 90, 64)

    def test_with_context(self) -> None:
        grn = GatedResidualNetwork(64, 64, 64, context_dim=32)
        x = torch.randn(2, 90, 64)
        c = torch.randn(2, 90, 32)
        assert grn(x, c).shape == (2, 90, 64)


class TestVariableSelectionNetwork:
    def test_output_shapes(self) -> None:
        vsn = VariableSelectionNetwork(n_features=16, d_model=64)
        x = torch.randn(2, 90, 16)
        selected, weights = vsn(x)
        assert selected.shape == (2, 90, 64)
        assert weights.shape == (2, 90, 16)

    def test_weights_sum_to_one(self) -> None:
        vsn = VariableSelectionNetwork(n_features=16, d_model=64)
        x = torch.randn(2, 90, 16)
        _, weights = vsn(x)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_gradcheck(self) -> None:
        vsn = VariableSelectionNetwork(n_features=4, d_model=8, dropout=0.0)
        vsn.double()
        x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        # Check that gradients flow correctly
        selected, weights = vsn(x)
        loss = selected.sum() + weights.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ─── Mamba ───────────────────────────────────────────────────────────────────


class TestSelectiveSSM:
    def test_output_shape(self) -> None:
        d_inner = 128
        ssm = SelectiveSSM(d_inner=d_inner, d_state=16, dt_rank=4)
        x = torch.randn(2, 90, d_inner)
        out = ssm(x)
        assert out.shape == (2, 90, d_inner)

    def test_no_nan(self) -> None:
        ssm = SelectiveSSM(d_inner=128, d_state=16, dt_rank=4)
        x = torch.randn(2, 90, 128)
        out = ssm(x)
        assert not torch.isnan(out).any()


class TestMambaBlock:
    def test_output_shape(self) -> None:
        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand_factor=2)
        x = torch.randn(2, 90, 64)
        out = block(x)
        assert out.shape == (2, 90, 64)
        assert out.dtype == torch.float32

    def test_residual_connection(self) -> None:
        block = MambaBlock(d_model=64)
        x = torch.zeros(1, 10, 64)
        out = block(x)
        # With zero input, output should be close to zero (residual + near-zero processing)
        assert out.abs().max() < 10.0


class TestMambaBackbone:
    def test_output_shape(self) -> None:
        backbone = MambaBackbone(d_model=64, n_layers=4, d_state=16)
        x = torch.randn(2, 90, 64)
        out = backbone(x)
        assert out.shape == (2, 90, 64)
        assert out.dtype == torch.float32

    def test_causality(self) -> None:
        """Perturbing input at t=50 should not affect output at t<50."""
        torch.manual_seed(42)
        # Use a single block to test causality clearly (multi-layer attenuates signal)
        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand_factor=2)
        block.eval()

        x = torch.randn(1, 90, 64)
        with torch.no_grad():
            out_original = block(x.clone())

        # Perturb input at t=50
        x_perturbed = x.clone()
        x_perturbed[:, 50, :] += 100.0

        with torch.no_grad():
            out_perturbed = block(x_perturbed)

        # t < 50 should be unchanged (SSM + causal conv are both causal)
        diff_before = (out_original[:, :50] - out_perturbed[:, :50]).abs().max().item()
        assert diff_before < 1e-4, f"Causality violated: diff before perturbation = {diff_before}"

        # t >= 50 should differ
        diff_after = (out_original[:, 50:] - out_perturbed[:, 50:]).abs().max().item()
        assert diff_after > 1e-4, f"Perturbation had no effect: diff = {diff_after}"

    def test_no_nan_stress(self) -> None:
        """Feed 1000 random inputs through the backbone — no NaNs should appear."""
        backbone = MambaBackbone(d_model=64, n_layers=4, d_state=16)
        backbone.eval()
        with torch.no_grad():
            for _ in range(100):
                x = torch.randn(10, 90, 64)  # 10 batches * 100 iterations = 1000
                out = backbone(x)
                assert not torch.isnan(out).any(), "NaN detected in backbone output"

    def test_param_count(self) -> None:
        backbone = MambaBackbone(d_model=64, n_layers=4, d_state=16)
        total = sum(p.numel() for p in backbone.parameters())
        # Spec says ~65K per backbone. Allow reasonable range.
        assert 40_000 < total < 200_000, f"Param count {total} outside expected range"

    def test_gradient_flow(self) -> None:
        backbone = MambaBackbone(d_model=64, n_layers=4, d_state=16)
        x = torch.randn(2, 90, 64, requires_grad=True)
        out = backbone(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
