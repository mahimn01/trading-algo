from __future__ import annotations

import torch
import pytest

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.attention import (
    DeStationaryModule,
    CausalSelfAttention,
    CrossAttention,
)
from trading_algo.quant_core.models.atlas.fusion import (
    ReturnConditionedFusion,
    ActionHead,
)


@pytest.fixture
def config() -> ATLASConfig:
    return ATLASConfig()


class TestDeStationaryModule:
    def test_tau_always_positive(self, config: ATLASConfig) -> None:
        mod = DeStationaryModule(config)
        mod.eval()
        for _ in range(1000):
            sigma = torch.randn(2, config.context_len, config.n_features)
            h = torch.randn(2, config.context_len, config.d_model)
            tau, _ = mod(sigma, h)
            assert (tau > 0).all(), "tau must be strictly positive"

    def test_output_shapes(self, config: ATLASConfig) -> None:
        mod = DeStationaryModule(config)
        sigma = torch.randn(2, config.context_len, config.n_features)
        h = torch.randn(2, config.context_len, config.d_model)
        tau, delta = mod(sigma, h)
        assert tau.shape == (2, config.context_len, 1)
        assert delta.shape == (2, config.context_len, config.d_model)


class TestCausalSelfAttention:
    def test_output_shape(self, config: ATLASConfig) -> None:
        mod = CausalSelfAttention(config)
        h = torch.randn(2, config.context_len, config.d_model)
        tau = torch.ones(2, config.context_len, 1)
        delta = torch.zeros(2, config.context_len, config.d_model)
        out = mod(h, tau, delta)
        assert out.shape == (2, config.d_model)

    def test_causal_masking(self, config: ATLASConfig) -> None:
        mod = CausalSelfAttention(config)
        mod.eval()
        h = torch.randn(2, config.context_len, config.d_model)
        tau = torch.ones(2, config.context_len, 1)
        delta = torch.zeros(2, config.context_len, config.d_model)

        # Hook to capture attention weights
        captured = {}

        def hook_fn(module, args, output):
            # Recompute attention weights to inspect
            B, L, _ = args[0].shape
            Q = module.W_q(args[0]).view(B, L, module.n_heads, module.head_dim).transpose(1, 2)
            K = module.W_k(args[0]).view(B, L, module.n_heads, module.head_dim).transpose(1, 2)
            tau_h = args[1].unsqueeze(1)
            logits = tau_h * torch.matmul(Q, K.transpose(-2, -1))
            mask = module.causal_mask[:L, :L]
            logits = (logits + mask) / (module.head_dim ** 0.5)
            weights = torch.softmax(logits, dim=-1)
            captured["weights"] = weights.detach()

        handle = mod.register_forward_hook(hook_fn)
        mod(h, tau, delta)
        handle.remove()

        weights = captured["weights"]
        L = config.context_len
        # Check upper triangle is effectively zero (future positions)
        for i in range(L):
            for j in range(i + 1, min(i + 3, L)):  # spot-check a few future positions
                assert weights[:, :, i, j].max().item() < 1e-6, (
                    f"Position {i} should not attend to future position {j}"
                )


class TestCrossAttention:
    def test_output_shapes(self, config: ATLASConfig) -> None:
        M = 50
        mod = CrossAttention(config)
        query = torch.randn(2, config.d_model)
        keys = torch.randn(M, config.d_model)
        values = torch.randn(M, config.d_model)
        out, alpha = mod(query, keys, values)
        assert out.shape == (2, config.d_model)
        assert alpha.shape == (2, M)

    def test_attention_weights_sum_to_one(self, config: ATLASConfig) -> None:
        M = 100
        mod = CrossAttention(config)
        query = torch.randn(2, config.d_model)
        keys = torch.randn(M, config.d_model)
        values = torch.randn(M, config.d_model)
        _, alpha = mod(query, keys, values)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestReturnConditionedFusion:
    def test_output_shape(self, config: ATLASConfig) -> None:
        mod = ReturnConditionedFusion(config)
        s = torch.randn(2, config.d_model)
        m = torch.randn(2, config.d_model)
        r = torch.randn(2)
        out = mod(s, m, r)
        assert out.shape == (2, config.d_model)


class TestActionHead:
    def test_output_shape(self, config: ATLASConfig) -> None:
        mod = ActionHead(config)
        z = torch.randn(2, config.d_model)
        out = mod(z)
        assert out.shape == (2, 5)

    def test_bounds_on_random_inputs(self, config: ATLASConfig) -> None:
        mod = ActionHead(config)
        mod.eval()
        z = torch.randn(10_000, config.d_model)
        actions = mod(z)

        delta = actions[:, 0]
        direction = actions[:, 1]
        leverage = actions[:, 2]
        dte = actions[:, 3]
        profit_tgt = actions[:, 4]

        assert delta.min() >= 0.0, f"delta min {delta.min().item()} < 0"
        assert delta.max() <= 0.50 + 1e-6, f"delta max {delta.max().item()} > 0.50"
        assert direction.min() >= -1.0 - 1e-6, f"direction min {direction.min().item()} < -1"
        assert direction.max() <= 1.0 + 1e-6, f"direction max {direction.max().item()} > 1"
        assert leverage.min() >= 0.0, f"leverage min {leverage.min().item()} < 0"
        assert leverage.max() <= 1.0 + 1e-6, f"leverage max {leverage.max().item()} > 1"
        assert dte.min() >= 14.0 - 1e-4, f"dte min {dte.min().item()} < 14"
        assert dte.max() <= 90.0 + 1e-4, f"dte max {dte.max().item()} > 90"
        assert profit_tgt.min() >= 0.0, f"profit_tgt min {profit_tgt.min().item()} < 0"
        assert profit_tgt.max() <= 1.0 + 1e-6, f"profit_tgt max {profit_tgt.max().item()} > 1"


class TestGradientFlow:
    def test_all_params_receive_gradients(self, config: ATLASConfig) -> None:
        B = 2
        L = config.context_len
        M = 50

        destat = DeStationaryModule(config)
        self_attn = CausalSelfAttention(config)
        cross_attn = CrossAttention(config)
        fusion = ReturnConditionedFusion(config)
        action_head = ActionHead(config)

        # Inputs
        sigma = torch.randn(B, L, config.n_features)
        h = torch.randn(B, L, config.d_model, requires_grad=True)
        mem_keys = torch.randn(M, config.d_model)
        mem_values = torch.randn(M, config.d_model)
        rtg = torch.randn(B)

        # Forward
        tau, delta = destat(sigma, h)
        s = self_attn(h, tau, delta)
        m, _ = cross_attn(s, mem_keys, mem_values)
        z = fusion(s, m, rtg)
        actions = action_head(z)

        loss = actions.sum()
        loss.backward()

        all_modules = [destat, self_attn, cross_attn, fusion, action_head]
        for mod in all_modules:
            for name, param in mod.named_parameters():
                assert param.grad is not None, f"{mod.__class__.__name__}.{name} has no gradient"
                assert not torch.all(param.grad == 0), (
                    f"{mod.__class__.__name__}.{name} has all-zero gradient"
                )
