from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from trading_algo.quant_core.models.atlas.config_v7 import ATLASv7Config
from trading_algo.quant_core.models.atlas.time_encoding import Time2Vec, CalendarEmbedding
from trading_algo.quant_core.models.atlas.vsn_v7 import VariableSelectionNetworkV7
from trading_algo.quant_core.models.atlas.backbone_v7 import HybridBackbone
from trading_algo.quant_core.models.atlas.attention_v7 import DeStationaryCausalAttention
from trading_algo.quant_core.models.atlas.fusion_v7 import (
    RevIN,
    ReturnConditionedFusionV7,
    ActionHeadV7,
)


class ATLASModelV7(nn.Module):
    """ATLAS v7: Adaptive Trading via Learned Action Sequences.

    Half the params of v6 (~410K vs 835K) with 2x d_model (128 vs 64).
    Hybrid mLSTM + Transformer backbone, patched input, FiLM fusion, RevIN.
    """

    def __init__(self, config: ATLASv7Config | None = None) -> None:
        super().__init__()
        self.config = config or ATLASv7Config()
        c = self.config

        token_dim = c.token_dim  # 34

        self.time2vec = Time2Vec(d_time=c.n_time_features)
        self.calendar_embed = CalendarEmbedding()

        self.revin = RevIN(n_features=c.n_features, affine=True)

        self.patch_embed = nn.Conv1d(
            in_channels=token_dim,
            out_channels=c.d_model,
            kernel_size=c.patch_size,
            stride=c.patch_size,
        )

        self.vsn = VariableSelectionNetworkV7(
            n_features=c.d_model,
            d_model=c.d_model,
            dropout=c.dropout,
        )

        self.backbone = HybridBackbone(
            d_model=c.d_model,
            n_heads=c.n_heads,
            n_mlstm_layers=c.n_mlstm_layers,
            n_transformer_layers=c.n_transformer_layers,
            ffn_mult=4,
            dropout=c.dropout,
        )

        self.attention = DeStationaryCausalAttention(c)

        self.fusion = ReturnConditionedFusionV7(c)
        self.action_head = ActionHeadV7(c)

        self.value_adapter = nn.Sequential(
            nn.Linear(c.d_model, c.d_model // 4),
            nn.GELU(),
            nn.Linear(c.d_model // 4, 1),
        )

        self.log_std = nn.Parameter(torch.full((c.action_dim,), -0.5))

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        with torch.no_grad():
            last_action = self.action_head.net[-1]
            last_action.weight.mul_(0.01)
            last_action.bias.mul_(0.01)
            self.value_adapter[-1].weight.mul_(0.01)
            self.value_adapter[-1].bias.mul_(0.01)

    def _encode_and_patch(
        self,
        features: Tensor,
        timestamps: Tensor,
        day_of_week: Tensor,
        month: Tensor,
        is_opex: Tensor,
        is_quarter_end: Tensor,
    ) -> Tensor:
        """Encode inputs, apply RevIN to features, concatenate, and patch.

        Returns:
            patches: (B, seq_len, d_model)
        """
        B, L, _ = features.shape

        normed_features = self.revin.normalize(features)  # (B, L, n_features)

        time_enc = self.time2vec(timestamps)  # (B, L, 8)
        cal_enc = self.calendar_embed(day_of_week, month, is_opex, is_quarter_end)  # (B, L, 10)

        token = torch.cat([normed_features, time_enc, cal_enc], dim=-1)  # (B, L, 34)

        # Conv1d expects (B, C, L)
        patches = self.patch_embed(token.transpose(1, 2)).transpose(1, 2)  # (B, seq_len, d_model)

        return patches

    def _backbone_forward(
        self,
        features: Tensor,
        timestamps: Tensor,
        day_of_week: Tensor,
        month: Tensor,
        is_opex: Tensor,
        is_quarter_end: Tensor,
        pre_norm_mu: Tensor,
        pre_norm_sigma: Tensor,
    ) -> Tensor:
        """Shared computation through backbone + attention. Returns (B, d_model)."""
        patches = self._encode_and_patch(features, timestamps, day_of_week, month, is_opex, is_quarter_end)

        selected, _ = self.vsn(patches)  # (B, seq_len, d_model)

        temporal = self.backbone(selected)  # (B, seq_len, d_model)

        # Pool pre_norm stats to match patch sequence length
        B, L, F = pre_norm_mu.shape
        seq_len = temporal.shape[1]
        ps = self.config.patch_size

        mu_patched = pre_norm_mu[:, :seq_len * ps, :].reshape(B, seq_len, ps, F).mean(dim=2)
        sigma_patched = pre_norm_sigma[:, :seq_len * ps, :].reshape(B, seq_len, ps, F).mean(dim=2)

        out = self.attention(temporal, mu_patched, sigma_patched)  # (B, d_model)

        return out

    def forward(
        self,
        features: Tensor,
        timestamps: Tensor,
        day_of_week: Tensor,
        month: Tensor,
        is_opex: Tensor,
        is_quarter_end: Tensor,
        pre_norm_mu: Tensor,
        pre_norm_sigma: Tensor,
        return_to_go: Tensor,
    ) -> Tensor:
        """Full forward pass. Same signature as ATLASModel.

        Returns:
            actions: (B, 5) bounded continuous action vector
        """
        backbone_out = self._backbone_forward(
            features, timestamps, day_of_week, month, is_opex, is_quarter_end,
            pre_norm_mu, pre_norm_sigma,
        )

        fused = self.fusion(backbone_out, return_to_go)

        actions = self.action_head(fused)

        return actions

    def forward_with_value(
        self,
        features: Tensor,
        timestamps: Tensor,
        day_of_week: Tensor,
        month: Tensor,
        is_opex: Tensor,
        is_quarter_end: Tensor,
        pre_norm_mu: Tensor,
        pre_norm_sigma: Tensor,
        return_to_go: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning both actions and value estimate (for PPO)."""
        backbone_out = self._backbone_forward(
            features, timestamps, day_of_week, month, is_opex, is_quarter_end,
            pre_norm_mu, pre_norm_sigma,
        )

        fused = self.fusion(backbone_out, return_to_go)
        actions = self.action_head(fused)

        value = self.value_adapter(backbone_out.detach()).squeeze(-1)

        return actions, value

    def get_action_distribution(
        self, actions_mean: Tensor,
    ) -> torch.distributions.Normal:
        std = torch.exp(self.log_std.clamp(-3, 1))
        return torch.distributions.Normal(actions_mean, std)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    config = ATLASv7Config()
    model = ATLASModelV7(config).to(device)

    # --- Param count per module ---
    print("\n=== Parameter Count ===")
    module_params: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        module_params[top] = module_params.get(top, 0) + param.numel()

    total = 0
    for mod, count in sorted(module_params.items(), key=lambda x: -x[1]):
        print(f"  {mod:25s} {count:>8,}")
        total += count
    print(f"  {'TOTAL':25s} {total:>8,}")

    # --- Forward pass test ---
    B, L = 4, 90
    inputs = {
        "features": torch.randn(B, L, config.n_features, device=device),
        "timestamps": torch.randn(B, L, device=device),
        "day_of_week": torch.randint(0, 5, (B, L), device=device),
        "month": torch.randint(0, 12, (B, L), device=device),
        "is_opex": torch.randint(0, 2, (B, L), device=device).float(),
        "is_quarter_end": torch.randint(0, 2, (B, L), device=device).float(),
        "pre_norm_mu": torch.randn(B, L, config.n_features, device=device),
        "pre_norm_sigma": torch.randn(B, L, config.n_features, device=device).abs() + 0.1,
        "return_to_go": torch.randn(B, device=device),
    }

    print("\n=== Forward Pass ===")
    actions = model(**inputs)
    print(f"  actions shape: {actions.shape}")
    print(f"  actions range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")

    # --- Forward with value ---
    print("\n=== Forward With Value ===")
    actions2, values = model.forward_with_value(**inputs)
    print(f"  actions shape: {actions2.shape}, values shape: {values.shape}")

    # --- Backward pass ---
    print("\n=== Backward Pass ===")
    loss = actions.sum() + values.sum()
    loss.backward()
    grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            top = name.split(".")[0]
            norm = param.grad.norm().item()
            grad_norms[top] = max(grad_norms.get(top, 0.0), norm)
    for mod, norm in sorted(grad_norms.items()):
        print(f"  {mod:25s} max_grad_norm={norm:.6f}")
    expected_no_grad = {"log_std"}
    no_grad = [n for n, p in model.named_parameters() if p.grad is None and p.requires_grad and n not in expected_no_grad]
    if no_grad:
        print(f"  WARNING: no gradient for: {no_grad}")
        sys.exit(1)
    else:
        print("  All parameters received gradients (log_std excluded — only used in PPO sampling).")

    # --- Distribution test ---
    print("\n=== Action Distribution ===")
    dist = model.get_action_distribution(actions.detach())
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    print(f"  sample shape: {sample.shape}, log_prob shape: {log_prob.shape}")

    # --- NaN stress test ---
    print("\n=== NaN Stress Test (100 random inputs) ===")
    model.eval()
    nan_count = 0
    for i in range(100):
        test_inputs = {
            "features": torch.randn(2, L, config.n_features, device=device) * (1 + i * 0.1),
            "timestamps": torch.randn(2, L, device=device) * 1e6,
            "day_of_week": torch.randint(0, 5, (2, L), device=device),
            "month": torch.randint(0, 12, (2, L), device=device),
            "is_opex": torch.randint(0, 2, (2, L), device=device).float(),
            "is_quarter_end": torch.randint(0, 2, (2, L), device=device).float(),
            "pre_norm_mu": torch.randn(2, L, config.n_features, device=device) * 10,
            "pre_norm_sigma": torch.randn(2, L, config.n_features, device=device).abs() + 0.01,
            "return_to_go": torch.randn(2, device=device) * 5,
        }
        with torch.no_grad():
            out = model(**test_inputs)
        if torch.isnan(out).any() or torch.isinf(out).any():
            nan_count += 1
    if nan_count > 0:
        print(f"  FAIL: {nan_count}/100 inputs produced NaN/Inf")
        sys.exit(1)
    else:
        print("  PASS: 0/100 NaN/Inf")

    # --- Output bounds check ---
    print("\n=== Output Bounds ===")
    with torch.no_grad():
        a = model(**inputs)
    delta, direction, leverage, dte, profit = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    checks = [
        ("delta",     delta.min().item() >= 0 and delta.max().item() <= 0.5),
        ("direction", direction.min().item() >= -1 and direction.max().item() <= 1),
        ("leverage",  leverage.min().item() >= 0 and leverage.max().item() <= 1),
        ("dte",       dte.min().item() >= 14 and dte.max().item() <= 90),
        ("profit",    profit.min().item() >= 0 and profit.max().item() <= 1),
    ]
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:12s} {status}")
        if not ok:
            all_ok = False
    if not all_ok:
        sys.exit(1)

    print("\n=== All tests passed. ===")
