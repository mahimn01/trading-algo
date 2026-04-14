"""
ATLAS: Adaptive Trading via Learned Action Sequences

Full model assembly connecting all modules:
    Input → VSN → Mamba Backbone → De-Stationary Attention
    → Self-Attention + Cross-Attention → Return-Conditioned Fusion → Action Head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.time_encoding import Time2Vec, CalendarEmbedding
from trading_algo.quant_core.models.atlas.vsn import VariableSelectionNetwork
from trading_algo.quant_core.models.atlas.mamba import MambaBackbone
from trading_algo.quant_core.models.atlas.attention import (
    DeStationaryModule,
    CausalSelfAttention,
    CrossAttention,
)
from trading_algo.quant_core.models.atlas.fusion import ReturnConditionedFusion, ActionHead


class ATLASModel(nn.Module):
    """
    Full ATLAS model: market features → continuous action vector.

    Parameters: ~119K (intentionally small to prevent overfitting).

    Input:
        features:        (B, L, n_features)      normalized market features
        timestamps:      (B, L)                   unix timestamps for Time2Vec
        day_of_week:     (B, L)                   int [0-4]
        month:           (B, L)                   int [0-11]
        is_opex:         (B, L)                   binary
        is_quarter_end:  (B, L)                   binary
        pre_norm_mu:     (B, L, n_features)       for de-stationary attention
        pre_norm_sigma:  (B, L, n_features)       for de-stationary attention
        return_to_go:    (B,)                     target Sharpe

    Output:
        actions:         (B, 5)                   [delta, direction, leverage, dte, profit_target]
    """

    def __init__(self, config: ATLASConfig | None = None):
        super().__init__()
        self.config = config or ATLASConfig()
        c = self.config

        # Token dimension = n_features + n_time_features + n_calendar_features
        token_dim = c.n_features + c.n_time_features + c.n_calendar_features

        # Input encoding
        self.time2vec = Time2Vec(d_time=c.n_time_features)
        self.calendar_embed = CalendarEmbedding()
        self.vsn = VariableSelectionNetwork(
            n_features=token_dim,
            d_model=c.d_model,
            dropout=c.dropout,
        )

        # Temporal backbone
        self.mamba = MambaBackbone(
            d_model=c.d_model,
            d_state=c.d_state,
            d_conv=c.d_conv,
            expand_factor=c.expand_factor,
            n_layers=c.n_mamba_layers,
        )

        # De-stationary attention
        self.de_stationary = DeStationaryModule(c)

        # Dual attention
        self.self_attention = CausalSelfAttention(c)
        self.cross_attention = CrossAttention(c)

        # Fusion + action head
        self.fusion = ReturnConditionedFusion(c)
        self.action_head = ActionHead(c)

        # Value head for PPO (separate from action head, shared backbone)
        self.value_head = nn.Sequential(
            nn.Linear(c.d_model, c.d_model // 2),
            nn.GELU(),
            nn.Linear(c.d_model // 2, 1),
        )

        # Memory bank (registered as buffer — not a parameter)
        # Xavier-scale init for proper attention magnitude
        import math
        mem_scale = 1.0 / math.sqrt(c.d_model)
        self.register_buffer(
            "memory_keys",
            torch.randn(c.memory_bank_size, c.d_model) * mem_scale,
        )
        self.register_buffer(
            "memory_values",
            torch.randn(c.memory_bank_size, c.d_model) * mem_scale,
        )

        # PPO log-std (learnable, not part of the model per se)
        self.log_std = nn.Parameter(torch.full((c.action_dim,), -0.5))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init for linear layers, scaled output heads."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale action head output layer by 0.01 (small initial actions)
        with torch.no_grad():
            last_linear = self.action_head.net[-1]
            last_linear.weight.mul_(0.01)
            last_linear.bias.mul_(0.01)
            # Scale value head similarly
            self.value_head[-1].weight.mul_(0.01)
            self.value_head[-1].bias.mul_(0.01)

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
        """
        Full forward pass.

        Returns:
            actions: (B, 5) bounded continuous action vector
        """
        B, L, _ = features.shape

        # 1. Encode time and calendar
        time_enc = self.time2vec(timestamps)              # (B, L, 8)
        cal_enc = self.calendar_embed(
            day_of_week, month, is_opex, is_quarter_end,
        )                                                  # (B, L, 10)

        # 2. Concatenate into token
        token = torch.cat([features, time_enc, cal_enc], dim=-1)  # (B, L, 34)

        # 3. Variable Selection Network
        selected, vsn_weights = self.vsn(token)            # (B, L, d_model), (B, L, 34)

        # 4. Mamba temporal backbone
        temporal = self.mamba(selected)                     # (B, L, d_model)

        # 5. De-stationary attention modulation
        tau, delta = self.de_stationary(pre_norm_sigma, temporal)

        # 6a. Causal self-attention (last position)
        self_attn_out = self.self_attention(temporal, tau, delta)  # (B, d_model)

        # 6b. Cross-attention to historical memory bank
        cross_attn_out, attn_weights = self.cross_attention(
            self_attn_out, self.memory_keys, self.memory_values,
        )                                                  # (B, d_model), (B, M)

        # 7. Return-conditioned fusion
        fused = self.fusion(self_attn_out, cross_attn_out, return_to_go)  # (B, d_model)

        # 8. Action head
        actions = self.action_head(fused)                  # (B, 5)

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
        B, L, _ = features.shape

        time_enc = self.time2vec(timestamps)
        cal_enc = self.calendar_embed(day_of_week, month, is_opex, is_quarter_end)
        token = torch.cat([features, time_enc, cal_enc], dim=-1)

        selected, _ = self.vsn(token)
        temporal = self.mamba(selected)
        tau, delta = self.de_stationary(pre_norm_sigma, temporal)
        self_attn_out = self.self_attention(temporal, tau, delta)
        cross_attn_out, _ = self.cross_attention(
            self_attn_out, self.memory_keys, self.memory_values,
        )
        fused = self.fusion(self_attn_out, cross_attn_out, return_to_go)

        actions = self.action_head(fused)
        value = self.value_head(fused).squeeze(-1)

        return actions, value

    def get_action_distribution(
        self, actions_mean: Tensor,
    ) -> torch.distributions.Normal:
        """Create action distribution for PPO (diagonal Gaussian)."""
        std = torch.exp(self.log_std.clamp(-3, 1))  # clamp before exp: [e^-3, e^1] ≈ [0.05, 2.7]
        return torch.distributions.Normal(actions_mean, std)

    def set_memory_bank(self, keys: Tensor, values: Tensor) -> None:
        """Load a pre-computed memory bank."""
        assert keys.shape == self.memory_keys.shape, (
            f"Expected keys shape {self.memory_keys.shape}, got {keys.shape}"
        )
        assert values.shape == self.memory_values.shape
        self.memory_keys.copy_(keys)
        self.memory_values.copy_(values)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
