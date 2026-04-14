from __future__ import annotations

import torch
import torch.nn as nn

from trading_algo.quant_core.models.atlas.config import ATLASConfig


class ReturnConditionedFusion(nn.Module):
    """Fuses self-attention and cross-attention outputs, conditioned on target Sharpe."""

    def __init__(self, config: ATLASConfig) -> None:
        super().__init__()
        d = config.d_model

        self.project = nn.Linear(2 * d, d)
        self.return_embed = nn.Linear(1, d)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        self_attn_out: torch.Tensor,
        cross_attn_out: torch.Tensor,
        return_to_go: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            self_attn_out: (B, d_model)
            cross_attn_out: (B, d_model)
            return_to_go: (B,) — target Sharpe ratio

        Returns:
            (B, d_model)
        """
        combined = torch.cat([self_attn_out, cross_attn_out], dim=-1)  # (B, 2d)
        projected = self.project(combined)  # (B, d)

        r_emb = self.return_embed(return_to_go.clamp(-5, 5).unsqueeze(-1))  # (B, d)

        return self.norm(projected + r_emb)  # (B, d)


class ActionHead(nn.Module):
    """Maps fused representation to bounded continuous action vector."""

    def __init__(self, config: ATLASConfig) -> None:
        super().__init__()
        d = config.d_model

        self.net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, config.action_dim),
        )

        self.delta_max = config.delta_max
        self.dte_min = config.dte_min
        self.dte_range = config.dte_max - config.dte_min

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_model) — fused representation

        Returns:
            (B, 5) — bounded action vector:
                [0] delta      in [0, 0.50]
                [1] direction  in [-1, +1]
                [2] leverage   in [0, 1]
                [3] dte        in [14, 90]
                [4] profit_tgt in [0, 1]
        """
        raw = self.net(z)  # (B, 5)

        delta = torch.sigmoid(raw[:, 0:1]) * self.delta_max
        direction = torch.tanh(raw[:, 1:2])
        leverage = torch.sigmoid(raw[:, 2:3])
        dte = torch.sigmoid(raw[:, 3:4]) * self.dte_range + self.dte_min
        profit_tgt = torch.sigmoid(raw[:, 4:5])

        return torch.cat([delta, direction, leverage, dte, profit_tgt], dim=-1)
