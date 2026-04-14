from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from trading_algo.quant_core.models.atlas.config_v7 import ATLASv7Config


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Stores per-instance statistics during normalize, restores during denormalize.
    """

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    def normalize(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, F)

        Returns:
            normalized x: (B, L, F)
        """
        self._mean = x.mean(dim=1, keepdim=True).detach()  # (B, 1, F)
        self._std = x.std(dim=1, keepdim=True).clamp(min=self.eps).detach()  # (B, 1, F)
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, F) or (B, F) — restores original scale

        Returns:
            denormalized x
        """
        assert self._mean is not None and self._std is not None
        if self.affine:
            x = (x - self.beta) / self.gamma.clamp(min=1e-8)
        if x.dim() == 2:
            mean = self._mean.squeeze(1)
            std = self._std.squeeze(1)
        else:
            mean = self._mean
            std = self._std
        return x * std + mean


class ReturnConditionedFusionV7(nn.Module):
    """FiLM-based fusion conditioned on target return-to-go."""

    def __init__(self, config: ATLASv7Config) -> None:
        super().__init__()
        d = config.d_model
        h = d // 4

        self.return_proj = nn.Sequential(
            nn.Linear(1, h),
            nn.GELU(),
        )
        self.to_scale = nn.Linear(h, d)
        self.to_shift = nn.Linear(h, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, return_to_go: Tensor) -> Tensor:
        """
        Args:
            x: (B, d_model) — backbone output
            return_to_go: (B,) — target Sharpe

        Returns:
            (B, d_model) — FiLM-modulated output
        """
        r = return_to_go.clamp(-5, 5).unsqueeze(-1)  # (B, 1)
        h = self.return_proj(r)  # (B, h)
        scale = self.to_scale(h)  # (B, d)
        shift = self.to_shift(h)  # (B, d)
        return self.norm(scale * x + shift)


class ActionHeadV7(nn.Module):
    """Maps fused representation to bounded actions using soft tanh bounds."""

    def __init__(self, config: ATLASv7Config) -> None:
        super().__init__()
        d = config.d_model

        self.net = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, config.action_dim),
        )

        self.delta_max = config.delta_max
        self.dte_min = config.dte_min
        self.dte_range = config.dte_max - config.dte_min

    @staticmethod
    def _soft_sigmoid(x: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.tanh(x / 2.0))

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, d_model)

        Returns:
            (B, 5) — [delta, direction, leverage, dte, profit_tgt]
        """
        raw = self.net(z)

        delta = self._soft_sigmoid(raw[:, 0:1]) * self.delta_max
        direction = torch.tanh(raw[:, 1:2])
        leverage = self._soft_sigmoid(raw[:, 2:3])
        dte = self._soft_sigmoid(raw[:, 3:4]) * self.dte_range + self.dte_min
        profit_tgt = self._soft_sigmoid(raw[:, 4:5])

        return torch.cat([delta, direction, leverage, dte, profit_tgt], dim=-1)
