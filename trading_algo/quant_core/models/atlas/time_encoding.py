from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Time2Vec(nn.Module):
    def __init__(self, d_time: int = 8) -> None:
        super().__init__()
        self.d_time = d_time
        self.omega = nn.Parameter(torch.randn(d_time) * 0.1)  # Small init
        self.phi = nn.Parameter(torch.randn(d_time) * 0.1)

    def forward(self, timestamps: Tensor) -> Tensor:
        # timestamps: (B, L)
        std = timestamps.std(dim=-1, keepdim=True).clamp(min=1e-4)
        tau = (timestamps - timestamps.mean(dim=-1, keepdim=True)) / std
        tau = tau.clamp(-10, 10).unsqueeze(-1)  # (B, L, 1)
        # Clamp omega/phi to prevent unbounded frequency growth
        raw = self.omega.clamp(-3, 3) * tau + self.phi.clamp(-3, 3)
        te = torch.cat([raw[..., :1], torch.sin(raw[..., 1:])], dim=-1)
        return te


class CalendarEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dow_embed = nn.Embedding(5, 4)
        self.month_embed = nn.Embedding(12, 4)

    def forward(
        self,
        dow: Tensor,
        month: Tensor,
        is_opex: Tensor,
        is_qtr_end: Tensor,
    ) -> Tensor:
        # dow: (B, L) long [0-4], month: (B, L) long [0-11]
        # is_opex, is_qtr_end: (B, L) float
        d = self.dow_embed(dow)        # (B, L, 4)
        m = self.month_embed(month)    # (B, L, 4)
        binary = torch.stack([is_opex, is_qtr_end], dim=-1)  # (B, L, 2)
        return torch.cat([d, m, binary], dim=-1)  # (B, L, 10)
