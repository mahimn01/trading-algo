from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def entmax15(z: Tensor, dim: int = -1, n_iter: int = 25) -> Tensor:
    """Entmax 1.5 via bisection. Sparse alternative to softmax."""
    z_max = z.max(dim=dim, keepdim=True).values
    z = z - z_max

    lo = z.min(dim=dim, keepdim=True).values - 1.0
    hi = z.max(dim=dim, keepdim=True).values + 1.0

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        p = F.relu(z - mid).pow(2)
        total = p.sum(dim=dim, keepdim=True)
        lo = torch.where(total > 1.0, mid, lo)
        hi = torch.where(total <= 1.0, mid, hi)

    tau = (lo + hi) / 2.0
    return F.relu(z - tau).pow(2).clamp(min=0.0)


class GRNLite(nn.Module):
    """Lightweight GRN for selection weights."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class VariableSelectionNetworkV7(nn.Module):
    """Lightweight VSN with shared projection + per-feature FiLM + entmax selection.

    ~20K params instead of ~579K in the original.
    """

    def __init__(self, n_features: int = 34, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.shared_proj = nn.Linear(1, d_model)

        self.film_scale = nn.Parameter(torch.ones(n_features, d_model))
        self.film_bias = nn.Parameter(torch.zeros(n_features, d_model))

        self.selection_grn = GRNLite(n_features, d_model // 4, n_features, dropout=dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, L, n_features) — raw token features

        Returns:
            selected: (B, L, d_model) — weighted combination
            weights: (B, L, n_features) — sparse selection weights
        """
        B, L, F = x.shape

        projected = self.shared_proj(x.unsqueeze(-1))  # (B, L, F, d_model)

        filmed = projected * self.film_scale + self.film_bias  # (B, L, F, d_model)

        raw_weights = self.selection_grn(x)  # (B, L, F)
        weights = entmax15(raw_weights.clamp(-20, 20), dim=-1)  # (B, L, F)

        selected = (weights.unsqueeze(-1) * filmed).sum(dim=-2)  # (B, L, d_model)

        return selected, weights
