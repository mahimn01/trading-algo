from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.w4 = nn.Linear(hidden_dim, output_dim)

        # GLU gate: two projections -> sigmoid(a) * b
        self.w5 = nn.Linear(output_dim, output_dim)
        self.w6 = nn.Linear(output_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip projection if dims differ
        self.skip_proj = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )

    def forward(self, a: Tensor, c: Tensor | None = None) -> Tensor:
        residual = self.skip_proj(a) if self.skip_proj else a

        eta_1 = self.w1(a)
        eta_2 = self.w2(eta_1)
        if self.w3 is not None and c is not None:
            eta_2 = eta_2 + self.w3(c)
        eta_2 = F.elu(eta_2)
        eta_3 = self.w4(eta_2)
        eta_3 = self.dropout(eta_3)

        # GLU gate
        gate = torch.sigmoid(self.w5(eta_3)) * self.w6(eta_3)

        return self.layer_norm(residual + gate)


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        n_features: int = 16,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature GRNs: each maps scalar (1,) -> (d_model,)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout=dropout)
            for _ in range(n_features)
        ])

        # Selection GRN: maps flattened features (n_features,) -> (n_features,)
        self.selection_grn = GatedResidualNetwork(
            n_features, d_model, n_features, dropout=dropout
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, L, n_features)
        B, L, _F = x.shape

        # Process each feature independently
        processed = torch.stack(
            [self.feature_grns[i](x[..., i : i + 1]) for i in range(self.n_features)],
            dim=-2,
        )  # (B, L, n_features, d_model)

        # Compute selection weights (clamp before softmax to prevent overflow)
        weights = self.selection_grn(x)  # (B, L, n_features)
        weights = F.softmax(weights.clamp(-20, 20), dim=-1)

        # Weighted combination: (B, L, n_features, 1) * (B, L, n_features, d_model)
        selected = (weights.unsqueeze(-1) * processed).sum(dim=-2)  # (B, L, d_model)

        return selected, weights
