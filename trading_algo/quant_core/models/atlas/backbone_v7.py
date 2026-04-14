from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class mLSTMCell(nn.Module):
    """Matrix LSTM cell with exponential gating.

    Cell state is a matrix C of shape (d_head, d_head) — lower rank for param efficiency.
    Uses exponential forget/input gates for stable long-range memory.
    """

    def __init__(self, d_model: int = 128, d_head: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head

        self.W_fi = nn.Linear(d_model, 2)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_head)
        self.W_k = nn.Linear(d_model, d_head)
        self.W_v = nn.Linear(d_model, d_head)
        self.W_up = nn.Linear(d_head, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            h_seq: (B, L, d_model)
        """
        B, L, D = x.shape
        dh = self.d_head
        device = x.device
        dtype = x.dtype

        C = torch.zeros(B, dh, dh, device=device, dtype=dtype)
        n = torch.zeros(B, 1, device=device, dtype=dtype)

        h_list: list[Tensor] = []

        for t in range(L):
            x_t = x[:, t, :]  # (B, D)

            fi = self.W_fi(x_t).clamp(-5, 5)  # (B, 2)
            f_t = torch.exp(fi[:, 0:1])  # (B, 1)
            i_t = torch.exp(fi[:, 1:2])  # (B, 1)

            v_t = self.W_v(x_t)  # (B, dh)
            k_t = self.W_k(x_t)  # (B, dh)
            q_t = self.W_q(x_t)  # (B, dh)
            o_t = torch.sigmoid(self.W_o(x_t))  # (B, D)

            outer = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))  # (B, dh, dh)
            C = f_t.unsqueeze(-1) * C + i_t.unsqueeze(-1) * outer

            n = f_t * n + i_t  # (B, 1)

            Cq = torch.bmm(C, q_t.unsqueeze(2)).squeeze(2)  # (B, dh)
            denom = torch.clamp(n.abs(), min=1.0)  # (B, 1)
            h_raw = Cq / denom  # (B, dh)
            h_t = o_t * self.W_up(h_raw)  # (B, D)

            h_list.append(h_t)

        return torch.stack(h_list, dim=1)  # (B, L, D)


class mLSTMLayer(nn.Module):
    """mLSTM with pre-norm and residual connection."""

    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlstm = mLSTMCell(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.mlstm(self.norm(x)))


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal Transformer block."""

    def __init__(self, d_model: int = 128, n_heads: int = 2, ffn_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1)
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class HybridBackbone(nn.Module):
    """2x mLSTM + 2x Transformer hybrid backbone."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 2,
        n_mlstm_layers: int = 2,
        n_transformer_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(n_mlstm_layers):
            layers.append(mLSTMLayer(d_model, dropout))
        for _ in range(n_transformer_layers):
            layers.append(CausalTransformerBlock(d_model, n_heads, 2, dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
