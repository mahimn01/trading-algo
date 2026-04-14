from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from trading_algo.quant_core.models.atlas.config_v7 import ATLASv7Config


class DeStationaryCausalAttention(nn.Module):
    """Multi-head causal self-attention with integrated de-stationary modulation.

    Fixes from original:
    - Accepts both mu AND sigma (original used sigma for both)
    - mu drives delta (bias), sigma drives tau (scale)
    - All numerical clamps applied
    """

    def __init__(self, config: ATLASv7Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        d_kv = config.d_model // 2
        self.d_kv = d_kv
        self.kv_head_dim = d_kv // config.n_heads
        self.seq_len = config.seq_len

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, d_kv, bias=False)
        self.W_v = nn.Linear(self.d_model, d_kv, bias=False)
        self.W_o = nn.Linear(d_kv, self.d_model)

        hidden = self.d_model // 8
        self.tau_net = nn.Sequential(
            nn.Linear(self.d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(self.d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_kv),
        )

        mask = torch.full((self.seq_len, self.seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        h: Tensor,
        pre_norm_mu: Tensor,
        pre_norm_sigma: Tensor,
    ) -> Tensor:
        """
        Args:
            h: (B, L, d_model) — hidden states from backbone
            pre_norm_mu: (B, L, n_features) — rolling mean before normalization
            pre_norm_sigma: (B, L, n_features) — rolling std before normalization

        Returns:
            (B, d_model) — last position output
        """
        B, L, _ = h.shape

        sigma_mean = pre_norm_sigma.mean(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, L, 1)
        mu_mean = pre_norm_mu.mean(dim=-1, keepdim=True)  # (B, L, 1)

        cat_tau = torch.cat([sigma_mean, h], dim=-1)  # (B, L, d+1)
        tau = torch.exp(self.tau_net(cat_tau).clamp(-3, 3))  # (B, L, 1)

        cat_delta = torch.cat([mu_mean, h], dim=-1)  # (B, L, d+1)
        delta = self.delta_net(cat_delta).clamp(-5, 5)  # (B, L, d)

        Q = self.W_q(h)  # (B, L, d_model)
        K = self.W_k(h)  # (B, L, d_kv)
        V = self.W_v(h)  # (B, L, d_kv)

        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, L, hd)
        K = K.view(B, L, self.n_heads, self.kv_head_dim).transpose(1, 2)  # (B, nh, L, kv_hd)
        V = V.view(B, L, self.n_heads, self.kv_head_dim).transpose(1, 2)  # (B, nh, L, kv_hd)

        # Q @ K^T with asymmetric dims: (B, nh, L, hd) @ (B, nh, kv_hd, L) -> need projection
        # Use only the first kv_head_dim dims of Q for attention scores
        Q_for_attn = Q[..., :self.kv_head_dim]
        attn_logits = torch.matmul(Q_for_attn, K.transpose(-2, -1)) / math.sqrt(self.kv_head_dim)

        tau_expanded = tau.unsqueeze(1)  # (B, 1, L, 1)
        attn_logits = tau_expanded * attn_logits

        delta_reshaped = delta.view(B, L, self.n_heads, self.kv_head_dim).transpose(1, 2)
        delta_last = delta_reshaped[:, :, -1:, :]
        delta_bias = torch.matmul(delta_last, K.transpose(-2, -1))
        attn_logits = attn_logits + delta_bias

        attn_logits = attn_logits.clamp(-50, 50)

        mask = self.causal_mask[:L, :L]
        attn_logits = attn_logits + mask

        attn_weights = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn_weights, V)  # (B, nh, L, kv_hd)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_kv)
        out = self.W_o(out)

        return out[:, -1, :]  # (B, d_model)
