from __future__ import annotations

import math

import torch
import torch.nn as nn

from trading_algo.quant_core.models.atlas.config import ATLASConfig


class DeStationaryModule(nn.Module):
    """Recovers non-stationary information lost during normalization.

    From Non-stationary Transformers (Liu et al., NeurIPS 2022).
    """

    def __init__(self, config: ATLASConfig) -> None:
        super().__init__()
        d = config.d_model
        hidden = d // 4

        # sigma_mean is scalar (1), concat with h gives d_model + 1
        self.mlp_tau = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.mlp_delta = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(
        self,
        pre_norm_sigma: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pre_norm_sigma: (B, L, n_features) — rolling std before normalization
            hidden_states: (B, L, d_model) — output from temporal backbone

        Returns:
            tau: (B, L, 1) — scaling factor, always positive
            delta: (B, L, d_model) — bias factor
        """
        # Average across features to get a scalar per timestep
        sigma_mean = pre_norm_sigma.mean(dim=-1, keepdim=True)  # (B, L, 1)
        mu_mean = pre_norm_sigma.mean(dim=-1, keepdim=True)  # (B, L, 1)

        cat_tau = torch.cat([sigma_mean, hidden_states], dim=-1)  # (B, L, d+1)
        tau = torch.exp(self.mlp_tau(cat_tau).clamp(-3, 3))  # (B, L, 1), bounded [e^-3, e^3] ≈ [0.05, 20]

        cat_delta = torch.cat([mu_mean, hidden_states], dim=-1)  # (B, L, d+1)
        delta = self.mlp_delta(cat_delta).clamp(-5, 5)  # (B, L, d_model), bounded bias

        return tau, delta


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with de-stationary modifications."""

    def __init__(self, config: ATLASConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.context_len = config.context_len

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        # Causal mask: lower-triangular, upper = -inf
        mask = torch.full((config.context_len, config.context_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, L, d_model) — hidden states
            tau: (B, L, 1) — de-stationary scaling factor
            delta: (B, L, d_model) — de-stationary bias

        Returns:
            (B, d_model) — last position output only
        """
        B, L, _ = h.shape

        Q = self.W_q(h)  # (B, L, d)
        K = self.W_k(h)  # (B, L, d)
        V = self.W_v(h)  # (B, L, d)

        # Reshape for multi-head: (B, n_heads, L, head_dim)
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # tau: (B, L, 1) -> (B, 1, L, 1) for broadcasting over heads and key dim
        tau_expanded = tau.unsqueeze(1)  # (B, 1, L, 1)

        # Standard QK^T: (B, n_heads, L, L)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1))  # (B, n_heads, L, L)

        # Scale FIRST, then mask (prevents -inf from interacting with scaling)
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        # De-stationary: scale by tau and add delta bias
        attn_logits = tau_expanded * attn_logits

        delta_reshaped = delta.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        delta_last = delta_reshaped[:, :, -1:, :]
        delta_bias = torch.matmul(delta_last, K.transpose(-2, -1))
        attn_logits = attn_logits + delta_bias

        # Clamp before mask+softmax to prevent overflow
        attn_logits = attn_logits.clamp(-50, 50)

        # Add causal mask
        mask = self.causal_mask[:L, :L]
        attn_logits = attn_logits + mask

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Attend
        out = torch.matmul(attn_weights, V)  # (B, n_heads, L, head_dim)

        # Concat heads: (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # Project
        out = self.W_o(out)

        # Return ONLY last position
        return out[:, -1, :]  # (B, d_model)


class CrossAttention(nn.Module):
    """Single-head cross-attention to historical memory bank."""

    def __init__(self, config: ATLASConfig) -> None:
        super().__init__()
        self.d_model = config.d_model

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, d_model) — current state representation
            memory_keys: (M, d_model) — memory bank context embeddings
            memory_values: (M, d_model) — memory bank value embeddings

        Returns:
            output: (B, d_model) — retrieved memory
            attention_weights: (B, M) — weights over memory entries
        """
        Q = self.W_q(query)  # (B, d_model)
        K = self.W_k(memory_keys)  # (M, d_model)
        V = self.W_v(memory_values)  # (M, d_model)

        # Q @ K^T: (B, d_model) @ (d_model, M) -> (B, M)
        scale = math.sqrt(self.d_model)
        alpha = (torch.matmul(Q, K.T) / scale).clamp(-20, 20)  # (B, M)
        alpha = torch.softmax(alpha, dim=-1)  # (B, M)

        # alpha @ V: (B, M) @ (M, d_model) -> (B, d_model)
        output = torch.matmul(alpha, V)  # (B, d_model)

        return output, alpha
