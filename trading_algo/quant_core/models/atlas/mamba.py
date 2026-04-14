from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CausalTransformerBlock(nn.Module):
    """Causal Transformer block — drop-in replacement for MambaBlock.

    Pre-norm architecture: LayerNorm → causal MHA → residual → LayerNorm → FFN → residual.
    torch.nn.MultiheadAttention dispatches to Apple Metal kernels on MPS, making
    this ~10-50× faster than Mamba's sequential scan for the same d_model and L.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model)
        B, L, D = x.shape
        # Additive causal mask: -inf in upper triangle prevents attending to future tokens
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        # Pre-norm causal self-attention + residual
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, attn_mask=mask, need_weights=False)
        x = x + attn_out
        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


# Alias kept so existing code importing MambaBlock still works
MambaBlock = CausalTransformerBlock


class MambaBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        # Legacy Mamba params — accepted for API compatibility, not used
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, ffn_mult)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
