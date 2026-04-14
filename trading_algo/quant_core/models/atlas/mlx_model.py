"""ATLAS model ported to Apple MLX for native Metal GPU training.

Unified memory (zero-copy), fused Metal kernels, lazy evaluation.
All modules match the PyTorch version exactly for weight compatibility.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Time2Vec + Calendar
# ---------------------------------------------------------------------------

class Time2Vec(nn.Module):
    def __init__(self, d_time: int = 8):
        super().__init__()
        self.d_time = d_time
        self.omega = mx.random.normal((d_time,)) * 0.01
        self.phi = mx.random.normal((d_time,)) * 0.01

    def __call__(self, timestamps: mx.array) -> mx.array:
        # timestamps: (B, L)
        mu = mx.mean(timestamps, axis=-1, keepdims=True)
        sigma = mx.maximum(mx.std(timestamps, axis=-1, keepdims=True), mx.array(1e-8))
        tau = mx.expand_dims((timestamps - mu) / sigma, axis=-1)  # (B, L, 1)
        raw = self.omega * tau + self.phi  # (B, L, d_time)
        return mx.concatenate([raw[..., :1], mx.sin(raw[..., 1:])], axis=-1)


class CalendarEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.dow_embed = nn.Embedding(5, 4)
        self.month_embed = nn.Embedding(12, 4)

    def __call__(self, dow: mx.array, month: mx.array,
                 is_opex: mx.array, is_qtr_end: mx.array) -> mx.array:
        d = self.dow_embed(dow)     # (B, L, 4)
        m = self.month_embed(month) # (B, L, 4)
        binary = mx.stack([is_opex, is_qtr_end], axis=-1)  # (B, L, 2)
        return mx.concatenate([d, m, binary], axis=-1)      # (B, L, 10)


# ---------------------------------------------------------------------------
# GRN + Variable Selection Network
# ---------------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w4 = nn.Linear(hidden_dim, output_dim)
        self.w5 = nn.Linear(output_dim, output_dim)
        self.w6 = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def __call__(self, a: mx.array) -> mx.array:
        residual = a if self.skip_proj is None else self.skip_proj(a)
        h = self.w1(a)
        h = nn.elu(self.w2(h))
        h = self.dropout(self.w4(h))
        gate = mx.sigmoid(self.w5(h)) * self.w6(h)
        return self.layer_norm(residual + gate)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.feature_grns = [
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ]
        self.selection_grn = GatedResidualNetwork(n_features, d_model, n_features, dropout)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B, L, n_features)
        processed = mx.stack(
            [self.feature_grns[i](x[..., i:i+1]) for i in range(self.n_features)],
            axis=-2,
        )  # (B, L, n_features, d_model)
        weights = mx.softmax(self.selection_grn(x), axis=-1)  # (B, L, n_features)
        w = mx.expand_dims(weights, axis=-1)                   # (B, L, n_features, 1)
        selected = mx.sum(w * processed, axis=-2)              # (B, L, d_model)
        return selected, weights


# ---------------------------------------------------------------------------
# Causal Transformer Block (replaces Mamba SSM)
# ---------------------------------------------------------------------------

class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, d_model)
        L = x.shape[1]
        mask = mx.triu(mx.full((L, L), float('-inf')), k=1)
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class MambaBackbone(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4, n_heads: int = 4,
                 ffn_mult: int = 4, **_kwargs):
        super().__init__()
        self.layers = [
            CausalTransformerBlock(d_model, n_heads, ffn_mult)
            for _ in range(n_layers)
        ]
        self.final_norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# De-Stationary Attention
# ---------------------------------------------------------------------------

class DeStationaryModule(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.mlp_tau = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.mlp_delta = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

    def __call__(self, pre_norm_sigma: mx.array, hidden: mx.array
                 ) -> tuple[mx.array, mx.array]:
        sigma_mean = mx.mean(pre_norm_sigma, axis=-1, keepdims=True)  # (B, L, 1)
        cat_tau = mx.concatenate([sigma_mean, hidden], axis=-1)       # (B, L, d+1)
        tau = mx.exp(self.mlp_tau(cat_tau))                           # (B, L, 1), > 0
        cat_delta = mx.concatenate([sigma_mean, hidden], axis=-1)
        delta = self.mlp_delta(cat_delta)                             # (B, L, d)
        return tau, delta


# ---------------------------------------------------------------------------
# Causal Self-Attention (with de-stationary modulation)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 2, context_len: int = 90):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        # Causal mask stored as frozen parameter
        self._causal_mask = mx.triu(mx.full((context_len, context_len), float('-inf')), k=1)

    def __call__(self, h: mx.array, tau: mx.array, delta: mx.array) -> mx.array:
        B, L, _ = h.shape
        Q = self.W_q(h).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.W_k(h).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.W_v(h).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # (B, n_heads, L, head_dim)

        attn = Q @ K.transpose(0, 1, 3, 2)  # (B, n_heads, L, L)

        # De-stationary: tau scaling
        tau_exp = mx.expand_dims(tau, axis=1)  # (B, 1, L, 1)
        attn = tau_exp * attn

        # De-stationary: delta bias (last position only)
        delta_h = delta.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        delta_last = delta_h[:, :, -1:, :]  # (B, n_heads, 1, head_dim)
        delta_bias = delta_last @ K.transpose(0, 1, 3, 2)  # (B, n_heads, 1, L)
        attn = attn + delta_bias

        # Causal mask + scale + softmax
        mask = self._causal_mask[:L, :L]
        attn = (attn + mask) / self.scale
        attn = mx.softmax(attn, axis=-1)

        out = attn @ V  # (B, n_heads, L, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        out = self.W_o(out)
        return out[:, -1, :]  # (B, d_model) — last position only


# ---------------------------------------------------------------------------
# Cross-Attention (to memory bank)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, query: mx.array, mem_k: mx.array, mem_v: mx.array
                 ) -> tuple[mx.array, mx.array]:
        Q = self.W_q(query)    # (B, d)
        K = self.W_k(mem_k)    # (M, d)
        V = self.W_v(mem_v)    # (M, d)
        scale = math.sqrt(self.d_model)
        alpha = mx.softmax(Q @ K.T / scale, axis=-1)  # (B, M)
        out = alpha @ V                                 # (B, d)
        return out, alpha


# ---------------------------------------------------------------------------
# Fusion + Action Head
# ---------------------------------------------------------------------------

class ReturnConditionedFusion(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.project = nn.Linear(2 * d_model, d_model)
        self.return_embed = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, self_attn: mx.array, cross_attn: mx.array,
                 rtg: mx.array) -> mx.array:
        combined = mx.concatenate([self_attn, cross_attn], axis=-1)
        projected = self.project(combined)
        r_emb = self.return_embed(mx.expand_dims(rtg, axis=-1))
        return self.norm(projected + r_emb)


class ActionHead(nn.Module):
    def __init__(self, d_model: int = 64, action_dim: int = 5,
                 delta_max: float = 0.50, dte_min: int = 14, dte_max: int = 90):
        super().__init__()
        self.delta_max = delta_max
        self.dte_min = dte_min
        self.dte_range = dte_max - dte_min
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
        )

    def __call__(self, z: mx.array) -> mx.array:
        raw = self.net(z)  # (B, 5)
        delta = mx.sigmoid(raw[:, 0:1]) * self.delta_max
        direction = mx.tanh(raw[:, 1:2])
        leverage = mx.sigmoid(raw[:, 2:3])
        dte = mx.sigmoid(raw[:, 3:4]) * self.dte_range + self.dte_min
        pt = mx.sigmoid(raw[:, 4:5])
        return mx.concatenate([delta, direction, leverage, dte, pt], axis=-1)


# ---------------------------------------------------------------------------
# Normal distribution utilities (MLX has no torch.distributions)
# ---------------------------------------------------------------------------

_LOG2PI = math.log(2 * math.pi)


def normal_log_prob(x: mx.array, mean: mx.array, log_std: mx.array) -> mx.array:
    """Per-dimension log probability under diagonal Gaussian."""
    std = mx.exp(log_std)
    return -0.5 * (((x - mean) / std) ** 2 + 2 * log_std + _LOG2PI)


def normal_entropy(log_std: mx.array) -> mx.array:
    """Per-dimension entropy of diagonal Gaussian."""
    return 0.5 + 0.5 * _LOG2PI + log_std


def normal_sample(mean: mx.array, log_std: mx.array) -> mx.array:
    """Reparameterized sample from diagonal Gaussian."""
    std = mx.clip(mx.exp(log_std), 0.05, 1.0)
    eps = mx.random.normal(mean.shape)
    return mean + std * eps


# ---------------------------------------------------------------------------
# Full ATLAS Model
# ---------------------------------------------------------------------------

class ATLASModel(nn.Module):
    def __init__(self, d_model: int = 64, n_features: int = 16,
                 n_time: int = 8, n_calendar: int = 10,
                 n_mamba_layers: int = 4, n_heads: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand_factor: int = 2,
                 context_len: int = 90, action_dim: int = 5,
                 delta_max: float = 0.50, dte_min: int = 14, dte_max: int = 90,
                 memory_bank_size: int = 10_000, dropout: float = 0.1):
        super().__init__()
        token_dim = n_features + n_time + n_calendar  # 34

        self.time2vec = Time2Vec(n_time)
        self.calendar_embed = CalendarEmbedding()
        self.vsn = VariableSelectionNetwork(token_dim, d_model, dropout)

        self.mamba = MambaBackbone(d_model, n_mamba_layers, n_heads)
        self.de_stationary = DeStationaryModule(d_model)
        self.self_attention = CausalSelfAttention(d_model, n_heads, context_len)
        self.cross_attention = CrossAttention(d_model)

        self.fusion = ReturnConditionedFusion(d_model)
        self.action_head = ActionHead(d_model, action_dim, delta_max, dte_min, dte_max)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )

        # Memory bank (frozen — not trainable)
        self.memory_keys = mx.random.normal((memory_bank_size, d_model)) * 0.01
        self.memory_values = mx.random.normal((memory_bank_size, d_model)) * 0.01

        # PPO log-std
        self.log_std = mx.full((action_dim,), -0.5)

        # Freeze non-trainable
        self.freeze(keys=["memory_keys", "memory_values", "_causal_mask"])

    def __call__(self, features, timestamps, dow, month,
                 is_opex, is_qtr, pre_mu, pre_sigma, rtg):
        time_enc = self.time2vec(timestamps)
        cal_enc = self.calendar_embed(dow, month, is_opex, is_qtr)
        token = mx.concatenate([features, time_enc, cal_enc], axis=-1)

        selected, _ = self.vsn(token)
        temporal = self.mamba(selected)
        tau, delta = self.de_stationary(pre_sigma, temporal)
        self_attn_out = self.self_attention(temporal, tau, delta)
        cross_attn_out, _ = self.cross_attention(
            self_attn_out, self.memory_keys, self.memory_values,
        )
        fused = self.fusion(self_attn_out, cross_attn_out, rtg)
        return self.action_head(fused)

    def forward_with_value(self, features, timestamps, dow, month,
                           is_opex, is_qtr, pre_mu, pre_sigma, rtg):
        time_enc = self.time2vec(timestamps)
        cal_enc = self.calendar_embed(dow, month, is_opex, is_qtr)
        token = mx.concatenate([features, time_enc, cal_enc], axis=-1)

        selected, _ = self.vsn(token)
        temporal = self.mamba(selected)
        tau, delta = self.de_stationary(pre_sigma, temporal)
        self_attn_out = self.self_attention(temporal, tau, delta)
        cross_attn_out, _ = self.cross_attention(
            self_attn_out, self.memory_keys, self.memory_values,
        )
        fused = self.fusion(self_attn_out, cross_attn_out, rtg)
        actions = self.action_head(fused)
        value = mx.squeeze(self.value_head(fused), axis=-1)
        return actions, value
