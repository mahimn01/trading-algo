from __future__ import annotations

import torch
from torch import Tensor


def differential_sharpe_ratio(returns: Tensor, eta: float = 2 / 64) -> Tensor:
    """Compute incremental DSR for a batch of return sequences.

    Args:
        returns: (B, T) tensor of return sequences.
        eta: EMA decay rate (default 2/64 ~ quarterly).

    Returns:
        Scalar: sum of DSR_t over time, averaged over batch.
    """
    B, T = returns.shape
    A = torch.zeros(B, device=returns.device, dtype=returns.dtype)
    B_val = torch.zeros(B, device=returns.device, dtype=returns.dtype)
    dsr_total = torch.zeros(B, device=returns.device, dtype=returns.dtype)

    for t in range(T):
        R_t = returns[:, t]
        dA = R_t - A
        dB = R_t ** 2 - B_val

        denom = (B_val - A ** 2).clamp(min=1e-12) ** 1.5
        dsr_t = (B_val * dA - 0.5 * A * dB) / denom
        dsr_total = dsr_total + dsr_t

        A = A + eta * dA
        B_val = B_val + eta * dB

    return dsr_total.mean()


def drawdown_penalty(equity_curve: Tensor, threshold: float = 0.20, lam: float = 10.0) -> Tensor:
    """Quadratic penalty when drawdown exceeds threshold.

    Args:
        equity_curve: (B, T) tensor of equity values.
        threshold: max acceptable drawdown fraction (default 0.20).
        lam: penalty weight.

    Returns:
        Scalar: mean penalty across batch.
    """
    # cummax along time dimension
    peak, _ = torch.cummax(equity_curve, dim=1)
    dd = (peak - equity_curve) / peak.clamp(min=1e-12)
    excess = torch.clamp(dd - threshold, min=0.0)
    penalty = lam * (excess ** 2).sum(dim=1)
    return penalty.mean()


def transaction_cost_penalty(actions: Tensor, lam: float = 0.01) -> Tensor:
    """L1 penalty on action changes between consecutive steps.

    Args:
        actions: (B, T, A) tensor of action sequences.
        lam: penalty weight.

    Returns:
        Scalar: mean penalty across batch.
    """
    diffs = (actions[:, 1:, :] - actions[:, :-1, :]).abs().sum(dim=(1, 2))
    return lam * diffs.mean()
