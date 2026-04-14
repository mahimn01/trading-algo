from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ATLASConfig:
    # Model
    d_model: int = 64
    n_mamba_layers: int = 4
    n_heads: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    context_len: int = 90
    n_features: int = 16
    n_time_features: int = 8
    n_calendar_features: int = 10
    memory_bank_size: int = 10_000

    # Action
    action_dim: int = 5
    delta_max: float = 0.50
    dte_min: int = 14
    dte_max: int = 90

    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    grad_clip: float = 0.5
    bc_epochs: int = 10
    bc_patience: int = 5

    # PPO
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    ppo_epochs: int = 4
    rollout_steps: int = 2048

    # DSR (Differential Sharpe Ratio)
    dsr_eta: float = 2 / 64
    dd_threshold: float = 0.20
    dd_lambda: float = 10.0
    tc_lambda: float = 0.01

    # EWC (Elastic Weight Consolidation)
    ewc_lambda_init: float = 100.0
    ewc_decay_days: int = 500
    ewc_update_interval: int = 30
    ewc_steps: int = 100

    # Risk
    max_position_pct: float = 0.25
    vol_target: float = 0.15

    # Regularization
    dropout: float = 0.1
    risk_free_rate: float = 0.045
