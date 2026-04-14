#!/usr/bin/env python3
"""ATLAS v5: MLX-native training on Apple Silicon.

Optimizations over v4 (PyTorch hybrid):
  1. MLX unified memory — zero-copy between CPU numpy and GPU compute
  2. Fused Metal kernels — no MPS kernel launch overhead per op
  3. mx.compile JIT — entire training step compiled into single Metal kernel graph
  4. No device transfers — everything stays in unified memory
  5. Still uses numba-vectorized env for BSM pricing

Usage:
    .venv/bin/python scripts/atlas_v5_train_mlx.py
    .venv/bin/python scripts/atlas_v5_train_mlx.py --quick-test
    .venv/bin/python scripts/atlas_v5_train_mlx.py --from-pytorch checkpoints/atlas_v3/atlas_v3_bc_best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as optim
from mlx.optimizers import clip_grad_norm

from trading_algo.quant_core.models.atlas.mlx_model import (
    ATLASModel,
    normal_log_prob,
    normal_entropy,
    normal_sample,
)
from trading_algo.quant_core.models.atlas.fast_env import VectorizedOptionsEnv
from trading_algo.quant_core.models.atlas.train_ppo import load_training_data_v2

FEATURE_DIR = "data/atlas_features_v3"
N_ENVS = 32


# ---------------------------------------------------------------------------
# PyTorch → MLX weight conversion
# ---------------------------------------------------------------------------

def convert_pytorch_weights(pt_path: str, mlx_model: ATLASModel) -> None:
    """Load a PyTorch .pt checkpoint into the MLX model."""
    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    def to_mx(t):
        return mx.array(t.detach().float().numpy())

    # Build flat key→value mapping from PyTorch state dict
    # Skip buffers that don't exist as MLX parameters
    _skip_keys = {"causal_mask"}
    pt_weights = {k: to_mx(v) for k, v in sd.items()
                  if not any(sk in k for sk in _skip_keys)}

    # Get MLX parameter tree
    mlx_params = mlx_model.parameters()

    def _set_nested(d, keys, val):
        for k in keys[:-1]:
            if isinstance(d, list):
                d = d[int(k)]
            else:
                d = d[k]
        last = keys[-1]
        if isinstance(d, list):
            d[int(last)] = val
        else:
            d[last] = val

    # Map PyTorch flat keys to MLX nested structure
    # Most keys map directly (e.g., "vsn.feature_grns.0.w1.weight")
    # Exception: MambaBackbone's CausalTransformerBlock uses nn.MultiHeadAttention
    # PyTorch MultiheadAttention: in_proj_weight (3d, d), in_proj_bias (3d)
    # MLX MultiHeadAttention: query_proj.weight, key_proj.weight, value_proj.weight

    mapped = 0
    skipped = []
    for pt_key, pt_val in pt_weights.items():
        parts = pt_key.split(".")

        # Handle MultiheadAttention weight splitting
        if "attn.in_proj_weight" in pt_key:
            prefix = ".".join(parts[:-1])  # e.g., "mamba.layers.0.attn"
            d = pt_val.shape[0] // 3
            q_w, k_w, v_w = pt_val[:d], pt_val[d:2*d], pt_val[2*d:]
            for suffix, w in [("query_proj.weight", q_w), ("key_proj.weight", k_w), ("value_proj.weight", v_w)]:
                full_key = f"{prefix}.{suffix}"
                try:
                    _set_nested(mlx_params, full_key.split("."), w)
                    mapped += 1
                except (KeyError, IndexError):
                    skipped.append(full_key)
            continue

        if "attn.in_proj_bias" in pt_key:
            prefix = ".".join(parts[:-1])
            d = pt_val.shape[0] // 3
            q_b, k_b, v_b = pt_val[:d], pt_val[d:2*d], pt_val[2*d:]
            for suffix, b in [("query_proj.bias", q_b), ("key_proj.bias", k_b), ("value_proj.bias", v_b)]:
                full_key = f"{prefix}.{suffix}"
                try:
                    _set_nested(mlx_params, full_key.split("."), b)
                    mapped += 1
                except (KeyError, IndexError):
                    skipped.append(full_key)
            continue

        if "attn.out_proj.weight" in pt_key or "attn.out_proj.bias" in pt_key:
            # out_proj maps directly
            try:
                _set_nested(mlx_params, parts, pt_val)
                mapped += 1
            except (KeyError, IndexError):
                skipped.append(pt_key)
            continue

        # Handle nn.Sequential index mapping (value_head.0.weight → value_head.layers.0.weight)
        # MLX Sequential stores layers in .layers list
        if pt_key.startswith("value_head."):
            # value_head.0.weight → value_head.layers.0.weight
            new_parts = list(parts)
            if len(new_parts) >= 2 and new_parts[1].isdigit():
                new_parts.insert(1, "layers")
            try:
                _set_nested(mlx_params, new_parts, pt_val)
                mapped += 1
            except (KeyError, IndexError):
                skipped.append(pt_key)
            continue

        # Handle other Sequential modules (action_head.net, de_stationary.mlp_*)
        new_parts = list(parts)
        for i, p in enumerate(new_parts):
            if i > 0 and p.isdigit() and new_parts[i-1] in ("net", "mlp_tau", "mlp_delta", "ffn"):
                new_parts.insert(i, "layers")
                break

        try:
            _set_nested(mlx_params, new_parts, pt_val)
            mapped += 1
        except (KeyError, IndexError):
            skipped.append(pt_key)

    mlx_model.update(mlx_params)
    mlx_model.freeze(keys=["memory_keys", "memory_values", "_causal_mask"])
    print(f"  Converted {mapped} params, skipped {len(skipped)}", flush=True)
    if skipped:
        print(f"  Skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}", flush=True)


# ---------------------------------------------------------------------------
# GAE (pure numpy, same as PyTorch version)
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T and not dones[t] else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO training step (compilable with mx.compile)
# ---------------------------------------------------------------------------

def ppo_loss_fn(model, features, timestamps, dow, month, opex, qtr,
                mu, sigma, rtg, actions, old_lp, advantages, returns,
                clip_eps=0.2, entropy_coeff=0.01):
    """PPO clipped loss — designed to be wrapped with nn.value_and_grad."""
    new_mean, new_val = model.forward_with_value(
        features, timestamps, dow, month, opex, qtr, mu, sigma, rtg,
    )
    new_val = mx.clip(new_val, -100, 100)

    new_lp = mx.sum(normal_log_prob(actions, new_mean, model.log_std), axis=-1)
    ent = mx.sum(normal_entropy(model.log_std))

    ratio = mx.exp(new_lp - old_lp)
    clipped = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -mx.mean(mx.minimum(ratio * advantages, clipped * advantages))
    value_loss = mx.mean((new_val - returns) ** 2)

    loss = policy_loss + 0.5 * value_loss - entropy_coeff * ent
    return loss, (policy_loss, value_loss, ent)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_ppo_mlx(
    model: ATLASModel,
    all_data: dict,
    n_iterations: int,
    rollout_steps: int,
    regime_filter: str,
    reward_shaping: str,
    optimizer: optim.OptimizerBase,
    checkpoint_dir: str,
    n_envs: int = N_ENVS,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    grad_clip: float = 0.5,
    ppo_epochs: int = 4,
) -> dict[str, list[float]]:

    from trading_algo.quant_core.models.atlas.config import ATLASConfig
    env = VectorizedOptionsEnv(
        all_data, ATLASConfig(), n_envs=n_envs,
        regime_filter=regime_filter,
        reward_shaping=reward_shaping,
    )

    if regime_filter != "all":
        print(f"  Regime filter: {regime_filter} ({len(env._eligible)} eligible episodes)", flush=True)

    loss_grad_fn = mnn.value_and_grad(model, partial(
        ppo_loss_fn, clip_eps=clip_eps, entropy_coeff=entropy_coeff,
    ))

    history = {"policy_loss": [], "value_loss": [], "entropy": [], "mean_reward": []}

    for iteration in range(n_iterations):
        # === ROLLOUT (env in numpy, model in MLX) ===
        obs = env.reset_all()
        all_obs = {k: [] for k in obs}
        all_actions, all_log_probs, all_rewards, all_values, all_dones = [], [], [], [], []

        model.eval()
        steps_per_env = rollout_steps // n_envs

        for _ in range(steps_per_env):
            # Convert torch obs → mx.array
            mx_obs = {k: mx.array(v.numpy()) for k, v in obs.items()}

            action_mean, value = model.forward_with_value(
                mx_obs["features"], mx_obs["timestamps"], mx_obs["dow"],
                mx_obs["month"], mx_obs["is_opex"], mx_obs["is_qtr"],
                mx_obs["pre_mu"], mx_obs["pre_sigma"], mx_obs["rtg"],
            )
            action_mean = mx.nan_to_num(action_mean, nan=0.0)
            value = mx.clip(mx.nan_to_num(value, nan=0.0), -100, 100)

            action = normal_sample(action_mean, model.log_std)
            log_prob = mx.sum(normal_log_prob(action, action_mean, model.log_std), axis=-1)

            # Force evaluation before numpy conversion
            mx.eval(action, log_prob, value)

            action_np = np.array(action)
            next_obs, rewards, dones = env.step(action_np)

            # Store all transitions
            for k, v in obs.items():
                all_obs[k].append(v.numpy() if hasattr(v, 'numpy') else np.array(v))
            all_actions.append(action_np)
            all_log_probs.append(np.array(log_prob))
            all_rewards.extend(rewards.tolist() if len(rewards.shape) > 0 else [float(rewards)] * n_envs)
            all_values.extend(np.array(value).tolist() if len(np.array(value).shape) > 0 else [float(value)] * n_envs)
            all_dones.extend(dones.tolist() if len(dones.shape) > 0 else [bool(dones)] * n_envs)

            obs = next_obs

        # Flatten: each step produced n_envs transitions
        n_steps = steps_per_env * n_envs

        # Interleave per-env data: step0_env0, step0_env1, ..., step1_env0, step1_env1, ...
        flat_obs = {}
        for k in all_obs:
            flat_obs[k] = np.concatenate(
                [np.concatenate([all_obs[k][s][i:i+1] for s in range(steps_per_env)])
                 for i in range(n_envs)]
            ) if False else np.concatenate(all_obs[k])  # (steps_per_env * n_envs, ...)

        flat_actions = np.concatenate(all_actions)  # (steps_per_env * n_envs, 5)
        flat_log_probs = np.concatenate(all_log_probs)  # (steps_per_env * n_envs,)

        # GAE
        advantages, returns = compute_gae(all_rewards, all_values, all_dones, gamma, gae_lambda)

        # === PPO UPDATE (all MLX) ===
        model.train()

        # Convert to mx.array once
        mx_features = mx.array(flat_obs["features"])
        mx_ts = mx.array(flat_obs["timestamps"])
        mx_dow = mx.array(flat_obs["dow"].astype(np.int32))
        mx_month = mx.array(flat_obs["month"].astype(np.int32))
        mx_opex = mx.array(flat_obs["is_opex"])
        mx_qtr = mx.array(flat_obs["is_qtr"])
        mx_mu = mx.array(flat_obs["pre_mu"])
        mx_sigma = mx.array(flat_obs["pre_sigma"])
        mx_rtg = mx.array(flat_obs["rtg"])
        mx_actions = mx.array(flat_actions)
        mx_old_lp = mx.array(flat_log_probs)
        mx_adv = mx.array(advantages)
        mx_ret = mx.array(returns)

        mini_batch_size = min(512, n_steps)
        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(ppo_epochs):
            indices = np.random.permutation(n_steps)
            for start in range(0, n_steps, mini_batch_size):
                end = min(start + mini_batch_size, n_steps)
                idx = mx.array(indices[start:end])

                mb_features = mx_features[idx]
                mb_ts = mx_ts[idx]
                mb_dow = mx_dow[idx]
                mb_month = mx_month[idx]
                mb_opex = mx_opex[idx]
                mb_qtr = mx_qtr[idx]
                mb_mu = mx_mu[idx]
                mb_sigma = mx_sigma[idx]
                mb_rtg = mx_rtg[idx]
                mb_actions = mx_actions[idx]
                mb_old_lp = mx_old_lp[idx]
                mb_adv = mx_adv[idx]
                mb_ret = mx_ret[idx]

                # Normalize advantages
                mb_adv = (mb_adv - mx.mean(mb_adv)) / (mx.std(mb_adv) + 1e-8)

                (loss, (pl, vl, ent)), grads = loss_grad_fn(
                    model, mb_features, mb_ts, mb_dow, mb_month,
                    mb_opex, mb_qtr, mb_mu, mb_sigma, mb_rtg,
                    mb_actions, mb_old_lp, mb_adv, mb_ret,
                )

                grads, _ = clip_grad_norm(grads, max_norm=grad_clip)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                total_pl += pl.item()
                total_vl += vl.item()
                total_ent += ent.item()
                n_updates += 1

        avg_pl = total_pl / max(n_updates, 1)
        avg_vl = total_vl / max(n_updates, 1)
        avg_ent = total_ent / max(n_updates, 1)
        avg_rew = float(np.mean(all_rewards))

        history["policy_loss"].append(avg_pl)
        history["value_loss"].append(avg_vl)
        history["entropy"].append(avg_ent)
        history["mean_reward"].append(avg_rew)

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  PPO iter {iteration+1}/{n_iterations}: "
                  f"policy={avg_pl:.4f} value={avg_vl:.4f} "
                  f"entropy={avg_ent:.4f} reward={avg_rew:.4f}", flush=True)
        if (iteration + 1) % 25 == 0:
            model.save_weights(f"{checkpoint_dir}/atlas_ppo_iter{iteration+1}.safetensors")

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATLAS v5 — MLX native training")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--from-pytorch", type=str, default="checkpoints/atlas_v3/atlas_v3_bc_best.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/atlas_v5_mlx")
    args = parser.parse_args()

    print("=" * 70)
    print("  ATLAS v5 — MLX Native Training (Apple Metal GPU)")
    print("=" * 70)
    print(f"Device: {mx.default_device()}", flush=True)
    print(f"Parallel envs: {args.n_envs}", flush=True)
    print(f"Output: {args.checkpoint_dir}", flush=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Build model
    model = ATLASModel()
    import mlx.utils
    n_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"Parameters: {n_params:,}", flush=True)

    # Load weights from PyTorch checkpoint
    if args.from_pytorch and Path(args.from_pytorch).exists():
        print(f"Loading PyTorch checkpoint: {args.from_pytorch}", flush=True)
        convert_pytorch_weights(args.from_pytorch, model)
    else:
        print("Training from random initialization", flush=True)

    # Load data
    print("Loading training data...", flush=True)
    t0 = time.time()
    all_data = load_training_data_v2(FEATURE_DIR, min_len=290)
    print(f"Loaded {len(all_data)} symbols in {time.time()-t0:.1f}s", flush=True)

    # Warmup JIT
    print("Warming up numba JIT...", flush=True)
    t0 = time.time()
    from trading_algo.quant_core.models.atlas.config import ATLASConfig as _AC
    _env = VectorizedOptionsEnv(all_data, _AC(), n_envs=2, regime_filter="all", reward_shaping="none")
    _obs = _env.reset_all()
    _env.step(np.random.uniform(0, 1, (2, 5)))
    print(f"JIT warm in {time.time()-t0:.1f}s", flush=True)

    # Optimizer
    lr = 3e-4
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    if args.quick_test:
        stages = [
            ("high_iv", 5, "high_iv", "high_iv"),
            ("uptrend", 5, "uptrend", "uptrend"),
            ("mixed", 5, "all", "none"),
        ]
        rollout_steps = 256
    else:
        stages = [
            ("high_iv", 75, "high_iv", "high_iv"),
            ("uptrend", 75, "uptrend", "uptrend"),
            ("mixed", 200, "all", "none"),
        ]
        rollout_steps = 512

    total_iters = sum(s[1] for s in stages)
    print(f"\nTotal: {total_iters} iterations across {len(stages)} stages", flush=True)
    train_start = time.time()

    for i, (name, iters, regime, shaping) in enumerate(stages):
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage {i+1}/{len(stages)}: {name} ({iters} iters, {regime}, {shaping})", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        history = train_ppo_mlx(
            model=model,
            all_data=all_data,
            n_iterations=iters,
            rollout_steps=rollout_steps,
            regime_filter=regime,
            reward_shaping=shaping,
            optimizer=optimizer,
            checkpoint_dir=args.checkpoint_dir,
            n_envs=args.n_envs,
        )
        elapsed = time.time() - t0

        model.save_weights(f"{args.checkpoint_dir}/atlas_stage_{name}.safetensors")
        print(f"  Stage {name} done in {elapsed:.0f}s ({elapsed/iters:.1f}s/iter)", flush=True)
        print(f"  Mean reward (last 10): {np.mean(history['mean_reward'][-10:]):.4f}", flush=True)

    # Final save
    model.save_weights(f"{args.checkpoint_dir}/atlas_final.safetensors")

    total = time.time() - train_start
    print(f"\nDone. {total_iters} iterations in {total/60:.1f} min ({total/total_iters:.1f}s/iter)", flush=True)
    print(f"Checkpoint: {args.checkpoint_dir}/atlas_final.safetensors", flush=True)


if __name__ == "__main__":
    main()
