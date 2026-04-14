#!/usr/bin/env python3
"""ATLAS v4: Optimized curriculum retrain with stop-loss-aware vectorized environment.

Optimizations over v3:
  1. Vectorized env: 32 parallel environments per rollout step (32x data collection)
  2. Numba JIT: BSM pricing compiled to native code (~100x vs Python)
  3. Hybrid device: CPU for rollouts (fast B=32), MPS for batched PPO updates
  4. Batched model inference: forward pass on (32, L, F) not (1, L, F)
  5. Pre-transfer: move entire rollout batch to MPS once, not per mini-batch

Architecture changes:
  - 3x stop-loss on short puts AND short calls
  - Crash shaping calibrated to ±0.5 (was ±2.0)
  - Retrains from BC checkpoint (learns stop-loss from scratch)

Usage:
    .venv/bin/python scripts/atlas_v4_train.py
    .venv/bin/python scripts/atlas_v4_train.py --quick-test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.model import ATLASModel
from trading_algo.quant_core.models.atlas.fast_env import VectorizedOptionsEnv
from trading_algo.quant_core.models.atlas.train_ppo import (
    load_training_data_v2,
    compute_gae,
)

BC_CHECKPOINT = "checkpoints/atlas_v3/atlas_v3_bc_best.pt"
CHECKPOINT_DIR = "checkpoints/atlas_v4"
FEATURE_DIR = "data/atlas_features_v3"
N_ENVS = 32


def train_ppo_fast(
    model: nn.Module,
    all_data: dict[str, dict],
    config: ATLASConfig,
    checkpoint_dir: str,
    device_train: str,
    n_iterations: int,
    rollout_steps: int,
    regime_filter: str,
    reward_shaping: str,
    optimizer: torch.optim.Optimizer,
    n_envs: int = N_ENVS,
) -> dict[str, list[float]]:
    """PPO with vectorized environment and hybrid CPU/MPS pipeline."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Vectorized env always runs on CPU (numpy)
    env = VectorizedOptionsEnv(
        all_data, config,
        n_envs=n_envs,
        regime_filter=regime_filter,
        reward_shaping=reward_shaping,
    )
    if regime_filter != "all":
        print(f"  Regime filter: {regime_filter} ({len(env._eligible)} eligible episodes)", flush=True)

    # Keep a CPU copy of the model for fast rollouts (avoids MPS dispatch overhead)
    model_cpu = ATLASModel(config).float().eval()
    model_cpu.load_state_dict(model.state_dict())

    # Training model on MPS
    model = model.to(device_train).float()

    history: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "mean_reward": [],
    }

    for iteration in range(n_iterations):
        # === ROLLOUT PHASE (CPU, batched across N_ENVS) ===
        obs = env.reset_all()
        all_features, all_ts, all_dow, all_month = [], [], [], []
        all_opex, all_qtr, all_mu, all_sigma, all_rtg = [], [], [], [], []
        all_actions, all_log_probs, all_rewards, all_values, all_dones = [], [], [], [], []

        model_cpu.eval()
        steps_per_env = rollout_steps // n_envs  # each env does this many steps

        for _ in range(steps_per_env):
            with torch.no_grad():
                # Batched forward pass: (N_ENVS, L, F) — much better than N_ENVS × (1, L, F)
                action_mean, value = model_cpu.forward_with_value(
                    obs["features"], obs["timestamps"], obs["dow"],
                    obs["month"], obs["is_opex"], obs["is_qtr"],
                    obs["pre_mu"], obs["pre_sigma"], obs["rtg"],
                )
                action_mean = torch.nan_to_num(action_mean, nan=0.0)
                value = torch.nan_to_num(value, nan=0.0).clamp(-100, 100)

                dist = model_cpu.get_action_distribution(action_mean)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

            action_np = action.numpy()  # (N_ENVS, 5)
            next_obs, rewards, dones = env.step(action_np)

            # Store transitions for ALL envs
            for i in range(n_envs):
                all_features.append(obs["features"][i:i+1])
                all_ts.append(obs["timestamps"][i:i+1])
                all_dow.append(obs["dow"][i:i+1])
                all_month.append(obs["month"][i:i+1])
                all_opex.append(obs["is_opex"][i:i+1])
                all_qtr.append(obs["is_qtr"][i:i+1])
                all_mu.append(obs["pre_mu"][i:i+1])
                all_sigma.append(obs["pre_sigma"][i:i+1])
                all_rtg.append(obs["rtg"][i:i+1])
                all_actions.append(action[i:i+1])
                all_log_probs.append(log_prob[i].item())
                all_rewards.append(float(rewards[i]))
                all_values.append(value[i].item())
                all_dones.append(bool(dones[i]))

            obs = next_obs

        # === GAE ===
        n_steps = len(all_rewards)
        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones,
            config.gamma, config.gae_lambda,
        )

        # === PPO UPDATE PHASE (MPS, batched) ===
        # Sync CPU model weights to training model is not needed (we update training model,
        # then sync back to CPU at the end)
        model.train()

        # Pre-concatenate and transfer entire batch to MPS ONCE
        batch_features = torch.cat(all_features).to(device_train)
        batch_ts = torch.cat(all_ts).to(device_train)
        batch_dow = torch.cat(all_dow).to(device_train)
        batch_month = torch.cat(all_month).to(device_train)
        batch_opex = torch.cat(all_opex).to(device_train)
        batch_qtr = torch.cat(all_qtr).to(device_train)
        batch_mu = torch.cat(all_mu).to(device_train)
        batch_sigma = torch.cat(all_sigma).to(device_train)
        batch_rtg = torch.cat(all_rtg).to(device_train)
        batch_actions = torch.cat(all_actions).to(device_train)
        batch_old_lp = torch.tensor(all_log_probs, device=device_train)
        batch_adv = advantages.to(device_train)
        batch_ret = returns.to(device_train)

        indices = np.arange(n_steps)
        mini_batch_size = min(512, n_steps)
        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, mini_batch_size):
                end = min(start + mini_batch_size, n_steps)
                mb_idx = indices[start:end]
                idx = torch.from_numpy(mb_idx).long()

                mb_features = batch_features[idx]
                mb_ts = batch_ts[idx]
                mb_dow = batch_dow[idx]
                mb_month = batch_month[idx]
                mb_opex = batch_opex[idx]
                mb_qtr = batch_qtr[idx]
                mb_mu = batch_mu[idx]
                mb_sigma = batch_sigma[idx]
                mb_rtg = batch_rtg[idx]
                mb_actions = batch_actions[idx]
                mb_old_lp = batch_old_lp[idx]
                mb_adv = batch_adv[idx]
                mb_ret = batch_ret[idx]

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_mean, new_val = model.forward_with_value(
                    mb_features, mb_ts, mb_dow, mb_month, mb_opex, mb_qtr,
                    mb_mu, mb_sigma, mb_rtg,
                )
                if torch.isnan(new_mean).any() or torch.isnan(new_val).any():
                    optimizer.zero_grad()
                    continue
                new_val = new_val.clamp(-100, 100)

                dist = model.get_action_distribution(new_mean)
                new_lp = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                if torch.isnan(new_lp).any() or torch.isnan(entropy):
                    optimizer.zero_grad()
                    continue

                ratio = (new_lp - mb_old_lp).exp()
                clipped = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
                value_loss = nn.functional.mse_loss(new_val, mb_ret)

                loss = policy_loss + 0.5 * value_loss - config.entropy_coeff * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        # Sync trained weights back to CPU model
        model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})

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
                  f"entropy={avg_ent:.4f} reward={avg_rew:.4f}",
                  flush=True)
        if (iteration + 1) % 25 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": config,
            }, f"{checkpoint_dir}/atlas_ppo_iter{iteration+1}.pt")

    return history


# --------------------------------------------------------------------------
# Curriculum stages
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS v4 — optimized curriculum")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("  ATLAS v4 — Optimized Curriculum (Vectorized Env + Hybrid Device)")
    print("=" * 70)
    print(f"Train device: {device}  |  Rollout device: cpu", flush=True)
    print(f"Parallel envs: {args.n_envs}", flush=True)
    print(f"BC checkpoint: {BC_CHECKPOINT}", flush=True)
    print(f"Output: {CHECKPOINT_DIR}", flush=True)
    print()
    print("Optimizations:", flush=True)
    print(f"  - {args.n_envs} parallel envs (vectorized numpy)", flush=True)
    print("  - Numba JIT-compiled BSM pricing", flush=True)
    print("  - CPU rollouts → MPS PPO updates", flush=True)
    print("  - 3x stop-loss on all short options", flush=True)
    print("  - Calibrated crash shaping (±0.5)", flush=True)

    config = ATLASConfig()
    model = ATLASModel(config)

    ckpt = torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded BC checkpoint", flush=True)

    # Load data once
    print("Loading training data...", flush=True)
    t0 = time.time()
    all_data = load_training_data_v2(FEATURE_DIR, min_len=config.context_len + 200)
    print(f"Loaded {len(all_data)} symbols in {time.time()-t0:.1f}s", flush=True)

    # Warmup numba JIT (first call compiles)
    print("Warming up numba JIT...", flush=True)
    t0 = time.time()
    _warmup_env = VectorizedOptionsEnv(all_data, config, n_envs=2, regime_filter="all", reward_shaping="none")
    _obs = _warmup_env.reset_all()
    _acts = np.random.uniform(0, 1, (2, 5)).astype(np.float64)
    _warmup_env.step(_acts)
    print(f"JIT compiled in {time.time()-t0:.1f}s", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if args.quick_test:
        stages = [
            ("high_iv", 5, "high_iv", "high_iv"),
            ("uptrend", 5, "uptrend", "uptrend"),
            ("mixed", 5, "all", "none"),
        ]
        rollout_steps = 256
    else:
        # No crash stage — the 3x stop-loss handles crashes mechanically.
        # Crash reward shaping was killing the policy (model learns never to trade).
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
        print(f"  Stage {i+1}/{len(stages)}: {name} "
              f"({iters} iterations, regime={regime}, shaping={shaping})", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        history = train_ppo_fast(
            model=model,
            all_data=all_data,
            config=config,
            checkpoint_dir=CHECKPOINT_DIR,
            device_train=device,
            n_iterations=iters,
            rollout_steps=rollout_steps,
            regime_filter=regime,
            reward_shaping=shaping,
            optimizer=optimizer,
            n_envs=args.n_envs,
        )
        elapsed = time.time() - t0

        torch.save({
            "stage": name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        }, f"{CHECKPOINT_DIR}/atlas_curriculum_stage_{name}.pt")

        print(f"  Stage {name} complete in {elapsed:.0f}s "
              f"({elapsed/iters:.1f}s/iter)", flush=True)
        print(f"  Mean reward (last 10): {np.mean(history['mean_reward'][-10:]):.4f}", flush=True)

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, f"{CHECKPOINT_DIR}/atlas_curriculum_final.pt")

    total_time = time.time() - train_start
    print(f"\nDone. {total_iters} iterations in {total_time/60:.1f} min "
          f"({total_time/total_iters:.1f}s/iter)", flush=True)
    print(f"Checkpoint: {CHECKPOINT_DIR}/atlas_curriculum_final.pt", flush=True)


if __name__ == "__main__":
    main()
