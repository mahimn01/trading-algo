#!/usr/bin/env python3
"""ATLAS v6: Full training pipeline on 2600+ symbols (9.2M windows).

Key changes over v3:
  - 2608 symbols (vs 444) — 10x more data diversity
  - Streaming BC: loads symbols in chunks to fit 32GB RAM
  - Numba-compiled hindsight actions (4500x speedup)
  - Same curriculum PPO structure (vectorized env, hybrid CPU/MPS)

Phases:
  1. BC (Behavioral Cloning): streaming symbol-chunked training
  2. Curriculum PPO: high_iv → uptrend → mixed

Usage:
    .venv/bin/python scripts/atlas_v6_train.py --phase bc
    .venv/bin/python scripts/atlas_v6_train.py --phase curriculum
    .venv/bin/python scripts/atlas_v6_train.py --phase all
    .venv/bin/python scripts/atlas_v6_train.py --quick-test
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

FEATURE_DIR = "data/atlas_features_v4"
CHECKPOINT_DIR = "checkpoints/atlas_v6"
N_ENVS = 32

# Streaming BC config
SYMBOLS_PER_CHUNK = 100   # symbols loaded into RAM at once
SYMBOLS_PER_EPOCH = 800   # random subset per epoch (diversity via shuffling across epochs)


def _extract_windows(
    data: dict[str, np.ndarray],
    context_len: int,
    stride: int = 3,
) -> dict[str, torch.Tensor] | None:
    """Extract sliding windows — fully vectorized, no Python loops."""
    normed = data["normed"]
    actions = data["actions"]
    T = len(normed)
    L = context_len
    F = normed.shape[1]

    if T < L + 252:
        return None

    # Sanitize
    normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).clip(-10, 10)
    actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clip(-100, 100)
    mu_arr = np.nan_to_num(data["mu"], nan=0.0, posinf=0.0, neginf=0.0).clip(-10, 10)
    sigma_arr = np.nan_to_num(data["sigma"], nan=0.0, posinf=0.0, neginf=0.0).clip(0, 10)

    # All candidate starts
    starts = np.arange(252, T - L + 1, stride)
    if len(starts) == 0:
        return None

    # Vectorized max-abs per window using stride_tricks (zero-copy view)
    from numpy.lib.stride_tricks import as_strided
    bs = normed.strides
    all_win = as_strided(normed, shape=(T - L + 1, L, F), strides=(bs[0], bs[0], bs[1]))
    max_abs = np.abs(all_win[starts]).reshape(len(starts), -1).max(axis=1)

    # Vectorized action validity
    act_end = actions[starts + L - 1]  # (N, 5)
    act_ok = (
        np.isfinite(act_end).all(axis=1)
        & (act_end[:, 0] >= 0) & (act_end[:, 0] <= 0.5)
        & (act_end[:, 1] >= -1) & (act_end[:, 1] <= 1)
        & (act_end[:, 2] >= 0) & (act_end[:, 2] <= 1)
        & (act_end[:, 3] >= 14) & (act_end[:, 3] <= 60)
        & (act_end[:, 4] >= 0) & (act_end[:, 4] <= 0.75)
    )
    mask = (max_abs < 9.9) & act_ok
    vs = starts[mask]

    if len(vs) < 10:
        return None

    # Batch index: (N, L)
    idx = vs[:, None] + np.arange(L)[None, :]

    acts = act_end[mask]
    norm_acts = acts.copy()
    norm_acts[:, 0] = (acts[:, 0] / 0.50).clip(0, 1)
    norm_acts[:, 1] = ((acts[:, 1] + 1.0) / 2.0).clip(0, 1)
    norm_acts[:, 3] = ((acts[:, 3] - 14.0) / 76.0).clip(0, 1)

    return {
        "features": torch.from_numpy(normed[idx].astype(np.float32)),
        "mu": torch.from_numpy(mu_arr[idx].astype(np.float32)),
        "sigma": torch.from_numpy(sigma_arr[idx].astype(np.float32)),
        "timestamps": torch.from_numpy(data["timestamps"][idx].astype(np.float32)),
        "dow": torch.from_numpy(data["dow"][idx].astype(np.int64)),
        "month": torch.from_numpy(data["month"][idx].astype(np.int64)),
        "actions": torch.from_numpy(norm_acts.astype(np.float32)),
        "rtg": torch.from_numpy(data["rtg"][vs + L - 1].astype(np.float32)),
    }


def _normalize_output(pred: torch.Tensor) -> torch.Tensor:
    n = pred.clone()
    n[:, 0] = pred[:, 0] / 0.50
    n[:, 1] = (pred[:, 1] + 1.0) / 2.0
    n[:, 3] = (pred[:, 3] - 14.0) / 76.0
    return n


def phase_bc(config: ATLASConfig, device: str, quick_test: bool = False) -> str:
    """Streaming Behavioral Cloning on 9.2M windows.

    Streams symbols in chunks of 50 to fit in 32GB RAM.
    Each chunk: load 50 symbols → extract windows → shuffle → train mini-batches.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    L = config.context_len

    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")])
    print(f"Feature files: {len(feature_files)}")

    model = ATLASModel(config).to(device).float()
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n_epochs = 2 if quick_test else config.bc_epochs
    batch_size = 512

    # Cosine annealing over total expected steps
    total_chunks = (len(feature_files) + SYMBOLS_PER_CHUNK - 1) // SYMBOLS_PER_CHUNK
    total_steps_est = total_chunks * n_epochs * 50  # rough estimate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps_est, 1), eta_min=1e-6,
    )

    best_val = float("inf")
    patience = 0
    best_path = f"{CHECKPOINT_DIR}/atlas_v6_bc_best.pt"

    import copy
    last_good_state = {k: v.clone() for k, v in model.state_dict().items()}
    last_good_optim = copy.deepcopy(optimizer.state_dict())

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()

        # Subsample symbols per epoch for speed — different subset each epoch
        # 800 symbols/epoch × 10 epochs covers most of 2608 symbols
        epoch_files = feature_files.copy()
        np.random.shuffle(epoch_files)
        n_files = 100 if quick_test else min(SYMBOLS_PER_EPOCH, len(epoch_files))
        epoch_files = epoch_files[:n_files]

        epoch_loss = 0.0
        epoch_batches = 0
        epoch_windows = 0

        from concurrent.futures import ThreadPoolExecutor

        def _load_symbol(fname: str) -> dict[str, torch.Tensor] | None:
            data = dict(np.load(f"{FEATURE_DIR}/{fname}"))
            return _extract_windows(data, L, stride=3)

        for chunk_start in range(0, len(epoch_files), SYMBOLS_PER_CHUNK):
            chunk_file_list = epoch_files[chunk_start:chunk_start + SYMBOLS_PER_CHUNK]

            # Parallel I/O: load + extract windows concurrently
            all_windows: list[dict[str, torch.Tensor]] = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for w in pool.map(_load_symbol, chunk_file_list):
                    if w is not None:
                        all_windows.append(w)

            if not all_windows:
                continue

            # Concatenate and transfer ENTIRE chunk to device ONCE
            chunk_data = {
                k: torch.cat([w[k] for w in all_windows], dim=0).to(device)
                for k in all_windows[0].keys()
            }
            N = chunk_data["features"].shape[0]
            epoch_windows += N

            # Pre-allocate opex/qtr zeros on device (reused every batch)
            opex_zeros = torch.zeros(N, L, device=device)
            qtr_zeros = torch.zeros(N, L, device=device)

            # Split 95/5 for val (more training data)
            val_n = max(1, N // 20)
            perm = torch.randperm(N, device=device)
            train_perm = perm[val_n:]
            val_perm = perm[:val_n]

            # Train on chunk — data already on device, no per-batch transfers
            model.train()
            train_perm = train_perm[torch.randperm(len(train_perm), device=device)]
            for bs in range(0, len(train_perm), batch_size):
                be = min(bs + batch_size, len(train_perm))
                idx = train_perm[bs:be]

                bf = chunk_data["features"][idx]
                bts = chunk_data["timestamps"][idx]
                bdow = chunk_data["dow"][idx]
                bmo = chunk_data["month"][idx]
                bop = opex_zeros[:len(idx)]
                bqt = qtr_zeros[:len(idx)]
                bmu = chunk_data["mu"][idx]
                bsi = chunk_data["sigma"][idx]
                brtg = chunk_data["rtg"][idx]
                ba = chunk_data["actions"][idx]

                pred = model(bf, bts, bdow, bmo, bop, bqt, bmu, bsi, brtg)
                normed_pred = _normalize_output(pred)

                # Skip batch if model output is degenerate
                if torch.isnan(normed_pred).any() or torch.isinf(normed_pred).any():
                    optimizer.zero_grad()
                    continue

                loss = nn.functional.mse_loss(normed_pred, ba)

                # Skip if loss is extreme (typical good loss is 0.05-0.15)
                if loss.item() > 1.0 or torch.isnan(loss):
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_batches += 1

                # Fast NaN check every 50 batches (check first param only — cheap)
                if epoch_batches % 50 == 0:
                    p0 = next(model.parameters())
                    if torch.isnan(p0).any():
                        print(f"  WARNING: NaN in weights at batch {epoch_batches} — rolling back model+optimizer", flush=True)
                        model.load_state_dict(last_good_state)
                        optimizer.load_state_dict(copy.deepcopy(last_good_optim))
                        break

            # Validate on chunk (data already on device)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for bs in range(0, len(val_perm), batch_size):
                    be = min(bs + batch_size, len(val_perm))
                    idx = val_perm[bs:be]
                    pred = model(
                        chunk_data["features"][idx], chunk_data["timestamps"][idx],
                        chunk_data["dow"][idx], chunk_data["month"][idx],
                        opex_zeros[:len(idx)], qtr_zeros[:len(idx)],
                        chunk_data["mu"][idx], chunk_data["sigma"][idx],
                        chunk_data["rtg"][idx],
                    )
                    vl = nn.functional.mse_loss(_normalize_output(pred), chunk_data["actions"][idx])
                    if not torch.isnan(vl):
                        val_losses.append(vl.item())

            # Health check: detect NaN in model weights and rollback (model + optimizer)
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f"  WARNING: NaN in model weights after chunk — rolling back model+optimizer", flush=True)
                model.load_state_dict(last_good_state)
                optimizer.load_state_dict(copy.deepcopy(last_good_optim))
            else:
                last_good_state = {k: v.clone() for k, v in model.state_dict().items()}
                last_good_optim = copy.deepcopy(optimizer.state_dict())

            chunk_idx = chunk_start // SYMBOLS_PER_CHUNK + 1
            if chunk_idx % 5 == 0 or chunk_idx == 1:
                avg_vl = float(np.mean(val_losses)) if val_losses else 0
                print(f"  Epoch {epoch+1} chunk {chunk_idx}/{total_chunks}: "
                      f"train={epoch_loss/max(epoch_batches,1):.6f} val={avg_vl:.6f} "
                      f"windows={epoch_windows:,}", flush=True)

        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{n_epochs}: avg_loss={avg_epoch_loss:.6f} "
              f"windows={epoch_windows:,} time={elapsed:.0f}s", flush=True)

        # Save best based on epoch average loss
        if avg_epoch_loss < best_val:
            best_val = avg_epoch_loss
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_epoch_loss,
                "config": config,
            }, best_path)
            print(f"    Saved best (loss={avg_epoch_loss:.6f})", flush=True)
        else:
            patience += 1
            if patience >= config.bc_patience:
                print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

    print(f"\nBC complete. Best loss: {best_val:.6f}")
    return best_path


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
    """PPO with vectorized environment and hybrid CPU/MPS pipeline.

    Identical to v4 — the PPO phase doesn't need to change, it uses the
    vectorized env which loads data from feature files directly.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = VectorizedOptionsEnv(
        all_data, config,
        n_envs=n_envs,
        regime_filter=regime_filter,
        reward_shaping=reward_shaping,
    )
    if regime_filter != "all":
        print(f"  Regime filter: {regime_filter} ({len(env._eligible)} eligible episodes)", flush=True)

    model_cpu = ATLASModel(config).float().eval()
    model_cpu.load_state_dict(model.state_dict())
    model = model.to(device_train).float()

    history: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "mean_reward": [],
    }

    for iteration in range(n_iterations):
        obs = env.reset_all()
        all_features, all_ts, all_dow, all_month = [], [], [], []
        all_opex, all_qtr, all_mu, all_sigma, all_rtg = [], [], [], [], []
        all_actions, all_log_probs, all_rewards, all_values, all_dones = [], [], [], [], []

        model_cpu.eval()
        steps_per_env = rollout_steps // n_envs

        for _ in range(steps_per_env):
            with torch.no_grad():
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

            action_np = action.numpy()
            next_obs, rewards, dones = env.step(action_np)

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

        n_steps = len(all_rewards)
        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones,
            config.gamma, config.gae_lambda,
        )

        model.train()

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


def phase_curriculum(
    config: ATLASConfig,
    device: str,
    bc_checkpoint: str,
    quick_test: bool = False,
) -> None:
    """Curriculum PPO fine-tuning using vectorized environments."""

    model = ATLASModel(config)
    ckpt = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded BC checkpoint: {bc_checkpoint}", flush=True)

    # Load ALL feature data for vectorized env (it loads npz files lazily)
    print("Loading training data for PPO envs...", flush=True)
    t0 = time.time()
    all_data = load_training_data_v2(FEATURE_DIR, min_len=config.context_len + 200)
    print(f"Loaded {len(all_data)} symbols in {time.time()-t0:.1f}s", flush=True)

    # Warmup numba JIT
    print("Warming up numba JIT...", flush=True)
    t0 = time.time()
    _warmup_env = VectorizedOptionsEnv(all_data, config, n_envs=2, regime_filter="all", reward_shaping="none")
    _obs = _warmup_env.reset_all()
    _acts = np.random.uniform(0, 1, (2, 5)).astype(np.float64)
    _warmup_env.step(_acts)
    print(f"JIT compiled in {time.time()-t0:.1f}s", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if quick_test:
        stages = [
            ("high_iv", 5, "high_iv", "high_iv"),
            ("uptrend", 5, "uptrend", "uptrend"),
            ("mixed", 10, "all", "none"),
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
            n_envs=N_ENVS,
        )
        elapsed = time.time() - t0

        torch.save({
            "stage": name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        }, f"{CHECKPOINT_DIR}/atlas_v6_stage_{name}.pt")

        print(f"  Stage {name} complete in {elapsed:.0f}s "
              f"({elapsed/iters:.1f}s/iter)", flush=True)
        print(f"  Mean reward (last 10): {np.mean(history['mean_reward'][-10:]):.4f}", flush=True)

    # Save final
    final_path = f"{CHECKPOINT_DIR}/atlas_v6_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, final_path)

    total_time = time.time() - train_start
    print(f"\nDone. {total_iters} iterations in {total_time/60:.1f} min "
          f"({total_time/total_iters:.1f}s/iter)", flush=True)
    print(f"Checkpoint: {final_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS v6 — 2600+ symbol training")
    parser.add_argument("--phase", choices=["bc", "curriculum", "all"], default="all")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    config = ATLASConfig()

    print("=" * 70)
    print("  ATLAS v6 — Full Training (2600+ symbols, 9.2M windows)")
    print("=" * 70)
    print(f"Device: {device}", flush=True)
    print(f"Features: {FEATURE_DIR}", flush=True)
    print(f"Output: {CHECKPOINT_DIR}", flush=True)

    bc_path = f"{CHECKPOINT_DIR}/atlas_v6_bc_best.pt"

    if args.phase in ("bc", "all"):
        print(f"\n{'='*60}")
        print("  Phase 1: Behavioral Cloning (streaming)")
        print(f"{'='*60}", flush=True)
        bc_path = phase_bc(config, device, quick_test=args.quick_test)

    if args.phase in ("curriculum", "all"):
        print(f"\n{'='*60}")
        print("  Phase 2: Curriculum PPO")
        print(f"{'='*60}", flush=True)
        if not os.path.exists(bc_path):
            print(f"ERROR: BC checkpoint not found at {bc_path}")
            print("Run --phase bc first.")
            return
        phase_curriculum(config, device, bc_path, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
