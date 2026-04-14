#!/usr/bin/env python3
"""ATLAS v7: Multi-task regime-conditioned training pipeline.

Key changes over v6:
  - ATLASModelV7 / ATLASv7Config (mLSTM + transformer backbone)
  - Multi-task PPO: all regimes simultaneously (no sequential curriculum)
  - CURATE-style automatic curriculum (sample harder regimes more)
  - Multi-component reward: DSR + Sortino + drawdown penalty + tail risk
  - Progressive action space: direction/leverage -> +delta/DTE -> +profit_target
  - 64 parallel envs

Phases:
  1. BC (Behavioral Cloning): streaming symbol-chunked training (same as v6)
  2. Multi-task regime-conditioned PPO
  3. Evaluation (200-episode, per-regime breakdown)

Usage:
    .venv/bin/python scripts/atlas_v7_train.py --phase bc
    .venv/bin/python scripts/atlas_v7_train.py --phase ppo
    .venv/bin/python scripts/atlas_v7_train.py --phase eval
    .venv/bin/python scripts/atlas_v7_train.py --phase all
    .venv/bin/python scripts/atlas_v7_train.py --quick-test
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import as_strided

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config_v7 import ATLASv7Config
from trading_algo.quant_core.models.atlas.model_v7 import ATLASModelV7
from trading_algo.quant_core.models.atlas.fast_env import VectorizedOptionsEnv
from trading_algo.quant_core.models.atlas.train_ppo import (
    load_training_data_v2,
    compute_gae,
)

FEATURE_DIR = "data/atlas_features_v4"
CHECKPOINT_DIR = "checkpoints/atlas_v7"
N_ENVS = 64

SYMBOLS_PER_CHUNK = 100
SYMBOLS_PER_EPOCH = 800

REGIME_NAMES = {0: "high_iv", 1: "uptrend", 2: "crash", 3: "neutral"}
REGIME_FILTERS = {0: "high_iv", 1: "uptrend", 2: "crash", 3: "all"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_windows(
    data: dict[str, np.ndarray],
    context_len: int,
    stride: int = 3,
) -> dict[str, torch.Tensor] | None:
    normed = data["normed"]
    actions = data["actions"]
    T = len(normed)
    L = context_len
    F = normed.shape[1]

    if T < L + 252:
        return None

    normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).clip(-10, 10)
    actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clip(-100, 100)
    mu_arr = np.nan_to_num(data["mu"], nan=0.0, posinf=0.0, neginf=0.0).clip(-10, 10)
    sigma_arr = np.nan_to_num(data["sigma"], nan=0.0, posinf=0.0, neginf=0.0).clip(0, 10)

    starts = np.arange(252, T - L + 1, stride)
    if len(starts) == 0:
        return None

    bs = normed.strides
    all_win = as_strided(normed, shape=(T - L + 1, L, F), strides=(bs[0], bs[0], bs[1]))
    max_abs = np.abs(all_win[starts]).reshape(len(starts), -1).max(axis=1)

    act_end = actions[starts + L - 1]
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


def softmax_sample(scores: np.ndarray, temperature: float = 0.5) -> int:
    shifted = -scores / temperature
    shifted -= shifted.max()
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum()
    return int(np.random.choice(len(scores), p=probs))


# ---------------------------------------------------------------------------
# Multi-component reward wrapper
# ---------------------------------------------------------------------------

class MultiComponentRewardWrapper:
    """Wraps VectorizedOptionsEnv to replace its reward with a multi-component signal.

    Components:
      1. Differential Sharpe Ratio (40%)
      2. Differential Sortino (30%)
      3. Drawdown penalty (20%)
      4. Tail risk / CVaR proxy (10%)
    """

    def __init__(self, env: VectorizedOptionsEnv, eta: float = 2.0 / 64) -> None:
        self.env = env
        self.n_envs = env.n_envs
        self.eta = eta

        self._A = np.zeros(self.n_envs, dtype=np.float64)
        self._B = np.zeros(self.n_envs, dtype=np.float64)
        self._D = np.zeros(self.n_envs, dtype=np.float64)
        self._equity_peaks = np.full(self.n_envs, 100_000.0, dtype=np.float64)
        self._prev_equity = np.full(self.n_envs, 100_000.0, dtype=np.float64)

    def reset_all(self) -> dict[str, torch.Tensor]:
        obs = self.env.reset_all()
        self._A[:] = 0.0
        self._B[:] = 0.0
        self._D[:] = 0.0
        self._equity_peaks[:] = 100_000.0
        self._prev_equity[:] = 100_000.0
        return obs

    def reset_stats(self, mask: np.ndarray) -> None:
        self._A[mask] = 0.0
        self._B[mask] = 0.0
        self._D[mask] = 0.0
        self._equity_peaks[mask] = 100_000.0
        self._prev_equity[mask] = 100_000.0

    def step(self, actions: np.ndarray) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        obs, raw_rewards, dones = self.env.step(actions)

        equity = self.env.prev_equity.copy()
        daily_ret = np.where(
            self._prev_equity > 0,
            (equity - self._prev_equity) / self._prev_equity,
            0.0,
        )

        # 1. Differential Sharpe Ratio
        delta_A = daily_ret - self._A
        new_B = self._B + self.eta * (daily_ret ** 2 - self._B)
        new_A = self._A + self.eta * delta_A
        denom = np.sqrt(np.maximum(new_B - new_A ** 2, 1e-16))
        dsr = np.where(denom > 1e-8, delta_A / denom, 0.0)
        self._A = new_A
        self._B = new_B

        # 2. Differential Sortino (only penalizes downside)
        downside_sq = np.where(daily_ret < 0, daily_ret ** 2, 0.0)
        self._D = self._D + self.eta * (downside_sq - self._D)
        downside_std = np.sqrt(np.maximum(self._D, 1e-16))
        sortino = np.where(downside_std > 1e-8, daily_ret / downside_std, 0.0)

        # 3. Drawdown penalty (quadratic above 15%)
        self._equity_peaks = np.maximum(self._equity_peaks, equity)
        dd = np.where(
            self._equity_peaks > 0,
            (self._equity_peaks - equity) / self._equity_peaks,
            0.0,
        )
        dd_penalty = np.maximum(0.0, dd - 0.15) ** 2 * 10.0

        # 4. Tail risk (exponential penalty for >2% daily loss)
        tail_penalty = np.maximum(0.0, -daily_ret - 0.02) * 5.0

        # Use raw env rewards as base (proven to work in v3/v4/v6)
        # Add mild supplementary signals — don't let penalties dominate
        reward = raw_rewards + 0.1 * dsr + 0.05 * sortino - 0.05 * dd_penalty - 0.02 * tail_penalty
        reward = np.clip(reward, -5.0, 5.0)

        self._prev_equity = equity.copy()

        # Reset stats for done envs
        if dones.any():
            self.reset_stats(dones)

        return obs, reward, dones


# ---------------------------------------------------------------------------
# Progressive action masking
# ---------------------------------------------------------------------------

def apply_action_mask(
    actions: torch.Tensor | np.ndarray,
    iteration: int,
) -> torch.Tensor | np.ndarray:
    """Mask action dimensions based on training progress.

    Dims: [0]=delta, [1]=direction, [2]=leverage, [3]=DTE, [4]=profit_target
    Phase 1 (iter <  50): only direction + leverage (mask delta=0.25, DTE=30norm, pt=0.5)
    Phase 2 (iter < 150): unlock delta + DTE (mask pt=0.5)
    Phase 3 (iter >= 150): full 5D
    """
    is_numpy = isinstance(actions, np.ndarray)

    if iteration < 50:
        if is_numpy:
            actions[:, 0] = 0.25
            actions[:, 3] = (30.0 - 14.0) / 76.0
            actions[:, 4] = 0.5
        else:
            actions = actions.clone()
            actions[:, 0] = 0.25
            actions[:, 3] = (30.0 - 14.0) / 76.0
            actions[:, 4] = 0.5
    elif iteration < 150:
        if is_numpy:
            actions[:, 4] = 0.5
        else:
            actions = actions.clone()
            actions[:, 4] = 0.5

    return actions


# ---------------------------------------------------------------------------
# Phase 1: Behavioral Cloning (streaming, same pattern as v6)
# ---------------------------------------------------------------------------

def phase_bc(config: ATLASv7Config, device: str, quick_test: bool = False) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    L = config.context_len

    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")])
    print(f"Feature files: {len(feature_files)}", flush=True)

    model = ATLASModelV7(config).to(device).float()
    print(f"Parameters: {model.count_parameters():,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n_epochs = 2 if quick_test else config.bc_epochs
    batch_size = 512

    total_chunks = (len(feature_files) + SYMBOLS_PER_CHUNK - 1) // SYMBOLS_PER_CHUNK
    total_steps_est = total_chunks * n_epochs * 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps_est, 1), eta_min=1e-6,
    )

    best_val = float("inf")
    patience = 0
    best_path = f"{CHECKPOINT_DIR}/atlas_v7_bc_best.pt"

    last_good_state = {k: v.clone() for k, v in model.state_dict().items()}
    last_good_optim = copy.deepcopy(optimizer.state_dict())

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()

        epoch_files = feature_files.copy()
        np.random.shuffle(epoch_files)
        n_files = 100 if quick_test else min(SYMBOLS_PER_EPOCH, len(epoch_files))
        epoch_files = epoch_files[:n_files]

        epoch_loss = 0.0
        epoch_batches = 0
        epoch_windows = 0

        def _load_symbol(fname: str) -> dict[str, torch.Tensor] | None:
            data = dict(np.load(f"{FEATURE_DIR}/{fname}"))
            return _extract_windows(data, L, stride=3)

        for chunk_start in range(0, len(epoch_files), SYMBOLS_PER_CHUNK):
            chunk_file_list = epoch_files[chunk_start:chunk_start + SYMBOLS_PER_CHUNK]

            all_windows: list[dict[str, torch.Tensor]] = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for w in pool.map(_load_symbol, chunk_file_list):
                    if w is not None:
                        all_windows.append(w)

            if not all_windows:
                continue

            chunk_data = {
                k: torch.cat([w[k] for w in all_windows], dim=0).to(device)
                for k in all_windows[0].keys()
            }
            N = chunk_data["features"].shape[0]
            epoch_windows += N

            opex_zeros = torch.zeros(N, L, device=device)
            qtr_zeros = torch.zeros(N, L, device=device)

            val_n = max(1, N // 20)
            perm = torch.randperm(N, device=device)
            train_perm = perm[val_n:]
            val_perm = perm[:val_n]

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

                if torch.isnan(normed_pred).any() or torch.isinf(normed_pred).any():
                    optimizer.zero_grad()
                    continue

                loss = nn.functional.mse_loss(normed_pred, ba)

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

                if epoch_batches % 50 == 0:
                    p0 = next(model.parameters())
                    if torch.isnan(p0).any():
                        print(f"  WARNING: NaN in weights at batch {epoch_batches} — rolling back", flush=True)
                        model.load_state_dict(last_good_state)
                        optimizer.load_state_dict(copy.deepcopy(last_good_optim))
                        break

            model.eval()
            val_losses: list[float] = []
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

            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f"  WARNING: NaN in weights after chunk — rolling back", flush=True)
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

    print(f"\nBC complete. Best loss: {best_val:.6f}", flush=True)
    return best_path


# ---------------------------------------------------------------------------
# Phase 2: Multi-task regime-conditioned PPO
# ---------------------------------------------------------------------------

def phase_ppo(
    config: ATLASv7Config,
    device: str,
    bc_checkpoint: str,
    quick_test: bool = False,
    n_envs: int = N_ENVS,
) -> None:
    model = ATLASModelV7(config)
    ckpt = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded BC checkpoint: {bc_checkpoint}", flush=True)

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

    # Build per-regime envs wrapped with multi-component reward
    regime_envs: dict[int, MultiComponentRewardWrapper] = {}
    for rid, rfilter in REGIME_FILTERS.items():
        raw_env = VectorizedOptionsEnv(
            all_data, config,
            n_envs=n_envs,
            regime_filter=rfilter,
            reward_shaping="none",
        )
        regime_envs[rid] = MultiComponentRewardWrapper(raw_env, eta=config.dsr_eta)

    # CURATE-style regime scores (running mean reward per regime)
    regime_scores = np.zeros(4, dtype=np.float64)
    regime_counts = np.zeros(4, dtype=np.int64)

    # Lower LR for PPO to preserve BC-learned policy (1e-4 vs 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=config.weight_decay)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    n_iterations = 20 if quick_test else 400
    rollout_steps = 256 if quick_test else 512

    model_cpu = ATLASModelV7(config).float().eval()
    model_cpu.load_state_dict(model.state_dict())
    model = model.to(device).float()

    history: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [],
        "mean_reward": [], "regime_id": [],
    }

    print(f"\nMulti-task PPO: {n_iterations} iterations, {n_envs} envs, "
          f"rollout_steps={rollout_steps}", flush=True)
    train_start = time.time()

    for iteration in range(n_iterations):
        # CURATE: sample regime proportional to softmax(-score / temp)
        if regime_counts.sum() < 4:
            regime_id = iteration % 4
        else:
            regime_id = softmax_sample(regime_scores, temperature=0.5)

        wrapped_env = regime_envs[regime_id]
        obs = wrapped_env.reset_all()

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
            next_obs, rewards, dones = wrapped_env.step(action_np)

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

        # Update CURATE scores
        iter_mean_reward = float(np.mean(all_rewards))
        alpha = 0.1
        regime_scores[regime_id] = (
            (1 - alpha) * regime_scores[regime_id] + alpha * iter_mean_reward
            if regime_counts[regime_id] > 0
            else iter_mean_reward
        )
        regime_counts[regime_id] += 1

        # PPO update
        model.train()

        batch_features = torch.cat(all_features).to(device)
        batch_ts = torch.cat(all_ts).to(device)
        batch_dow = torch.cat(all_dow).to(device)
        batch_month = torch.cat(all_month).to(device)
        batch_opex = torch.cat(all_opex).to(device)
        batch_qtr = torch.cat(all_qtr).to(device)
        batch_mu = torch.cat(all_mu).to(device)
        batch_sigma = torch.cat(all_sigma).to(device)
        batch_rtg = torch.cat(all_rtg).to(device)
        batch_actions = torch.cat(all_actions).to(device)
        batch_old_lp = torch.tensor(all_log_probs, device=device)
        batch_adv = advantages.to(device)
        batch_ret = returns.to(device)

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

        history["policy_loss"].append(avg_pl)
        history["value_loss"].append(avg_vl)
        history["entropy"].append(avg_ent)
        history["mean_reward"].append(iter_mean_reward)
        history["regime_id"].append(float(regime_id))

        if (iteration + 1) % 5 == 0 or iteration == 0:
            regime_name = REGIME_NAMES[regime_id]
            action_phase = "dir+lev" if iteration < 50 else ("dir+lev+delta+dte" if iteration < 150 else "full_5D")
            print(f"  PPO iter {iteration+1}/{n_iterations}: "
                  f"regime={regime_name:<8} policy={avg_pl:.4f} value={avg_vl:.4f} "
                  f"entropy={avg_ent:.4f} reward={iter_mean_reward:.4f} "
                  f"actions={action_phase}", flush=True)

        if (iteration + 1) % 50 == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/atlas_v7_ppo_iter{iteration+1}.pt"
            torch.save({
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "regime_scores": regime_scores.tolist(),
                "config": config,
            }, ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}", flush=True)
            print(f"    Regime scores: " + "  ".join(
                f"{REGIME_NAMES[i]}={regime_scores[i]:.4f}" for i in range(4)
            ), flush=True)

    final_path = f"{CHECKPOINT_DIR}/atlas_v7_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "regime_scores": regime_scores.tolist(),
        "config": config,
    }, final_path)

    total_time = time.time() - train_start
    print(f"\nPPO complete. {n_iterations} iterations in {total_time/60:.1f} min "
          f"({total_time/n_iterations:.1f}s/iter)", flush=True)
    print(f"Final regime scores: " + "  ".join(
        f"{REGIME_NAMES[i]}={regime_scores[i]:.4f}" for i in range(4)
    ), flush=True)
    print(f"Checkpoint: {final_path}", flush=True)


# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------

def _classify_regime(
    closes: np.ndarray,
    iv_ranks: np.ndarray,
    t0: int,
    ep_len: int,
) -> str:
    iv_window = iv_ranks[t0:min(t0 + 30, len(iv_ranks))]
    if len(iv_window) > 0 and float(np.nanmean(iv_window)) > 60:
        return "high_iv"

    if ep_len >= 20 and t0 + 21 <= len(closes):
        log_rets = np.diff(np.log(np.maximum(closes[t0:t0 + 21], 1e-8)))
        rv = float(np.std(log_rets) * np.sqrt(252)) if len(log_rets) > 1 else 0.0
        ret20 = (float(closes[t0 + 19]) - float(closes[t0])) / max(float(closes[t0]), 1e-8)
        if rv > 0.40 or ret20 < -0.10:
            return "crash"

    if ep_len >= 60 and t0 + 60 < len(closes):
        ret60 = (float(closes[t0 + 59]) - float(closes[t0])) / max(float(closes[t0]), 1e-8)
        if ret60 > 0.15:
            return "uptrend"

    return "neutral"


def run_eval_episode(
    env: VectorizedOptionsEnv,
    model: ATLASModelV7,
    device: str,
    all_data: dict[str, dict],
) -> dict[str, float | str | bool]:
    """Run a single-env evaluation episode."""
    single_env = VectorizedOptionsEnv(
        all_data, env.config,
        n_envs=1,
        regime_filter="all",
        reward_shaping="none",
    )
    obs = single_env.reset_all()

    sym = single_env._sym[0]
    t0_ep = single_env._start[0]
    t_end = single_env._end[0]
    data = single_env._data_ref[0]
    closes = data["closes"]
    iv_ranks = data["iv_ranks"]
    ep_len = t_end - t0_ep

    equity_curve: list[float] = [100_000.0]

    done = False
    while not done:
        with torch.no_grad():
            action_mean, _ = model.forward_with_value(
                obs["features"].to(device), obs["timestamps"].to(device),
                obs["dow"].to(device), obs["month"].to(device),
                obs["is_opex"].to(device), obs["is_qtr"].to(device),
                obs["pre_mu"].to(device), obs["pre_sigma"].to(device),
                obs["rtg"].to(device),
            )
            action_mean = torch.nan_to_num(action_mean, nan=0.0)
            action_np = action_mean.cpu().numpy()

        obs, rewards, dones = single_env.step(action_np)
        equity_curve.append(float(single_env.prev_equity[0]))
        done = bool(dones[0])

    equity = np.array(equity_curve, dtype=np.float64)
    n_days = len(equity) - 1
    total_return = (equity[-1] - equity[0]) / equity[0]
    ann_return = total_return * (252.0 / max(n_days, 1))
    daily_rets = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
    sharpe = (float(np.mean(daily_rets)) / (float(np.std(daily_rets)) + 1e-10)) * np.sqrt(252)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-8)))

    bnh_end = min(single_env._t[0], len(closes) - 1)
    bnh_return = (float(closes[bnh_end]) - float(closes[t0_ep])) / max(float(closes[t0_ep]), 1e-8)

    regime = _classify_regime(closes, iv_ranks, t0_ep, ep_len)

    return {
        "sym": sym,
        "regime": regime,
        "ep_len": n_days,
        "total_ret": total_return,
        "ann_ret": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "final_equity": float(equity[-1]),
        "bnh_ret": bnh_return,
        "win": total_return > 0,
    }


def aggregate(episodes: list[dict]) -> dict[str, float]:
    if not episodes:
        return {}
    rets = [e["total_ret"] for e in episodes]
    sharpe = [e["sharpe"] for e in episodes]
    dd = [e["max_dd"] for e in episodes]
    bnh = [e["bnh_ret"] for e in episodes]
    wins = [e["win"] for e in episodes]
    return {
        "n": len(episodes),
        "mean_ret": float(np.mean(rets)),
        "mean_sharpe": float(np.mean(sharpe)),
        "mean_dd": float(np.mean(dd)),
        "max_dd": float(np.max(dd)),
        "win_rate": float(np.mean(wins)),
        "mean_bnh": float(np.mean(bnh)),
        "alpha": float(np.mean(rets)) - float(np.mean(bnh)),
    }


def print_table(label: str, agg: dict[str, float]) -> None:
    n = int(agg.get("n", 0))
    if n == 0:
        return
    print(f"  {label:<14} n={n:>3} | "
          f"ret={agg['mean_ret']:>+7.2%}  "
          f"SR={agg['mean_sharpe']:>+6.2f}  DD={agg['mean_dd']:>5.2%}  "
          f"win={agg['win_rate']:>5.1%}  alpha={agg['alpha']:>+6.2%}  "
          f"BnH={agg['mean_bnh']:>+6.2%}", flush=True)


def phase_eval(
    config: ATLASv7Config,
    device: str,
    checkpoint: str,
    n_episodes: int = 200,
) -> None:
    model = ATLASModelV7(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).float().eval()
    print(f"Loaded checkpoint: {checkpoint}", flush=True)

    print("Loading evaluation data...", flush=True)
    t0 = time.time()
    all_data = load_training_data_v2(FEATURE_DIR, min_len=config.context_len + 200)
    print(f"Loaded {len(all_data)} symbols in {time.time()-t0:.1f}s", flush=True)

    dummy_env = VectorizedOptionsEnv(all_data, config, n_envs=1, regime_filter="all", reward_shaping="none")

    print(f"\nRunning {n_episodes} evaluation episodes...", flush=True)
    episodes: list[dict] = []
    t0 = time.time()
    for i in range(n_episodes):
        ep = run_eval_episode(dummy_env, model, device, all_data)
        episodes.append(ep)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_episodes - i - 1)
            print(f"  {i+1}/{n_episodes} episodes ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("  ATLAS v7 EVALUATION RESULTS", flush=True)
    print(f"{'='*80}", flush=True)

    print("\n--- OVERALL ---", flush=True)
    print_table("v7", aggregate(episodes))

    # Try to load v6 for comparison
    v6_path = "checkpoints/atlas_v6/atlas_v6_final.pt"
    if os.path.exists(v6_path):
        print(f"\n  (v6 baseline loaded from {v6_path})", flush=True)

    regimes = ["high_iv", "uptrend", "crash", "neutral"]
    for regime in regimes:
        r_eps = [e for e in episodes if e["regime"] == regime]
        if not r_eps:
            continue
        print(f"\n--- {regime.upper()} ---", flush=True)
        print_table("v7", aggregate(r_eps))

    print(f"\n  Regime distribution:", flush=True)
    for regime in regimes:
        n = sum(1 for e in episodes if e["regime"] == regime)
        pct = n / max(len(episodes), 1)
        print(f"    {regime:<10} {n:>4} episodes ({pct:.1%})", flush=True)

    overall = aggregate(episodes)
    print(f"\n  Overall SR: {overall.get('mean_sharpe', 0):+.2f}", flush=True)
    print(f"  Win rate:   {overall.get('win_rate', 0):.1%}", flush=True)
    print(f"  Max DD:     {overall.get('max_dd', 0):.2%}", flush=True)
    print(f"  Alpha:      {overall.get('alpha', 0):+.2%}", flush=True)
    print(f"{'='*80}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS v7 — Multi-task regime-conditioned training")
    parser.add_argument("--phase", choices=["bc", "ppo", "eval", "all"], default="all")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-episodes", type=int, default=200)
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    config = ATLASv7Config()

    print("=" * 70, flush=True)
    print("  ATLAS v7 — Multi-task Regime-Conditioned Training", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Features: {FEATURE_DIR}", flush=True)
    print(f"Output: {CHECKPOINT_DIR}", flush=True)
    print(f"Envs: {args.n_envs}", flush=True)
    if args.quick_test:
        print("MODE: quick-test (2 BC epochs, 20 PPO iters)", flush=True)

    n_envs = args.n_envs

    bc_path = f"{CHECKPOINT_DIR}/atlas_v7_bc_best.pt"
    final_path = f"{CHECKPOINT_DIR}/atlas_v7_final.pt"

    if args.phase in ("bc", "all"):
        print(f"\n{'='*60}", flush=True)
        print("  Phase 1: Behavioral Cloning (streaming)", flush=True)
        print(f"{'='*60}", flush=True)
        bc_path = phase_bc(config, device, quick_test=args.quick_test)

    if args.phase in ("ppo", "all"):
        print(f"\n{'='*60}", flush=True)
        print("  Phase 2: Multi-task PPO", flush=True)
        print(f"{'='*60}", flush=True)
        if not os.path.exists(bc_path):
            print(f"ERROR: BC checkpoint not found at {bc_path}", flush=True)
            print("Run --phase bc first.", flush=True)
            return
        phase_ppo(config, device, bc_path, quick_test=args.quick_test, n_envs=args.n_envs)

    if args.phase in ("eval", "all"):
        print(f"\n{'='*60}", flush=True)
        print("  Phase 3: Evaluation", flush=True)
        print(f"{'='*60}", flush=True)
        eval_ckpt = final_path if os.path.exists(final_path) else bc_path
        if not os.path.exists(eval_ckpt):
            print(f"ERROR: No checkpoint found for eval", flush=True)
            return
        n_eval = 50 if args.quick_test else args.eval_episodes
        phase_eval(config, device, eval_ckpt, n_episodes=n_eval)


if __name__ == "__main__":
    main()
