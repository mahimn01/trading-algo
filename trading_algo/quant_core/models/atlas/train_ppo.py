"""Phase 2: PPO RL fine-tuning for ATLAS."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from trading_algo.quant_core.models.atlas.config import ATLASConfig


@dataclass
class Rollout:
    """Stores a batch of rollout data for PPO."""
    features: list[Tensor] = field(default_factory=list)
    timestamps: list[Tensor] = field(default_factory=list)
    dow: list[Tensor] = field(default_factory=list)
    month: list[Tensor] = field(default_factory=list)
    is_opex: list[Tensor] = field(default_factory=list)
    is_qtr: list[Tensor] = field(default_factory=list)
    pre_mu: list[Tensor] = field(default_factory=list)
    pre_sigma: list[Tensor] = field(default_factory=list)
    rtg: list[Tensor] = field(default_factory=list)
    actions: list[Tensor] = field(default_factory=list)
    log_probs: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


class TradingEnvironment:
    """Simulates single-symbol options trading for PPO rollouts."""

    def __init__(
        self,
        all_features: dict[str, np.ndarray],
        config: ATLASConfig,
    ):
        self.all_features = all_features
        self.config = config
        self.symbols = list(all_features.keys())
        self.reset()

    def reset(self) -> dict[str, Tensor]:
        sym = random.choice(self.symbols)
        data = self.all_features[sym]
        T = data.shape[0]
        max_ep_len = min(500, T - self.config.context_len - 60)
        if max_ep_len < 100:
            max_ep_len = 100
        start = random.randint(self.config.context_len, max(self.config.context_len, T - max_ep_len - 1))
        self._data = data
        self._sym = sym
        self._t = start
        self._start = start
        self._end = min(start + max_ep_len, T - 1)
        self._equity = 1.0
        self._peak = 1.0
        self._prev_action = np.zeros(5)
        self._ema_ret = 0.0
        self._ema_ret2 = 0.0
        self._target_sharpe = random.uniform(0.3, 2.0)
        return self._get_obs()

    def _get_obs(self) -> dict[str, Tensor]:
        L = self.config.context_len
        window = self._data[self._t - L + 1 : self._t + 1]  # (L, F)
        features = torch.tensor(window[:, :16], dtype=torch.float32).unsqueeze(0)
        ts = torch.arange(L, dtype=torch.float32).unsqueeze(0)
        dow = torch.zeros(1, L, dtype=torch.long)
        mo = torch.zeros(1, L, dtype=torch.long)
        opex = torch.zeros(1, L)
        qtr = torch.zeros(1, L)
        mu = torch.zeros(1, L, 16)
        sigma = torch.ones(1, L, 16) * 0.2
        rtg = torch.tensor([self._target_sharpe])
        return {
            "features": features, "timestamps": ts, "dow": dow, "month": mo,
            "is_opex": opex, "is_qtr": qtr, "pre_mu": mu, "pre_sigma": sigma, "rtg": rtg,
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, Tensor], float, bool]:
        direction = action[1]
        leverage = action[2]
        if self._t + 1 >= len(self._data):
            return self._get_obs(), 0.0, True

        price_now = self._data[self._t, 0] if self._data.shape[1] > 0 else 0.0
        price_next = self._data[self._t + 1, 0] if self._t + 1 < len(self._data) else price_now
        daily_ret = (price_next - price_now) / max(abs(price_now), 1e-8) if price_now != 0 else 0.0
        trade_ret = direction * leverage * daily_ret

        self._equity *= (1 + trade_ret)
        self._peak = max(self._peak, self._equity)

        # DSR reward
        eta = 2.0 / 64.0
        self._ema_ret += eta * (trade_ret - self._ema_ret)
        self._ema_ret2 += eta * (trade_ret ** 2 - self._ema_ret2)
        var = self._ema_ret2 - self._ema_ret ** 2
        dsr = 0.0
        if var > 1e-12:
            dsr = (self._ema_ret2 * eta * (trade_ret - self._ema_ret) -
                   0.5 * self._ema_ret * eta * (trade_ret**2 - self._ema_ret2)) / max(var ** 1.5, 1e-12)

        # Drawdown penalty
        dd = (self._peak - self._equity) / max(self._peak, 1e-8)
        dd_pen = 10.0 * max(0.0, dd - 0.20) ** 2

        # Transaction cost
        tc = 0.01 * float(np.abs(action - self._prev_action).sum())
        self._prev_action = action.copy()

        reward = dsr - dd_pen - tc
        self._t += 1
        done = self._t >= self._end

        return self._get_obs(), reward, done


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T and not dones[t] else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def train_ppo(
    model: nn.Module,
    train_features: dict[str, np.ndarray],
    config: ATLASConfig,
    checkpoint_dir: str = "checkpoints/atlas",
    device: str = "auto",
    n_iterations: int = 500,
    rollout_steps: int = 2048,
) -> dict:
    """
    Phase 2: PPO RL fine-tuning with DSR reward.

    Args:
        model: ATLASModel (pre-trained via BC).
        train_features: symbol -> (T, F) array of normalized features.
        config: ATLASConfig.
        n_iterations: Number of PPO iterations.
        rollout_steps: Steps per rollout collection.

    Returns:
        Training history dict.
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).float()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    env = TradingEnvironment(train_features, config)
    history = {"policy_loss": [], "value_loss": [], "entropy": [], "mean_reward": []}

    for iteration in range(n_iterations):
        # Collect rollout
        rollout = Rollout()
        obs = env.reset()
        ep_rewards = []

        model.eval()
        for _ in range(rollout_steps):
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in obs.items()}
                action_mean, value = model.forward_with_value(
                    inputs["features"], inputs["timestamps"], inputs["dow"],
                    inputs["month"], inputs["is_opex"], inputs["is_qtr"],
                    inputs["pre_mu"], inputs["pre_sigma"], inputs["rtg"],
                )
                dist = model.get_action_distribution(action_mean)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done = env.step(action_np)

            for k in obs:
                getattr(rollout, {
                    "features": "features", "timestamps": "timestamps",
                    "dow": "dow", "month": "month", "is_opex": "is_opex",
                    "is_qtr": "is_qtr", "pre_mu": "pre_mu",
                    "pre_sigma": "pre_sigma", "rtg": "rtg",
                }.get(k, k)).append(obs[k].cpu())

            rollout.actions.append(action.cpu())
            rollout.log_probs.append(log_prob.cpu())
            rollout.rewards.append(reward)
            rollout.values.append(value.item())
            rollout.dones.append(done)
            ep_rewards.append(reward)

            obs = next_obs
            if done:
                obs = env.reset()

        # GAE
        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones,
            config.gamma, config.gae_lambda,
        )

        # PPO update
        model.train()
        n_steps = len(rollout.rewards)
        indices = np.arange(n_steps)
        mini_batch_size = min(512, n_steps)

        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, mini_batch_size):
                end = min(start + mini_batch_size, n_steps)
                mb_idx = indices[start:end]

                mb_features = torch.cat([rollout.features[i] for i in mb_idx]).to(device)
                mb_ts = torch.cat([rollout.timestamps[i] for i in mb_idx]).to(device)
                mb_dow = torch.cat([rollout.dow[i] for i in mb_idx]).to(device)
                mb_month = torch.cat([rollout.month[i] for i in mb_idx]).to(device)
                mb_opex = torch.cat([rollout.is_opex[i] for i in mb_idx]).to(device)
                mb_qtr = torch.cat([rollout.is_qtr[i] for i in mb_idx]).to(device)
                mb_mu = torch.cat([rollout.pre_mu[i] for i in mb_idx]).to(device)
                mb_sigma = torch.cat([rollout.pre_sigma[i] for i in mb_idx]).to(device)
                mb_rtg = torch.cat([rollout.rtg[i] for i in mb_idx]).to(device)
                mb_actions = torch.cat([rollout.actions[i] for i in mb_idx]).to(device)
                mb_old_lp = torch.tensor([rollout.log_probs[i].item() for i in mb_idx], device=device)
                mb_adv = advantages[mb_idx].to(device)
                mb_ret = returns[mb_idx].to(device)

                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_mean, new_val = model.forward_with_value(
                    mb_features, mb_ts, mb_dow, mb_month, mb_opex, mb_qtr,
                    mb_mu, mb_sigma, mb_rtg,
                )
                dist = model.get_action_distribution(new_mean)
                new_lp = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

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

        avg_pl = total_pl / max(n_updates, 1)
        avg_vl = total_vl / max(n_updates, 1)
        avg_ent = total_ent / max(n_updates, 1)
        avg_rew = float(np.mean(ep_rewards))

        history["policy_loss"].append(avg_pl)
        history["value_loss"].append(avg_vl)
        history["entropy"].append(avg_ent)
        history["mean_reward"].append(avg_rew)

        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(f"  PPO iter {iteration+1}/{n_iterations}: "
                  f"policy={avg_pl:.4f} value={avg_vl:.4f} "
                  f"entropy={avg_ent:.4f} reward={avg_rew:.4f}")
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"{checkpoint_dir}/atlas_ppo_iter{iteration+1}.pt")

    return history
