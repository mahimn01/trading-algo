"""Curriculum learning for ATLAS: teach regime-adaptive behavior via staged PPO training.

Stage 1 (high_iv):  High IV rank episodes — learn to sell aggressively at higher delta
Stage 2 (uptrend):  Strong uptrend episodes — learn to go long / positive direction
Stage 3 (crash):    Crash / high-vol episodes — learn to go to cash (delta < 0.10)
Stage 4 (mixed):    All regimes, no reward shaping — generalize without crutches
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.train_ppo import (
    load_training_data_v2,
    train_ppo,
)


@dataclass
class CurriculumStage:
    name: str
    iterations: int
    regime_filter: str
    reward_shaping: str


DEFAULT_STAGES: list[CurriculumStage] = [
    CurriculumStage(name="high_iv", iterations=50, regime_filter="high_iv", reward_shaping="high_iv"),
    CurriculumStage(name="uptrend", iterations=50, regime_filter="uptrend", reward_shaping="uptrend"),
    CurriculumStage(name="crash", iterations=50, regime_filter="crash", reward_shaping="crash"),
    CurriculumStage(name="mixed", iterations=100, regime_filter="all", reward_shaping="none"),
]


def train_curriculum(
    model: torch.nn.Module,
    config: ATLASConfig | None = None,
    checkpoint_dir: str = "checkpoints/atlas_curriculum",
    device: str = "cpu",
    feature_dir: str = "data/atlas_features_v2",
    stages: list[CurriculumStage] | None = None,
    rollout_steps: int = 512,
) -> dict[str, list]:
    """Run curriculum learning across all stages sequentially.

    The optimizer is shared across stages so momentum carries forward.
    Data is loaded once and reused.

    Returns:
        Combined training history across all stages.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config is None:
        config = ATLASConfig()
    if stages is None:
        stages = DEFAULT_STAGES

    model = model.to(device).float()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    print("Loading training data for curriculum learning...", flush=True)
    all_data = load_training_data_v2(feature_dir, min_len=config.context_len + 200)
    print(f"Loaded {len(all_data)} symbols.", flush=True)

    combined_history: dict[str, list] = {
        "stage": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "mean_reward": [],
    }

    for i, stage in enumerate(stages):
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage {i+1}/{len(stages)}: {stage.name} "
              f"({stage.iterations} iterations, regime={stage.regime_filter}, "
              f"shaping={stage.reward_shaping})", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        history = train_ppo(
            model=model,
            config=config,
            checkpoint_dir=checkpoint_dir,
            device=device,
            n_iterations=stage.iterations,
            rollout_steps=rollout_steps,
            regime_filter=stage.regime_filter,
            reward_shaping=stage.reward_shaping,
            all_data_preloaded=all_data,
            optimizer=optimizer,
        )
        elapsed = time.time() - t0

        for k in ("policy_loss", "value_loss", "entropy", "mean_reward"):
            combined_history[k].extend(history[k])
        combined_history["stage"].extend([stage.name] * len(history["mean_reward"]))

        torch.save({
            "stage": stage.name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        }, f"{checkpoint_dir}/atlas_curriculum_stage_{stage.name}.pt")

        print(f"  Stage {stage.name} complete in {elapsed:.0f}s", flush=True)
        print(f"  Mean reward (last 10): {np.mean(history['mean_reward'][-10:]):.4f}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": combined_history,
        "config": config,
    }, f"{checkpoint_dir}/atlas_curriculum_final.pt")

    print(f"\nCurriculum training complete. Saved to {checkpoint_dir}/atlas_curriculum_final.pt", flush=True)
    return combined_history
