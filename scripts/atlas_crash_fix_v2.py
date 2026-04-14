"""ATLAS Crash Fix v2 — minimal-disruption approach.

The v1 attempt used aggressive ±2.0 crash shaping which drove the policy to
max-entropy and required 75 mixed iters to partially recover. The result was
worse overall behavior even though crash DD improved.

v2 insight: the stop-loss (2× entry premium) is the structural fix. It caps
downside mechanically, regardless of policy behavior. All we need is to let the
model adapt to the new stop-loss mechanic via 100 mixed iterations from the
original good checkpoint — no crash-specific shaping, no regime filter.

The model keeps its high_iv/neutral behavior (SR+0.36, win 72%) while the
stop-loss caps crash losses. 100 mixed iters teaches the policy to account for
the new close-at-2× dynamic without disturbing learned behavior.

Usage:
    .venv/bin/python scripts/atlas_crash_fix_v2.py
    .venv/bin/python scripts/atlas_crash_fix_v2.py --quick-test
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.model import ATLASModel
from trading_algo.quant_core.models.atlas.train_curriculum import CurriculumStage, train_curriculum

CHECKPOINT_IN  = "checkpoints/atlas_v3_curriculum/atlas_curriculum_final.pt"
CHECKPOINT_DIR = "checkpoints/atlas_v3_crash_fix_v3"
FEATURE_DIR    = "data/atlas_features_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS Crash Fix v2")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("  ATLAS Crash Fix v2 — Stop-Loss Adaptation")
    print("=" * 60)
    print(f"Device: {device}", flush=True)
    print(f"Source: {CHECKPOINT_IN}", flush=True)
    print("Strategy: preserve existing policy, adapt to 2x stop-loss", flush=True)

    config = ATLASConfig()
    model  = ATLASModel(config)

    ckpt = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded original curriculum checkpoint (SR+0.36, win 72%).", flush=True)

    if args.quick_test:
        stages = [
            CurriculumStage(name="mixed_adapt", iterations=5, regime_filter="all", reward_shaping="none"),
        ]
        rollout_steps = 256
        print("Quick-test: 5 iterations", flush=True)
    else:
        stages = [
            CurriculumStage(name="mixed_adapt", iterations=100, regime_filter="all", reward_shaping="none"),
        ]
        rollout_steps = 512
        print("Full: 100 mixed iterations (no reward shaping)", flush=True)

    train_curriculum(
        model=model,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        device=device,
        feature_dir=FEATURE_DIR,
        stages=stages,
        rollout_steps=rollout_steps,
    )

    print(f"\nDone. Checkpoint: {CHECKPOINT_DIR}/atlas_curriculum_final.pt", flush=True)


if __name__ == "__main__":
    main()
