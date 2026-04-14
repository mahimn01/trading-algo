"""ATLAS Crash Fix — targeted crash-focused re-training from atlas_curriculum_final.pt.

Three root-cause fixes landed in train_ppo.py before this runs:
  1. 2× stop-loss for short_put — model can now escape losing positions
  2. Position-aware crash shaping — +2.0 for flat during crash, -2.0 for holding short puts
  3. Broader crash filter — RV30 > 30% OR 20d < -8% (was 40% / -10%)

Curriculum: 150 crash + 75 mixed from the existing curriculum checkpoint.
Goal: crash Sharpe from -0.14 → positive, win rate from 40% → 60%+.

Usage:
    .venv/bin/python scripts/atlas_crash_fix.py
    .venv/bin/python scripts/atlas_crash_fix.py --quick-test
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
CHECKPOINT_DIR = "checkpoints/atlas_v3_crash_fix"
FEATURE_DIR    = "data/atlas_features_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS Crash Fix Training")
    parser.add_argument("--quick-test", action="store_true", help="5 iters per stage")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("  ATLAS Crash Fix — Targeted Crash Re-Training")
    print("=" * 60)
    print(f"Device: {device}", flush=True)
    print(f"Source checkpoint: {CHECKPOINT_IN}", flush=True)

    config = ATLASConfig()
    model  = ATLASModel(config)

    ckpt = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded curriculum checkpoint.", flush=True)

    if args.quick_test:
        stages = [
            CurriculumStage(name="crash_fix", iterations=5,  regime_filter="crash", reward_shaping="crash"),
            CurriculumStage(name="mixed_re",  iterations=5,  regime_filter="all",   reward_shaping="none"),
        ]
        rollout_steps = 256
        print("Quick-test mode: 5 iterations per stage", flush=True)
    else:
        stages = [
            CurriculumStage(name="crash_fix", iterations=150, regime_filter="crash", reward_shaping="crash"),
            CurriculumStage(name="mixed_re",  iterations=75,  regime_filter="all",   reward_shaping="none"),
        ]
        rollout_steps = 512
        print("Full mode: 150 crash + 75 mixed iterations", flush=True)

    train_curriculum(
        model=model,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        device=device,
        feature_dir=FEATURE_DIR,
        stages=stages,
        rollout_steps=rollout_steps,
    )

    print(f"\nCrash fix training complete.", flush=True)
    print(f"Final checkpoint: {CHECKPOINT_DIR}/atlas_curriculum_final.pt", flush=True)


if __name__ == "__main__":
    main()
