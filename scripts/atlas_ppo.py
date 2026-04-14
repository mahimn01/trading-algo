"""ATLAS PPO Training — Run as: .venv/bin/python scripts/atlas_ppo.py"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from trading_algo.quant_core.models.atlas.model import ATLASModel
from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.train_ppo import train_ppo

FEATURE_DIR = "data/atlas_features"
CHECKPOINT_DIR = "checkpoints/atlas"

def main():
    print("=" * 60)
    print("  ATLAS Phase 2: PPO RL Fine-Tuning")
    print("=" * 60)

    config = ATLASConfig()

    # Load BC model with memory bank
    model = ATLASModel(config)
    ckpt_path = f"{CHECKPOINT_DIR}/atlas_bc_with_memory.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{CHECKPOINT_DIR}/atlas_bc_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Parameters: {model.count_parameters():,}")

    # Load training features for environment
    print("  Loading training features for environment...")
    train_features = {}
    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")])

    for fname in feature_files:
        sym = fname.replace("_features.npz", "")
        data = np.load(f"{FEATURE_DIR}/{fname}")
        normed = data["normed"]
        if len(normed) < config.context_len + 200:
            continue
        train_features[sym] = normed
    print(f"  Loaded {len(train_features)} symbols")

    # Run PPO
    print(f"\n  Starting PPO training ({500} iterations)...")
    print(f"  This will take ~4-8 hours on MPS")
    print(f"  Checkpoints saved every 50 iterations to {CHECKPOINT_DIR}/")
    print()

    start = time.time()
    history = train_ppo(
        model=model,
        train_features=train_features,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        device="cpu",  # CPU is more numerically stable than MPS for RL
        n_iterations=200,  # reduced from 500 — can continue from checkpoint
        rollout_steps=512,  # smaller for faster iterations
    )

    elapsed = time.time() - start
    print(f"\n  PPO training complete in {elapsed/3600:.1f} hours")
    print(f"  Final policy loss: {history['policy_loss'][-1]:.6f}")
    print(f"  Final mean reward: {history['mean_reward'][-1]:.6f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "ppo_history": history,
        "config": config,
    }, f"{CHECKPOINT_DIR}/atlas_ppo_final.pt")
    print(f"  Saved final model to {CHECKPOINT_DIR}/atlas_ppo_final.pt")

if __name__ == "__main__":
    main()
