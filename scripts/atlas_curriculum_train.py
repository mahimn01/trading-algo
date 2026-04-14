"""ATLAS Curriculum Training — teach regime-adaptive behavior via staged PPO.

Usage:
    .venv/bin/python scripts/atlas_curriculum_train.py
    .venv/bin/python scripts/atlas_curriculum_train.py --quick-test
    .venv/bin/python scripts/atlas_curriculum_train.py --from-bc checkpoints/atlas_v2/atlas_v2_bc_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.model import ATLASModel
from trading_algo.quant_core.models.atlas.train_curriculum import (
    CurriculumStage,
    train_curriculum,
)
from trading_algo.quant_core.models.atlas.train_ppo import load_training_data_v2

CHECKPOINT_DIR = "checkpoints/atlas_v3_curriculum"
FEATURE_DIR = "data/atlas_features_v3"


def quick_test(device: str = "auto"):
    """5 iterations per stage — verify regime differentiation."""
    print("=" * 60)
    print("  ATLAS Curriculum — Quick Test (5 iters/stage)")
    print("=" * 60)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    config = ATLASConfig()
    model = ATLASModel(config)

    bc_path = "checkpoints/atlas_v3/atlas_v3_bc_best.pt"
    if os.path.exists(bc_path):
        ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded BC checkpoint (val_loss={ckpt['val_loss']:.6f})")
    else:
        print("No BC checkpoint found — training from scratch (test only)")

    test_stages = [
        CurriculumStage(name="high_iv", iterations=5, regime_filter="high_iv", reward_shaping="high_iv"),
        CurriculumStage(name="uptrend", iterations=5, regime_filter="uptrend", reward_shaping="uptrend"),
        CurriculumStage(name="crash", iterations=5, regime_filter="crash", reward_shaping="crash"),
        CurriculumStage(name="mixed", iterations=5, regime_filter="all", reward_shaping="none"),
    ]

    history = train_curriculum(
        model=model,
        config=config,
        checkpoint_dir=f"{CHECKPOINT_DIR}_test",
        device=device,
        feature_dir=FEATURE_DIR,
        stages=test_stages,
        rollout_steps=256,
    )

    print("\n" + "=" * 60)
    print("  Regime Differentiation Test")
    print("=" * 60)
    _evaluate_regime_differentiation(model, config)


def full_train(from_bc: str | None = None, device: str = "auto"):
    """Full curriculum training: 50+50+50+100 iterations."""
    print("=" * 60)
    print("  ATLAS Curriculum — Full Training (250 iterations)")
    print("=" * 60)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    config = ATLASConfig()
    model = ATLASModel(config)

    if from_bc:
        ckpt = torch.load(from_bc, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded BC checkpoint from {from_bc} (val_loss={ckpt.get('val_loss', 'N/A')})")
    else:
        bc_path = "checkpoints/atlas_v3/atlas_v3_bc_best.pt"
        if os.path.exists(bc_path):
            ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded BC checkpoint (val_loss={ckpt['val_loss']:.6f})")
        else:
            print("WARNING: No BC checkpoint — starting from random init")

    history = train_curriculum(
        model=model,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        device=device,
        feature_dir=FEATURE_DIR,
        rollout_steps=512,
    )

    print("\n" + "=" * 60)
    print("  Post-Training Regime Differentiation Test")
    print("=" * 60)
    _evaluate_regime_differentiation(model, config)


def _evaluate_regime_differentiation(model: torch.nn.Module, config: ATLASConfig) -> None:
    """Feed the model different regime inputs and check if actions vary."""
    all_data = load_training_data_v2(FEATURE_DIR, min_len=config.context_len + 200)
    L = config.context_len
    model.eval()

    regime_actions: dict[str, list[np.ndarray]] = {
        "high_iv": [],
        "low_iv": [],
        "uptrend": [],
        "downtrend": [],
        "high_vol": [],
        "low_vol": [],
    }

    for sym, data in all_data.items():
        closes = data["closes"]
        iv_ranks = data["iv_ranks"]
        normed = data["normed"]
        mu = data["mu"]
        sigma = data["sigma"]
        timestamps = data["timestamps"]
        dow = data["dow"]
        month = data["month"]
        T = len(closes)

        for t in range(L + 252, T - 60, 100):
            window_normed = normed[t - L + 1: t + 1].copy()
            if np.isnan(window_normed).any():
                continue
            window_normed = np.nan_to_num(window_normed, nan=0.0)

            f_t = torch.tensor(window_normed, dtype=torch.float32).unsqueeze(0)
            ts_t = torch.tensor(timestamps[t - L + 1: t + 1], dtype=torch.float32).unsqueeze(0)
            dow_t = torch.tensor(dow[t - L + 1: t + 1], dtype=torch.long).unsqueeze(0)
            mo_t = torch.tensor(month[t - L + 1: t + 1], dtype=torch.long).unsqueeze(0)
            op_t = torch.zeros(1, L)
            qt_t = torch.zeros(1, L)
            mu_t = torch.tensor(np.nan_to_num(mu[t - L + 1: t + 1], nan=0.0), dtype=torch.float32).unsqueeze(0)
            si_t = torch.tensor(np.nan_to_num(sigma[t - L + 1: t + 1], nan=1.0), dtype=torch.float32).unsqueeze(0)
            rtg_t = torch.tensor([1.0])

            with torch.no_grad():
                action = model(f_t, ts_t, dow_t, mo_t, op_t, qt_t, mu_t, si_t, rtg_t)
            a = action.squeeze(0).numpy()

            ivr = float(iv_ranks[t])
            if ivr > 70:
                regime_actions["high_iv"].append(a)
            elif ivr < 30:
                regime_actions["low_iv"].append(a)

            if t >= 60:
                ret_60 = (closes[t] - closes[t - 60]) / max(closes[t - 60], 1e-8)
                if ret_60 > 0.15:
                    regime_actions["uptrend"].append(a)
                elif ret_60 < -0.10:
                    regime_actions["downtrend"].append(a)

            if t >= 30:
                log_rets = np.diff(np.log(np.maximum(closes[t - 30:t + 1], 1e-8)))
                rv = float(np.std(log_rets) * np.sqrt(252)) if len(log_rets) > 1 else 0.0
                if rv > 0.40:
                    regime_actions["high_vol"].append(a)
                elif rv < 0.15:
                    regime_actions["low_vol"].append(a)

    labels = ["delta", "direction", "leverage", "dte", "profit_tgt"]
    print(f"\n  {'Regime':<12} {'N':>5}  {'delta':>7} {'dir':>7} {'lev':>7} {'dte':>7} {'pt':>7}")
    print(f"  {'-'*60}")

    for regime, actions in regime_actions.items():
        if not actions:
            print(f"  {regime:<12} {'0':>5}  (no samples)")
            continue
        arr = np.array(actions)
        means = arr.mean(axis=0)
        print(f"  {regime:<12} {len(actions):>5}  "
              f"{means[0]:>7.3f} {means[1]:>7.3f} {means[2]:>7.3f} "
              f"{means[3]:>7.1f} {means[4]:>7.3f}")

    print()
    if regime_actions["high_iv"] and regime_actions["low_iv"]:
        hi = np.mean([a[0] for a in regime_actions["high_iv"]])
        lo = np.mean([a[0] for a in regime_actions["low_iv"]])
        diff = hi - lo
        print(f"  Delta diff (high_iv - low_iv): {diff:+.4f} {'PASS' if diff > 0.02 else 'FAIL (should be positive)'}")

    if regime_actions["uptrend"] and regime_actions["downtrend"]:
        up = np.mean([a[1] for a in regime_actions["uptrend"]])
        down = np.mean([a[1] for a in regime_actions["downtrend"]])
        diff = up - down
        print(f"  Direction diff (uptrend - downtrend): {diff:+.4f} {'PASS' if diff > 0.1 else 'FAIL (should be positive)'}")

    if regime_actions["high_vol"] and regime_actions["low_vol"]:
        hv = np.mean([a[0] for a in regime_actions["high_vol"]])
        lv = np.mean([a[0] for a in regime_actions["low_vol"]])
        diff = hv - lv
        print(f"  Delta diff (high_vol - low_vol): {diff:+.4f} {'PASS' if diff < -0.02 else 'FAIL (should be negative / go to cash)'}")


def main():
    parser = argparse.ArgumentParser(description="ATLAS Curriculum Training")
    parser.add_argument("--quick-test", action="store_true", help="5 iters/stage, verify regime differentiation")
    parser.add_argument("--from-bc", type=str, default=None, help="Path to BC checkpoint to start from")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, mps, cpu, cuda")
    args = parser.parse_args()

    if args.quick_test:
        quick_test(device=args.device)
    else:
        full_train(from_bc=args.from_bc, device=args.device)


if __name__ == "__main__":
    main()
