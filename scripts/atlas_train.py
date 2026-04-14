"""
ATLAS Training Pipeline — Full end-to-end training script.

Usage:
    .venv/bin/python scripts/atlas_train.py --phase all
    .venv/bin/python scripts/atlas_train.py --phase features
    .venv/bin/python scripts/atlas_train.py --phase bc
    .venv/bin/python scripts/atlas_train.py --phase ppo
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.features import ATLASFeatureComputer, RollingNormalizer
from trading_algo.quant_core.models.atlas.hindsight_actions import compute_hindsight_actions
from trading_algo.quant_core.strategies.options.iv_rank import iv_series_from_prices, iv_rank as compute_iv_rank


CACHE_DIR = "data/atlas_cache"
FEATURE_DIR = "data/atlas_features"
CHECKPOINT_DIR = "checkpoints/atlas"


def phase_features():
    """Step 2: Compute features + hindsight actions for all cached symbols."""
    os.makedirs(FEATURE_DIR, exist_ok=True)
    config = ATLASConfig()
    fc = ATLASFeatureComputer()
    norm = RollingNormalizer()

    symbols = sorted([f.replace(".parquet", "") for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")])
    print(f"Computing features for {len(symbols)} symbols...")
    total_windows = 0
    start = time.time()

    for i, sym in enumerate(symbols):
        feat_path = f"{FEATURE_DIR}/{sym}_features.npz"
        if os.path.exists(feat_path):
            data = np.load(feat_path)
            total_windows += max(0, len(data["normed"]) - config.context_len + 1)
            print(f"  [{i+1}/{len(symbols)}] {sym}: CACHED")
            continue

        try:
            df = pd.read_parquet(f"{CACHE_DIR}/{sym}.parquet")
            closes = df["Close"].values.astype(np.float64)
            highs = df["High"].values.astype(np.float64)
            lows = df["Low"].values.astype(np.float64)
            volumes = df["Volume"].values.astype(np.float64)
            timestamps = np.array([ts.timestamp() for ts in df.index])

            raw = fc.compute_features(closes, highs, lows, volumes)
            full = np.concatenate([raw, np.zeros((len(raw), 4))], axis=1)
            normed, mu_arr, sigma_arr = norm.normalize(full)

            iv_series = iv_series_from_prices(closes, rv_window=30, dynamic=True)
            iv_ranks = np.array([compute_iv_rank(iv_series, t, 252) for t in range(len(closes))])

            # Hindsight actions — run single-threaded to avoid multiprocessing issues
            optimal = compute_hindsight_actions(closes, iv_series, iv_ranks, config)

            rtg = np.zeros(len(closes), dtype=np.float32)
            for t in range(len(closes) - 45):
                fr = np.diff(np.log(closes[t : t + 46]))
                if len(fr) > 0 and np.std(fr) > 1e-8:
                    rtg[t] = float(np.mean(fr) / np.std(fr) * np.sqrt(252))

            dow = np.array([ts.weekday() for ts in df.index], dtype=np.int32)
            month = np.array([ts.month - 1 for ts in df.index], dtype=np.int32)

            np.savez_compressed(
                feat_path,
                normed=normed.astype(np.float32),
                mu=mu_arr.astype(np.float32),
                sigma=sigma_arr.astype(np.float32),
                timestamps=timestamps,
                actions=optimal.astype(np.float32),
                rtg=rtg,
                dow=dow,
                month=month,
            )

            n_w = max(0, normed.shape[0] - config.context_len + 1)
            total_windows += n_w
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(symbols) - i - 1)
            print(f"  [{i+1}/{len(symbols)}] {sym}: {normed.shape[0]} days, {n_w} windows ({elapsed:.0f}s, ETA ~{eta:.0f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym}: ERROR - {e}")

    print(f"\nDone: {total_windows:,} windows in {time.time() - start:.0f}s")
    return total_windows


def phase_bc():
    """Step 3: Behavioral Cloning training."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = ATLASConfig()
    L = config.context_len

    # Build dataset from precomputed features
    print("Loading feature files...")
    all_features, all_mu, all_sigma, all_ts, all_actions, all_rtg, all_dow, all_month = [], [], [], [], [], [], [], []

    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")])
    for f in feature_files:
        data = np.load(f"{FEATURE_DIR}/{f}")
        normed = data["normed"]
        if len(normed) < L + 252:
            continue
        # Create windows starting from index 252 (warmup) to end
        start_idx = 252
        for t in range(start_idx, len(normed) - L + 1):
            window = normed[t : t + L]
            if np.isnan(window).any():
                continue
            all_features.append(window)
            all_mu.append(data["mu"][t : t + L])
            all_sigma.append(data["sigma"][t : t + L])
            all_ts.append(data["timestamps"][t : t + L])
            all_actions.append(data["actions"][t + L - 1])  # action for last day
            all_rtg.append(data["rtg"][t + L - 1])
            all_dow.append(data["dow"][t : t + L])
            all_month.append(data["month"][t : t + L])

    N = len(all_features)
    print(f"Total training windows: {N:,}")

    if N == 0:
        print("ERROR: No valid windows. Run --phase features first.")
        return

    # Convert to tensors
    features_t = torch.tensor(np.array(all_features), dtype=torch.float32)
    mu_t = torch.tensor(np.array(all_mu), dtype=torch.float32)
    sigma_t = torch.tensor(np.array(all_sigma), dtype=torch.float32)
    ts_t = torch.tensor(np.array(all_ts), dtype=torch.float32)
    actions_t = torch.tensor(np.array(all_actions), dtype=torch.float32)
    rtg_t = torch.tensor(np.array(all_rtg), dtype=torch.float32)
    dow_t = torch.tensor(np.array(all_dow), dtype=torch.long)
    month_t = torch.tensor(np.array(all_month), dtype=torch.long)

    # Train/val split by index (last 15% = val)
    split = int(N * 0.85)
    train_idx = list(range(split))
    val_idx = list(range(split, N))
    print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}")

    # Model
    from trading_algo.quant_core.models.atlas.model import ATLASModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = ATLASModel(config).to(device).float()
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    batch_size = config.batch_size
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.bc_epochs):
        model.train()
        np.random.shuffle(train_idx)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(train_idx))
            idx = train_idx[batch_start:batch_end]

            bf = features_t[idx].to(device)
            bts = ts_t[idx].to(device)
            bdow = dow_t[idx].to(device)
            bmo = month_t[idx].to(device)
            bop = torch.zeros(len(idx), L, device=device)
            bqt = torch.zeros(len(idx), L, device=device)
            bmu = mu_t[idx].to(device)
            bsi = sigma_t[idx].to(device)
            brtg = rtg_t[idx].to(device)
            ba = actions_t[idx].to(device)

            pred = model(bf, bts, bdow, bmo, bop, bqt, bmu, bsi, brtg)
            loss = torch.nn.functional.mse_loss(pred, ba)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_start in range(0, len(val_idx), batch_size):
                batch_end = min(batch_start + batch_size, len(val_idx))
                idx = val_idx[batch_start:batch_end]
                bf = features_t[idx].to(device)
                bts = ts_t[idx].to(device)
                bdow = dow_t[idx].to(device)
                bmo = month_t[idx].to(device)
                bop = torch.zeros(len(idx), L, device=device)
                bqt = torch.zeros(len(idx), L, device=device)
                bmu = mu_t[idx].to(device)
                bsi = sigma_t[idx].to(device)
                brtg = rtg_t[idx].to(device)
                ba = actions_t[idx].to(device)
                pred = model(bf, bts, bdow, bmo, bop, bqt, bmu, bsi, brtg)
                val_loss += torch.nn.functional.mse_loss(pred, ba).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        print(f"  Epoch {epoch+1}/{config.bc_epochs}: train_loss={avg_train:.6f} val_loss={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "config": config,
            }, f"{CHECKPOINT_DIR}/atlas_bc_best.pt")
            print(f"    Saved best model (val_loss={avg_val:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.bc_patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    print(f"\nBC training complete. Best val loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="ATLAS Training Pipeline")
    parser.add_argument("--phase", choices=["features", "bc", "ppo", "all"], default="all")
    args = parser.parse_args()

    if args.phase in ("features", "all"):
        print("=" * 60)
        print("  PHASE: Feature Computation + Hindsight Actions")
        print("=" * 60)
        phase_features()

    if args.phase in ("bc", "all"):
        print("\n" + "=" * 60)
        print("  PHASE: Behavioral Cloning Training")
        print("=" * 60)
        phase_bc()

    if args.phase in ("ppo", "all"):
        print("\n" + "=" * 60)
        print("  PHASE: PPO RL Fine-Tuning")
        print("  (Run separately — takes ~8 hours)")
        print("=" * 60)


if __name__ == "__main__":
    main()
