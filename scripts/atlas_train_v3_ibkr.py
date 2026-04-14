"""
ATLAS v3 Training — Using IBKR data with real historical IV.

Key improvements over v2:
  - 445+ symbols (vs 50 from yfinance)
  - Real historical IV from IBKR (vs synthetic BSM estimates)
  - 915K+ training windows (vs 108K)
  - 1.2:1+ data:param ratio (vs 0.14:1)

Usage:
    .venv/bin/python scripts/atlas_train_v3_ibkr.py --phase features
    .venv/bin/python scripts/atlas_train_v3_ibkr.py --phase bc
    .venv/bin/python scripts/atlas_train_v3_ibkr.py --phase curriculum
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
from trading_algo.quant_core.strategies.options.iv_rank import iv_rank as compute_iv_rank

IBKR_DIR = "data/atlas_ibkr"
FEATURE_DIR = "data/atlas_features_v3"
CHECKPOINT_DIR = "checkpoints/atlas_v3"


def phase_features():
    """Compute features using REAL IBKR IV data (not synthetic estimates)."""
    os.makedirs(FEATURE_DIR, exist_ok=True)
    config = ATLASConfig()
    fc = ATLASFeatureComputer()
    norm = RollingNormalizer()

    # Find all symbols with both price and IV data
    price_files = {f.replace(".parquet", ""): f for f in os.listdir(IBKR_DIR)
                   if f.endswith(".parquet") and "_iv" not in f and f != "manifest.json"}
    iv_files = {f.replace("_iv.parquet", ""): f for f in os.listdir(IBKR_DIR)
                if f.endswith("_iv.parquet")}

    symbols_with_iv = sorted(set(price_files.keys()) & set(iv_files.keys()))
    symbols_no_iv = sorted(set(price_files.keys()) - set(iv_files.keys()))
    print(f"Symbols with price+IV: {len(symbols_with_iv)}")
    print(f"Symbols with price only: {len(symbols_no_iv)}")
    print(f"Processing all {len(price_files)} symbols...")

    total_windows = 0
    start = time.time()

    for i, sym in enumerate(sorted(price_files.keys())):
        feat_path = f"{FEATURE_DIR}/{sym}_features.npz"
        if os.path.exists(feat_path):
            data = np.load(feat_path)
            total_windows += max(0, len(data["normed"]) - config.context_len + 1 - 252)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(price_files)}] {sym}: CACHED", flush=True)
            continue

        try:
            # Load IBKR price data
            df = pd.read_parquet(f"{IBKR_DIR}/{sym}.parquet")
            if len(df) < 500:
                continue

            closes = df["close"].values.astype(np.float64)
            highs = df["high"].values.astype(np.float64)
            lows = df["low"].values.astype(np.float64)
            volumes = df["volume"].values.astype(np.float64)
            timestamps = np.array([ts.timestamp() for ts in df.index])
            dows = np.array([ts.weekday() for ts in df.index], dtype=np.int32)
            months = np.array([ts.month - 1 for ts in df.index], dtype=np.int32)

            # Compute 12 market features
            raw = fc.compute_features(closes, highs, lows, volumes)
            full = np.concatenate([raw, np.zeros((len(raw), 4))], axis=1)
            normed, mu_arr, sigma_arr = norm.normalize(full)

            # Use REAL IV from IBKR if available, else fall back to synthetic
            if sym in iv_files:
                iv_df = pd.read_parquet(f"{IBKR_DIR}/{sym}_iv.parquet")
                # Align IV data with price data by date
                real_iv = np.full(len(closes), np.nan)
                price_dates = {ts.date(): idx for idx, ts in enumerate(df.index)}
                for ts, row in iv_df.iterrows():
                    d = ts.date() if hasattr(ts, 'date') else ts
                    if d in price_dates:
                        real_iv[price_dates[d]] = row["iv"]
                # Forward-fill NaN
                for j in range(1, len(real_iv)):
                    if np.isnan(real_iv[j]) and not np.isnan(real_iv[j-1]):
                        real_iv[j] = real_iv[j-1]
                # Fill remaining with synthetic
                from trading_algo.quant_core.strategies.options.iv_rank import iv_series_from_prices
                synthetic_iv = iv_series_from_prices(closes, rv_window=30, dynamic=True)
                iv_series = np.where(np.isnan(real_iv), synthetic_iv, real_iv)
            else:
                from trading_algo.quant_core.strategies.options.iv_rank import iv_series_from_prices
                iv_series = iv_series_from_prices(closes, rv_window=30, dynamic=True)

            iv_series = np.nan_to_num(iv_series, nan=0.25)
            iv_ranks = np.array([compute_iv_rank(iv_series, t, 252) for t in range(len(closes))])
            iv_ranks = np.nan_to_num(iv_ranks, nan=50.0)

            # Hindsight-optimal actions using BSM with real IV
            optimal = compute_hindsight_actions(closes, iv_series, iv_ranks, config)
            optimal = np.nan_to_num(optimal, nan=0.0)

            # Return-to-go
            rtg = np.zeros(len(closes), dtype=np.float32)
            for t in range(len(closes) - 45):
                fr = np.diff(np.log(closes[t:t + 46]))
                if len(fr) > 0 and np.std(fr) > 1e-8:
                    rtg[t] = float(np.mean(fr) / np.std(fr) * np.sqrt(252))

            np.savez_compressed(
                feat_path,
                normed=normed.astype(np.float32), mu=mu_arr.astype(np.float32),
                sigma=sigma_arr.astype(np.float32), timestamps=timestamps,
                actions=optimal.astype(np.float32), rtg=rtg,
                dow=dows, month=months,
                closes=closes.astype(np.float32), ivs=iv_series.astype(np.float32),
                iv_ranks=iv_ranks.astype(np.float32),
            )

            n_w = max(0, normed.shape[0] - config.context_len + 1 - 252)
            total_windows += n_w
            elapsed = time.time() - start
            per = elapsed / (i + 1)
            eta = per * (len(price_files) - i - 1)
            print(f"  [{i+1}/{len(price_files)}] {sym}: {normed.shape[0]} days, {n_w} windows ({elapsed:.0f}s, ETA ~{eta/60:.0f}min)", flush=True)

        except Exception as e:
            print(f"  [{i+1}/{len(price_files)}] {sym}: ERROR — {e}", flush=True)

    print(f"\nDone: {total_windows:,} windows in {(time.time()-start)/60:.1f} min")


def phase_bc(resume: bool = False):
    """BC training on IBKR data."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = ATLASConfig()
    L = config.context_len

    # Two-pass memmap loader: avoids holding ~17GB in RAM.
    # Pass 1 counts valid windows; Pass 2 fills pre-allocated disk-backed arrays.
    # Training indexes into memmaps batch-by-batch via torch.from_numpy() — zero-copy.
    # Disk cost: ~17GB in data/atlas_memmap/. Re-used on subsequent runs.
    MEMMAP_DIR = "data/atlas_memmap"
    os.makedirs(MEMMAP_DIR, exist_ok=True)
    F = config.n_features  # 16

    feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith("_features.npz")])
    meta_path = f"{MEMMAP_DIR}/meta.npy"

    if os.path.exists(meta_path):
        N = int(np.load(meta_path))
        print(f"Found existing memmaps: {N:,} windows. Skipping generation.", flush=True)
    else:
        # --- Pass 1: count valid windows (no allocation) ---
        print("Pass 1: counting valid windows...", flush=True)
        N = 0
        for fname in feature_files:
            data = np.load(f"{FEATURE_DIR}/{fname}")
            normed = data["normed"]
            actions = data["actions"]
            if len(normed) < L + 252:
                continue
            for t in range(252, len(normed) - L + 1, 1):
                if np.isnan(normed[t:t + L]).any():
                    continue
                if np.isnan(actions[t + L - 1]).any():
                    continue
                N += 1
        print(f"Total valid windows: {N:,}", flush=True)
        if N == 0:
            print("ERROR: No windows.")
            return

        # --- Allocate memmaps on disk (~17GB total) ---
        print("Allocating memmaps on disk...", flush=True)
        mm_f   = np.memmap(f"{MEMMAP_DIR}/features.mmap",     dtype='float32', mode='w+', shape=(N, L, F))
        mm_mu  = np.memmap(f"{MEMMAP_DIR}/mu.mmap",           dtype='float32', mode='w+', shape=(N, L, F))
        mm_si  = np.memmap(f"{MEMMAP_DIR}/sigma.mmap",        dtype='float32', mode='w+', shape=(N, L, F))
        mm_ts  = np.memmap(f"{MEMMAP_DIR}/timestamps.mmap",   dtype='float32', mode='w+', shape=(N, L))
        mm_act = np.memmap(f"{MEMMAP_DIR}/actions.mmap",      dtype='float32', mode='w+', shape=(N, 5))
        mm_rtg = np.memmap(f"{MEMMAP_DIR}/rtg.mmap",          dtype='float32', mode='w+', shape=(N,))
        mm_dow = np.memmap(f"{MEMMAP_DIR}/dow.mmap",          dtype='int64',   mode='w+', shape=(N, L))
        mm_mo  = np.memmap(f"{MEMMAP_DIR}/month.mmap",        dtype='int64',   mode='w+', shape=(N, L))

        # --- Pass 2: fill memmaps symbol by symbol (low peak RAM) ---
        print("Pass 2: filling memmaps...", flush=True)
        wi = 0
        for fname in feature_files:
            data = np.load(f"{FEATURE_DIR}/{fname}")
            normed = data["normed"]
            actions = data["actions"]
            if len(normed) < L + 252:
                continue
            for t in range(252, len(normed) - L + 1, 1):
                window = normed[t:t + L]
                if np.isnan(window).any():
                    continue
                act = actions[t + L - 1]
                if np.isnan(act).any():
                    continue
                mm_f[wi]   = window
                mm_mu[wi]  = data["mu"][t:t + L]
                mm_si[wi]  = data["sigma"][t:t + L]
                mm_ts[wi]  = data["timestamps"][t:t + L].astype(np.float32)
                mm_act[wi] = act
                mm_rtg[wi] = data["rtg"][t + L - 1]
                mm_dow[wi] = data["dow"][t:t + L].astype(np.int64)
                mm_mo[wi]  = data["month"][t:t + L].astype(np.int64)
                wi += 1
            for mm in (mm_f, mm_mu, mm_si, mm_ts, mm_act, mm_rtg, mm_dow, mm_mo):
                mm.flush()

        N = wi
        print(f"Training windows: {N:,}", flush=True)

        # Normalize actions into a separate memmap (actions array is ~18MB — fine to load)
        raw_act = np.array(mm_act[:N])
        norm_act = raw_act.copy()
        norm_act[:, 0] = raw_act[:, 0] / 0.50
        norm_act[:, 1] = (raw_act[:, 1] + 1.0) / 2.0
        norm_act[:, 3] = (raw_act[:, 3] - 14.0) / 76.0
        mm_nact = np.memmap(f"{MEMMAP_DIR}/norm_actions.mmap", dtype='float32', mode='w+', shape=(N, 5))
        mm_nact[:] = norm_act
        mm_nact.flush()

        np.save(meta_path, np.array(N))

    # Force memmaps into private RAM so batch indexing is fast (no memmap protocol overhead).
    # macOS reclaims the ~14GB of inactive page cache from the old process as we allocate.
    # Steady state: ~17.6GB private + ~9GB OS/apps = ~27GB, fits in 32GB.
    print("Loading data into RAM (~17.6GB, may take a few minutes)...", flush=True)
    features_t = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/features.mmap",     dtype='float32', mode='r', shape=(N, L, F))))
    print("  features (5.3GB) loaded", flush=True)
    mu_t       = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/mu.mmap",           dtype='float32', mode='r', shape=(N, L, F))))
    print("  mu (5.3GB) loaded", flush=True)
    sigma_t    = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/sigma.mmap",        dtype='float32', mode='r', shape=(N, L, F))))
    print("  sigma (5.3GB) loaded", flush=True)
    ts_t       = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/timestamps.mmap",   dtype='float32', mode='r', shape=(N, L))))
    actions_t  = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/norm_actions.mmap", dtype='float32', mode='r', shape=(N, 5))))
    rtg_t      = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/rtg.mmap",          dtype='float32', mode='r', shape=(N,))))
    dow_t      = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/dow.mmap",          dtype='int64',   mode='r', shape=(N, L))))
    month_t    = torch.from_numpy(np.array(np.memmap(f"{MEMMAP_DIR}/month.mmap",        dtype='int64',   mode='r', shape=(N, L))))
    print("All data in RAM — fast batch indexing ready.", flush=True)

    split = int(N * 0.85)
    train_idx = list(range(split))
    val_idx = list(range(split, N))
    print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}", flush=True)

    from trading_algo.quant_core.models.atlas.model import ATLASModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = ATLASModel(config).to(device).float()
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.bc_epochs, eta_min=1e-6)
    batch_size = 256  # 256: best MPS throughput with CausalTransformer backbone (155ms/batch vs 88ms at 128)
    best_val = float("inf")
    patience = 0
    start_epoch = 0

    if resume:
        ckpt_path = f"{CHECKPOINT_DIR}/atlas_v3_bc_best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1  # ckpt["epoch"] is 0-indexed last completed epoch
        best_val = ckpt["val_loss"]
        # Fast-forward cosine scheduler to match where it would be after start_epoch steps
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from checkpoint: start_epoch={start_epoch}, best_val={best_val:.6f}", flush=True)

    def normalize_output(pred):
        n = pred.clone()
        n[:, 0] = pred[:, 0] / 0.50
        n[:, 1] = (pred[:, 1] + 1.0) / 2.0
        n[:, 3] = (pred[:, 3] - 14.0) / 76.0
        return n

    for epoch in range(start_epoch, config.bc_epochs):
        model.train()
        np.random.shuffle(train_idx)
        epoch_loss, n_b = 0.0, 0

        for bs in range(0, len(train_idx), batch_size):
            be = min(bs + batch_size, len(train_idx))
            idx = train_idx[bs:be]
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
            loss = torch.nn.functional.mse_loss(normalize_output(pred), ba)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_b, 1)

        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for bs in range(0, len(val_idx), batch_size):
                be = min(bs + batch_size, len(val_idx))
                idx = val_idx[bs:be]
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
                val_loss += torch.nn.functional.mse_loss(normalize_output(pred), ba).item()
                n_v += 1

        avg_val = val_loss / max(n_v, 1)
        print(f"  Epoch {epoch+1}/{config.bc_epochs}: train={avg_train:.6f} val={avg_val:.6f}", flush=True)

        if avg_val < best_val:
            best_val = avg_val
            patience = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": avg_val, "config": config,
            }, f"{CHECKPOINT_DIR}/atlas_v3_bc_best.pt")
            print(f"    Saved best (val={avg_val:.6f})", flush=True)
        else:
            patience += 1
            if patience >= config.bc_patience:
                print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

    print(f"\nBC complete. Best val loss: {best_val:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["features", "bc", "curriculum", "all"], default="all")
    parser.add_argument("--resume-bc", action="store_true", help="Resume BC from best checkpoint")
    args = parser.parse_args()

    if args.phase in ("features", "all"):
        print("=" * 60)
        print("  ATLAS v3: IBKR Features (real IV)")
        print("=" * 60)
        phase_features()

    if args.phase in ("bc", "all"):
        print("\n" + "=" * 60)
        print("  ATLAS v3: Behavioral Cloning")
        print("=" * 60)
        phase_bc(resume=args.resume_bc)


if __name__ == "__main__":
    main()
