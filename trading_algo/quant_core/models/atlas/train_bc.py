from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.data_pipeline import ATLASDataset


def _get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def _warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step / max(warmup_steps, 1))


def train_behavioral_cloning(
    model: nn.Module,
    train_dataset: ATLASDataset,
    val_dataset: ATLASDataset,
    config: ATLASConfig,
    checkpoint_dir: str = "checkpoints/atlas",
    device: str = "auto",
) -> dict:
    """Phase 1: Behavioral Cloning.

    Returns dict with training history (losses, best epoch, etc.)
    """
    dev = _get_device(device)
    model = model.to(dev).float()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=2, eta_min=1e-6
    )

    loss_fn = nn.MSELoss()
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    warmup_steps = 500
    global_step = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(config.bc_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Warmup LR
            if global_step < warmup_steps:
                lr = _warmup_lr(global_step, warmup_steps, config.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            features = batch["features"].to(dev, dtype=torch.float32)
            action_label = batch["action_label"].to(dev, dtype=torch.float32)

            pred = model(features)
            loss = loss_fn(pred, action_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            if global_step >= warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(dev, dtype=torch.float32)
                action_label = batch["action_label"].to(dev, dtype=torch.float32)
                pred = model(features)
                loss = loss_fn(pred, action_label)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        # Checkpoint best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_loss": best_val_loss,
                },
                ckpt_path / "best_bc.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.bc_patience:
            break

    return {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
    }
