"""Memory bank construction for ATLAS cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from trading_algo.quant_core.models.atlas.config import ATLASConfig


def build_memory_bank(
    model: nn.Module,
    dataset: "ATLASDataset",
    config: ATLASConfig,
    n_entries: int = 10_000,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """
    Build the historical memory bank by running the backbone on training data.

    Steps:
        1. Run VSN + Mamba on all windows to get temporal embeddings
        2. Pair with action labels and return-to-go
        3. Oversample tail events (top/bottom 10% by return_to_go)
        4. K-means cluster to n_entries

    Returns:
        keys:   (n_entries, d_model) — context embeddings
        values: (n_entries, d_model) — projected (action, reward) pairs
    """
    model.eval()
    d = config.d_model

    embeddings: list[Tensor] = []
    rtg_vals: list[float] = []
    action_labels: list[Tensor] = []

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    value_proj = nn.Linear(config.action_dim + 1, d).to(device)
    nn.init.orthogonal_(value_proj.weight)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            timestamps = batch["timestamps"].to(device)
            dow = batch["day_of_week"].to(device)
            month = batch["month"].to(device)
            is_opex = batch["is_opex"].to(device)
            is_qtr = batch["is_quarter_end"].to(device)

            # Run through VSN + Mamba only (backbone)
            time_enc = model.time2vec(timestamps)
            cal_enc = model.calendar_embed(dow, month, is_opex, is_qtr)
            token = torch.cat([features, time_enc, cal_enc], dim=-1)
            selected, _ = model.vsn(token)
            temporal = model.mamba(selected)
            h_last = temporal[:, -1, :]  # (B, d_model)

            embeddings.append(h_last.cpu())
            rtg_vals.extend(batch["return_to_go"].squeeze(-1).tolist())
            action_labels.append(batch["action_label"].cpu())

    all_embeddings = torch.cat(embeddings, dim=0)  # (N, d)
    all_actions = torch.cat(action_labels, dim=0)  # (N, 5)
    all_rtg = torch.tensor(rtg_vals, dtype=torch.float32)  # (N,)
    N = all_embeddings.shape[0]

    # Oversample tail events
    rtg_np = all_rtg.numpy()
    p10 = np.percentile(rtg_np, 10)
    p90 = np.percentile(rtg_np, 90)
    tail_mask = (rtg_np <= p10) | (rtg_np >= p90)
    tail_indices = np.where(tail_mask)[0]

    if len(tail_indices) > 0:
        oversample_indices = np.repeat(tail_indices, 4)  # 5x total for tails
        all_indices = np.concatenate([np.arange(N), oversample_indices])
        all_embeddings = all_embeddings[all_indices]
        all_actions = all_actions[all_indices]
        all_rtg = all_rtg[all_indices]

    # Project (action, rtg) → value embedding
    action_rtg = torch.cat([all_actions, all_rtg.unsqueeze(-1)], dim=-1)  # (N', 6)
    with torch.no_grad():
        all_values = value_proj(action_rtg.to(device)).cpu()  # (N', d)

    # K-means clustering to n_entries
    from sklearn.cluster import MiniBatchKMeans

    n_samples = all_embeddings.shape[0]
    n_clusters = min(n_entries, n_samples)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(1024, n_samples),
        n_init=3,
        max_iter=100,
        random_state=42,
    )
    labels = kmeans.fit_predict(all_embeddings.numpy())

    keys = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # (n_clusters, d)

    # For values: average the projected action embeddings per cluster
    values = torch.zeros(n_clusters, d, dtype=torch.float32)
    counts = torch.zeros(n_clusters, dtype=torch.float32)
    for i in range(n_samples):
        values[labels[i]] += all_values[i]
        counts[labels[i]] += 1
    counts = counts.clamp(min=1)
    values = values / counts.unsqueeze(-1)

    # Pad if n_clusters < n_entries
    if n_clusters < n_entries:
        pad_k = keys.mean(0, keepdim=True).expand(n_entries - n_clusters, -1)
        pad_v = values.mean(0, keepdim=True).expand(n_entries - n_clusters, -1)
        keys = torch.cat([keys, pad_k], dim=0)
        values = torch.cat([values, pad_v], dim=0)

    return keys, values
