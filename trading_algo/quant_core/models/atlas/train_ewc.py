"""Phase 3: Online Elastic Weight Consolidation for ATLAS."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from trading_algo.quant_core.models.atlas.config import ATLASConfig


class EWCAdapter:
    """Manages Fisher Information and EWC fine-tuning for continuous adaptation."""

    def __init__(self, model: nn.Module, config: ATLASConfig):
        self.config = config
        self.fisher_diag: dict[str, Tensor] = {}
        self.reference_params: dict[str, Tensor] = {}
        self.lambda_ewc: float = config.ewc_lambda_init
        self._days_elapsed: int = 0

    def compute_fisher(self, model: nn.Module, data_loader: DataLoader, device: str = "cpu") -> None:
        """Compute diagonal Fisher Information from recent data."""
        model.eval()
        self.fisher_diag = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}

        n_samples = 0
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, Tensor)}
            model.zero_grad()
            actions = model(
                inputs["features"], inputs["timestamps"], inputs["day_of_week"],
                inputs["month"], inputs["is_opex"], inputs["is_quarter_end"],
                inputs["pre_norm_mu"], inputs["pre_norm_sigma"], inputs["return_to_go"],
            )
            # Use sum of squared actions as pseudo-log-likelihood
            pseudo_ll = actions.pow(2).sum()
            pseudo_ll.backward()

            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher_diag[name] += p.grad.pow(2).detach()
            n_samples += 1

        for name in self.fisher_diag:
            self.fisher_diag[name] /= max(n_samples, 1)

        # Store reference parameters
        self.reference_params = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    def ewc_loss(self, model: nn.Module) -> Tensor:
        """Compute EWC penalty: (lambda/2) * sum(F_i * (theta_i - theta*_i)^2)."""
        if not self.fisher_diag:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self.fisher_diag and p.requires_grad:
                fisher = self.fisher_diag[name].to(p.device)
                ref = self.reference_params[name].to(p.device)
                loss = loss + (fisher * (p - ref).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss

    def adapt(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        n_steps: int = 100,
    ) -> dict:
        """Fine-tune model with EWC regularization on recent data."""
        self.compute_fisher(model, data_loader, device)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        losses = []

        step = 0
        while step < n_steps:
            for batch in data_loader:
                if step >= n_steps:
                    break
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, Tensor)}
                actions = model(
                    inputs["features"], inputs["timestamps"], inputs["day_of_week"],
                    inputs["month"], inputs["is_opex"], inputs["is_quarter_end"],
                    inputs["pre_norm_mu"], inputs["pre_norm_sigma"], inputs["return_to_go"],
                )
                task_loss = nn.functional.mse_loss(actions, inputs["action_label"])
                ewc_pen = self.ewc_loss(model)
                total_loss = task_loss + ewc_pen

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

                losses.append(total_loss.item())
                step += 1

        return {"losses": losses, "lambda_ewc": self.lambda_ewc, "steps": step}

    def decay_lambda(self, days_elapsed: int) -> None:
        """Decay lambda: lambda = init * exp(-days / decay_rate)."""
        self._days_elapsed += days_elapsed
        self.lambda_ewc = self.config.ewc_lambda_init * math.exp(
            -self._days_elapsed / self.config.ewc_decay_days
        )

    def save(self, path: str) -> None:
        torch.save({
            "fisher_diag": self.fisher_diag,
            "reference_params": self.reference_params,
            "lambda_ewc": self.lambda_ewc,
            "days_elapsed": self._days_elapsed,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        self.fisher_diag = data["fisher_diag"]
        self.reference_params = data["reference_params"]
        self.lambda_ewc = data["lambda_ewc"]
        self._days_elapsed = data["days_elapsed"]
