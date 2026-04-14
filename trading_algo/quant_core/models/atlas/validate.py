"""Validation suite for ATLAS — all verification gates."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from trading_algo.quant_core.models.atlas.config import ATLASConfig


def v1_param_count(model: nn.Module, target: int = 766_000, tolerance: float = 0.15) -> dict:
    """V1: Parameter count within tolerance."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = abs(n - target) / target
    return {"test": "V1_param_count", "passed": pct <= tolerance, "params": n, "target": target, "deviation_pct": round(pct * 100, 1)}


def v2_no_nan(model: nn.Module, config: ATLASConfig, n_trials: int = 1000, device: str = "cpu") -> dict:
    """V2: No NaN/Inf in random forward passes."""
    model = model.to(device).eval()
    nan_count = 0
    B, L = 1, config.context_len

    with torch.no_grad():
        for _ in range(n_trials):
            f = torch.randn(B, L, config.n_features, device=device)
            ts = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0)
            dow = torch.randint(0, 5, (B, L), device=device)
            mo = torch.randint(0, 12, (B, L), device=device)
            op = torch.zeros(B, L, device=device)
            qt = torch.zeros(B, L, device=device)
            mu = torch.randn(B, L, config.n_features, device=device)
            si = torch.abs(torch.randn(B, L, config.n_features, device=device)) + 0.01
            rtg = torch.randn(B, device=device)
            out = model(f, ts, dow, mo, op, qt, mu, si, rtg)
            if torch.isnan(out).any() or torch.isinf(out).any():
                nan_count += 1

    return {"test": "V2_no_nan", "passed": nan_count == 0, "nan_count": nan_count, "trials": n_trials}


def v3_gradient_health(model: nn.Module, config: ATLASConfig, device: str = "cpu") -> dict:
    """V3: Gradient flow — all trainable params get non-zero gradients."""
    model = model.to(device).train()
    B, L = 2, config.context_len

    f = torch.randn(B, L, config.n_features, device=device)
    ts = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
    dow = torch.randint(0, 5, (B, L), device=device)
    mo = torch.randint(0, 12, (B, L), device=device)
    op = torch.zeros(B, L, device=device)
    qt = torch.zeros(B, L, device=device)
    mu = torch.randn(B, L, config.n_features, device=device)
    si = torch.abs(torch.randn(B, L, config.n_features, device=device)) + 0.01
    rtg = torch.randn(B, device=device)

    model.zero_grad()
    out = model(f, ts, dow, mo, op, qt, mu, si, rtg)
    out.sum().backward()

    total = 0
    with_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                with_grad += 1

    pct = with_grad / total if total > 0 else 0
    return {"test": "V3_gradient_health", "passed": pct > 0.90, "with_grad": with_grad, "total": total, "pct": round(pct * 100, 1)}


def v4_action_bounds(model: nn.Module, config: ATLASConfig, n_trials: int = 10_000, device: str = "cpu") -> dict:
    """V4: All action outputs within specified bounds."""
    model = model.to(device).eval()
    B, L = 1, config.context_len
    violations = 0

    with torch.no_grad():
        for _ in range(n_trials):
            f = torch.randn(B, L, config.n_features, device=device)
            ts = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0)
            dow = torch.randint(0, 5, (B, L), device=device)
            mo = torch.randint(0, 12, (B, L), device=device)
            op = torch.zeros(B, L, device=device)
            qt = torch.zeros(B, L, device=device)
            mu = torch.randn(B, L, config.n_features, device=device)
            si = torch.abs(torch.randn(B, L, config.n_features, device=device)) + 0.01
            rtg = torch.randn(B, device=device)
            a = model(f, ts, dow, mo, op, qt, mu, si, rtg).squeeze(0)

            if a[0] < -0.01 or a[0] > 0.51:
                violations += 1
            if a[1] < -1.01 or a[1] > 1.01:
                violations += 1
            if a[2] < -0.01 or a[2] > 1.01:
                violations += 1
            if a[3] < 13.5 or a[3] > 90.5:
                violations += 1
            if a[4] < -0.01 or a[4] > 1.01:
                violations += 1

    return {"test": "V4_action_bounds", "passed": violations == 0, "violations": violations, "trials": n_trials}


def run_validation_suite(model: nn.Module, config: ATLASConfig | None = None, device: str = "cpu") -> dict:
    """Run all verification gates. Returns dict with results per gate."""
    config = config or ATLASConfig()
    results = {}

    print("Running ATLAS Validation Suite...")

    print("  V1: Parameter count...", end=" ")
    r = v1_param_count(model)
    results["V1"] = r
    print(f"{'PASS' if r['passed'] else 'FAIL'} ({r['params']:,} params)")

    print("  V2: NaN check (1000 trials)...", end=" ")
    r = v2_no_nan(model, config, n_trials=1000, device=device)
    results["V2"] = r
    print(f"{'PASS' if r['passed'] else 'FAIL'} ({r['nan_count']} NaN)")

    print("  V3: Gradient health...", end=" ")
    r = v3_gradient_health(model, config, device=device)
    results["V3"] = r
    print(f"{'PASS' if r['passed'] else 'FAIL'} ({r['pct']}% receive gradients)")

    print("  V4: Action bounds (10K trials)...", end=" ")
    r = v4_action_bounds(model, config, n_trials=10_000, device=device)
    results["V4"] = r
    print(f"{'PASS' if r['passed'] else 'FAIL'} ({r['violations']} violations)")

    n_pass = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    print(f"\n  Result: {n_pass}/{n_total} gates passed")

    results["summary"] = {"passed": n_pass, "total": n_total, "all_passed": n_pass == n_total}
    return results
