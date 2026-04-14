#!/usr/bin/env python3
"""
ATLAS v3 Curriculum Model — RL Environment Evaluation

Evaluates the trained model using the OptionsEnvironment directly.
Compares against random policy and cash (no-trade) baselines.

Metrics: episode return%, annualized return, Sharpe ratio,
         max drawdown, win rate — broken down by market regime.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict

from scipy.special import erfinv as _scipy_erfinv

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.model import ATLASModel
import trading_algo.quant_core.models.atlas.train_ppo as _ppo_mod
from trading_algo.quant_core.models.atlas.train_ppo import OptionsEnvironment, load_training_data_v2
from trading_algo.quant_core.strategies.options.wheel import _round_strike


# ---------------------------------------------------------------------------
# Fast inline BSM — replaces scipy.stats.norm.cdf (~50µs) with math.erfc (~300ns)
# and replaces scipy.brentq strike search with direct analytical inversion.
# Monkey-patched into train_ppo module namespace so OptionsEnvironment picks them up.
# ---------------------------------------------------------------------------

_ISQRT2 = 1.0 / math.sqrt(2.0)
_SQRT2  = math.sqrt(2.0)
_R      = 0.045   # risk-free rate (matches train_ppo._RISK_FREE_RATE)
_SKEW   = 0.8     # skew slope (matches train_ppo._SKEW_SLOPE)


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x * _ISQRT2)


def _ncdf_inv(p: float) -> float:
    return _SQRT2 * float(_scipy_erfinv(2.0 * p - 1.0))


def _fast_price_option(
    spot: float, strike: float, tte_years: float, vol: float,
    rate: float, option_type: str,
    dividend_yield: float = 0.0,
    skew_adjust: bool = True,
    skew_slope: float = _SKEW,
) -> float:
    if tte_years <= 1e-6:
        return max(strike - spot, 0.0) if option_type == "put" else max(spot - strike, 0.0)
    if skew_adjust and spot > 0:
        if option_type == "put":
            vol *= 1.0 + skew_slope * max(0.0, (spot - strike) / spot)
        else:
            vol *= 1.0 - skew_slope * 0.3 * max(0.0, (strike - spot) / spot)
    sq = vol * math.sqrt(tte_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tte_years) / sq
    d2 = d1 - sq
    er = math.exp(-rate * tte_years)
    if option_type == "put":
        return max(strike * er * _ncdf(-d2) - spot * _ncdf(-d1), 0.0)
    return max(spot * _ncdf(d1) - strike * er * _ncdf(d2), 0.0)


def _fast_strike_by_delta(
    spot: float, target_abs_delta: float, tte_years: float, vol: float,
    rate: float, option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    """Direct analytical strike from delta — no brentq iteration needed.

    Derivation: d1 = (ln(S/K) + (r+0.5σ²)T) / σ√T
    → K = S * exp(-d1*σ√T + (r+0.5σ²)T)

    Put:  |Δ| = N(-d1)  → d1 = -N_inv(|Δ|)
    Call:  Δ  = N(d1)   → d1 =  N_inv(Δ)
    """
    if tte_years <= 1e-6:
        return spot
    sq   = vol * math.sqrt(tte_years)
    drift = (rate + 0.5 * vol * vol) * tte_years
    if option_type == "put":
        d1 = -_ncdf_inv(float(target_abs_delta))
    else:
        d1 =  _ncdf_inv(float(target_abs_delta))
    k_raw = spot * math.exp(-d1 * sq + drift)
    return _round_strike(k_raw, spot)


# Patch train_ppo module so OptionsEnvironment uses fast versions
_ppo_mod._price_option          = _fast_price_option
_ppo_mod._find_strike_by_delta  = _fast_strike_by_delta

import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--checkpoint", default="checkpoints/atlas_v4/atlas_curriculum_final.pt")
_parser.add_argument("--episodes", type=int, default=100)
_args, _ = _parser.parse_known_args()

CHECKPOINT = _args.checkpoint
FEATURE_DIR = "data/atlas_features_v3"
N_EPISODES  = _args.episodes
_INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(device: str) -> ATLASModel:
    config = ATLASConfig()
    model = ATLASModel(config)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).float().eval()
    print(f"Loaded checkpoint: {CHECKPOINT}", flush=True)

    # Pre-project memory bank once — eliminates 2×Linear(10K,64) per step
    with torch.no_grad():
        _K = model.cross_attention.W_k(model.memory_keys).detach()   # (M, d_model)
        _V = model.cross_attention.W_v(model.memory_values).detach()  # (M, d_model)
    _scale = model.cross_attention.d_model ** 0.5
    _W_q = model.cross_attention.W_q

    def _cached_cross_attn(query, _mk, _mv):
        Q = _W_q(query)                              # (B, d_model)
        alpha = torch.matmul(Q, _K.T) / _scale       # (B, M)
        alpha = torch.softmax(alpha, dim=-1)
        output = torch.matmul(alpha, _V)             # (B, d_model)
        return output, alpha

    model.cross_attention.forward = _cached_cross_attn

    # Fuse per-feature GRN loop in VSN into batched bmm — 2× speedup on VSN (5.8ms → 2.8ms)
    import torch.nn.functional as _F
    vsn = model.vsn
    nf, D = vsn.n_features, vsn.d_model
    grns = vsn.feature_grns
    _W1f = torch.stack([g.w1.weight.detach().squeeze(-1) for g in grns])       # (nf, D)
    _B1  = torch.stack([g.w1.bias.detach() for g in grns])
    _W2  = torch.stack([g.w2.weight.detach() for g in grns])                   # (nf, D, D)
    _B2  = torch.stack([g.w2.bias.detach() for g in grns])
    _W4  = torch.stack([g.w4.weight.detach() for g in grns])
    _B4  = torch.stack([g.w4.bias.detach() for g in grns])
    _W5  = torch.stack([g.w5.weight.detach() for g in grns])
    _B5  = torch.stack([g.w5.bias.detach() for g in grns])
    _W6  = torch.stack([g.w6.weight.detach() for g in grns])
    _B6  = torch.stack([g.w6.bias.detach() for g in grns])
    _Wsk = torch.stack([g.skip_proj.weight.detach().squeeze(-1) for g in grns])  # (nf, D)
    _Bsk = torch.stack([g.skip_proj.bias.detach() for g in grns])
    _LNg = torch.stack([g.layer_norm.weight.detach() for g in grns])           # (nf, D)
    _LNb = torch.stack([g.layer_norm.bias.detach() for g in grns])
    _sel_grn = vsn.selection_grn

    def _bmm_linear(h, W, B):
        # h: (B, L, nf, D), W: (nf, D, D), B: (nf, D)
        Bsz, L, nf_, D_ = h.shape
        h2  = h.permute(2, 0, 1, 3).reshape(nf_, Bsz * L, D_)
        out = torch.bmm(h2, W.permute(0, 2, 1)).reshape(nf_, Bsz, L, D_).permute(1, 2, 0, 3)
        return out + B

    def _fused_vsn(x):
        # x: (B, L, nf)
        h    = x.unsqueeze(-1) * _W1f + _B1           # scalar-multiply: (B,L,nf,D)
        h    = _F.elu(_bmm_linear(h, _W2, _B2))
        h    = _bmm_linear(h, _W4, _B4)
        g5   = _bmm_linear(h, _W5, _B5)
        g6   = _bmm_linear(h, _W6, _B6)
        gate = torch.sigmoid(g5) * g6
        skip = x.unsqueeze(-1) * _Wsk + _Bsk
        out  = skip + gate                             # (B, L, nf, D)
        mean = out.mean(-1, keepdim=True)
        var  = out.var(-1, keepdim=True, unbiased=False)
        out  = (out - mean) / (var + 1e-5).sqrt() * _LNg + _LNb
        sel_w = _F.softmax(_sel_grn(x), dim=-1).unsqueeze(-1)  # (B, L, nf, 1)
        return (sel_w * out).sum(dim=-2), sel_w.squeeze(-1)

    model.vsn.forward = _fused_vsn
    print("VSN fused (batched bmm) — 2× speedup on VSN", flush=True)
    return model


# ---------------------------------------------------------------------------
# Episode runner — tracks equity curve, computes trading metrics
# ---------------------------------------------------------------------------

def run_episode(
    env: OptionsEnvironment,
    model: ATLASModel | None,
    device: str,
    greedy: bool = True,
) -> dict:
    """
    Run one episode and return a dict of trading metrics.

    model=None → random policy (uniform action sampling).
    greedy=True → use policy mean (no exploration noise) for model eval.
    """
    obs = env.reset()

    # Capture episode start state for regime classification and BnH baseline
    sym   = env._sym
    t0    = env._start
    t_end = env._end
    closes = env._closes

    equity_curve: list[float] = [_INITIAL_CAPITAL]
    step = 0

    while True:
        if model is not None:
            inputs = {k: v.to(device) for k, v in obs.items()}
            with torch.no_grad():
                action_mean, _ = model.forward_with_value(
                    inputs["features"], inputs["timestamps"], inputs["dow"],
                    inputs["month"], inputs["is_opex"], inputs["is_qtr"],
                    inputs["pre_mu"], inputs["pre_sigma"], inputs["rtg"],
                )
                if greedy:
                    action = action_mean
                else:
                    dist = model.get_action_distribution(action_mean)
                    action = dist.sample()
            action_np = action.squeeze(0).cpu().numpy()
        else:
            # Random policy: uniform over [0,1] for all 5 dims
            action_np = np.random.uniform(0.0, 1.0, size=5)

        obs, reward, done = env.step(action_np)
        step += 1

        # Recover current equity from env state
        price  = float(env._closes[env._t])
        iv_now = float(env._ivs[env._t])
        eq     = env._compute_equity(price, iv_now)
        equity_curve.append(eq)

        if done:
            break

    # --- Trading metrics ---
    equity = np.array(equity_curve, dtype=np.float64)
    ep_len = len(equity) - 1  # number of trading days

    total_return   = (equity[-1] - equity[0]) / equity[0]
    ann_return     = total_return * (252.0 / max(ep_len, 1))
    daily_rets     = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
    sharpe         = (float(np.mean(daily_rets)) / (float(np.std(daily_rets)) + 1e-10)) * np.sqrt(252)
    peak           = np.maximum.accumulate(equity)
    drawdowns      = (peak - equity) / np.maximum(peak, 1e-8)
    max_dd         = float(np.max(drawdowns))

    # Buy-and-hold return over same period
    bnh_return = (float(closes[min(env._t, len(closes)-1)]) - float(closes[t0])) / max(float(closes[t0]), 1e-8)

    # Regime classification based on episode start window
    regime = _classify_regime(env, t0)

    return {
        "sym":          sym,
        "regime":       regime,
        "ep_len":       ep_len,
        "total_ret":    total_return,
        "ann_ret":      ann_return,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "final_equity": float(equity[-1]),
        "bnh_ret":      bnh_return,
        "win":          total_return > 0,
    }


def _classify_regime(env: OptionsEnvironment, t0: int) -> str:
    closes   = env._closes
    iv_ranks = env._iv_ranks
    ep_len   = env._end - t0

    # High IV: mean IV rank in first 30 days > 60
    iv_window = iv_ranks[t0 : min(t0 + 30, len(iv_ranks))]
    if len(iv_window) > 0 and float(np.nanmean(iv_window)) > 60:
        return "high_iv"

    # Crash: high realized vol OR sharp drop in first 20 days
    if ep_len >= 20:
        log_rets = np.diff(np.log(np.maximum(closes[t0 : t0 + 21], 1e-8)))
        rv = float(np.std(log_rets) * np.sqrt(252)) if len(log_rets) > 1 else 0.0
        ret20 = (float(closes[t0 + 19]) - float(closes[t0])) / max(float(closes[t0]), 1e-8) if len(closes) > t0 + 19 else 0.0
        if rv > 0.40 or ret20 < -0.10:
            return "crash"

    # Uptrend: 60-day return > 15%
    if ep_len >= 60 and t0 + 60 < len(closes):
        ret60 = (float(closes[t0 + 59]) - float(closes[t0])) / max(float(closes[t0]), 1e-8)
        if ret60 > 0.15:
            return "uptrend"

    return "neutral"


# ---------------------------------------------------------------------------
# Aggregate metrics across episodes
# ---------------------------------------------------------------------------

def aggregate(episodes: list[dict]) -> dict:
    if not episodes:
        return {}
    rets   = [e["total_ret"] for e in episodes]
    ann    = [e["ann_ret"]   for e in episodes]
    sharpe = [e["sharpe"]    for e in episodes]
    dd     = [e["max_dd"]    for e in episodes]
    bnh    = [e["bnh_ret"]   for e in episodes]
    wins   = [e["win"]       for e in episodes]
    return {
        "n":          len(episodes),
        "mean_ret":   float(np.mean(rets)),
        "median_ret": float(np.median(rets)),
        "mean_ann":   float(np.mean(ann)),
        "mean_sharpe":float(np.mean(sharpe)),
        "mean_dd":    float(np.mean(dd)),
        "max_dd":     float(np.max(dd)),
        "win_rate":   float(np.mean(wins)),
        "mean_bnh":   float(np.mean(bnh)),
        "alpha":      float(np.mean(rets)) - float(np.mean(bnh)),
    }


def print_table(label: str, agg: dict) -> None:
    n = agg.get("n", 0)
    if n == 0:
        return
    print(f"  {label:<14} n={n:>3} | "
          f"ret={agg['mean_ret']:>+7.2%}  ann={agg['mean_ann']:>+7.2%}  "
          f"SR={agg['mean_sharpe']:>+6.2f}  DD={agg['mean_dd']:>5.2%}  "
          f"win={agg['win_rate']:>5.1%}  alpha={agg['alpha']:>+6.2%}  "
          f"BnH={agg['mean_bnh']:>+6.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cpu"  # B=1 inference: CPU avoids MPS kernel dispatch overhead (~10x faster per step)
    config = ATLASConfig()

    print("=" * 100, flush=True)
    print("  ATLAS v3 Curriculum Model — RL Environment Evaluation", flush=True)
    print(f"  Checkpoint: {CHECKPOINT}", flush=True)
    print(f"  Device: {device}  |  Episodes per policy: {N_EPISODES}", flush=True)
    print("=" * 100, flush=True)

    # Load model
    model = load_model(device)

    # Load v3 data
    print(f"\nLoading v3 features from {FEATURE_DIR}...", flush=True)
    t0 = time.time()
    all_data = load_training_data_v2(FEATURE_DIR, min_len=config.context_len + 200)
    print(f"Loaded {len(all_data)} symbols in {time.time()-t0:.1f}s", flush=True)

    # Build environment (full data, no regime filter for eval)
    env = OptionsEnvironment(all_data, config, regime_filter="all", reward_shaping="none")

    # -----------------------------------------------------------------------
    # Run episodes: trained model (greedy)
    # -----------------------------------------------------------------------
    print(f"\nRunning {N_EPISODES} episodes — ATLAS model (greedy)...", flush=True)
    model_episodes: list[dict] = []
    t0 = time.time()
    for i in range(N_EPISODES):
        ep = run_episode(env, model, device, greedy=True)
        model_episodes.append(ep)
        if (i + 1) % 50 == 0:
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (N_EPISODES - done)
            print(f"  {done}/{N_EPISODES} episodes ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # -----------------------------------------------------------------------
    # Run episodes: random policy baseline
    # -----------------------------------------------------------------------
    print(f"\nRunning {N_EPISODES} episodes — random policy baseline...", flush=True)
    random_episodes: list[dict] = []
    for i in range(N_EPISODES):
        ep = run_episode(env, None, device)
        random_episodes.append(ep)

    # -----------------------------------------------------------------------
    # Run episodes: cash policy baseline (delta=0, never trades)
    # -----------------------------------------------------------------------
    print(f"\nRunning {N_EPISODES} episodes — cash (no-trade) baseline...", flush=True)

    class CashEnv:
        """Wraps OptionsEnvironment with forced cash action."""
        pass

    cash_episodes: list[dict] = []
    for i in range(N_EPISODES):
        # Override: run episode but always pass action with delta < 0.05 (no trade)
        obs = env.reset()
        sym   = env._sym
        t0_ep = env._start
        closes = env._closes
        equity_curve = [_INITIAL_CAPITAL]
        while True:
            cash_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # delta=0 → no trade
            obs, reward, done = env.step(cash_action)
            price  = float(env._closes[env._t])
            iv_now = float(env._ivs[env._t])
            equity_curve.append(env._compute_equity(price, iv_now))
            if done:
                break
        equity = np.array(equity_curve)
        ep_len = len(equity) - 1
        total_return = (equity[-1] - equity[0]) / equity[0]
        ann_return = total_return * (252.0 / max(ep_len, 1))
        daily_rets = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
        sharpe = (float(np.mean(daily_rets)) / (float(np.std(daily_rets)) + 1e-10)) * np.sqrt(252)
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-8)))
        bnh_ret = (float(closes[min(env._t, len(closes)-1)]) - float(closes[t0_ep])) / max(float(closes[t0_ep]), 1e-8)
        regime = _classify_regime(env, t0_ep)
        cash_episodes.append({
            "sym": sym, "regime": regime, "ep_len": ep_len,
            "total_ret": total_return, "ann_ret": ann_return,
            "sharpe": sharpe, "max_dd": max_dd,
            "final_equity": float(equity[-1]), "bnh_ret": bnh_ret,
            "win": total_return > 0,
        })

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("  RESULTS", flush=True)
    print("  Columns: mean_ret  ann_ret  SR  max_DD  win_rate  alpha_vs_BnH  BnH_ret", flush=True)
    print("=" * 100, flush=True)

    regimes = ["high_iv", "uptrend", "crash", "neutral"]

    # Overall
    print("\n--- OVERALL ---", flush=True)
    print_table("ATLAS",  aggregate(model_episodes))
    print_table("Random", aggregate(random_episodes))
    print_table("Cash",   aggregate(cash_episodes))

    # By regime
    for regime in regimes:
        m_eps = [e for e in model_episodes  if e["regime"] == regime]
        r_eps = [e for e in random_episodes if e["regime"] == regime]
        c_eps = [e for e in cash_episodes   if e["regime"] == regime]
        if not m_eps:
            continue
        print(f"\n--- {regime.upper()} ---", flush=True)
        print_table("ATLAS",  aggregate(m_eps))
        print_table("Random", aggregate(r_eps))
        print_table("Cash",   aggregate(c_eps))

    # ATLAS beats random?
    print("\n" + "=" * 100, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 100, flush=True)

    ma = aggregate(model_episodes)
    ra = aggregate(random_episodes)
    ca = aggregate(cash_episodes)

    def _cmp(label: str, m_val: float, b_val: float, higher_better: bool = True) -> None:
        diff = m_val - b_val
        sign = diff > 0 if higher_better else diff < 0
        result = "ATLAS >" if sign else "ATLAS <"
        print(f"  {result} {label}: ATLAS={m_val:>+.4f}  baseline={b_val:>+.4f}  diff={diff:>+.4f}")

    print("\n  vs Random policy:", flush=True)
    _cmp("mean_ret",    ma["mean_ret"],    ra["mean_ret"])
    _cmp("mean_sharpe", ma["mean_sharpe"], ra["mean_sharpe"])
    _cmp("win_rate",    ma["win_rate"],    ra["win_rate"])
    _cmp("mean_dd",     ma["mean_dd"],     ra["mean_dd"],    higher_better=False)

    print("\n  vs Cash (no-trade) policy:", flush=True)
    _cmp("mean_ret",    ma["mean_ret"],    ca["mean_ret"])
    _cmp("mean_sharpe", ma["mean_sharpe"], ca["mean_sharpe"])

    print(f"\n  ATLAS mean alpha vs buy-and-hold: {ma['alpha']:>+.2%}")
    print(f"  Random mean alpha vs buy-and-hold: {ra['alpha']:>+.2%}")
    print(f"  ATLAS win rate: {ma['win_rate']:.1%}  |  Random win rate: {ra['win_rate']:.1%}", flush=True)

    # Regime distribution of test episodes
    print("\n  Regime distribution (ATLAS episodes):", flush=True)
    for regime in regimes:
        n = sum(1 for e in model_episodes if e["regime"] == regime)
        pct = n / len(model_episodes)
        print(f"    {regime:<10} {n:>4} episodes ({pct:.1%})", flush=True)

    print("\n" + "=" * 100, flush=True)


if __name__ == "__main__":
    main()
