# ATLAS v2 — Revised Architecture After Training Failure Analysis

## What Failed in v1

1. **Hindsight actions broken**: delta and profit_target always zero because the
   scoring function didn't use them. BC learned to output delta=0 (don't trade).
2. **Environment doesn't model options**: reward = direction × leverage × daily_return
   is stock trading, not options. Delta doesn't affect reward → model can't learn it.
3. **7x overparameterized**: 766K params vs 108K windows. Model memorizes, then PPO
   collapses to a constant policy.
4. **BC loss scale mismatch**: DTE in [14,90] dominates MSE over delta in [0,0.5].
5. **Cross-attention memory bank built from broken BC** — encodes "do nothing".

## v2 Structural Changes

### Change 1: Smaller Model (~50K params)

Reduce to match the data budget (108K windows ÷ 10 = ~10K target params, but
50K is workable with dropout):

```
d_model:     64 → 32
n_mamba_layers: 4 → 2
n_heads:     2 → 1
VSN:         34 individual GRNs → single linear projection (34 → 32)
Memory bank: 10K entries → REMOVE entirely (first get basic learning working)
Return conditioning: REMOVE (adds complexity, model doesn't use it)
```

Estimated: ~50K params. Data:param ratio = 2.2:1 (still low but viable with
regularization).

### Change 2: Simplified Action Space

Instead of 5 continuous outputs, reduce to 3:
```
action = [strategy_weight, aggressiveness, hold_duration]

strategy_weight ∈ [-1, 1]:
  -1 = full Wheel (premium selling)
   0 = cash (do nothing)
  +1 = full PMCC (directional)

aggressiveness ∈ [0, 1]:
  0 = most conservative (20-delta, far OTM)
  1 = most aggressive (50-delta, near ATM)

hold_duration ∈ [0, 1]:
  0 = shortest (close at 25% profit or 21 DTE)
  1 = longest (let expire)
```

The execution bridge maps these 3 numbers to concrete strategy parameters:
  - strategy_weight → selects Wheel vs PMCC vs Cash
  - aggressiveness → maps to delta (0.20 + 0.30 * aggressiveness)
  - hold_duration → maps to DTE (21 + 69 * hold_duration) and profit_target

This is MUCH easier to learn than 5 heterogeneous-scale continuous outputs.

### Change 3: Options-Aware Environment

The TradingEnvironment must simulate ACTUAL options mechanics:

```python
def step(action):
    strategy = action_to_strategy(action)

    if strategy == "wheel":
        # Use the actual WheelStrategy.on_bar() from our backtested code
        events = self.wheel.on_bar(date, price, iv, iv_rank)
        reward = equity_change / equity * 100  # percentage return

    elif strategy == "pmcc":
        events = self.pmcc.on_bar(date, price, iv, iv_rank)
        reward = equity_change / equity * 100

    else:  # cash
        reward = risk_free_daily_rate * 100  # opportunity cost of cash
```

This reuses the ALREADY BACKTESTED Wheel and PMCC strategy code. The model
learns WHEN to use which strategy, not HOW to execute the strategy.

### Change 4: Fix Hindsight Actions

The scoring function must evaluate actual options outcomes:

```python
def _score_combo(combo, closes, ivs, iv_ranks, t):
    strategy_weight, aggressiveness, hold_duration = combo

    # Map to concrete strategy params
    delta = 0.20 + 0.30 * aggressiveness
    dte = int(21 + 69 * hold_duration)

    # Simulate using actual Wheel/PMCC backtest code
    if strategy_weight < -0.3:
        # Run Wheel on this window with these params
        cfg = WheelConfig(put_delta=delta, target_dte=dte, ...)
        strat = WheelStrategy(cfg)
        for day in range(min(dte, len(closes) - t)):
            strat.on_bar(dates[t+day], closes[t+day], ivs[t+day], iv_ranks[t+day])
        sharpe = compute_sharpe(strat.equity_curve)
    elif strategy_weight > 0.3:
        # Run PMCC similarly
        ...
    else:
        sharpe = risk_free_rate / sqrt(252)  # cash Sharpe

    return sharpe
```

This is slower but CORRECT — the labels reflect actual options outcomes.

### Change 5: Normalize Action Labels for BC

Before computing BC loss, normalize all action dimensions to [0, 1]:
```
strategy_weight: (x + 1) / 2  → [0, 1]
aggressiveness: already [0, 1]
hold_duration: already [0, 1]
```

MSE is now equally weighted across all 3 outputs.

### Change 6: Simpler Backbone — MLP with Attention, Not Mamba

For 90-step sequences with 32-dim features, a 2-layer MLP with a single
attention head is sufficient and more sample-efficient:

```
Input: (B, 90, 32) — 90 days of 32-dim features
  → Flatten last 20 days: (B, 20*32=640)  [recent context]
  → MLP: Linear(640, 128) → GELU → Linear(128, 64)
  → Add: mean of full 90-day window (captures long-term regime)
  → MLP: Linear(64, 32) → GELU → Linear(32, 3) → bounded activations
```

Why MLP over Mamba: With only 108K samples, the Mamba's sequential processing
of 90 timesteps is a waste — a simple MLP over a flattened recent window is
more sample-efficient. Research shows MLPs match or beat transformers for RL
with limited data.

### Change 7: PPO with Proper Reward Scaling

```
reward = options_pnl_percentage  # already in [-5%, +5%] range
       - 0.5 * max(0, drawdown - 0.15)^2  # mild DD penalty
       - 0.001 * |action_change|  # tiny transaction cost
```

No multiplication by 100. No DSR (too noisy for small sample RL). Just
direct percentage P&L from actual options simulation.

## v2 Training Pipeline

1. Fix hindsight_actions to use actual Wheel/PMCC simulation
2. Build dataset with 3-dim normalized action labels
3. Train BC on smaller model (~50K params, 20 epochs, normalized MSE)
4. PPO with options-aware environment (reuses Wheel/PMCC code)
5. 1000 iterations × 1024 steps = 1M environment steps
6. Validate on out-of-sample 2023-2026

## Expected Outcome

The model should learn:
- "When trend is strong and IV is low → use PMCC (ride the trend)"
- "When range-bound and IV is high → use Wheel (collect premium)"
- "When crashing → go to cash"

This is what the Meta-Strategy TRIED to do with hard-coded rules but failed
due to switching costs and lag. The learned model should do it with smooth
continuous weights and no switching costs.
