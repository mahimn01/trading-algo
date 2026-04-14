# ATLAS: Adaptive Trading via Learned Action Sequences

## A Purpose-Built Transformer for Options Strategy Evolution

### Version 0.1 — Mathematical Specification
### Author: Mahimn Patel
### Date: March 2026

---

## 1. Problem Statement

Given a sequence of market observations over the past N trading days, output a
continuous action vector that parameterizes an options trading strategy, such
that risk-adjusted return (Sharpe ratio) is maximized over the strategy's
lifetime while respecting drawdown constraints.

The model must:
- Adapt to changing market regimes without retraining (in-context learning)
- Generalize across symbols, sectors, and time periods
- Be interpretable (we must understand WHY it makes each decision)
- Learn online from its own trading outcomes

---

## 2. Architecture Overview

```
           ┌─────────────────────────────────┐
           │         Input Pipeline           │
           │   (Section 3)                    │
           │                                  │
           │   Raw features per day:          │
           │   x_t in R^F  (F=16 features)   │
           └──────────┬──────────────────────┘
                      │
                      ▼
           ┌─────────────────────────────────┐
           │   Variable Selection Network     │
           │   (Section 4)                    │
           │                                  │
           │   Learns WHICH features matter   │
           │   in the current regime.         │
           │                                  │
           │   Output: x~_t in R^d  (d=64)   │
           └──────────┬──────────────────────┘
                      │
                      ▼
           ┌─────────────────────────────────┐
           │   Temporal Backbone (Mamba)      │
           │   (Section 5)                    │
           │                                  │
           │   4-layer selective SSM.         │
           │   Captures multi-scale temporal  │
           │   dependencies at O(N) cost.     │
           │                                  │
           │   Output: h_t in R^d  (d=64)    │
           └──────────┬──────────────────────┘
                      │
                      ▼
           ┌─────────────────────────────────┐
           │   De-Stationary Attention        │
           │   (Section 6)                    │
           │                                  │
           │   Recovers non-stationary info   │
           │   lost during normalization.     │
           │   Plug-in module.                │
           │                                  │
           │   Output: h'_t in R^d            │
           └──────────┬──────────────────────┘
                      │
           ┌──────────┴──────────────────────┐
           │                                  │
           ▼                                  ▼
┌─────────────────────┐         ┌─────────────────────────┐
│  Self-Attention      │         │  Cross-Attention         │
│  (Section 7a)        │         │  to Historical Memory    │
│                      │         │  (Section 7b)            │
│  2 heads, causal     │         │                          │
│  mask. Captures      │         │  "What worked in         │
│  intra-window        │         │   situations like this?" │
│  dependencies.       │         │                          │
│                      │         │  Q = h'_90 (current)     │
│  Output: s_t in R^d  │         │  K,V = Memory Bank M     │
└──────────┬──────────┘         │  Output: m_t in R^d      │
           │                     └──────────┬──────────────┘
           │                                 │
           └──────────┬──────────────────────┘
                      │ concat + project
                      ▼
           ┌─────────────────────────────────┐
           │   Return-Conditioned Fusion      │
           │   (Section 8)                    │
           │                                  │
           │   Conditions on desired Sharpe   │
           │   via return-to-go embedding.    │
           │                                  │
           │   Output: z_t in R^d             │
           └──────────┬──────────────────────┘
                      │
                      ▼
           ┌─────────────────────────────────┐
           │   Action Head                    │
           │   (Section 9)                    │
           │                                  │
           │   z_t -> [delta, direction,      │
           │           leverage, dte,         │
           │           profit_target]         │
           │                                  │
           │   Bounded continuous outputs.    │
           └──────────┬──────────────────────┘
                      │
                      ▼
           ┌─────────────────────────────────┐
           │   Execution Layer                │
           │   (Section 10)                   │
           │                                  │
           │   Maps continuous actions to     │
           │   concrete options trades via    │
           │   existing Wheel/PMCC infra.     │
           └─────────────────────────────────┘
```

---

## 3. Input Pipeline

### 3.1 Feature Set

Each trading day t is represented by a feature vector x_t in R^16:

```
x_t = [
    r_1,        # 1-day log return: ln(p_t / p_{t-1})
    r_5,        # 5-day cumulative return (weekly)
    r_21,       # 21-day cumulative return (monthly)
    r_63,       # 63-day cumulative return (quarterly)
    rv_30,      # 30-day realized volatility (annualized)
    iv_est,     # estimated implied volatility (from iv_rank module)
    iv_rank,    # IV rank: percentile of current IV vs 252-day history [0, 100]
    rsi_14,     # RSI(14), rescaled to [-1, 1] via (RSI - 50) / 50
    adx_14,     # ADX(14), rescaled to [0, 1] via ADX / 100
    vol_ratio,  # volume ratio: 5-day avg / 20-day avg volume
    ts_ratio,   # term structure: 10-day RV / 60-day RV
    skew_est,   # estimated vol skew (from dynamic model)
    pos_state,  # current position: -1 (short options), 0 (none), +1 (long/stock)
    pos_pnl,    # current trade unrealized P&L as fraction of capital
    days_in,    # days in current trade / target DTE (0 if no trade)
    cash_pct,   # cash as fraction of total equity
]
```

**Justification for each feature:**
- r_1 through r_63: Multi-scale returns capture momentum at different horizons.
  Research (X-Trend) showed that multi-scale returns are the single most important
  input for trend-following models. Using log returns ensures stationarity.
- rv_30: Realized vol is the strongest predictor of future vol (GARCH effect).
  Annualized for interpretability.
- iv_est, iv_rank: Options-specific. IV rank determines whether premium is rich
  or cheap — the primary signal for premium selling strategies.
- rsi_14: Mean reversion indicator. Rescaled to [-1, 1] for symmetric treatment.
- adx_14: Trend strength. Determines whether to trend-follow or mean-revert.
- vol_ratio: Volume confirmation of price moves. High ratio = institutional participation.
- ts_ratio: Vol term structure. >1 = contango (short-term vol elevated), <1 = backwardation.
- skew_est: Put skew level. Higher skew = puts are expensive = good for selling.
- pos_state through cash_pct: Self-state features. The model must know its own
  position to avoid contradictory actions (e.g., selling a put while already short one).

**Why 16 features, not more:** With ~500K training samples and ~134K parameters,
the model's capacity is limited. Adding features without corresponding signal
increases overfitting risk. Each feature above was validated in our backtesting
pipeline as having measurable impact on strategy performance.

**Why not price levels:** Raw prices are non-stationary and scale-dependent
($5 stock vs $500 stock). All price information enters through returns and
derived indicators, which are stationary and scale-invariant.

### 3.2 Normalization

Each feature is normalized using a ROLLING z-score over the trailing 252 days:

```
x_t^norm[i] = (x_t[i] - mu_{t,252}[i]) / (sigma_{t,252}[i] + epsilon)
```

where mu and sigma are the rolling mean and std of feature i over days [t-252, t],
and epsilon = 1e-8 prevents division by zero.

**Why rolling, not global:** Global normalization uses future data (look-ahead bias).
Rolling normalization only uses past data and adapts to changing distributions —
the same normalization scheme applies in training and live inference.

**Why 252 days:** One trading year. Short enough to adapt to regime changes,
long enough for stable estimates. Matches the IV rank lookback window.

### 3.3 Temporal Encoding

**Time2Vec** (Kazemi et al., 2019) encodes the timestamp of each day:

```
tau_t = unix_timestamp(date_t)

te_t[0] = omega_0 * tau_t + phi_0                     (linear component)
te_t[i] = sin(omega_i * tau_t + phi_i)    for i > 0   (periodic components)

te_t in R^8  (1 linear + 7 periodic)
```

where omega_i and phi_i are LEARNABLE parameters.

**Why Time2Vec over sinusoidal PE:** Trading days are irregularly spaced
(weekends, holidays). Sinusoidal PE assumes uniform spacing. Time2Vec uses
actual timestamps, naturally handling gaps. The learnable frequencies can
discover relevant periodicities (weekly, monthly, quarterly, annual cycles).

**Why 8 dimensions:** 1 linear trend + 7 periodic components can capture:
weekly (5-day), biweekly (10-day), monthly (21-day), quarterly (63-day),
semi-annual (126-day), annual (252-day), and options-expiry (monthly Friday)
cycles. More dimensions would overfit; fewer would miss important cycles.

**Calendar embeddings** are added as additional features:

```
cal_t = [
    embed_dow(day_of_week_t),    # in R^4 (learned embedding, 5 possible values)
    embed_month(month_t),        # in R^4 (learned, 12 possible values)
    is_opex_week_t,              # binary: 1 if options expiry week
    is_quarter_end_t,            # binary: 1 if within 5 days of quarter end
]

cal_t in R^10
```

### 3.4 Day Token Construction

Each day's input token is the concatenation:

```
token_t = [x_t^norm ; te_t ; cal_t] in R^(16 + 8 + 10) = R^34
```

**Context window:** N = 90 trading days (~4.3 months).

**Why 90 days:** Must be long enough to capture regime changes (typically 20-60 days)
but short enough for computational efficiency. Research (X-Trend) found that
63-day windows capture most actionable patterns; 90 days provides buffer.
The SSM backbone processes this at O(N) cost regardless.

---

## 4. Variable Selection Network (VSN)

Adapted from Temporal Fusion Transformer (Lim et al., 2019).

**Purpose:** Learn which features matter in the current regime. During a crash,
volatility features dominate. During range-bound markets, IV rank and term
structure matter more. The VSN dynamically reweights features.

### 4.1 Gated Residual Network (GRN)

The building block for all VSN computations:

```
GRN(a, c) =
    eta_1 = W_1 * a + b_1                                    in R^d
    eta_2 = ELU(W_2 * eta_1 + W_3 * c + b_2)    if c given  in R^d
          = ELU(W_2 * eta_1 + b_2)               otherwise
    eta_3 = W_4 * eta_2 + b_4                                in R^d
    output = LayerNorm(a_proj + GLU(eta_3))

    where a_proj = W_0 * a  (project a to R^d if dimensions differ)
    GLU(x) = sigmoid(W_5 * x) * (W_6 * x)       (gated linear unit)
```

**Why GRN over simple MLP:** The gating mechanism (GLU) allows the network
to suppress irrelevant information — a feature can be completely zeroed out
by the gate. LayerNorm + residual skip ensures gradient flow. ELU activation
avoids dead neurons (unlike ReLU).

### 4.2 Feature-Level Variable Selection

```
For each feature i in {1, ..., 16}:
    xi_i = GRN_i(x_t[i])                    # feature-specific processing

# Compute selection weights:
v_t = Softmax(GRN_v(flatten(x_t)))          # v_t in R^16, sums to 1

# Weighted combination:
x~_t = sum_{i=1}^{16} v_t[i] * xi_i        # x~_t in R^d
```

**Interpretability:** The weights v_t directly tell us which features the model
considers important at time t. We can log these and analyze:
- "On day X, the model weighted IV rank at 0.35 and vol at 0.30"
- "During the crash, vol weight jumped from 0.15 to 0.45"

### 4.3 Dimension

```
Input:  token_t in R^34
Output: x~_t in R^64  (d_model = 64)
```

Parameters: 16 feature GRNs + 1 selection GRN = ~8K parameters.

---

## 5. Temporal Backbone (Selective SSM / Mamba)

### 5.1 The Selective State Space Model

Mamba (Gu & Dao, 2023) — discretized from continuous-time SSM:

```
Continuous:
    h'(t) = A * h(t) + B * x(t)
    y(t) = C * h(t)

Discretized (zero-order hold with step size Delta):
    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C_t * h_t

    where A_bar = exp(Delta * A)
          B_bar = (Delta * A)^{-1} * (exp(Delta * A) - I) * Delta * B
```

**The "selective" mechanism** — what makes Mamba different from S4:

```
B_t = Linear_B(x_t)       # B is a function of the INPUT, not fixed
C_t = Linear_C(x_t)       # C is a function of the INPUT
Delta_t = softplus(Linear_Delta(x_t) + bias_Delta)  # step size is input-dependent
```

**Why this matters for trading:** In calm markets, Delta_t is small — the model
filters noise aggressively, maintaining a stable hidden state. During volatility
spikes, Delta_t increases — the model becomes more responsive to new information.
This is EXACTLY the adaptive behavior we want: ignore noise in calm markets,
react quickly in crises.

### 5.2 Mamba Block Architecture

```
MambaBlock(x):
    # Expand dimension for richer processing
    z = Linear_expand(x)           # R^d -> R^(E*d), expansion factor E=2
    z = SiLU(Conv1D(z, k=4))      # local convolution for short-term patterns
    z = SelectiveSSM(z)            # state space processing
    z = z * SiLU(Linear_gate(x))   # output gate
    z = Linear_contract(z)         # R^(E*d) -> R^d
    return x + z                   # residual connection
```

**Why Conv1D with kernel 4:** Captures very short-term dependencies (1-4 days)
before the SSM. This handles daily noise patterns (e.g., mean reversion,
bid-ask bounce) that the SSM would otherwise waste capacity modeling.

### 5.3 Stack of 4 Layers

```
h_t^(0) = x~_t                    # from VSN
h_t^(l) = MambaBlock^(l)(h_t^(l-1))   for l = 1, 2, 3, 4
h_t = h_t^(4)                     # final temporal representation
```

**Why 4 layers:** Each layer doubles the effective receptive field.
- Layer 1: ~4-day patterns (Conv1D kernel)
- Layer 2: ~16-day patterns
- Layer 3: ~64-day patterns
- Layer 4: ~256-day patterns (exceeds our 90-day window — full coverage)

Stacking more layers provides diminishing returns and increases overfitting risk.
Algorithm Distillation paper used L=4 with d=64 — our exact configuration.

### 5.4 State Space Dimension

```
A in R^{d x N_state}  where N_state = 16 (state dimension per feature)
```

**Why N_state = 16:** The state dimension determines how many "memory slots"
the SSM has per feature. For financial data with clear periodic structure
(weekly, monthly, quarterly cycles), 16 states can capture the dominant
frequency components (analogous to 16 Fourier modes). S4 paper showed
N_state = 16 is sufficient for most sequence modeling tasks.

Parameters per Mamba layer: ~16K. Total backbone: ~65K parameters.

---

## 6. De-Stationary Attention Module

From Non-stationary Transformers (Liu et al., NeurIPS 2022).

### 6.1 The Problem

Section 3.2's rolling z-score normalization makes the input stationary — good
for modeling but destroys non-stationary information (absolute vol levels,
trend magnitudes) that matters for trading decisions.

### 6.2 The Solution

```
# Store pre-normalization statistics:
mu_t = rolling_mean(x_t, 252)     # what was removed
sigma_t = rolling_std(x_t, 252)   # what was removed

# Learn to recover useful non-stationary information:
tau_t = exp(MLP_tau(sigma_t, h_t))   # scaling factor (always positive via exp)
Delta_t = MLP_Delta(mu_t, h_t)       # bias factor

# Modified attention (applied in Section 7a):
Attn(Q, K, V, tau, Delta) = Softmax((tau * Q @ K^T + 1 * Delta^T) / sqrt(d)) @ V
```

**Why this works:** The tau and Delta terms re-inject non-stationary information
into the attention computation. tau scales the attention logits (higher absolute
vol → sharper attention, less averaging). Delta shifts attention toward specific
patterns associated with the current mean level.

**Research validation:** Reduces MSE by 46-49% across transformer variants.
This is a plug-in module — no changes to the rest of the architecture.

Parameters: 2 small MLPs, ~2K parameters total.

---

## 7. Dual Attention Module

### 7a. Intra-Window Self-Attention

Standard multi-head causal self-attention over the 90-day context:

```
For each head j in {1, 2}:
    Q_j = W_Q^j @ H       # H = [h_1, ..., h_90] in R^{d x 90}
    K_j = W_K^j @ H
    V_j = W_V^j @ H

    # Causal mask: day t can only attend to days [1, ..., t]
    mask[i,j] = 0 if j <= i, else -inf

    # De-stationary attention:
    A_j = Softmax((tau * Q_j @ K_j^T + Delta^T + mask) / sqrt(d/2)) @ V_j

# Concatenate heads and project:
s_t = W_O @ concat(A_1[:, 90], A_2[:, 90]) + b_O    # take last position
```

**Why only 2 heads:** With d=64 and 2 heads, each head has dimension 32.
This is sufficient for the attention to specialize:
- Head 1: short-term patterns (high attention on recent 5-10 days)
- Head 2: structural patterns (attention on specific past events)

More heads with smaller dimension lose representational capacity per head.

**Why causal mask:** During training, we process the full 90-day window at once.
The causal mask ensures day t only sees past days — no future leakage. At
inference, this is naturally satisfied since we process sequentially.

Parameters: ~16K.

### 7b. Cross-Attention to Historical Memory Bank

This is the **novel core** — where the model learns from historical analogues.

#### Memory Bank Construction (Training Time)

```
M = {(c_i, a_i, r_i)} for i = 1, ..., |M|

where:
    c_i in R^d  = SSM output at day i (historical context embedding)
    a_i in R^5  = action taken at day i (delta, direction, leverage, dte, profit_tgt)
    r_i in R    = realized risk-adjusted reward over the following DTE days
```

The memory bank is constructed from training data:
1. Run the SSM backbone over all training sequences
2. At each day, record (embedding, hindsight-optimal action, realized reward)
3. Cluster into ~10,000 representative entries via k-means on c_i

**Why 10,000 entries:** Cross-attention cost is O(N * |M|). With N=90 and |M|=10K,
that's 900K operations — negligible. Fewer entries lose coverage of rare regimes
(crashes, flash crashes, V-shaped recoveries). More entries add redundancy.

**Why cluster:** Raw training data has ~500K entries. Most are from "normal" markets
and are redundant. Clustering preserves diversity while reducing redundancy.
We oversample tail events (top/bottom 10% of returns) to ensure crash patterns
are well-represented.

#### Cross-Attention Mechanism

```
# Current state as query:
q = W_Q_cross @ h'_90                   # h'_90 = current day's representation after de-stationary attention

# Memory bank as keys and values:
K_mem = W_K_cross @ stack([c_i for (c_i, _, _) in M])     # R^{d x |M|}
V_mem = W_V_cross @ stack([embed(a_i, r_i) for (_, a_i, r_i) in M])  # R^{d x |M|}

# Attend:
alpha = Softmax(q^T @ K_mem / sqrt(d))   # attention weights over memory, in R^|M|
m_t = alpha @ V_mem^T                     # weighted average of historical actions, in R^d
```

**What this computes:** The model finds the historical situations most similar
to "right now" (via learned similarity in embedding space) and retrieves the
actions that worked in those situations.

**Interpretability:** The attention weights alpha tell us WHICH historical periods
the model considers analogous. We can report: "The model sees the current
market as most similar to July 2020 (post-COVID recovery, moderate vol, strong
uptrend) and is recommending the action that worked best then."

#### Regime Segmentation via Change-Point Detection

Following X-Trend, we segment historical data into regimes before building
the memory bank:

```
1. Run Gaussian Process Change-Point Detection on each training symbol
2. Segment into regime windows (typical length: 21-63 days)
3. For each regime window, compute:
   - The average SSM embedding (context)
   - The hindsight-optimal action for that regime
   - The realized Sharpe for that action
4. Store as memory bank entries
```

**Why CPD over fixed windows:** Regimes have variable length. A fixed 30-day
window might split a crash across two windows, losing the pattern. CPD detects
natural breakpoints where the data-generating process changes.

Parameters: ~12K for cross-attention projections.

---

## 8. Return-Conditioned Fusion

Adapted from Decision Transformer (Chen et al., 2021).

### 8.1 Concept

We want the model to generate actions conditioned on a DESIRED performance level.
At training time, we pair each state with the return-to-go (actual future Sharpe
achieved from that point). At inference, we set a target Sharpe and the model
generates actions to achieve it.

### 8.2 Mechanism

```
# Return-to-go embedding:
R_hat_t = target_sharpe     # at inference: set to desired Sharpe (e.g., 1.0)
                             # at training: use actual realized Sharpe from day t forward

r_emb = Linear_R(R_hat_t)   # R -> R^d, learned embedding of the scalar target

# Fusion:
combined = concat(s_t, m_t)              # self-attention output + memory output, R^(2d)
projected = Linear_fuse(combined)         # R^(2d) -> R^d
z_t = LayerNorm(projected + r_emb)       # add return conditioning
```

**Why this works:** By conditioning on return-to-go, the model learns a SPECTRUM
of behaviors:
- R_hat = 0.5: conservative actions (wide delta, low leverage)
- R_hat = 1.0: moderate actions (standard parameters)
- R_hat = 2.0: aggressive actions (tight delta, high leverage)

At inference, we set R_hat to a reasonable target (e.g., 1.0 Sharpe) and the model
generates actions calibrated to that ambition level. If we set it too high,
the model takes excessive risk — the drawdown penalty in the loss function
teaches it that unrealistic targets lead to blowups.

### 8.3 Target Update During Live Trading

```
After each realized trade with return r_t:
    R_hat_{t+1} = R_hat_t - r_t / sigma_t    # decrement by realized contribution to Sharpe
```

This tracks how much Sharpe is "left to achieve" in the current window.

Parameters: ~8K.

---

## 9. Action Head

### 9.1 Architecture

```
z_t in R^d                           # from fusion module

a_hidden = GELU(Linear_1(z_t))       # R^d -> R^(d/2) = R^32
a_raw = Linear_2(a_hidden)            # R^32 -> R^5

# Bounded activations (each justified below):
delta       = sigmoid(a_raw[0]) * 0.50      # [0, 0.50]
direction   = tanh(a_raw[1])                 # [-1, +1]
leverage    = sigmoid(a_raw[2])              # [0, 1]
dte         = sigmoid(a_raw[3]) * 76 + 14    # [14, 90]
profit_tgt  = sigmoid(a_raw[4])              # [0, 1]
```

### 9.2 Activation Justifications

**delta in [0, 0.50]:**
- 0 = don't trade (go to cash). The sigmoid naturally allows this.
- 0.50 = ATM. Never sell beyond 50-delta (that's ATM or ITM — not a premium selling strategy).
- Sigmoid provides smooth gradient everywhere — no dead zones.

**direction in [-1, +1]:**
- -1 = pure premium selling (Wheel: sell puts, sell calls)
- 0 = neutral (no directional bias, e.g., iron condor)
- +1 = pure directional (PMCC: buy LEAPS)
- tanh allows the model to express the full spectrum of directional views.
- The SIGN determines strategy type; the MAGNITUDE determines conviction.

**leverage in [0, 1]:**
- 0 = all cash (no position). Critical for crash protection.
- 1 = fully deployed (max contracts given capital).
- Sigmoid is monotonic — higher raw values always mean more leverage.

**dte in [14, 90]:**
- 14 = minimum tradeable DTE (options with < 14 days have excessive gamma risk
  and wide bid-ask spreads).
- 90 = maximum (beyond 90 DTE, theta decay is negligible — not worth the capital lockup).
- sigmoid * 76 + 14 maps smoothly to this range.

**profit_tgt in [0, 1]:**
- 0 = let expire (no early close). Our backtests showed this is often optimal.
- 1 = close immediately at any profit (too conservative for most situations).
- Intermediate values (e.g., 0.50 = close at 50% of max profit) provide flexibility.

### 9.3 Position Size Calculation (Volatility Targeting)

Following X-Trend's approach:

```
raw_size = leverage * capital * max_allocation_per_trade
vol_target = 0.15   # 15% annualized target volatility
vol_current = rv_30  # current 30-day realized vol

# Scale position inversely with volatility:
vol_adjusted_size = raw_size * (vol_target / max(vol_current, 0.05))

# Clamp to risk limits:
final_size = min(vol_adjusted_size, capital * 0.25)  # max 25% per trade
```

**Why volatility targeting:** A fixed leverage in a low-vol environment (RV=10%)
deploys the same capital as in a high-vol environment (RV=50%), even though
the risk is 5x different. Volatility targeting normalizes risk: in high vol,
smaller positions; in low vol, larger positions. This is standard in systematic
trading (used by most CTAs and trend-followers).

Parameters: ~6K for action head.

---

## 10. Training Objective

### 10.1 Primary: Differential Sharpe Ratio (Moody & Saffell, 1998)

```
A_t = A_{t-1} + eta * (R_t - A_{t-1})           # EMA of returns
B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})         # EMA of squared returns

DSR_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

where:
    dA_t = R_t - A_{t-1}
    dB_t = R_t^2 - B_{t-1}
    eta = 2 / (T_ema + 1),  T_ema = 63 (quarterly Sharpe estimation)
    R_t = realized return at time t from the model's action
```

**Why DSR:** It's the EXACT gradient of the Sharpe ratio, computed incrementally.
No surrogate losses or approximations. The model directly optimizes for
risk-adjusted return at every time step.

### 10.2 Drawdown Penalty

```
DD_t = max(0, peak_equity_{0:t} - equity_t) / peak_equity_{0:t}

L_dd = lambda_dd * max(0, DD_t - DD_threshold)^2

where DD_threshold = 0.20 (20% max acceptable drawdown)
      lambda_dd = 10.0
```

**Why quadratic:** Linear penalty treats a 21% drawdown and a 50% drawdown
almost equally. Quadratic penalty makes deep drawdowns MUCH more costly,
teaching the model to avoid catastrophic losses aggressively.

### 10.3 Transaction Cost Penalty

```
L_tc = lambda_tc * |a_t - a_{t-1}|_1

where lambda_tc = 0.01
```

**Why L1 norm:** L1 encourages SPARSE changes — the model learns to hold steady
rather than constantly adjusting. This matches real trading where every action
has a cost (slippage + commission).

### 10.4 Combined Loss

```
L = -DSR_t + L_dd + L_tc
```

Negative DSR because we MAXIMIZE Sharpe (minimize negative Sharpe).

---

## 11. Training Regime

### Phase 1: Behavioral Cloning (Warmstart)

```
Data: For each (day, symbol) in training set:
    x_t = market features
    a*_t = hindsight-optimal action (computed by brute-force search over action space
           using future returns)

Loss: L_BC = MSE(model(x_{t-90:t}), a*_t)

Duration: ~10 epochs over full training set
Purpose: Give the model a reasonable initialization before RL fine-tuning
```

**Why behavioral cloning first:** RL from scratch in a complex action space is
extremely sample-inefficient. Starting from a supervised warmstart reduces
training time by 10-50x (standard practice in Decision Transformer literature).

### Phase 2: RL Fine-Tuning via Policy Gradient

```
Algorithm: PPO (Proximal Policy Optimization) with clipped objective

For each training episode (random symbol, random 2-year window):
    Run model on window, collecting (state, action, reward) trajectory
    Compute advantages A_t using GAE (lambda=0.95)
    Update model:
        L_PPO = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
        where r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
        eps = 0.2 (clipping range)

Randomization:
    - Random start date (prevents overfitting to specific calendar effects)
    - Random symbol from training universe
    - Random target Sharpe (R_hat sampled from [0.3, 2.0])
```

**Why PPO over SAC/DDPG:** PPO is more stable for continuous action spaces
with bounded outputs. SAC requires entropy tuning; DDPG is brittle. PPO's
clipped objective prevents catastrophically large policy updates.
Research (FinRL) confirmed PPO outperforms for options-related tasks.

### Phase 3: Online Adaptation (Elastic Weight Consolidation)

```
After every K=30 trading days of live data:

1. Compute Fisher Information Matrix on recent 90 days:
   F_i = E[(d log pi(a|s) / d theta_i)^2]  # diagonal approximation

2. Fine-tune on last 90 days with EWC regularization:
   L_EWC = L_current(theta) + (lambda_ewc / 2) * sum_i F_i * (theta_i - theta*_i)^2

   where theta* = parameters before this update
         lambda_ewc = 100 * exp(-t / T_decay)  # DECAYING lambda

3. T_decay = 500 trading days (~2 years)
   After ~2 years, lambda_ewc has decayed to ~37% of its initial value,
   allowing the model to gradually forget very old regime patterns.
```

**Why decaying lambda:** Standard EWC prevents ALL forgetting. But markets
change — a pattern from 2016 may not be relevant in 2026. Decaying lambda
lets the model gradually release old knowledge while still preventing
catastrophic forgetting of recent lessons.

---

## 12. Data Requirements and Compute Budget

### Training Data

```
Symbols: 100+ liquid US equities + ETFs + commodity ETFs
History: 20 years (2006-2026), covering:
    - 2008 financial crisis (bear + recovery)
    - 2010 flash crash
    - 2015-2016 China devaluation / oil crash
    - 2018 vol explosion (Volmageddon)
    - 2020 COVID crash + recovery
    - 2022 bear market
    - 2024-2025 bull market
Days: ~5,000 per symbol
Total: ~500,000 (state, action, reward) training samples
```

### Model Size

```
Module                  Parameters
---------------------------------
Feature Encoder (VSN)        8,192
Temporal Backbone (4x SSM)  65,536
De-Stationary Attention      2,048
Self-Attention (2 heads)    16,384
Cross-Attention             12,288
Return Conditioning          8,192
Action Head                  6,144
---------------------------------
TOTAL                      118,784  (~119K parameters)
```

**Data-to-parameter ratio:** 500,000 / 119,000 = 4.2x.
This is tight but workable with:
- Dropout (p=0.1) on all linear layers
- Weight decay (1e-4)
- Early stopping on validation set
- Data augmentation (random window offsets, symbol permutation)

### Compute

```
Phase 1 (BC):   ~2 hours on single GPU (RTX 3090 or equivalent)
Phase 2 (PPO):  ~8 hours on single GPU
Phase 3 (live): ~5 minutes per monthly update on CPU

Total training: ~10 hours one-time, ~5 min/month ongoing
Inference: <10ms per day (single CPU forward pass)
```

**Why this is feasible on consumer hardware:** The model is 119K parameters —
smaller than a single layer of GPT-2. The SSM backbone runs at O(N) not O(N^2).
Training on 500K samples with batch size 64 = 7,800 steps per epoch.
No need for cloud GPU clusters.

---

## 13. What Makes This Novel

1. **Hybrid SSM + Cross-Attention for options:** Mamba/S4 has not been applied to
   options strategy generation. The combination with historical analogue
   matching via cross-attention is architecturally new.

2. **Continuous action space for options strategies:** Existing trading RL outputs
   buy/sell/hold. ATLAS outputs (delta, direction, leverage, DTE, profit_target)
   as a continuous vector that directly parameterizes ANY options strategy
   (Wheel, PMCC, spreads, or novel combinations).

3. **Return conditioning for risk calibration:** Decision Transformer's return-to-go
   mechanism adapted for Sharpe targeting. The model can express a SPECTRUM
   of risk-return tradeoffs via a single scalar input.

4. **Built-in interpretability:** VSN weights reveal feature importance.
   Cross-attention weights reveal historical analogues. No post-hoc
   interpretability tools needed — understanding is architectural.

5. **Online EWC with decaying lambda:** Novel combination of continual learning
   with intentional forgetting, specifically designed for non-stationary markets.

6. **The action-to-strategy mapping:** The continuous action vector maps to
   existing, backtested infrastructure (Wheel, PMCC, spreads). This means
   every action the model takes is executable through proven code paths,
   not through a fragile end-to-end neural network.

---

## 14. Execution Layer (Bridging Model to Market)

The continuous action vector maps to concrete trades via the existing infrastructure:

```
if direction < -0.3:
    # Premium selling regime
    if pos_state == 0:
        # No position -> sell put (Wheel CSP phase)
        execute_wheel_csp(delta=delta, dte=dte, leverage=leverage)
    elif pos_state == 1:
        # Holding stock -> sell call (Wheel CC phase)
        execute_wheel_cc(delta=delta, dte=dte)

elif direction > 0.3:
    # Directional regime
    execute_pmcc(leaps_delta=0.80, short_delta=delta, short_dte=dte, leverage=leverage)

else:
    # Neutral regime
    if iv_rank > 50:
        execute_put_spread(delta=delta, dte=dte, leverage=leverage)
    elif leverage < 0.1:
        go_to_cash()
    else:
        hold_current_position()
```

**Why this mapping instead of end-to-end:** The model doesn't need to learn
how to construct an OPT InstrumentSpec or compute bid-ask slippage. That's
engineering, not intelligence. The model focuses on the STRATEGIC decision
(what kind of trade, at what parameters) and delegates execution to code
that's already tested and debugged.

---

## 15. Validation Protocol

Before live deployment, ATLAS must pass:

1. **Out-of-sample backtest:** Train on 2006-2022, test on 2023-2026.
   Must achieve Sharpe > 0.5 on test set across 20+ symbols.

2. **Walk-forward:** 4-fold expanding window. Sharpe degradation < 30%.

3. **Stress test:** Inject synthetic crashes (30%, 50% drawdowns).
   Max drawdown must stay below 25%.

4. **Regime test:** Separately evaluate on bull/bear/range/crisis periods.
   Must be profitable in at least 3 of 4 regimes.

5. **Comparison:** Must outperform the base Wheel (Sharpe 0.30, 11.8% ann)
   and match or exceed buy-and-hold risk-adjusted return.

6. **Paper trade:** 3 months on IBKR paper account before any live capital.

---

## 16. Risk Management Invariants (Hard-Coded, Not Learned)

These are NEVER overridden by the model:

1. Max position size: 25% of capital per trade
2. Max drawdown: 25% triggers forced liquidation (circuit breaker)
3. Max leverage: 1.0 (no margin)
4. No selling naked calls (undefined risk)
5. Must be able to survive assignment (sufficient cash for shares)
6. Daily equity logged to SQLite (full audit trail)
7. Paper trading for 3 months before any live capital
