# Quantitative Trading System

Multi-asset quantitative trading system with enterprise-grade backtesting
infrastructure, a full IBKR live-trading stack, an options-income strategy
suite (Wheel / PMCC / Enhanced Wheel / Jade Lizard), the ATLAS deep-RL
trading model, and a comprehensive IBKR data + Flex Web Service CLI toolkit.

Equity strategies run on IBKR 5-minute data (10 years) and crypto edges run
on Binance perpetual futures.

## Results

All numbers from enterprise-grade backtesting: next-bar-open execution, VWAP
position tracking, backward-only data lookups, and validated by randomized
signal null tests.

| System | Sharpe | Return | Max DD | Period | Notes |
|--------|--------|--------|--------|--------|-------|
| **Equity V11** | +0.480 | +151.3% | 18.5% | 2016–2026 | 5 strategies, 7 symbols, vol-gated intraday signals |
| **Crypto 9-Edge** | +0.277 | +7.7% | 21.8% | 2022–2026 | 9 edges, 5 perps, below BTC buy-and-hold (+0.80) |

**Equity integrity**: DSR 15.988 (p=0.000), OOS SR +0.547, alpha vs SPY +4.06%
annualized. Beta 0.119. Enterprise runner with next-bar-open execution
eliminates same-bar look-ahead bias.

**Crypto integrity**: Fraud detection v2 confirms honest infrastructure.
Random signals produce SR -0.38 (near zero minus costs). Only 3/9 edges
individually positive: IMC (+0.72), CED (+0.40), PBMR (+0.33).

## Architecture

```
randomThings/
├── trading_algo/                        # Equity + options + live trading
│   ├── cli.py                           # Main trading engine CLI (place/modify/cancel/bracket/scan/backtest/chat/…)
│   ├── ibkr_tool.py                     # 46-command IBKR data/ops CLI (quotes, chains, depth, ticks, fundamentals, news, orders, whatIf)
│   ├── flex_tool.py                     # 31-command Flex Web Service CLI (send+parse+aggregate statements)
│   ├── broker/
│   │   ├── ibkr.py                      # ib_async broker adapter: contract cache, rate limiter, circuit breaker, health monitor
│   │   └── sim.py                       # Deterministic simulation broker for backtests
│   ├── config.py                        # TradingConfig + IBKRConfig with safety rails
│   ├── engine.py                        # Polling engine + risk manager
│   ├── oms.py                           # Order manager + state machine
│   ├── orders.py                        # Order validation, TradeIntent
│   ├── risk.py                          # Risk limits
│   ├── persistence.py                   # Sqlite audit trail
│   ├── multi_strategy/                  # Multi-strategy controller, walk-forward, adapters (14 equity strategies)
│   ├── orchestrator/                    # Original 6-edge orchestrator
│   ├── strategies/                      # Legacy strategy modules
│   ├── quant_core/
│   │   ├── strategies/options/          # Wheel, PMCC, Enhanced Wheel, Jade Lizard, hybrid regime, defined-risk
│   │   ├── strategies/intraday/         # ORB + intraday modules
│   │   ├── edge_validation/             # Statistical edge-validation pipeline (ORB/gap fade/VWAP patterns)
│   │   ├── models/atlas/                # ATLAS: Mamba-SSM + Cross-Attention hybrid RL trader (v1–v7)
│   │   └── validation/                  # PBO, Deflated Sharpe, White's Reality Check
│   ├── llm/                             # Gemini 3 chat + trader loop + tools
│   ├── dashboard/                       # TUI dashboard (equity curve, trades, backtest panel)
│   └── rat/                             # RAT framework (Reflexive Attention Topology)
├── crypto_alpha/                        # Crypto multi-edge trading system
│   ├── edges/                           # 9 crypto-native edge implementations
│   ├── adapters/                        # Edge → controller adapters
│   ├── backtest/crypto_runner.py        # Enterprise crypto backtest runner
│   ├── data/                            # CCXT loader + numpy cache
│   └── scripts/                         # Fraud detection, deep analysis
├── backtest/                            # Shared metrics (Sharpe, Sortino, VaR, DSR)
├── scripts/
│   ├── run_10yr_backtest.py             # 10-year equity backtest entry point
│   ├── validate_edge.py                 # Edge validation runner (ORB/gap/VWAP)
│   ├── atlas_*.py                       # ATLAS training/eval scripts (v1–v7, IBKR curriculum, R3000)
│   └── wheel_*.py, pmcc_sweep, …        # Options strategy backtests
├── tests/                               # Unit + integration tests (incl. test_edge_validation/)
├── docs/                                # Architecture, safety, RAT research, traditional vs AI analysis
└── CLAUDE.md                            # Project trading rules
```

The equity + crypto systems share the same `MultiStrategyController` and
`StrategySignal` protocol. Strategies implement a common adapter interface
that maps edge-specific logic to unified signals.

## Strategy Catalog

### Equity Strategies (14 adapters)

| Strategy | Description |
|----------|-------------|
| **PureMomentum** | Cross-sectional momentum with trend confirmation |
| **MeanReversion** | Statistical extremes (z-score) with regime filtering |
| **PairsTrading** | Cointegrated pair spreads with Kalman filter |
| **FlowPressure** | Volume-weighted price pressure signals |
| **Orchestrator** | 6-edge ensemble (regime, relative strength, stats, volume, cross-asset, time) |
| **IntradayMomentum** | Session momentum with breakout detection |
| **OvernightGap** | Overnight gap fade/continuation |
| **ORB** | Opening range breakout |
| **LeadLag** | Cross-asset lead-lag relationships |
| **HurstExponent** | Persistence/mean-reversion regime detection |
| **LiquidityCycle** | Volume cycle timing |
| **RegimeTransition** | HMM-based regime change detection |
| **TimeAdaptive** | Time-of-day pattern exploitation |
| **CrossAsset** | Correlated asset confirmation |

**V11 config**: `signal_interval_bars=156, intraday_vol_threshold=0.15, max_gross_exposure=1.5, vol_target=0.18`

### Options Income Strategies

| Strategy | Module | Description |
|---|---|---|
| **Wheel** | `quant_core/strategies/options/wheel.py` | Cash-secured put → covered call cycle |
| **PMCC** | `quant_core/strategies/options/pmcc.py` | Long deep-ITM LEAP + rolled short call |
| **Enhanced Wheel** | `quant_core/strategies/options/enhanced_wheel.py` | IV rank gating, profit target close, regime filter |
| **Portfolio Wheel** | `quant_core/strategies/options/portfolio_wheel.py` | Multi-symbol capital allocation |
| **Jade Lizard** | `quant_core/strategies/options/jade_lizard.py` | No-upside-risk short put + short call spread |
| **Hybrid Regime** | `quant_core/strategies/options/hybrid_regime.py` | Strategy selection by volatility regime |
| **Put Spread** | `quant_core/strategies/options/put_spread.py` | Defined-risk credit put spreads |

### Crypto Edges (9)

| Edge | Standalone SR | Description |
|------|--------------|-------------|
| **IMC** (Intermarket Cascade) | +0.72 | Cross-asset momentum cascades (BTC→ETH→alts) |
| **CED** (Cross-Exchange Divergence) | +0.40 | Price divergence across exchanges |
| **PBMR** (Perpetual Basis Mean Reversion) | +0.33 | Futures-spot basis mean reversion |
| **FRM** (Funding Rate Momentum) | -0.02 | Funding rate trends |
| **VTS** (Volatility Term Structure) | -0.11 | IV term structure signals |
| **LCP** (Liquidation Cascade Predictor) | -0.20 | Liquidation cascade detection |
| **VF** (Volume Flow) | -0.31 | Directional volume analysis |
| **VV** (Volume Velocity) | -0.42 | Volume acceleration |
| **RADL** (Regime-Adaptive Dynamic Leverage) | -0.58 | Regime-based leverage adjustment |

## ATLAS — Deep-RL Trading Model

`trading_algo/quant_core/models/atlas/` implements a hybrid Mamba-SSM +
Cross-Attention architecture trained via PPO with EWC (Elastic Weight
Consolidation) on curriculum-structured episodes. Now at v7 with new
attention/fusion/backbone modules, a fast env, fast hindsight rewarding,
and MLX support. Training scripts live in `scripts/atlas_*.py`.

## Edge Validation Infrastructure

`trading_algo/quant_core/edge_validation/` answers the question "does this
pattern exist in historical data with positive expected value?" before any
backtesting infrastructure is built, using academically rigorous methods:

- **Pattern detectors**: ORB (15/30/60 min), gap fade, VWAP reversion
- **Excursion analysis**: MFE/MAE computation + random-entry comparison (KS test)
- **Significance**: binomial, t-test, PSR, DSR (multi-testing deflated), MinTRL
- **Monte Carlo**: bootstrap CI, permutation test, White's Reality Check
- **Walk-forward**: rolling edge stability
- **Regime conditioning**: ATR/trend classification + Hurst exponent + Lo-MacKinlay variance ratio test

Entry point: `python scripts/validate_edge.py`.

## IBKR Toolkit CLIs

Two standalone CLIs complement the main trading engine:

### `trading_algo.ibkr_tool` — 46-command data/ops CLI

```bash
python -m trading_algo.ibkr_tool <command> [--format json|csv|table] [args]
```

Uses client id **177** by default so it never collides with the trading
engine (client id 7). Reads `IBKR_HOST/IBKR_PORT` from `.env`.

**Commands:**

| Group | Commands |
|---|---|
| **Meta** | `connect`, `time`, `accounts`, `user-info` |
| **Account** | `summary`, `values`, `positions`, `portfolio`, `pnl`, `pnl-single` |
| **Quotes** | `quote`, `quotes` (batch via `reqTickers`), `stream`, `fx` |
| **Depth / realtime** | `depth`, `depth-exchanges`, `realtime-bars`, `ticks` (tick-by-tick) |
| **Historical** | `history`, `history-ticks`, `head-timestamp`, `histogram`, `schedule` |
| **Options** | `chain`, `chain-quote`, `calc-iv`, `calc-price` |
| **Discovery** | `search`, `contract`, `smart-components`, `market-rule` |
| **Fundamentals** | `fundamentals` (ReportsFinSummary, ReportSnapshot, RESC, etc.) |
| **News** | `news-providers`, `news`, `article`, `news-bulletins` |
| **Scanner** | `scanner-params`, `scan` |
| **Orders** | `open-orders`, `completed-orders`, `executions`, `whatif`, `place`, `combo`, `cancel`, `cancel-all` |
| **WSH** | `wsh-meta` |

Order-placing commands require an explicit `--yes` flag.

Examples:
```bash
python -m trading_algo.ibkr_tool summary --account U12345678 --format table
python -m trading_algo.ibkr_tool quote --kind OPT --symbol NVDA --expiry 20260515 --right C --strike 195
python -m trading_algo.ibkr_tool chain-quote --symbol NVDA --expiry 20260515 --min-strike 190 --max-strike 200
python -m trading_algo.ibkr_tool executions --account U12345678 --format csv
python -m trading_algo.ibkr_tool whatif --kind STK --symbol NVDA --side BUY --qty 100 --type LMT --limit-price 150 --account U12345678
```

### `trading_algo.flex_tool` — 31-command Flex Web Service CLI

```bash
python -m trading_algo.flex_tool <command> [--format json|csv|table] [args]
```

Reads `IBKR_FLEX_TOKEN` and `IBKR_FLEX_QUERY_ACTIVITY` from `.env`. Caches
XML to `data/flex/` (gitignored).

**Commands:**

| Group | Commands |
|---|---|
| **Meta** | `list-queries`, `send`, `cached`, `latest`, `info`, `accounts`, `auto`, `grep` |
| **Sections** | `trades`, `open-positions`, `prior-positions`, `cash-report`, `cash`, `change-in-nav`, `change-in-position-values`, `complex-positions`, `conversion-rates`, `corporate-actions`, `securities-info`, `stmt-funds`, `transfers`, `slb-activities` |
| **Analytics** | `pnl-by-symbol`, `pnl-by-account`, `commissions-total`, `dividends`, `interest`, `fees`, `summary`, `symbols` |

Every section command supports `--account`, `--symbol` (matches underlying
too, so `--symbol AMZN` catches all AMZN options), `--date-from`,
`--date-to`, `--asset-category`, `--currency`, `--limit`.

Examples:
```bash
python -m trading_algo.flex_tool send                          # fetch fresh XML
python -m trading_algo.flex_tool summary                       # NAV, P&L, TWR per account
python -m trading_algo.flex_tool pnl-by-symbol --top 20
python -m trading_algo.flex_tool trades --symbol AMZN --account U12345678
python -m trading_algo.flex_tool cash-report --base-only
python -m trading_algo.flex_tool auto                          # send + cache + summary in one shot
```

## Enterprise Backtest Methodology

Both runners implement the same integrity guarantees:

1. **Next-bar-open execution**: Signals generated on bar N are queued and fill at bar N+1's OPEN price. Eliminates look-ahead bias from same-bar execution.
2. **Natural equity compounding**: Position sizes scale with current equity. Returns `r_t = equity_t / equity_{t-1} - 1` are stationary, making Sharpe ratios meaningful.
3. **VWAP position tracking**: When adding to an existing position in the same direction, entry price is the volume-weighted average of all fills (not overwritten with latest).
4. **Backward-only data**: All funding rate, spot price, and OI lookups search only backward in time. Verified by fraud detection suite.
5. **Crypto-specific**: Settlement-aligned funding (00:00/08:00/16:00 UTC), leverage-aware liquidation simulation, margin accounting on both longs and shorts.
6. **Fraud detection suite**: Random signal null test (infrastructure must produce SR~0), reversed signal test, cost sensitivity, single-asset isolation, buy-and-hold benchmark comparison.

## How to Run

### Prerequisites

- Python 3.11+
- For equity backtesting: cached IBKR 5-min data in `ibkr_data_cache/`
- For crypto backtesting: internet connection (CCXT) or cached data
- For live trading: TWS or IB Gateway running (see ports below)
- For Flex CLI: IBKR Flex Web Service token + Activity query configured in
  Account Management (see `docs/` for setup)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Equity Backtest (10-year)

```bash
python scripts/run_10yr_backtest.py
```

Outputs to `backtest_results/`:
- `10yr_validated_report.txt` — full text report
- `10yr_validated_report.json` — machine-readable
- `10yr_equity_curve.csv`
- `10yr_monthly_returns.csv`

### Run Crypto Backtest

```bash
python crypto_alpha/scripts/deep_analysis.py
```

Comprehensive analysis: crisis periods, rolling Sharpe, walk-forward, monthly
breakdown, per-edge solo performance, correlation matrix.

### Run Crypto Fraud Detection

```bash
python crypto_alpha/scripts/fraud_detection.py
```

8 integrity tests including random signals, reversed signals, doubled costs,
single-asset, and buy-and-hold comparison.

### Run Edge Validation

```bash
python scripts/validate_edge.py --symbols NQ,ES --fetch
```

Statistical validation for ORB/gap/VWAP patterns on futures data. Writes a
PASS/WEAK/FAIL verdict per pattern with the full stat battery.

### Live Trading (IBKR)

```bash
# Paper trading via IB Gateway (port 4002)
python run.py --paper

# Live account via IB Gateway (port 4001, requires confirmation for every order)
python run.py --live

# Explicit port override
python run.py --ibkr-port 4001

# Dry run (stage orders only, no execution)
python run.py --dry-run
```

#### Connection Shortcuts

| Flag | Port | Gateway mode | Notes |
|------|------|------|-------|
| `--paper` | 4002 | IB Gateway paper | Default safe mode |
| `--live`  | 4001 | IB Gateway live  | Implies `--allow-live`, requires YES confirmation |
| `--ibkr-port <N>` | custom | any | Overrides `--paper`/`--live` |

TWS uses ports 7497 (paper) / 7496 (live); IB Gateway uses 4002 / 4001. Most
of the live trading in this repo runs through IB Gateway via IBC in tmux with
`AutoRestartTime=23:55` for 24/7 availability through IBKR's nightly
disconnect.

Both `--paper` and `--live` can run simultaneously on different ports,
allowing paper testing and live analysis in parallel.

#### Live Account Safety Guards

The system supports connecting to live IBKR accounts with multiple layers of
protection:

| Guard | Description |
|-------|-------------|
| **`TRADING_ALLOW_LIVE`** | Env var must be `true` to allow live connections (default: `false`) |
| **`--allow-live`** | CLI flag must be passed explicitly |
| **`require_paper`** | Automatically enforced unless `allow_live` is set |
| **Interactive confirmation** | Every `place_order`, `modify_order`, `cancel_order`, and `place_bracket_order` requires typing `YES` at the prompt |
| **Callback architecture** | The `live_confirm_callback` is injected via the CLI — no callback means all orders are blocked |

When connected to a live account, every mutating operation displays a
confirmation prompt:

```
============================================================
  *** LIVE ACCOUNT ORDER CONFIRMATION ***
  Action: PLACE ORDER
  BUY 10 STK NVDA @ LIMIT limit=120.00 tif=DAY
============================================================
Type 'YES' to confirm:
```

Environment variables for live trading:

```bash
TRADING_ALLOW_LIVE=true       # Allow live account connections
TRADING_LIVE_ENABLED=false    # Additional gate (not required for allow_live)
TRADING_REQUIRE_PAPER=false   # Disable paper-only enforcement
IBKR_PORT=4001                # IB Gateway live API port (4002 for paper)
IBKR_CLIENT_ID=7              # Trading engine client id

# Flex Web Service (for the flex_tool CLI)
IBKR_FLEX_TOKEN=...
IBKR_FLEX_QUERY_ACTIVITY=...

# Optional: Gemini LLM trader / chat
LLM_ENABLED=true
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3-pro-preview
```

See `.env.example` for the full list with defaults.

## Key Learnings

- **Vol gating is critical**: Intraday signals help in high-vol periods but devastate low-vol. Gate on 20-day annualized vol > 15%.
- **Vol targeting beats Kelly**: Vol is 10× easier to forecast than returns. Size by inverse vol, not estimated edge.
- **Regime detection**: Simple vol thresholds capture 80%+ of HMM/ML classifier value.
- **Diversification math**: `SR_combined = sqrt(N) * avg_SR / sqrt(1 + (N-1)*rho)`. Breadth matters.
- **Alpha decay is real**: 32%+ post-publication for academic anomalies. Structural edges (funding arb, momentum) survive; statistical patterns don't.
- **Transaction costs kill**: Of 120 published anomalies, <15 survive realistic costs (Novy-Marx & Velikov, 2016).
- **Don't chase Sharpe**: With 5 strategies on 7 symbols at 5-min frequency, SR ~1.0 is the practical ceiling.
- **Don't roll a working short**: Cycling short-DTE positions dominates a single long-DTE position on per-day theta capture. Only roll on defensive triggers (ITM, delta > 0.40, IV spike, gap, margin). See `CLAUDE.md` for the full options-management ruleset.

## Known Limitations

- Crypto system underperforms BTC buy-and-hold (SR 0.28 vs 0.80)
- `_strategy_positions` in controller never populated — per-strategy position limits dormant
- Equity trailing stops disabled in V11 (degrades Sharpe on all tested configurations)
- Crypto funding rate data may have gaps depending on exchange API availability
- ATLAS v7 training requires the R3000 dataset (not shipped in repo)
- Edge validation needs `pandas>=2.0.0` and `statsmodels>=0.14.0`

## Project Rules

`CLAUDE.md` at the repo root documents non-obvious trading rules,
paper/live port conventions, and a self-improvement log of corrections to
prevent repeat mistakes. Read it before touching options positions or
routing live orders.

## Further Reading

`docs/` contains:
- `ARCHITECTURE.md` — deeper system architecture
- `SAFETY.md` — live-trading safety model
- `LLM_TRADER.md` — Gemini trader loop + chat
- `WORKFLOWS.md` — common dev/ops workflows
- `DB_SCHEMA.md` — sqlite audit schema
- `NOVEL_ALGORITHM_PROPOSAL.md` / `RAT` research docs
- `TRADITIONAL_VS_AI_TRADING_VERDICT.md` — research comparison
- `HOW_LLMS_WERE_USED_IN_RESEARCH.md` — critique of LLM-assisted alpha discovery literature

See `CHANGELOG.md` for the full dated commit history.
