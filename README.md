# Quantitative Trading System

Multi-asset quantitative trading system with enterprise-grade backtesting infrastructure. Equity strategies on IBKR 5-minute data (10 years) and crypto edges on Binance perpetual futures.

## Results

All numbers from enterprise-grade backtesting: next-bar-open execution, VWAP position tracking, backward-only data lookups, and validated by randomized signal null tests.

| System | Sharpe | Return | Max DD | Period | Notes |
|--------|--------|--------|--------|--------|-------|
| **Equity V11** | +0.480 | +151.3% | 18.5% | 2016-2026 | 5 strategies, 7 symbols, vol-gated intraday signals |
| **Crypto 9-Edge** | +0.277 | +7.7% | 21.8% | 2022-2026 | 9 edges, 5 perps, below BTC buy-and-hold (+0.80) |

**Equity integrity**: DSR 15.988 (p=0.000), OOS SR +0.547, alpha vs SPY +4.06% annualized. Beta 0.119. Enterprise runner with next-bar-open execution eliminates same-bar look-ahead bias.

**Crypto integrity**: Fraud detection v2 confirms honest infrastructure. Random signals produce SR -0.38 (near zero minus costs). Only 3/9 edges individually positive: IMC (+0.72), CED (+0.40), PBMR (+0.33).

## Architecture

```
randomThings/
├── trading_algo/                    # Equity trading system
│   ├── multi_strategy/
│   │   ├── controller.py            # Multi-strategy controller (shared)
│   │   ├── protocol.py              # StrategySignal protocol (shared)
│   │   ├── backtest_runner.py       # Enterprise equity backtest runner
│   │   └── adapters/                # 14 strategy adapters
│   ├── quant_core/strategies/       # Core strategy implementations
│   ├── orchestrator/                # Original 6-edge orchestrator
│   └── broker/                      # IBKR + simulation brokers (live account safety guards)
├── crypto_alpha/                    # Crypto trading system
│   ├── edges/                       # 9 crypto-native edge implementations
│   ├── adapters/                    # Edge → controller adapters
│   ├── backtest/
│   │   └── crypto_runner.py         # Enterprise crypto backtest runner
│   ├── data/
│   │   ├── ccxt_loader.py           # CCXT data downloader with caching
│   │   └── cache_manager.py         # Numpy archive cache
│   ├── scripts/
│   │   ├── fraud_detection.py       # Infrastructure integrity tests
│   │   └── deep_analysis.py         # Comprehensive backtest analysis
│   └── types.py                     # CryptoBar dataclass
├── backtest/
│   └── metrics.py                   # Shared metrics (Sharpe, Sortino, VaR, etc.)
├── scripts/
│   └── run_10yr_backtest.py         # 10-year equity backtest entry point
└── ibkr_data_cache/                 # Cached IBKR 5-min bars (gitignored)
```

Both systems share the same `MultiStrategyController` and `StrategySignal` protocol. Strategies implement a common adapter interface that maps edge-specific logic to unified signals.

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

### Crypto Edges (9 edges)

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

- Python 3.10+
- For equity backtesting: cached IBKR 5-min data in `ibkr_data_cache/`
- For crypto backtesting: internet connection (downloads via CCXT) or cached data
- For live trading: TWS or IB Gateway running

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

Runs comprehensive analysis: crisis periods, rolling Sharpe, walk-forward, monthly breakdown, per-edge solo performance, correlation matrix.

### Run Crypto Fraud Detection

```bash
python crypto_alpha/scripts/fraud_detection.py
```

8 integrity tests including random signals, reversed signals, doubled costs, single-asset, and buy-and-hold comparison.

### Live Trading (IBKR)

```bash
# Paper trading (dry run)
python run.py --dry-run

# Paper trading (live orders on paper account)
python run.py

# Live account (requires interactive confirmation for every order)
python run.py --allow-live
```

Requires TWS/Gateway running on port 7497 (paper) or 7496 (live).

#### Live Account Safety Guards

The system supports connecting to live IBKR accounts with multiple layers of protection:

| Guard | Description |
|-------|-------------|
| **`TRADING_ALLOW_LIVE`** | Env var must be `true` to allow live connections (default: `false`) |
| **`--allow-live`** | CLI flag must be passed explicitly |
| **`require_paper`** | Automatically enforced unless `allow_live` is set |
| **Interactive confirmation** | Every `place_order`, `modify_order`, `cancel_order`, and `place_bracket_order` requires typing `YES` at the prompt |
| **Callback architecture** | The `live_confirm_callback` is injected via the CLI — no callback means all orders are blocked |

When connected to a live account, every mutating operation displays a confirmation prompt:

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
IBKR_PORT=7496                # TWS live API port (7497 for paper)
```

## Key Learnings

- **Vol gating is critical**: Intraday signals help in high-vol periods but devastate low-vol. Gate on 20-day annualized vol > 15%.
- **Vol targeting beats Kelly**: Vol is 10x easier to forecast than returns. Size by inverse vol, not estimated edge.
- **Regime detection**: Simple vol thresholds capture 80%+ of HMM/ML classifier value.
- **Diversification math**: SR_combined = sqrt(N) * avg_SR / sqrt(1 + (N-1)*rho). Breadth matters.
- **Alpha decay is real**: 32%+ post-publication for academic anomalies. Structural edges (funding arb, momentum) survive; statistical patterns don't.
- **Transaction costs kill**: Of 120 published anomalies, <15 survive realistic costs (Novy-Marx & Velikov, 2016).
- **Don't chase Sharpe**: With 5 strategies on 7 symbols at 5-min frequency, SR ~1.0 is the practical ceiling.

## Known Limitations

- Crypto system underperforms BTC buy-and-hold (SR 0.28 vs 0.80)
- `_strategy_positions` in controller never populated — per-strategy position limits dormant
- Equity trailing stops disabled in V11 (degrades Sharpe on all tested configurations)
- Crypto funding rate data may have gaps depending on exchange API availability

## License

MIT
