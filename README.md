# trading-algo

Multi-strategy quant trading system I've been building for a while. Runs on US stocks through Interactive Brokers, and on crypto through Binance, Kraken, and Hyperliquid. Runs a bunch of strategies at once, has its own backtester, and can go live through IBKR with heavy safety wiring in the way.

## What's inside

The equity side has around a dozen strategies that all plug into one controller. Momentum, mean reversion, cointegrated pairs, opening range breakout, flow-pressure patterns, regime transitions, Hurst-based adaptive allocation, and a few more. The controller blends their signals, resolves conflicts, sizes positions, and enforces risk limits.

The crypto side has nine edges aimed at perpetual futures. Most are structural (funding rate dynamics, perp-spot basis mean reversion, cross-exchange price lag, intermarket cascades from BTC leading ETH leading alts, that sort of thing). Being honest, only three of the nine have positive standalone Sharpe. The other six exist because I wanted to test whether you could stack uncorrelated negative-SR edges into something useful. You can't really, but the tests had to happen.

There are options income strategies in there too. Wheel, PMCC, enhanced wheel, jade lizard, portfolio wheel, hybrid regime, defined-risk put spreads. They all run on the same backtester and can go live through IBKR.

ATLAS is an experimental deep-RL trader with a Mamba state-space backbone and cross-attention, trained with PPO and EWC on curriculum episodes. Currently v7. Lives in `trading_algo/quant_core/models/atlas/`, with training scripts in `scripts/atlas_*.py`.

On the tooling side there are two big IBKR CLIs. `trading_algo.ibkr_tool` covers around 46 commands across IBKR's API. Quotes for stocks and options, option chains, depth of book, historical bars, tick data, fundamentals, news, market scanners, order placement, what-if margin calculations. `trading_algo.flex_tool` does the Flex Web Service side. Account statements, trades, P&L rolled up by symbol or account, cash reports, commissions, dividends, transfers. Both output json, csv, or a plain table.

There's a Gemini chat and trader loop in `trading_algo/llm/`. I used to drive live trades through it directly. I stopped because the research on LLM-generated trading signals just isn't there yet. The chat side still works if you want to ask the model for market context or poke at positions.

RAT ("Reflexive Attention Topology") is experimental research code. Attention tracking, Soros-style reflexivity detection via Granger causality, topological regime classification, adversarial algo fingerprinting, alpha decay monitoring. Mostly a playground for ideas I wanted to try.

## How honest the backtester is

This got a lot of attention because the whole system is worthless if the backtests lie.

Signals generated on bar N fill at the open of bar N+1, so there's no same-bar look-ahead. When you add to an existing position in the same direction, the entry price becomes the VWAP of all fills, not the latest one. All data lookups (prices, funding rates, open interest) go backward in time only. Commissions and slippage are realistic, around $0.0035 per share and 2 basis points. Walk-forward validation runs sequential folds and wants less than 30% in-sample to out-of-sample degradation before it accepts a config. Overfitting gets checked with PBO and deflated Sharpe (which accounts for multiple testing), plus White's reality check.

The crypto runner adds a few things on top. Funding settles at the correct UTC hours. Leverage is tracked properly. Liquidations get simulated with a 0.5% penalty. Returns annualize on 365 days instead of 252.

There's also a fraud detection suite for the crypto side. Random signals should produce Sharpe near zero once costs are subtracted. Reversed signals should flip the sign of PnL. Doubling the costs should crush the edges. Single-asset isolation should show where alpha is coming from. If any of those tests fail, the infrastructure is lying and the rest of the numbers don't matter.

## Running it

Python 3.11+. Install and you're set.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ten-year equity backtest.

```bash
python scripts/run_10yr_backtest.py
```

Crypto deep analysis.

```bash
python crypto_alpha/scripts/deep_analysis.py
```

Crypto fraud detection.

```bash
python crypto_alpha/scripts/fraud_detection.py
```

Edge validation for ORB, gap fade, and VWAP patterns on futures.

```bash
python scripts/validate_edge.py --symbols NQ,ES --fetch
```

Results land in `backtest_results/` as text reports, JSON, equity curves, and monthly breakdowns.

## Going live

Run paper first. Paper goes through IB Gateway on port 4002.

```bash
python run.py --paper
```

For live, you have to clear a few gates. `TRADING_ALLOW_LIVE` has to be true in the environment. `--allow-live` has to be on the command line. Paper-only enforcement has to be explicitly disabled. Every order (place, modify, cancel, bracket) prompts you to type `YES` at the terminal before anything transmits. If no confirmation callback is wired into the CLI, orders get blocked outright.

```bash
python run.py --live
```

Most of my live trading runs through IB Gateway via IBC inside tmux, with AutoRestartTime set to 23:55 so the gateway comes back up through IBKR's nightly disconnect. Paper and live can run simultaneously on their own ports if you want to paper-test a new strategy while existing positions stay live.

## Latest equity results (V11 config)

Sharpe 0.48. Total return 151.3% over ten years (2016 to 2026). Max drawdown 18.5%. Five strategies on seven symbols with vol-gated intraday signals. Deflated Sharpe is 16 with p ≈ 0. Out-of-sample Sharpe is 0.55. Alpha versus SPY is about 4% annualized. Beta is 0.12.

Crypto is rougher. Sharpe 0.28. Return 7.7% over four years (2022 to 2026). Max drawdown 21.8%. That's below BTC buy-and-hold's 0.80 Sharpe, which is fine. I'm not trying to beat hodling BTC, I'm trying to generate uncorrelated PnL. Only three of the nine edges are positive as standalones. IMC (+0.72), CED (+0.40), PBMR (+0.33).

## Things I've learned

Vol targeting beats Kelly for position sizing. Vol is maybe ten times easier to forecast than returns, and sizing by inverse vol sidesteps the whole "my edge estimates are wrong" problem.

Intraday signals are a high-vol regime thing. In low-vol they actively hurt you. Gating on 20-day annualized vol above 15% cleans that up.

Simple vol thresholds catch most of what a full HMM regime classifier gives you. The last sliver of accuracy isn't worth the complexity.

Alpha decay is real and public. Academic anomalies decay about a third post-publication (McLean & Pontiff, 2016). Structural edges like momentum and funding arb survive. Pure statistical patterns mostly don't.

Out of 120 published anomalies, fewer than 15 survive realistic transaction costs (Novy-Marx & Velikov, 2016). Most published alpha is dead on arrival once real costs are in, which is why backtests need honest slippage baked in from the start.

Don't roll a working short option. Cycling shorter-dated shorts captures more per-day theta than one long-dated short. The math is in `CLAUDE.md`. Only roll defensively (short goes ITM, delta past 0.40, IV spike, gap against you, or margin needed elsewhere).

## Known issues

The crypto system underperforms BTC buy-and-hold. I know. It's a diversification play, not a BTC replacement, but the headline comparison looks worse than it really is.

`_strategy_positions` in the controller never gets populated, so per-strategy position limits are dormant. Something to fix.

Equity trailing stops are off in V11 because they hurt Sharpe on every config I tested. I'll turn them back on if I find a better trigger.

Crypto funding data has gaps depending on which exchange API it pulls from. Usually minor but worth knowing.

ATLAS v7 training needs the R3000 dataset, which isn't shipped in the repo.

Edge validation wants `pandas>=2.0.0` and `statsmodels>=0.14.0`.

## Docs

More detail lives in `docs/`.

- `ARCHITECTURE.md` is the deep dive
- `SAFETY.md` is the full live-trading safety model
- `LLM_TRADER.md` covers the Gemini loop and chat interface
- `WORKFLOWS.md` is the day-to-day
- `DB_SCHEMA.md` is the sqlite audit schema
- `TRADITIONAL_VS_AI_TRADING_VERDICT.md` is why the LLM direction got shelved
- `HOW_LLMS_WERE_USED_IN_RESEARCH.md` is a critique of LLM-assisted alpha discovery papers
- `CLAUDE.md` at the repo root has the non-negotiable trading rules

`CHANGELOG.md` has the full commit history.
