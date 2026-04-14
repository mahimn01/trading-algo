# Changelog

Complete project history, regenerated from `git log` on 2026-04-14. Every
commit on `main` (and every commit pending merge) is listed with its author
date and short SHA.

## 2026-04

### Unreleased — IBKR CLIs + edge validation + ATLAS v7

- **2026-04-14** `7f2a289` Add project trading rules (CLAUDE.md): paper/live port conventions, IBKR API usage rules, risk limit policy, self-improvement log.
- **2026-04-14** `f3807d8` ATLAS v7: new model architecture (attention_v7, backbone_v7, config_v7, fusion_v7, model_v7, vsn_v7) + fast_env, fast_hindsight, MLX model + training scripts for v3–v7 including IBKR curriculum and R3000 data pipeline.
- **2026-04-14** `6dca538` Add edge validation infrastructure: statistical pipeline for candidate futures day-trading patterns (ORB, gap fade, VWAP reversion) with MFE/MAE excursion analysis, probabilistic/deflated Sharpe, Monte Carlo permutation, walk-forward stability, regime-conditional testing.
- **2026-04-14** `9c98956` Add comprehensive IBKR data CLI (`trading_algo.ibkr_tool`, 46 commands: accounts, positions, PnL streaming, quotes with greeks, chains, depth, realtime bars, tick-by-tick, history, fundamentals, news, scanner, executions, whatIf preview, combo orders, FX, histogram, head timestamp) and Flex Web Service CLI (`trading_algo.flex_tool`, 31 commands: send+poll+cache, then parse every FlexQueryResponse section with per-section filters plus PnL-by-symbol/account aggregation).
- **2026-04-14** `7fcc443` chore(gitignore): exclude ATLAS data caches, checkpoints, Flex XML exports (13 GB+ of local state kept off origin).
- **2026-04-02** `270480d` ATLAS curriculum training: regime-adaptive behavior achieved.
- **2026-04-01** `6ef0d23` ATLAS v2: fix training pipeline, BSM-based environment + evaluation.

## 2026-03

- **2026-03-29** `553195a` Complete ATLAS training pipeline: PPO, EWC, inference, validation.
- **2026-03-29** `f597a2d` Add ATLAS model: hybrid Mamba-SSM + Cross-Attention trading transformer.
- **2026-03-28** `3137cd7` Evolve options strategies: hybrid regime, defined-risk, live trading, monitoring.
- **2026-03-28** `330fe00` Merge pull request #22 from mahimn01/options-income-strategies.
- **2026-03-25** `1075e49` Add options income strategies: Wheel, PMCC, and Enhanced Wheel.
- **2026-03-20** `5464622` Merge pull request #21 from mahimn01/paper-live-port-flags.
- **2026-03-20** `a9e96df` Add `--paper` and `--live` CLI flags for quick port switching.
- **2026-03-17** `dcdab9c` Merge pull request #20 from mahimn01/live-account-safety-guards.
- **2026-03-17** `5bf0aa1` Add live account support with multi-layer safety guards: `TRADING_ALLOW_LIVE`, `--allow-live`, interactive YES confirmation on every mutating call, callback architecture.
- **2026-03-15** `7417199` Merge pull request #19 from mahimn01/ibkr-market-scanner.
- **2026-03-15** `631e54c` Add IBKR market-wide scanner for dynamic stock discovery.
- **2026-03-05** `8091ad8` Merge pull request #18 from mahimn01/upgrade-ib-async.
- **2026-03-05** `cb35668` Upgrade `ib_insync` → `ib_async` (actively maintained fork).
- **2026-03-03** `0657b84` Merge pull request #17 from mahimn01/enterprise-backtest-v2.
- **2026-03-03** `2506cbe` Comprehensive README, updated backtest results, project config.
- **2026-03-03** `cadc9bb` Add crypto alpha multi-edge trading system (9 edges, enterprise runner).
- **2026-03-03** `cadb24a` Enterprise backtest: next-bar-open execution, VWAP tracking, shared metrics.
- **2026-03-02** `03f1144` V11: vol-gated intraday signals, parallel execution, trailing stop infrastructure (#16).

## 2026-02

- **2026-02-26** `be43ca8` Fix Sortino ratio double-annualization in backtest_runner.
- **2026-02-26** `5dfaf32` Fix PairsTrading `IndexError` when price arrays have different lengths (#15).
- **2026-02-26** `c522928` Merge pull request #14 from mahimn01/claude/phase4-5-ml-validation.
- **2026-02-26** `168d7a4` 10-year enterprise backtest: IBKR data + validated results.
- **2026-02-25** `8eb7cfb` Merge pull request #13 from mahimn01/claude/phase4-5-ml-validation.
- **2026-02-25** `2d2a995` Fix 5 metric bugs, add institutional metrics, build 10yr enterprise backtest.
- **2026-02-25** `c595d67` Merge pull request #12 from mahimn01/claude/phase4-5-ml-validation.
- **2026-02-25** `9d4a28f` Phase 4-5: ML signal framework, walk-forward validation, critical bug fixes.
- **2026-02-19** `8491842` Merge pull request #11 from mahimn01/claude/volatile-backtest-results.
- **2026-02-19** `d9c1878` Add volatile backtest suite, results, and IBKR cached data.
- **2026-02-18** `7253fd2` Merge pull request #10 from mahimn01/claude/backtest-fixes.
- **2026-02-18** `0ec91a0` Fix 6 critical backtest bugs; add comprehensive strategy backtest suite.
- **2026-02-18** `9fff2f6` Merge pull request #9 from mahimn01/claude/novel-pattern-discovery-engine.
- **2026-02-18** `d1e81f3` Novel pattern discovery engine: 8 strategy modules, 4 adapters, full integration.
- **2026-02-18** `5742fd9` Fix Sharpe calculation for zero-variance edge case.
- **2026-02-18** `62ba369` Merge pull request #8 from mahimn01/claude/market-pattern-discovery-ZNxBU.
- **2026-02-18** `72041a6` Fix Sharpe calculation for zero-variance edge case.
- **2026-02-09** `def7ee8` Merge pull request #7 from mahimn01/claude/analyze-algorithm-structure-A1kov.
- **2026-02-09** `69f9943` Phase 4-5: backtesting validation, walk-forward analysis, regime-adaptive allocation.
- **2026-02-09** `71d39a3` Phase 3: research-backed alpha sources and portfolio vol management.
- **2026-02-09** `ac442e4` Phase 2: multi-strategy portfolio controller with 4 strategy adapters.
- **2026-02-09** `26fb067` Phase 1: aggressive position sizing, Kelly criterion, leverage for 25-50% annual returns.
- **2026-02-07** `3064986` Merge pull request #6 from mahimn01/claude/review-codex-algorithm-Fdd3c.
- **2026-02-07** `8d3c2d6` Update review doc with implementation status for all 5 priorities.
- **2026-02-07** `8cd02b4` Bridge `quant_core` into production Orchestrator with full infrastructure.
- **2026-02-07** `a88fe28` Add review of Codex algorithm changes with next steps.
- **2026-02-03** `8910505` Reconcile trade PnL with equity and fix daily stats marking.
- **2026-02-03** `4dea07f` Fix backtest trade accounting and export reconciliation.
- **2026-02-03** `b63a0a3` Improve return quality across orchestrator and quant engine.
- **2026-02-03** `6f2653a` Merge pull request #5 from mahimn01/algorithm/quantitative-framework-v2.
- **2026-02-03** `25f58c5` Add comprehensive quantitative trading framework.
- **2026-02-01** `1ad495e` Merge pull request #4 from mahimn01/feature/ibkr-speed-optimizations.
- **2026-02-01** `2618b32` Add enterprise-grade IBKR speed optimizations.
- **2026-02-01** `84f59ec` Fix graceful shutdown to handle Ctrl+C properly.
- **2026-02-01** `ccabdeb` Merge pull request #3 from mahimn01/feature/trading-dashboard.
- **2026-02-01** `ae9ff3c` Use real IBKR data for backtests, fix dashboard integration.

## 2026-01

- **2026-01-31** `b4de32f` Fix BacktestEngine usage and add comprehensive tests.
- **2026-01-31** `0f03097` Fix `BacktestConfig` parameter name and make backtest panel scrollable.
- **2026-01-31** `b3d8501` Add enterprise-level backtest system with dashboard integration.
- **2026-01-31** `843573e` Fix trades widget tab panes.
- **2026-01-31** `61a5c08` Add enterprise-level trading dashboard with TUI interface.
- **2026-01-31** `38482f7` Merge pull request #2 from mahimn01/claude/ai-async-trading-analysis-epxMW.
- **2026-01-31** `4ab98b5` Merge main into feature branch (keeping our changes).
- **2026-01-31** `9618438` Reorganize repository: modularize Orchestrator and archive old code.
- **2026-01-30** `67c4c32` Fix division by zero after market close and add auto-stop.
- **2026-01-30** `cbcb80b` Add Orchestrator: multi-edge ensemble day trading system.
- **2026-01-27** `ed39b8a` Fix division by zero after market close and add auto-stop.
- **2026-01-26** `001d8d3` Add multi-market support (NYSE, HKEX, TSE, LSE, ASX).
- **2026-01-26** `7bd634a` Add lunch break (12-1pm) to day trader time-of-day filters.
- **2026-01-26** `8b172f2` Upgrade `ChameleonDayTrader` to v2 with adaptive risk management.
- **2026-01-22** `99da876` Add AI-driven day trading stock selector with multi-factor analysis.
- **2026-01-22** `e9715fa` Add aggressive day trading system with intraday backtester.
- **2026-01-22** `6472496` Add data CSV files to gitignore.
- **2026-01-22** `67e352c` Add IBKR real data backtest for Chameleon Strategy.
- **2026-01-16** `a96723b` Add Chameleon Strategy: adaptive alpha in both bull and bear markets.
- **2026-01-16** `0defc13` Add realistic market simulation backtest.
- **2026-01-16** `3ddfde4` Add research-backed trading enhancements to RAT framework.
- **2026-01-15** `8942378` Merge pull request #1 from mahimn01/claude/ai-async-trading-analysis-epxMW.
- **2026-01-15** `0e5e9af` Add IBKR data pull and RAT trading scripts.
- **2026-01-15** `febbd88` Add backtest runner and fix missing exports.
- **2026-01-15** `71d278a` Add comprehensive RAT framework tests and fix 4 bugs.
- **2026-01-15** `3644775` Complete RAT framework: enterprise-grade quantitative trading system.
- **2026-01-15** `9cad8bc` Add RAT framework core modules: adversarial detector and alpha tracker.
- **2026-01-14** `89560f1` Add RAT: Reflexive Attention Topology — genuinely novel trading framework.
- **2026-01-14** `3354b47` Add critical analysis of how LLMs were used in trading research.
- **2026-01-14** `28116c1` Add comprehensive Traditional vs AI trading analysis with profit verdict.
- **2026-01-14** `02fe8e0` Add comprehensive AI async trading analysis and novel architecture proposal.
- **2026-01-14** `6961fff` Gemini: fix duplicate thoughts, smarter search prefetch.

## 2025-12

- **2025-12-26** `644eefa` Upgrade Gemini chat TUI + function calling (streamed thought summaries, Google Search/URL context/code execution tools, structured outputs, official google-genai SDK, token-aware history compaction, explicit caching).
- **2025-12-26** `8276b52` Fix chat UI output for streamed JSON.
- **2025-12-26** `147eaa7` Fix Gemini SSE streaming parsing and chat fallback.
- **2025-12-26** `434f032` Harden chat against Gemini HTTP errors.
- **2025-12-26** `7fb4d70` Clarify Gemini Google Search grounding keys.
- **2025-12-26** `1a717d2` Enforce Gemini 3 model for LLM features.
- **2025-12-26** `d8be423` Add interactive Gemini streaming chat with OMS tools.
- **2025-12-26** `8b98b42` Add optional Gemini LLM trader loop.
- **2025-12-24** `5ff334f` Add IBKR historical export + deterministic backtests.
- **2025-12-24** `1e76f50` Remove README next steps.
- **2025-12-24** `931d358` Initial IBKR paper OMS skeleton.
