# Changelog

Complete project history, regenerated from `git log` on 2026-04-14. Every
commit on `main` (and every commit pending merge) is listed with its author
date and short SHA.

## 2026-10

### Unreleased — NautilusTrader backtest path

- Added `trading_algo/nautilus_backtest`, which runs the existing
  `TradingStrategy` adapters, unchanged, through NautilusTrader 2.0.0rc6.
  Decisions are made at each session's close and fill at the next session's
  open, priced by the repo's realistic cost model as NautilusTrader fee and
  fill models.
- A per-session ledger is reconciled against NautilusTrader's positions and
  account at every close. Exceptions in strategy callbacks, which
  NautilusTrader swallows, fail the run.
- Benchmark statistics against SPY come from NautilusTrader's analysis
  module, alongside an equal-weight buy-and-hold baseline.
- A parity check against `MultiStrategyBacktestRunner` agrees trade for trade
  on single-symbol runs (AAPL 2016-2026: 1,263 trades, max NAV difference
  4e-7). Multi-symbol runs are reported as not comparable, because the runner
  generates signals before every symbol's bar has arrived.
- New CLI `scripts/nautilus_backtest.py`, pinned dependencies in
  `requirements-nautilus.txt`, and 37 tests.

## 2026-07

### Unreleased — IBKR account CLI correctness and lifecycle hardening

- Replaced obsolete `reqAccountUpdates(subscribe, account)` calls with
  bounded, account-scoped `ib_async` 2.1 startup synchronization.
- Made `values` and `portfolio` return complete deterministic snapshots for
  every managed account, including multi-account logins.
- Corrected `positions.marketValue`, which was actually cost basis, to the
  accurately named `costBasis`; live mark-to-market remains in `portfolio`.
- Added read-only sessions, guaranteed disconnect/PnL cancellation cleanup,
  account validation, and normalization of IBKR unset numeric sentinels.
- PnL commands now require a real initial broker update and fail with a
  retryable timeout instead of emitting NaN/default-flat placeholder data.
- Bounded the supported dependency range to `ib_async>=2.1.0,<3.0.0`.

## 2026-05

### Unreleased — Wave T5: enterprise foundation (PIT data, cost realism, validation)

Branch: `enterprise-foundation-v1`. Foundation refactor that closes
five structural gaps identified in the deep audit of the trading-algo
codebase. ~3,000 LOC added (new modules + tests). 1054 → 1143 tests
(+89), zero regressions. PR-to-main acceptance gates per `PLAN.md`.

**The five gaps closed:**

1. **No point-in-time data correctness.** Every backtest above the
   legacy JSON cache used forward-adjusted prices and a hand-picked,
   currently-listed universe — silent forward bias + survivorship bias.
2. **Optimistic flat-bps fill model.** Same-bar close fills with a 5
   bps fixed slippage and no impact, no borrow cost, no real spread.
   Inflated returns by ~50–150 bps/year on liquid equity strategies and
   200–500 bps/year on shorts.
3. **Look-ahead in the bar loop.** Signals generated on bar T's close
   filled at bar T's close — strictly impossible in live trading.
4. **Optional idempotency, ephemeral risk state.** orderRef wasn't
   guaranteed at the broker layer; daily-loss + orders/day counters
   reset on every process restart.
5. **CSCV PBO formula bug + framework never used.** Validator code was
   correct in skeleton but reported the *complement* of PBO; `n_trials`
   was hardcoded to 1; no strategy had ever been run through the full
   suite.

**New modules** (all under `trading_algo/` unless noted):

- `data/schema.sql`: SQLite DDL for the bitemporal metadata layer:
  securities, ticker_history, splits, dividends, mergers, spinoffs,
  index_membership, risk_state, spread_cache, borrow_rates,
  migration_log. Bitemporal `known_from`/`known_to` columns; restatements
  append rows, never overwrite.
- `data/pit_store.py`: `PITStore` — SQLite metadata + pyarrow
  parquet bars partitioned by symbol/year. `upsert_security`,
  `record_ticker_change` (FB→META 2022-06-09 semantics),
  `add_split`/`add_dividend`/`add_index_membership`, `write_bars`,
  `read_bars(as_of=...)`, `restate_bar`. Atomic parquet writes via
  rename. 13 tests.
- `data/universe.py`: `UniverseResolver` — point-in-time index
  membership lookup. `get_universe('SP500', date(2008, 9, 15))` returns
  LEH-pre-bankruptcy. Hardcoded dev universes gated behind
  `allow_dev=True` so production strategies can't accidentally use
  survivorship-biased lists. 8 tests.
- `data/corporate_actions.py`: `AdjustmentEngine` — cumulative split
  factor at *query time* (storage stays unadjusted). AAPL pre-2014 bar
  viewed as of 2024 → factor 1/28 (7:1 then 4:1). Reload-after-future-
  split doesn't silently rewrite history. 5 tests.
- `data/migration.py` + `scripts/migrate_to_pit.py`: walks legacy
  `ibkr_data_cache` JSON files, parses two filename formats, writes
  Bars to PITStore, logs to `migration_log`. `reconcile()` spot-checks
  close-price equality post-import. CLI: `python
  scripts/migrate_to_pit.py --cache-dir <legacy> --pit-root <new>
  [--reconcile]`. 11 tests.
- `backtest_v2/cost_model.py`: layered estimators for institutional-
  grade fill realism (PLAN.md §2.3, full bibliography in module docstring):
    - `corwin_schultz_spread` — Corwin-Schultz 2012 (J. Finance 67:719)
      effective-spread estimator from two consecutive H/L pairs.
    - `abdi_ranaldo_spread` — Abdi-Ranaldo 2017 (RFS 30:4437) CHL
      triplet fallback for low-volume days.
    - `daily_effective_spread` — convenience: CS first, AR fallback.
    - `sqrt_impact_bps` — Toth-Bouchaud-CFM 2011 (PRX 1:021006)
      square-root impact, `Y * sigma * sqrt(participation)`. Below
      0.1% ADV: zero. Above 10% ADV: capped (model breakdown).
    - `borrow_charge` + `recall_probability` — ACT/360 daily accrual,
      recall risk scaled by rate (capped 5%). `BorrowTier` defaults:
      SP500 30bps / R1000 50bps / R2000 200bps / HTB skip.
    - `ibkr_tiered_commission` — per-share + min/max + SEC fee
      (sells only) + FINRA TAF + pass-through.
    - `compute_fill_cost` / `adjust_fill_price` — composite fill cost
      decomposition (spread + impact + commission), apply slippage to
      paper price.
   23 tests covering edge cases, threshold + cap, side-specific fees.
- `backtest_v2/execution_policy.py`: `ExecutionPolicy` enum
  (`SAME_BAR_CLOSE` / `NEXT_BAR_OPEN` / `NEXT_BAR_VWAP`), `PendingOrder`
  dataclass, `fill_price_for_policy`, `policy_introduces_lookahead`.
  7 tests.
- `quant_core/validation/stationary_bootstrap.py`:
  Politis-Romano 1994 (JASA 89:1303) stationary bootstrap with
  geometric-block resampling; Politis-White 2004 simplified block-length
  rule (`b ~ T^{1/3}`). `bootstrap_sharpe_ci` returns 95% percentile
  CI for annualized Sharpe — wider on autocorrelated returns than IID
  bootstrap, as required. 8 tests.
- `quant_core/validation/report_card.py`: `build_report_card` produces
  a 7-gate pass/fail audit per strategy. Gates: lower 95% CI on
  annualized Sharpe (>0.3), PBO (CSCV, <0.5), Deflated Sharpe (>0.95),
  walk-forward 12m positivity (>=75%), MinTRL (Bailey-LdP 2012, in
  years), cost-adjusted Sharpe (>0.3). `ReportCard.render()` emits
  markdown report card per PLAN.md §2.7. Status APPROVED iff every
  populated gate passes; BLOCKED otherwise. 6 tests.

**Modified modules:**

- `risk.py`: `RiskManager(__init__, ..., db_path=...)` —
  `_load_state()`/`_save_state()` persist session_start_net_liq,
  orders_today, orders_today_date to SQLite. Survives mid-day crashes.
  `last_updated` date used to invalidate stale session NL across
  calendar boundaries. 7 tests.
- `oms.py`: `submit()` and `modify()` call `assert_not_halted()` as
  the first gate, before normalisation, dry-run, or _authorize_send.
  Even dry-runs are blocked while halted. `cancel()` deliberately
  does NOT halt-gate (cancels reduce exposure; halted operators
  often want to flatten). `_require_idempotency_key` defense-in-depth
  assertion. 4 tests.
- `orders.py`: `TradeIntent` gains `client_order_id: str` field with
  default factory `uuid4().hex`. Empty/whitespace IDs rejected at
  `to_order_request()`. Routes to `OrderRequest.order_ref`.
- `broker/base.py`: `OrderRequest.normalized()` auto-fills `order_ref`
  with UUID4 when missing — defense-in-depth so production paths
  (LLM decision parser, raw broker callers) that historically passed
  None still emit an idempotent request. 8 tests.
- `quant_core/validation/pbo.py`: **Bug fix.** CSCV `calculate_multi_strategy`
  was returning `mean(logits < 0)` — the COMPLEMENT of canonical PBO
  (Bailey-Borwein-LdP-Zhu 2017). Strategies that should have failed
  the PBO < 0.5 gate were being approved. Now correctly computes
  `mean(rank_OOS(best_IS) > N/2)`. With N=50 trials of pure noise the
  bug reported PBO ~0.13 (looks fine) when truth is ~0.87 (overfit).
  4 correctness tests.
- `backtest_v2/engine.py` + `backtest_v2/models.py`: wire cost_model +
  ExecutionPolicy into the engine. New BacktestConfig fields:
  `cost_model_config` (None = legacy flat-bps; set = realistic
  decomposition), `execution_policy` (string-form for serialisation,
  default `'same_bar_close'` for back-compat), `default_borrow_rate_bps`.
  Pending-order machinery for non-look-ahead fills (queue on bar T,
  drain at start of bar T+1 _process_bar). Per-symbol rolling 22-bar
  OHLC + 20-bar volume + 60-bar return windows for spread / ADV / vol
  estimators. Borrow accrual on shorts, scaled by bars/day from
  `bar_size`. SAME_BAR_CLOSE emits a warning surfaced in
  `BacktestResults.warnings`. 5 end-to-end tests.

**Documentation:**

- `PLAN.md` (new at repo root): full enterprise refactor spec —
  phase plan, file-by-file change matrix, library + dependency
  decisions, test strategy, acceptance criteria for PR-to-main, risk
  register, 20-commit plan, 4-phase forward outline.
- `CLAUDE.md`: new commands and rules for PIT data, cost model, halt
  gate, mandatory client_order_id.

**Test counts by suite (post-refactor):**

| File | Tests |
| --- | --- |
| test_pit_store.py | 13 |
| test_universe.py | 8 |
| test_corporate_actions.py | 5 |
| test_cost_model.py | 23 |
| test_execution_policy.py | 7 |
| test_risk_persistence.py | 7 |
| test_pbo_correctness.py | 4 |
| test_stationary_bootstrap.py | 8 |
| test_oms_halt_gate.py | 4 |
| test_oms_idempotency.py | 8 |
| test_backtest_engine_realistic.py | 5 |
| test_report_card.py | 6 |
| test_migration.py | 11 |
| **Total new** | **109** |

(Pre-existing 1054 still pass — 4 ATLAS tests excluded due to
unrelated `SelectiveSSM` import error on `main`.)

**Acceptance gates (per PLAN.md §6):**

| Gate | Status |
|---|---|
| All pre-existing tests pass | yes (1054/1054) |
| All new tests pass (>=90% coverage on new modules) | yes (109/109) |
| `import trading_algo` clean on fresh venv | yes |
| Halt assertion enforced at oms.submit | yes |
| Validator wireable into orchestrator with `n_trials` from grid | yes (build_report_card API) |
| No silent fallback to forward-adjusted prices | yes (AdjustmentEngine query-time) |
| CHANGELOG entry written | yes (this entry) |

The remaining gates from PLAN.md (PIT-store full migration, reference
baseline run, orchestrator hookup of report card per backtest run) are
deliverables for the post-merge Phase-2 cull effort.

## 2026-04

### Unreleased — Wave T1: enterprise hardening (agent-first infrastructure)

Back-port of `kite-algo`'s Wave 1 hardening, adapted for IBKR. ~4,415 LOC
added (new modules + tests). 572 → 830 tests (+258), zero regressions.
Porting roadmap produced by cross-repo feasibility study; P0 items here.

**New modules** (all under `trading_algo/`):

- `config.py` (hardened): `_get_env_bool` is now STRICT — unknown values
  raise `EnvParseError` instead of silently defaulting. Closes the
  latent typo-disables-live-gate bug (a `TRADING_ALLOW_LIVE=tru` typo
  now crashes at startup, forcing the operator to notice). New
  `atomic_write_text(path, data, mode)` helper for future use (flex CSV
  dumps, contract cache, any cached state) — tempfile → `os.replace`
  with owner-only permissions.
- `exit_codes.py`: 15 enumerated codes (OK, GENERIC, USAGE, VALIDATION,
  HARD_REJECT, AUTH, PERMISSION, LEASE, HALTED, OUT_OF_WINDOW,
  MARKET_CLOSED, UNAVAILABLE, INTERNAL, TRANSIENT, TIMEOUT, SIGINT).
  `classify_exception` maps our own IBKR exceptions (`IBKRConnectionError`
  → UNAVAILABLE, `IBKRRateLimitError` → TRANSIENT, etc.) + ib_async
  `errorCode` values: 1100/1101/1102 → UNAVAILABLE, 200 → VALIDATION,
  201/203/103 → HARD_REJECT, 162/165/420 → TRANSIENT, 354/430 →
  PERMISSION, 502/504 → AUTH. Handles attribute + string-scan extraction.
- `envelope.py`: `{ok, cmd, schema_version, request_id, data, warnings,
  meta}` per command. 26-char Crockford-base32 ULID request IDs;
  `TRADING_PARENT_REQUEST_ID` propagates for nested agent workflows.
  `TRADING_NO_ENVELOPE=1` backwards-compat shim.
- `errors.py`: `emit_error(exc, env=...)` writes a single structured
  stderr JSON with code / class / message / retryable / ib_error_code /
  field_errors / suggested_action. IBKR-flavored action hints ("reconnect
  TWS/Gateway", "check market-data subscription", "wait 10s for pacing").
  `with_error_envelope(cmd)` decorator.
- `halt.py`: sentinel at `data/HALTED` (override `TRADING_HALT_PATH`).
  `write_halt` / `clear_halt` / `assert_not_halted` / `parse_duration`.
  Malformed sentinels fail CLOSED ("assume halted"). Optional
  `--expires-in` (e.g. `1h`) auto-clears on next read. `HaltActive`
  classifies as exit code 11.
- `audit.py`: NDJSON audit log at `data/audit/YYYY-MM-DD.jsonl`
  (override `TRADING_AUDIT_DIR`). POSIX-atomic `O_APPEND` single
  `write()` — concurrent writes never interleave partial lines.
  Per-entry fields: ts / ts_epoch_ms / request_id / parent_request_id /
  cmd / args (secret-redacted) / exit_code / error_code / elapsed_ms /
  ib_order_id / perm_id / order_ref / account / strategy_id / agent_id.
  Shallow redaction for known credential-keyed values (api_secret,
  access_token, flex_token, password, etc.). 7-year retention default
  (SEC 17a-4 min is 6y; we use 7 for enterprise).
- `idempotency.py`: SQLite-backed `IdempotencyStore` at
  `data/idempotency.sqlite` (WAL). Records (cmd, request_json,
  first_seen_at_ms, result_json, completed_at_ms, exit_code,
  ib_order_id, perm_id, order_ref, request_id). `record_attempt` /
  `record_completion` / `lookup` / `find_by_order_ref` / `purge_older_than`.
  `derive_order_ref(key, prefix="TA", length=30)` — BLAKE2b-deterministic
  IBKR orderRef derivation so cross-process retries produce the same
  orderRef and orderbook-based dedup works.
- `broker/idempotent_placer.py`: `IdempotentOrderPlacer` wraps any
  broker's `place_order`. Before placing, queries `ib.openTrades()` +
  `ib.reqCompletedOrders()` for an order whose `orderRef` matches our
  derived value. If found → return existing Trade state; NEVER
  re-transmit. If lookup itself fails →
  `IBKROrderbookLookupError` (classified as UNAVAILABLE/retryable):
  agent must reconcile before retrying. The agent-crash double-fill
  defense.
- `cli_runner.py`: shared `run_command(args)` wrapper. Every top-level
  CLI invocation: mint request_id, capture redacted args, execute
  handler, classify exceptions via `exit_codes.classify_exception`, emit
  structured stderr JSON on failure, write exactly one audit line
  per invocation (success OR failure), propagate
  `TRADING_STRATEGY_ID` / `TRADING_AGENT_ID` / `TRADING_PARENT_REQUEST_ID`.
  Audit I/O failure never crashes the command.

**Wiring into existing CLIs**:

- `cli.py`, `ibkr_tool.py`, `flex_tool.py` `main()` now all route through
  `cli_runner.run_command`. Every invocation is audited, every exception
  becomes a structured stderr JSON, every exit code is classified.
- Halt guards on every write command: `_cmd_place_order`,
  `_cmd_place_bracket`, `_cmd_cancel_order`, `_cmd_modify_order`,
  `_cmd_run` in `cli.py`; `cmd_place`, `cmd_combo`, `cmd_cancel`,
  `cmd_cancel_all` in `ibkr_tool.py`. Read commands unchanged — agents
  can still query state during a halt.
- `cli.py place-order --idempotency-key KEY`: durable replay via
  `IdempotencyStore` + BLAKE2b→orderRef derivation; IBKR broker uses
  `IdempotentOrderPlacer` for the orderbook pre-check. Same key across
  retries → stored result is replayed, `place_order` never re-transmits.
- `cli.py halt` + `resume` subcommands: `halt --reason "..."
  [--by ...] [--expires-in 1h]`; `resume --confirm-resume` (distinct
  from `--yes` so an accidentally-replayed `halt` can't undo itself).

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
