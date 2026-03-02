#!/usr/bin/env python3
"""
10-Year Enterprise-Grade Validated Backtest

Comprehensive walk-forward backtest of Combo B (8-strategy ensemble) over
10 years of 5-minute IBKR data (2016-01-01 to 2026-02-25).

Outputs:
  - backtest_results/10yr_validated_report.txt    (full text report)
  - backtest_results/10yr_validated_report.json   (machine-readable)
  - backtest_results/10yr_equity_curve.csv
  - backtest_results/10yr_monthly_returns.csv
  - backtest_results/10yr_walk_forward.csv

Usage:
    python scripts/run_10yr_backtest.py
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_algo.multi_strategy.controller import (
    ControllerConfig,
    MultiStrategyController,
    StrategyAllocation,
)
from trading_algo.multi_strategy.backtest_runner import (
    MultiStrategyBacktestConfig,
    MultiStrategyBacktestResults,
    MultiStrategyBacktestRunner,
)
from trading_algo.multi_strategy.adapters import (
    OrchestratorStrategyAdapter,
    PairsStrategyAdapter,
    MomentumStrategyAdapter,
    RegimeTransitionAdapter,
    CrossAssetDivergenceAdapter,
    FlowPressureAdapter,
)
from trading_algo.quant_core.data.ibkr_data_loader import load_universe_data
from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig
from trading_algo.quant_core.strategies.regime_transition import TransitionConfig
from trading_algo.orchestrator.config import OrchestratorConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "SMCI", "IWM"]
REFERENCE_SYMBOLS = ["HYG", "LQD", "TLT", "GLD"]
ALL_SYMBOLS = TRADING_SYMBOLS + REFERENCE_SYMBOLS

START_DATE = "2016-01-01"
END_DATE = "2026-02-25"

INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.045

RESULTS_DIR = PROJECT_ROOT / "backtest_results"

# Walk-forward parameters
BARS_PER_DAY = 78            # 5-min bars in a regular trading day
IS_WINDOW_DAYS = 504         # 2 years of trading days
OOS_STEP_DAYS = 126          # 6 months of trading days
IS_WINDOW_BARS = IS_WINDOW_DAYS * BARS_PER_DAY   # ~39,312
OOS_STEP_BARS = OOS_STEP_DAYS * BARS_PER_DAY     # ~9,828
WARMUP_DAYS = 252            # 1 year warmup excluded from metrics

# Hard IS/OOS split
HARD_IS_END = "2022-12-31"
HARD_OOS_START = "2023-01-01"

# Combo B strategy names (removed ORB and LiquidityCycles — they require
# intraday signal generation which we don't use)
COMBO_B_STRATEGIES = [
    "Orchestrator", "PairsTrading", "PureMomentum",
    "CrossAssetDivergence", "FlowPressure",
]

# Market regime date ranges
REGIME_DEFINITIONS = [
    ("Low-Vol Bull (2016-2019)",    date(2016, 1, 1),  date(2019, 12, 31)),
    ("COVID Crash (2020 Q1)",       date(2020, 1, 1),  date(2020, 3, 31)),
    ("Recovery/Bull (2020Q2-2021)", date(2020, 4, 1),  date(2021, 12, 31)),
    ("Bear Market (2022)",          date(2022, 1, 1),  date(2022, 12, 31)),
    ("AI Recovery (2023-2024)",     date(2023, 1, 1),  date(2024, 12, 31)),
    ("Recent (2025-2026)",          date(2025, 1, 1),  date(2026, 12, 31)),
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("10yr_backtest")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Shared state for parallel workers (inherited via fork, not pickled)
# ---------------------------------------------------------------------------
_SHARED_BAR_DATA: Optional[Dict[str, list]] = None
_MP_CONTEXT = multiprocessing.get_context("fork")
_MAX_WORKERS = min(8, os.cpu_count() or 4)


def _run_single_backtest(args: tuple):
    """
    Worker for parallel backtest execution.

    Reads bar slices from the module-level _SHARED_BAR_DATA (inherited via
    fork -- no pickling of large data).  ``args`` carries only lightweight
    slice indices / config.

    args layout:
        (slice_spec, strategy_names, symbols, initial_capital,
         risk_free_rate, entropy_filter)

    slice_spec is either:
        ("index", start_idx, end_idx)          -- slice by bar index
        ("date", start_date, end_date)         -- slice by date objects
    """
    slice_spec, strategy_names, symbols, initial_capital, risk_free_rate, entropy_filter = args
    try:
        bar_data = _SHARED_BAR_DATA
        if bar_data is None:
            return "ERROR: _SHARED_BAR_DATA is None in worker"

        # Resolve the slice
        if slice_spec[0] == "index":
            _, start_idx, end_idx = slice_spec
            bar_slices = {sym: bars[start_idx:end_idx] for sym, bars in bar_data.items()}
        elif slice_spec[0] == "date":
            _, start_dt, end_dt = slice_spec
            ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
            ref_bars = bar_data[ref_sym]
            start_idx = _date_to_bar_index_fast(ref_bars, start_dt)
            end_idx = min(_date_to_bar_index_fast(ref_bars, end_dt), len(ref_bars))
            bar_slices = {sym: bars[start_idx:end_idx] for sym, bars in bar_data.items()}
        else:
            return f"ERROR: unknown slice_spec type: {slice_spec[0]}"

        _, runner = build_controller_and_runner(
            strategy_names, entropy_filter=entropy_filter,
            symbols=symbols, initial_capital=initial_capital,
        )
        result = runner.run(bar_slices)
        return result
    except Exception as exc:
        return f"ERROR: {exc}"


def _date_to_bar_index_fast(bars: list, target_date: date) -> int:
    """Find the first bar index at or after the target date (for workers)."""
    for i, bar in enumerate(bars):
        if bar.timestamp.date() >= target_date:
            return i
    return len(bars)


# =========================================================================
# Data helpers
# =========================================================================

@dataclass
class BarObject:
    """Lightweight bar compatible with MultiStrategyBacktestRunner."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_bar_data() -> Tuple[Dict[str, List[BarObject]], Dict[str, Dict]]:
    """
    Load 10 years of OHLCV data for all 11 symbols.

    Returns:
        (bar_data dict, coverage_info dict)
    """
    print("Loading 10-year universe data for all 11 symbols ...")
    print(f"  Symbols: {', '.join(ALL_SYMBOLS)}")
    print(f"  Date range: {START_DATE} to {END_DATE}")

    aligned_data, timestamps = load_universe_data(
        symbols=ALL_SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        bar_size="5mins",
    )

    bar_data: Dict[str, List[BarObject]] = {}
    coverage: Dict[str, Dict] = {}

    for symbol, ohlcv in aligned_data.items():
        bars: List[BarObject] = []
        first_valid = None
        last_valid = None
        valid_count = 0

        for i, ts in enumerate(timestamps):
            if np.isnan(ohlcv[i, 0]):
                continue
            bars.append(BarObject(
                timestamp=ts,
                open=float(ohlcv[i, 0]),
                high=float(ohlcv[i, 1]),
                low=float(ohlcv[i, 2]),
                close=float(ohlcv[i, 3]),
                volume=float(ohlcv[i, 4]),
            ))
            if first_valid is None:
                first_valid = ts
            last_valid = ts
            valid_count += 1

        bar_data[symbol] = bars
        coverage[symbol] = {
            "first_date": first_valid.date() if first_valid else None,
            "last_date": last_valid.date() if last_valid else None,
            "bar_count": valid_count,
            "total_bars": len(timestamps),
            "pct_coverage": valid_count / len(timestamps) * 100 if len(timestamps) > 0 else 0,
        }

    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    total_bars = len(bar_data[ref_sym])
    print(f"  Loaded {len(bar_data)} symbols, reference ({ref_sym}) has {total_bars} bars")
    print(f"  Period: {bar_data[ref_sym][0].timestamp.date()} to "
          f"{bar_data[ref_sym][-1].timestamp.date()}")

    for sym in ALL_SYMBOLS:
        if sym in coverage:
            c = coverage[sym]
            print(f"    {sym:>5}: {c['bar_count']:>9,} bars  "
                  f"{c['first_date']} to {c['last_date']}  "
                  f"({c['pct_coverage']:.1f}% coverage)")
        else:
            print(f"    {sym:>5}: NOT LOADED")

    return bar_data, coverage


def slice_bar_data(
    data: Dict[str, List[BarObject]],
    start_idx: int,
    end_idx: int,
) -> Dict[str, List[BarObject]]:
    """Slice all symbols' bar lists by index range."""
    return {sym: bars[start_idx:end_idx] for sym, bars in data.items()}


def filter_trading_symbols(data: Dict[str, List[BarObject]]) -> Dict[str, List[BarObject]]:
    """Return only trading symbols (exclude reference)."""
    return {sym: bars for sym, bars in data.items() if sym in TRADING_SYMBOLS}


def date_to_bar_index(bars: List[BarObject], target_date: date) -> int:
    """Find the first bar index at or after the target date."""
    for i, bar in enumerate(bars):
        if bar.timestamp.date() >= target_date:
            return i
    return len(bars)


def extract_spy_daily_returns(bar_data: Dict[str, List[BarObject]]) -> Tuple[np.ndarray, List[date]]:
    """
    Compute SPY daily returns from bar data.

    Returns last close of each day / last close of prev day - 1.
    """
    spy_bars = bar_data.get("SPY", [])
    if len(spy_bars) < 2:
        return np.array([]), []

    # Group by day, take last close
    daily_closes: Dict[date, float] = {}
    for bar in spy_bars:
        daily_closes[bar.timestamp.date()] = bar.close

    sorted_dates = sorted(daily_closes.keys())
    returns: List[float] = []
    dates_out: List[date] = []

    for i in range(1, len(sorted_dates)):
        prev_close = daily_closes[sorted_dates[i - 1]]
        curr_close = daily_closes[sorted_dates[i]]
        if prev_close > 0:
            returns.append(curr_close / prev_close - 1.0)
            dates_out.append(sorted_dates[i])

    return np.array(returns), dates_out


def extract_daily_returns_from_equity(equity_curve: List[float]) -> np.ndarray:
    """Compute daily returns from an equity curve."""
    ec = np.array(equity_curve)
    if len(ec) < 2:
        return np.array([0.0])
    returns = ec[1:] / ec[:-1] - 1.0
    # Remove any inf/nan
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


# =========================================================================
# Strategy combination factory
# =========================================================================

def _make_adapters(names: List[str]) -> List[Any]:
    """Instantiate adapters by canonical name, skipping failures."""
    # Orchestrator config tuned for 5-minute bar data with daily-only signals:
    # - ATR thresholds scaled down ~20x (daily ATR% ≈ 20× 5-min ATR%)
    # - Lower consensus bar: at market open, intraday regime is always
    #   RANGE_BOUND (day_return ≈ 0), so we need a lower bar to enter
    # - Lower mean reversion z-score: 1.0 vs 1.5 to trigger in range-bound
    orch_cfg = OrchestratorConfig(
        min_atr_pct=0.000001,      # effectively disabled for 5-min bars
        max_atr_pct=0.05,          # 5% (very permissive)
        min_consensus_edges=2,     # 2 of 6-7 edges (was 4)
        min_consensus_score=0.15,  # (was 0.5) — lower for 5-min
        max_opposition_score=0.60, # (was 0.35) — more tolerant
        min_directional_quality=0.35,  # (was 0.6) — more tolerant
        max_position_pct=0.10,     # 10% max (was 3%)
        mean_reversion_zscore=0.8, # (was 1.5) — trigger more in range-bound
        warmup_bars=20,            # (was 30) — ready faster
    )
    registry: Dict[str, Callable[[], Any]] = {
        "Orchestrator": lambda: OrchestratorStrategyAdapter(config=orch_cfg),
        "PairsTrading": lambda: PairsStrategyAdapter(),
        "PureMomentum": lambda: MomentumStrategyAdapter(
            MomentumConfig(allow_short=False),
        ),
        "RegimeTransition": lambda: RegimeTransitionAdapter(
            config=TransitionConfig(
                min_signal_strength=0.001,     # 10bps daily return minimum
                transition_threshold=0.15,     # require meaningful transition prob
                velocity_threshold=0.005,      # velocity ~0.001 between retrains
                max_position=0.10,             # 10% max position (was 20%)
            ),
        ),
        "CrossAssetDivergence": lambda: CrossAssetDivergenceAdapter(),
        "FlowPressure": lambda: FlowPressureAdapter(),
    }
    adapters: List[Any] = []
    for n in names:
        factory = registry.get(n)
        if factory is None:
            logger.warning("Unknown adapter name: %s -- skipping", n)
            continue
        try:
            adapters.append(factory())
        except Exception as exc:
            logger.warning("Failed to create adapter %s: %s", n, exc)
    return adapters


def build_controller_and_runner(
    strategy_names: List[str],
    entropy_filter: bool = False,
    symbols: List[str] = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[MultiStrategyController, MultiStrategyBacktestRunner]:
    """Build a fresh controller + runner for the given strategy set."""
    if symbols is None:
        symbols = TRADING_SYMBOLS

    cfg = ControllerConfig(
        enable_entropy_filter=entropy_filter,
        enable_vol_management=True,
        vol_target=0.18,
        vol_scale_min=0.50,      # was 0.25 — less aggressive shrinkage in high-vol
        vol_scale_max=2.0,       # cap leverage in low-vol periods
        max_drawdown=1.0,       # Disable drawdown halt (was causing whipsaw)
        daily_loss_limit=1.0,   # Disable daily loss halt
        enable_regime_adaptation=False,
        # Redistribute allocation from removed ORB (10%) and LiquidityCycles (10%)
        # to the 6 remaining strategies
        # Keep v4-like allocations.  The ~12% unallocated from
        # removed RegimeTransition acts as an implicit cash buffer,
        # reducing position sizes and drawdowns.
        allocations={
            "Orchestrator": StrategyAllocation(weight=0.25, max_positions=12),
            "PairsTrading": StrategyAllocation(weight=0.15, max_positions=10),
            "PureMomentum": StrategyAllocation(weight=0.18, max_positions=10),
            "CrossAssetDivergence": StrategyAllocation(weight=0.13, max_positions=8),
            "FlowPressure": StrategyAllocation(weight=0.17, max_positions=10),
        },
    )
    controller = MultiStrategyController(cfg)
    for adapter in _make_adapters(strategy_names):
        controller.register(adapter)

    bt_cfg = MultiStrategyBacktestConfig(
        initial_capital=initial_capital,
        symbols=symbols,
        risk_free_rate=RISK_FREE_RATE,
        signal_interval_bars=156,  # ~hourly (78 bars/day × 11 symbols ÷ ~5.5 signals/day)
        intraday_vol_threshold=0.15,  # only intraday signals when ann vol > 15%
        max_gross_exposure=1.5,    # match controller's 150% limit
    )
    runner = MultiStrategyBacktestRunner(controller, bt_cfg)
    return controller, runner


# =========================================================================
# Metric helpers
# =========================================================================

def sharpe_from_daily(daily_returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe (excess over risk-free rate)."""
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    std = float(np.std(excess, ddof=1))
    if std < 1e-8:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def total_return_from_daily(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return float(np.prod(1 + daily_returns) - 1)


def annualized_return_from_daily(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    total = np.prod(1 + daily_returns)
    n_years = len(daily_returns) / 252
    if n_years < 1 / 252:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def max_drawdown_from_daily(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    equity = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    return float(np.max(dd))


def compute_benchmark_metrics(
    algo_daily: np.ndarray,
    bench_daily: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """Compute beta, alpha, information ratio, correlation externally."""
    min_len = min(len(algo_daily), len(bench_daily))
    if min_len < 10:
        return {"beta": 0.0, "alpha_annual": 0.0, "information_ratio": 0.0, "benchmark_correlation": 0.0}

    algo_r = algo_daily[:min_len]
    bench_r = bench_daily[:min_len]

    cov_matrix = np.cov(algo_r, bench_r)
    var_bench = cov_matrix[1, 1]
    beta = float(cov_matrix[0, 1] / var_bench) if var_bench > 1e-10 else 0.0

    bench_ann = float(np.mean(bench_r) * 252)
    algo_ann = float(np.mean(algo_r) * 252)
    alpha_annual = algo_ann - risk_free_rate - beta * (bench_ann - risk_free_rate)

    active_returns = algo_r - bench_r
    tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(252))
    information_ratio = float(np.mean(active_returns) * 252 / tracking_error) if tracking_error > 1e-10 else 0.0

    corr = np.corrcoef(algo_r, bench_r)
    benchmark_correlation = float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0

    return {
        "beta": beta,
        "alpha_annual": alpha_annual,
        "information_ratio": information_ratio,
        "benchmark_correlation": benchmark_correlation,
    }


# =========================================================================
# Walk-forward protocol (2yr IS, 6mo OOS, rolling)
# =========================================================================

@dataclass
class WFWindowResult:
    """Metrics for one walk-forward window."""
    window_idx: int
    is_start_date: Optional[date] = None
    is_end_date: Optional[date] = None
    oos_start_date: Optional[date] = None
    oos_end_date: Optional[date] = None
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_return: float = 0.0
    oos_return: float = 0.0
    ratio: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None


def run_walk_forward(
    bar_data: Dict[str, List[BarObject]],
) -> List[WFWindowResult]:
    """
    Rolling walk-forward: 2-year IS window, 6-month OOS step.

    Yields ~16 windows covering 2016-2026.
    Uses ProcessPoolExecutor with fork context for parallelism.
    """
    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    total_bars = len(bar_data[ref_sym])
    ref_bars = bar_data[ref_sym]

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD ANALYSIS  (parallel, {_MAX_WORKERS} workers)")
    print(f"  IS window: ~{IS_WINDOW_DAYS} trading days ({IS_WINDOW_DAYS//252} years)")
    print(f"  OOS step:  ~{OOS_STEP_DAYS} trading days ({OOS_STEP_DAYS//252*6} months)")
    print(f"  Total bars: {total_bars:,}")
    print(f"{'='*70}")

    # ── Build all window specs upfront ────────────────────────────────
    window_specs: List[dict] = []
    window_start = 0
    window_idx = 0

    while window_start + IS_WINDOW_BARS < total_bars:
        is_s = window_start
        is_e = window_start + IS_WINDOW_BARS
        oos_s = is_e
        oos_e = min(oos_s + OOS_STEP_BARS, total_bars)

        if oos_e <= oos_s:
            break

        is_start_dt = ref_bars[is_s].timestamp.date() if is_s < len(ref_bars) else None
        is_end_dt = ref_bars[min(is_e - 1, len(ref_bars) - 1)].timestamp.date()
        oos_start_dt = ref_bars[min(oos_s, len(ref_bars) - 1)].timestamp.date()
        oos_end_dt = ref_bars[min(oos_e - 1, len(ref_bars) - 1)].timestamp.date()

        window_specs.append({
            "window_idx": window_idx,
            "is_s": is_s, "is_e": is_e,
            "oos_s": oos_s, "oos_e": oos_e,
            "is_start_dt": is_start_dt, "is_end_dt": is_end_dt,
            "oos_start_dt": oos_start_dt, "oos_end_dt": oos_end_dt,
        })

        window_start += OOS_STEP_BARS
        window_idx += 1

    n_windows = len(window_specs)
    print(f"  Submitting {n_windows} windows ({n_windows * 2} IS+OOS runs) ...")

    # ── Submit all IS + OOS runs in parallel ──────────────────────────
    # Each window produces two futures: (window_idx, "is"/"oos", future)
    t_wf_start = time.time()

    # Pre-build WFWindowResult shells
    wf_results: Dict[int, WFWindowResult] = {}
    for spec in window_specs:
        wf_results[spec["window_idx"]] = WFWindowResult(
            window_idx=spec["window_idx"],
            is_start_date=spec["is_start_dt"],
            is_end_date=spec["is_end_dt"],
            oos_start_date=spec["oos_start_dt"],
            oos_end_date=spec["oos_end_dt"],
        )

    common_args = (COMBO_B_STRATEGIES, TRADING_SYMBOLS, INITIAL_CAPITAL,
                   RISK_FREE_RATE, False)

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS,
                             mp_context=_MP_CONTEXT) as pool:
        future_to_key = {}
        for spec in window_specs:
            widx = spec["window_idx"]
            # IS future
            is_args = (("index", spec["is_s"], spec["is_e"]), *common_args)
            f_is = pool.submit(_run_single_backtest, is_args)
            future_to_key[f_is] = (widx, "is")
            # OOS future
            oos_args = (("index", spec["oos_s"], spec["oos_e"]), *common_args)
            f_oos = pool.submit(_run_single_backtest, oos_args)
            future_to_key[f_oos] = (widx, "oos")

        completed = 0
        total_futures = len(future_to_key)
        for future in as_completed(future_to_key):
            widx, phase = future_to_key[future]
            wfr = wf_results[widx]
            completed += 1

            try:
                result = future.result()
                if isinstance(result, str):
                    # Error string returned from worker
                    wfr.error = result
                    logger.warning("WF window %d %s failed: %s", widx, phase, result)
                else:
                    if phase == "is":
                        wfr.is_sharpe = result.sharpe_ratio
                        wfr.is_return = result.total_return
                    else:
                        wfr.oos_sharpe = result.sharpe_ratio
                        wfr.oos_return = result.total_return
            except Exception as exc:
                wfr.error = str(exc)
                logger.warning("WF window %d %s exception: %s", widx, phase, exc)

            print(f"\r  Completed {completed}/{total_futures} WF runs ...", end="", flush=True)

    # ── Post-process: compute ratios, print summary ───────────────────
    print()  # newline after progress
    windows: List[WFWindowResult] = []
    for spec in window_specs:
        widx = spec["window_idx"]
        wfr = wf_results[widx]
        wfr.elapsed_sec = time.time() - t_wf_start  # wall-clock for whole batch

        if wfr.error is None:
            wfr.ratio = (wfr.oos_sharpe / wfr.is_sharpe
                         if abs(wfr.is_sharpe) > 0.01 else 0.0)

        status = "OK" if wfr.error is None else "FAIL"
        print(f"  Window {widx:>2}: IS [{wfr.is_start_date} .. {wfr.is_end_date}]  "
              f"OOS [{wfr.oos_start_date} .. {wfr.oos_end_date}]  "
              f"IS SR={wfr.is_sharpe:+.3f}  OOS SR={wfr.oos_sharpe:+.3f}  "
              f"Ratio={wfr.ratio:.2f}  {status}")
        windows.append(wfr)

    print(f"\n  Walk-forward complete: {len(windows)} windows processed "
          f"({time.time() - t_wf_start:.1f}s wall-clock)")
    return windows


# =========================================================================
# Hard IS/OOS split
# =========================================================================

@dataclass
class HardSplitResults:
    """Results from the hard IS/OOS split."""
    is_result: Optional[MultiStrategyBacktestResults] = None
    oos_result: Optional[MultiStrategyBacktestResults] = None
    is_date_range: Tuple[Optional[date], Optional[date]] = (None, None)
    oos_date_range: Tuple[Optional[date], Optional[date]] = (None, None)


def run_hard_split(
    bar_data: Dict[str, List[BarObject]],
) -> HardSplitResults:
    """Run hard IS (2016-2022) and OOS (2023-2026) backtests in parallel."""
    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    ref_bars = bar_data[ref_sym]

    split_date = date.fromisoformat(HARD_OOS_START)
    split_idx = date_to_bar_index(ref_bars, split_date)

    result = HardSplitResults()
    result.is_date_range = (ref_bars[0].timestamp.date(), ref_bars[split_idx - 1].timestamp.date())
    result.oos_date_range = (ref_bars[split_idx].timestamp.date(), ref_bars[-1].timestamp.date())

    print(f"\n{'='*70}")
    print(f"  HARD IS/OOS SPLIT  (parallel, 2 workers)")
    print(f"  IS:  {result.is_date_range[0]} to {result.is_date_range[1]} ({split_idx:,} bars)")
    print(f"  OOS: {result.oos_date_range[0]} to {result.oos_date_range[1]} "
          f"({len(ref_bars) - split_idx:,} bars)")
    print(f"{'='*70}")

    common_args = (COMBO_B_STRATEGIES, TRADING_SYMBOLS, INITIAL_CAPITAL,
                   RISK_FREE_RATE, False)

    print("  Running IS + OOS backtests in parallel ...", end=" ", flush=True)
    t0 = time.time()

    is_args = (("index", 0, split_idx), *common_args)
    oos_args = (("index", split_idx, len(ref_bars)), *common_args)

    with ProcessPoolExecutor(max_workers=2, mp_context=_MP_CONTEXT) as pool:
        f_is = pool.submit(_run_single_backtest, is_args)
        f_oos = pool.submit(_run_single_backtest, oos_args)

        try:
            is_res = f_is.result()
            if isinstance(is_res, str):
                print(f"\n  IS FAILED: {is_res}")
            else:
                result.is_result = is_res
        except Exception as exc:
            print(f"\n  IS FAILED: {exc}")
            traceback.print_exc()

        try:
            oos_res = f_oos.result()
            if isinstance(oos_res, str):
                print(f"\n  OOS FAILED: {oos_res}")
            else:
                result.oos_result = oos_res
        except Exception as exc:
            print(f"\n  OOS FAILED: {exc}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    if result.is_result:
        print(f"    IS  -- SR={result.is_result.sharpe_ratio:+.3f}  "
              f"Ret={result.is_result.total_return*100:+.2f}%")
    if result.oos_result:
        print(f"    OOS -- SR={result.oos_result.sharpe_ratio:+.3f}  "
              f"Ret={result.oos_result.total_return*100:+.2f}%")

    return result


# =========================================================================
# Full-period backtest
# =========================================================================

def run_full_period(
    bar_data: Dict[str, List[BarObject]],
) -> MultiStrategyBacktestResults:
    """Run Combo B on the full 10-year dataset."""
    print(f"\n{'='*70}")
    print(f"  FULL-PERIOD BACKTEST (Combo B)")
    print(f"{'='*70}")

    _, runner = build_controller_and_runner(COMBO_B_STRATEGIES)

    def progress(pct: float, msg: str) -> None:
        bar_len = 40
        filled = int(pct * bar_len)
        bar_str = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar_str}] {pct*100:5.1f}%  {msg[:50]:<50}", end="", flush=True)

    t0 = time.time()
    result = runner.run(bar_data, progress_callback=progress)
    elapsed = time.time() - t0
    print(f"\n  Full-period backtest complete ({elapsed:.1f}s)")
    print(f"  Total Return: {result.total_return * 100:+.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:+.3f}")
    print(f"  Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"  Total Trades: {result.total_trades}")

    return result


# =========================================================================
# Market regime breakdown
# =========================================================================

@dataclass
class RegimeResult:
    """Backtest results for a single market regime."""
    name: str
    start_date: date
    end_date: date
    sharpe: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    volatility: float = 0.0
    error: Optional[str] = None


def run_regime_breakdown(
    bar_data: Dict[str, List[BarObject]],
) -> List[RegimeResult]:
    """Run separate backtest for each market regime in parallel."""
    ref_sym = max(bar_data, key=lambda s: len(bar_data[s]))
    ref_bars = bar_data[ref_sym]

    print(f"\n{'='*70}")
    print(f"  MARKET REGIME BREAKDOWN  (sequential)")
    print(f"{'='*70}")

    # ── Build regime specs, skipping insufficient-data regimes ────────
    regime_specs: List[Tuple[int, str, date, date, int, int]] = []
    skipped: List[RegimeResult] = []

    for idx, (regime_name, start_dt, end_dt) in enumerate(REGIME_DEFINITIONS):
        start_idx = date_to_bar_index(ref_bars, start_dt)
        end_idx = date_to_bar_index(ref_bars, end_dt + timedelta(days=1))
        end_idx = min(end_idx, len(ref_bars))

        if end_idx <= start_idx or end_idx - start_idx < BARS_PER_DAY * 5:
            rr = RegimeResult(name=regime_name, start_date=start_dt, end_date=end_dt,
                              error="Insufficient data")
            skipped.append(rr)
            print(f"  {regime_name:<35} -- insufficient data")
        else:
            regime_specs.append((idx, regime_name, start_dt, end_dt, start_idx, end_idx))
            print(f"  {regime_name:<35} bars {start_idx:>9,} .. {end_idx:>9,}  [queued]")

    print(f"  Submitting {len(regime_specs)} regime runs ...")
    t_regime_start = time.time()

    common_args = (COMBO_B_STRATEGIES, TRADING_SYMBOLS, INITIAL_CAPITAL,
                   RISK_FREE_RATE, False)

    # ── Submit to pool ────────────────────────────────────────────────
    regime_results_map: Dict[int, RegimeResult] = {}
    for rr in skipped:
        # Find the original index for ordering
        for orig_idx, (rn, sd, ed) in enumerate(REGIME_DEFINITIONS):
            if rn == rr.name:
                regime_results_map[orig_idx] = rr
                break

    # Run sequentially to avoid memory pressure from parallel intraday backtests
    for idx_in_batch, (orig_idx, regime_name, start_dt, end_dt, start_idx, end_idx) in enumerate(regime_specs):
        rr = RegimeResult(name=regime_name, start_date=start_dt, end_date=end_dt)
        try:
            args = (("index", start_idx, end_idx), *common_args)
            res = _run_single_backtest(args)
            if isinstance(res, str):
                rr.error = res
                logger.warning("Regime %s failed: %s", regime_name, res)
            else:
                rr.sharpe = res.sharpe_ratio
                rr.total_return = res.total_return
                rr.annualized_return = res.annualized_return
                rr.max_drawdown = res.max_drawdown
                rr.total_trades = res.total_trades
                rr.volatility = res.volatility
        except Exception as exc:
            rr.error = str(exc)
            logger.warning("Regime %s exception: %s", regime_name, exc)

        regime_results_map[orig_idx] = rr

        status = "OK" if rr.error is None else f"FAIL: {rr.error}"
        print(f"  [{idx_in_batch+1}/{len(regime_specs)}] {regime_name:<35} "
              f"SR={rr.sharpe:+.3f}  Ret={rr.total_return*100:+.1f}%  "
              f"DD={rr.max_drawdown*100:.1f}%  {status}")

    elapsed = time.time() - t_regime_start
    print(f"  Regime breakdown complete ({elapsed:.1f}s wall-clock)")

    # Return in original REGIME_DEFINITIONS order
    results: List[RegimeResult] = []
    for i in range(len(REGIME_DEFINITIONS)):
        if i in regime_results_map:
            results.append(regime_results_map[i])
    return results


# =========================================================================
# Monthly / Yearly return helpers
# =========================================================================

def compute_monthly_returns(
    daily_returns: List[float],
    timestamps: List[datetime],
) -> Dict[Tuple[int, int], float]:
    """
    Compound daily returns by calendar month.

    Returns dict of (year, month) -> monthly return.
    """
    monthly: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for i, dr in enumerate(daily_returns):
        if i < len(timestamps):
            dt = timestamps[i]
            monthly[(dt.year, dt.month)].append(dr)

    result: Dict[Tuple[int, int], float] = {}
    for key, rets in sorted(monthly.items()):
        result[key] = float(np.prod(1 + np.array(rets)) - 1)

    return result


def compute_yearly_returns(
    daily_returns: List[float],
    timestamps: List[datetime],
) -> Dict[int, float]:
    """Compound daily returns by calendar year."""
    yearly: Dict[int, List[float]] = defaultdict(list)

    for i, dr in enumerate(daily_returns):
        if i < len(timestamps):
            yearly[timestamps[i].year].append(dr)

    result: Dict[int, float] = {}
    for year, rets in sorted(yearly.items()):
        result[year] = float(np.prod(1 + np.array(rets)) - 1)

    return result


def compute_yearly_spy_returns(
    spy_daily: np.ndarray,
    spy_dates: List[date],
) -> Dict[int, float]:
    """Compute SPY annual returns for alpha comparison."""
    yearly: Dict[int, List[float]] = defaultdict(list)
    for i, dr in enumerate(spy_daily):
        if i < len(spy_dates):
            yearly[spy_dates[i].year].append(dr)

    result: Dict[int, float] = {}
    for year, rets in sorted(yearly.items()):
        result[year] = float(np.prod(1 + np.array(rets)) - 1)
    return result


# =========================================================================
# Deflated Sharpe Ratio
# =========================================================================

def compute_deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_daily_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute Deflated Sharpe Ratio.

    Returns (DSR, p-value).
    """
    try:
        from trading_algo.quant_core.validation.pbo import DeflatedSharpe
        ds = DeflatedSharpe()
        result = ds.calculate(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            n_observations=n_daily_obs,
            skewness=skewness,
            kurtosis=kurtosis,
        )
        return result.deflated_sharpe, result.p_value
    except Exception:
        # Fallback simplified DSR
        if n_trials < 2 or n_daily_obs < 10:
            return observed_sharpe, 1.0
        sr_benchmark = np.sqrt(2 * np.log(n_trials)) / np.sqrt(n_daily_obs)
        dsr = (observed_sharpe - sr_benchmark) / (1.0 / np.sqrt(n_daily_obs))
        p_value = 1.0 - float(sp_stats.norm.cdf(dsr))
        return float(dsr), float(p_value)


# =========================================================================
# Report generation — Text
# =========================================================================

def generate_text_report(
    full_result: MultiStrategyBacktestResults,
    hard_split: HardSplitResults,
    wf_windows: List[WFWindowResult],
    regime_results: List[RegimeResult],
    spy_daily: np.ndarray,
    spy_dates: List[date],
    coverage: Dict[str, Dict],
    benchmark_metrics: Dict[str, float],
    dsr_value: float,
    dsr_pval: float,
    monthly_returns: Dict[Tuple[int, int], float],
    yearly_returns: Dict[int, float],
    yearly_spy: Dict[int, float],
    total_elapsed: float,
) -> str:
    """Build the full enterprise text report."""
    lines: List[str] = []
    W = 100

    def sep(char: str = "=") -> str:
        return char * W

    def heading(num: int, title: str) -> None:
        lines.append("")
        lines.append(sep())
        lines.append(f"  {num}. {title}")
        lines.append(sep())

    def subheading(title: str) -> None:
        lines.append("")
        lines.append(f"  --- {title} ---")

    # ── Section 0: Header ─────────────────────────────────────────────
    lines.append(sep("#"))
    lines.append(f"#  10-YEAR ENTERPRISE-GRADE VALIDATED BACKTEST REPORT")
    lines.append(f"#  Combo B: 8-Strategy Ensemble")
    lines.append(f"#  {START_DATE} to {END_DATE}")
    lines.append(sep("#"))
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Runtime:   {total_elapsed/3600:.1f} hours ({total_elapsed:.0f}s)")
    lines.append(f"  Strategies: {', '.join(COMBO_B_STRATEGIES)}")
    lines.append(f"  Trading Symbols: {', '.join(TRADING_SYMBOLS)}")
    lines.append(f"  Reference Symbols: {', '.join(REFERENCE_SYMBOLS)}")
    lines.append(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    lines.append(f"  Risk-Free Rate: {RISK_FREE_RATE*100:.1f}%")

    # ── Section 1: Executive Summary ──────────────────────────────────
    heading(1, "EXECUTIVE SUMMARY")
    fr = full_result
    final_equity = fr.equity_curve[-1] if fr.equity_curve else INITIAL_CAPITAL
    lines.append(f"  Total Return (10yr):     {fr.total_return * 100:+.2f}%")
    lines.append(f"  Annualized Return:       {fr.annualized_return * 100:+.2f}%")
    lines.append(f"  Final Equity:            ${final_equity:,.2f}")
    lines.append(f"  Sharpe Ratio:            {fr.sharpe_ratio:+.3f}")
    lines.append(f"  Sortino Ratio:           {fr.sortino_ratio:+.3f}")
    lines.append(f"  Calmar Ratio:            {fr.calmar_ratio:+.3f}")
    lines.append(f"  Max Drawdown:            {fr.max_drawdown * 100:.2f}%")
    lines.append(f"  Max DD Duration:         {fr.max_drawdown_duration_days} equity-curve points")
    lines.append(f"  Annual Volatility:       {fr.volatility * 100:.2f}%")
    lines.append(f"  Total Trades:            {fr.total_trades}")
    lines.append(f"  Win Rate:                {fr.win_rate * 100:.1f}%")
    lines.append(f"  Profit Factor:           {fr.profit_factor:.3f}")

    spy_total = total_return_from_daily(spy_daily) if len(spy_daily) > 0 else 0
    lines.append(f"  SPY Total Return (B&H):  {spy_total * 100:+.2f}%")
    lines.append(f"  Alpha vs SPY (annual):   {benchmark_metrics.get('alpha_annual', 0) * 100:+.2f}%")
    lines.append(f"  Beta:                    {benchmark_metrics.get('beta', 0):.3f}")

    if hard_split.oos_result:
        lines.append(f"  OOS Sharpe (2023-2026):  {hard_split.oos_result.sharpe_ratio:+.3f}")
    lines.append(f"  Deflated Sharpe Ratio:   {dsr_value:+.3f}  (p={dsr_pval:.4f})")

    # ── Section 2: Data Coverage Table ────────────────────────────────
    heading(2, "DATA COVERAGE TABLE")
    lines.append(f"  {'Symbol':<8} {'First Date':<14} {'Last Date':<14} "
                 f"{'Bar Count':>12} {'Coverage':>10}")
    lines.append("  " + "-" * (W - 4))
    for sym in ALL_SYMBOLS:
        if sym in coverage:
            c = coverage[sym]
            lines.append(
                f"  {sym:<8} {str(c['first_date']):<14} {str(c['last_date']):<14} "
                f"{c['bar_count']:>12,} {c['pct_coverage']:>9.1f}%"
            )
        else:
            lines.append(f"  {sym:<8} {'N/A':<14} {'N/A':<14} {'0':>12} {'0.0':>9}%")

    # ── Section 3: Hard IS/OOS Results ────────────────────────────────
    heading(3, "HARD IN-SAMPLE / OUT-OF-SAMPLE RESULTS")
    if hard_split.is_result and hard_split.oos_result:
        is_r = hard_split.is_result
        oos_r = hard_split.oos_result
        lines.append(f"  IS period:  {hard_split.is_date_range[0]} to {hard_split.is_date_range[1]}")
        lines.append(f"  OOS period: {hard_split.oos_date_range[0]} to {hard_split.oos_date_range[1]}")
        lines.append("")
        metrics_list = [
            ("Total Return",       is_r.total_return,       oos_r.total_return,       True),
            ("Annualized Return",  is_r.annualized_return,  oos_r.annualized_return,  True),
            ("Sharpe Ratio",       is_r.sharpe_ratio,       oos_r.sharpe_ratio,       False),
            ("Sortino Ratio",      is_r.sortino_ratio,      oos_r.sortino_ratio,      False),
            ("Calmar Ratio",       is_r.calmar_ratio,       oos_r.calmar_ratio,       False),
            ("Max Drawdown",       is_r.max_drawdown,       oos_r.max_drawdown,       True),
            ("Volatility",         is_r.volatility,         oos_r.volatility,         True),
            ("Win Rate",           is_r.win_rate,           oos_r.win_rate,           True),
            ("Total Trades",       is_r.total_trades,       oos_r.total_trades,       False),
            ("Profit Factor",      is_r.profit_factor,      oos_r.profit_factor,      False),
            ("VaR 95%",            is_r.var_95,             oos_r.var_95,             True),
            ("CVaR 95%",           is_r.cvar_95,            oos_r.cvar_95,            True),
        ]

        lines.append(f"  {'Metric':<22} {'IS':>14} {'OOS':>14} {'OOS/IS Ratio':>14}")
        lines.append("  " + "-" * (W - 4))
        for label, is_val, oos_val, is_pct in metrics_list:
            if "Trades" in label:
                is_str = f"{int(is_val)}"
                oos_str = f"{int(oos_val)}"
            elif is_pct:
                is_str = f"{is_val * 100:.2f}%"
                oos_str = f"{oos_val * 100:.2f}%"
            else:
                is_str = f"{is_val:+.3f}"
                oos_str = f"{oos_val:+.3f}"
            ratio = oos_val / is_val if abs(is_val) > 1e-8 else 0.0
            lines.append(f"  {label:<22} {is_str:>14} {oos_str:>14} {ratio:>14.3f}")

        sr_degrade = oos_r.sharpe_ratio / is_r.sharpe_ratio if abs(is_r.sharpe_ratio) > 0.01 else 0.0
        lines.append(f"\n  Sharpe Degradation Ratio: {sr_degrade:.3f}")
        if sr_degrade > 0.5:
            lines.append("  Assessment: ACCEPTABLE -- OOS retains >50% of IS edge")
        else:
            lines.append("  Assessment: DEGRADED -- significant overfitting risk")
    else:
        lines.append("  Hard split failed -- one or both periods produced no results.")

    # ── Section 4: Walk-Forward Results Table ─────────────────────────
    heading(4, "WALK-FORWARD RESULTS TABLE")
    valid_windows = [w for w in wf_windows if w.error is None]
    lines.append(f"  {'Win':>4} {'IS Period':<25} {'OOS Period':<25} "
                 f"{'IS SR':>7} {'OOS SR':>8} {'Ratio':>7} {'IS Ret':>8} {'OOS Ret':>8}")
    lines.append("  " + "-" * (W - 4))
    for w in wf_windows:
        is_period = f"{w.is_start_date}..{w.is_end_date}" if w.is_start_date else "N/A"
        oos_period = f"{w.oos_start_date}..{w.oos_end_date}" if w.oos_start_date else "N/A"
        if w.error:
            lines.append(f"  {w.window_idx:>4} {is_period:<25} {oos_period:<25} "
                         f"{'FAILED':>7} {'':>8} {'':>7} {'':>8} {'':>8}")
        else:
            lines.append(
                f"  {w.window_idx:>4} {is_period:<25} {oos_period:<25} "
                f"{w.is_sharpe:>+7.3f} {w.oos_sharpe:>+8.3f} {w.ratio:>7.2f} "
                f"{w.is_return * 100:>+7.1f}% {w.oos_return * 100:>+7.1f}%"
            )

    # ── Section 5: Walk-Forward Stability Analysis ────────────────────
    heading(5, "WALK-FORWARD STABILITY ANALYSIS")
    if valid_windows:
        oos_sharpes = np.array([w.oos_sharpe for w in valid_windows])
        oos_rets = np.array([w.oos_return for w in valid_windows])
        ratios = np.array([w.ratio for w in valid_windows])
        pct_positive = float(np.sum(oos_sharpes > 0) / len(oos_sharpes) * 100)
        pct_profitable = float(np.sum(oos_rets > 0) / len(oos_rets) * 100)

        lines.append(f"  Total windows:             {len(wf_windows)}")
        lines.append(f"  Valid windows:             {len(valid_windows)}")
        lines.append(f"  Failed windows:            {len(wf_windows) - len(valid_windows)}")
        lines.append(f"")
        lines.append(f"  OOS Sharpe mean:           {float(np.mean(oos_sharpes)):+.3f}")
        lines.append(f"  OOS Sharpe median:         {float(np.median(oos_sharpes)):+.3f}")
        lines.append(f"  OOS Sharpe std:            {float(np.std(oos_sharpes)):.3f}")
        lines.append(f"  OOS Sharpe min:            {float(np.min(oos_sharpes)):+.3f}")
        lines.append(f"  OOS Sharpe max:            {float(np.max(oos_sharpes)):+.3f}")
        lines.append(f"  % windows OOS SR > 0:      {pct_positive:.1f}%")
        lines.append(f"  % windows OOS profitable:  {pct_profitable:.1f}%")
        lines.append(f"  Avg OOS/IS ratio:          {float(np.mean(ratios)):.3f}")
        lines.append(f"  Median OOS/IS ratio:       {float(np.median(ratios)):.3f}")

        if float(np.mean(ratios)) > 0.5:
            lines.append("  Assessment: ROBUST -- avg OOS/IS ratio above 0.5")
        else:
            lines.append("  Assessment: MARGINAL -- avg OOS/IS ratio below 0.5")
    else:
        lines.append("  No valid walk-forward windows.")

    # ── Section 6: Full-Period Institutional Metrics ──────────────────
    heading(6, "FULL-PERIOD INSTITUTIONAL METRICS")
    lines.append(f"  {'Metric':<32} {'Value':>18}")
    lines.append("  " + "-" * (W - 4))
    inst_metrics = [
        ("Total Return",              f"{fr.total_return * 100:+.2f}%"),
        ("Annualized Return",         f"{fr.annualized_return * 100:+.2f}%"),
        ("Sharpe Ratio",              f"{fr.sharpe_ratio:+.3f}"),
        ("Sortino Ratio",             f"{fr.sortino_ratio:+.3f}"),
        ("Calmar Ratio",              f"{fr.calmar_ratio:+.3f}"),
        ("Max Drawdown",              f"{fr.max_drawdown * 100:.2f}%"),
        ("Max DD Duration",           f"{fr.max_drawdown_duration_days} points"),
        ("Annual Volatility",         f"{fr.volatility * 100:.2f}%"),
        ("Total Trades",              f"{fr.total_trades}"),
        ("Win Rate",                  f"{fr.win_rate * 100:.1f}%"),
        ("Profit Factor",             f"{fr.profit_factor:.3f}"),
        ("Expectancy/Trade",          f"${fr.expectancy_per_trade:.2f}"),
        ("Skewness",                  f"{fr.skewness:+.3f}"),
        ("Excess Kurtosis",           f"{fr.kurtosis:+.3f}"),
        ("VaR 95% (daily)",           f"{fr.var_95 * 100:.3f}%"),
        ("CVaR 95% (daily)",          f"{fr.cvar_95 * 100:.3f}%"),
        ("Annual Turnover",           f"{fr.annual_turnover:.2f}x"),
        ("Diversification Ratio",     f"{fr.diversification_ratio:.3f}"),
    ]
    for label, val in inst_metrics:
        lines.append(f"  {label:<32} {val:>18}")

    # ── Section 7: Strategy Attribution ───────────────────────────────
    heading(7, "STRATEGY ATTRIBUTION")
    total_signals = sum(a.n_signals for a in fr.strategy_attribution.values())
    lines.append(f"  {'Strategy':<30} {'Signals':>10} {'% Contribution':>16}")
    lines.append("  " + "-" * (W - 4))
    for name, attr in sorted(
        fr.strategy_attribution.items(), key=lambda x: -x[1].n_signals
    ):
        pct = attr.n_signals / total_signals * 100 if total_signals > 0 else 0
        lines.append(f"  {name:<30} {attr.n_signals:>10,} {pct:>15.1f}%")
    lines.append(f"  {'TOTAL':<30} {total_signals:>10,} {'100.0':>15}%")

    # ── Section 8: Market Regime Breakdown ────────────────────────────
    heading(8, "MARKET REGIME BREAKDOWN")
    lines.append(f"  {'Regime':<38} {'Sharpe':>8} {'Return':>10} {'Ann Ret':>10} "
                 f"{'Max DD':>8} {'Vol':>8} {'Trades':>8}")
    lines.append("  " + "-" * (W - 4))
    for rr in regime_results:
        if rr.error:
            lines.append(f"  {rr.name:<38} {'ERR':>8} {'---':>10} {'---':>10} "
                         f"{'---':>8} {'---':>8} {'---':>8}")
        else:
            lines.append(
                f"  {rr.name:<38} {rr.sharpe:>+8.3f} {rr.total_return*100:>+9.1f}% "
                f"{rr.annualized_return*100:>+9.1f}% {rr.max_drawdown*100:>7.1f}% "
                f"{rr.volatility*100:>7.1f}% {rr.total_trades:>8}"
            )

    # ── Section 9: Alpha vs SPY ──────────────────────────────────────
    heading(9, "ALPHA vs SPY BUY-AND-HOLD")

    subheading("Full Period")
    lines.append(f"  SPY Total Return:    {spy_total * 100:+.2f}%")
    lines.append(f"  Algo Total Return:   {fr.total_return * 100:+.2f}%")
    lines.append(f"  Alpha (total):       {(fr.total_return - spy_total) * 100:+.2f}%")
    lines.append(f"  Alpha (annualized):  {benchmark_metrics.get('alpha_annual', 0) * 100:+.2f}%")

    subheading("Per-Year Alpha Table")
    lines.append(f"  {'Year':>6} {'Algo Return':>14} {'SPY Return':>14} {'Alpha':>10}")
    lines.append("  " + "-" * (W - 4))

    all_years = sorted(set(list(yearly_returns.keys()) + list(yearly_spy.keys())))
    for year in all_years:
        algo_yr = yearly_returns.get(year, 0)
        spy_yr = yearly_spy.get(year, 0)
        alpha = algo_yr - spy_yr
        lines.append(f"  {year:>6} {algo_yr*100:>+13.2f}% {spy_yr*100:>+13.2f}% {alpha*100:>+9.2f}%")

    # ── Section 10: Risk Analysis ─────────────────────────────────────
    heading(10, "RISK ANALYSIS")
    lines.append(f"  VaR 95% (daily):            {fr.var_95 * 100:.3f}%")
    lines.append(f"  CVaR 95% (daily):           {fr.cvar_95 * 100:.3f}%")
    lines.append(f"  Beta vs SPY:                {benchmark_metrics.get('beta', 0):.3f}")
    lines.append(f"  Benchmark Correlation:      {benchmark_metrics.get('benchmark_correlation', 0):.3f}")
    lines.append(f"  Information Ratio:          {benchmark_metrics.get('information_ratio', 0):+.3f}")
    lines.append(f"  Max Drawdown:               {fr.max_drawdown * 100:.2f}%")
    lines.append(f"  Max DD Duration:            {fr.max_drawdown_duration_days} equity-curve points")
    lines.append(f"  Return Skewness:            {fr.skewness:+.3f}")
    lines.append(f"  Return Excess Kurtosis:     {fr.kurtosis:+.3f}")
    lines.append(f"  Annual Volatility:          {fr.volatility * 100:.2f}%")

    fat_tail = "YES" if fr.kurtosis > 1.0 else "NO"
    neg_skew = "YES" if fr.skewness < -0.5 else "NO"
    lines.append(f"\n  Fat tails (kurtosis > 1):   {fat_tail}")
    lines.append(f"  Negative skew (< -0.5):     {neg_skew}")

    # ── Section 11: Statistical Validation ────────────────────────────
    heading(11, "STATISTICAL VALIDATION")
    n_daily_obs = len(fr.daily_returns)
    lines.append(f"  Observed Sharpe Ratio:       {fr.sharpe_ratio:+.3f}")
    lines.append(f"  Number of trials tested:     1 (single combo)")
    lines.append(f"  Number of daily observations:{n_daily_obs}")
    lines.append(f"  Deflated Sharpe Ratio:       {dsr_value:+.3f}")
    lines.append(f"  DSR p-value:                 {dsr_pval:.4f}")
    if dsr_pval < 0.05:
        lines.append("  Conclusion: SIGNIFICANT at 5% level -- low overfitting risk")
    elif dsr_pval < 0.10:
        lines.append("  Conclusion: MARGINAL -- moderate overfitting risk")
    else:
        lines.append("  Conclusion: NOT SIGNIFICANT -- possible overfitting")

    # t-test on daily excess returns
    if n_daily_obs > 2:
        daily_rf = (1 + RISK_FREE_RATE) ** (1/252) - 1
        excess = np.array(fr.daily_returns) - daily_rf
        t_stat, t_pval = sp_stats.ttest_1samp(excess, 0.0)
        lines.append(f"\n  t-test on daily excess returns:")
        lines.append(f"    t-statistic: {t_stat:.3f}")
        lines.append(f"    p-value:     {t_pval:.6f}")
        lines.append(f"    Significant at 5%: {'YES' if t_pval < 0.05 else 'NO'}")

    # ── Section 12: Monthly Return Heatmap ────────────────────────────
    heading(12, "MONTHLY RETURN HEATMAP")
    if monthly_returns:
        years = sorted(set(k[0] for k in monthly_returns.keys()))
        months = list(range(1, 13))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        lines.append(f"  {'Year':>6} " + "".join(f"{m:>8}" for m in month_names) + f"{'Annual':>10}")
        lines.append("  " + "-" * (W - 4))

        for year in years:
            row = f"  {year:>6} "
            year_rets = []
            for m in months:
                val = monthly_returns.get((year, m))
                if val is not None:
                    row += f"{val*100:>+7.1f}%"
                    year_rets.append(val)
                else:
                    row += f"{'---':>8}"
            annual = float(np.prod(1 + np.array(year_rets)) - 1) if year_rets else 0
            row += f"{annual*100:>+9.1f}%"
            lines.append(row)

    # ── Section 13: Per-Year Annual Returns Table ─────────────────────
    heading(13, "PER-YEAR ANNUAL RETURNS TABLE")
    lines.append(f"  {'Year':>6} {'Algo Return':>14} {'SPY Return':>14} {'Alpha':>10} "
                 f"{'Cum Algo':>12} {'Cum SPY':>12}")
    lines.append("  " + "-" * (W - 4))

    cum_algo = 1.0
    cum_spy = 1.0
    for year in all_years:
        algo_yr = yearly_returns.get(year, 0)
        spy_yr = yearly_spy.get(year, 0)
        alpha = algo_yr - spy_yr
        cum_algo *= (1 + algo_yr)
        cum_spy *= (1 + spy_yr)
        lines.append(
            f"  {year:>6} {algo_yr*100:>+13.2f}% {spy_yr*100:>+13.2f}% "
            f"{alpha*100:>+9.2f}% {(cum_algo-1)*100:>+11.2f}% {(cum_spy-1)*100:>+11.2f}%"
        )

    # ── Section 14: Honest Assessment ─────────────────────────────────
    heading(14, "HONEST ASSESSMENT -- CAVEATS AND LIMITATIONS")
    caveats = [
        "1. SURVIVORSHIP BIAS: All symbols in the universe survived to 2026. Delisted or "
        "bankrupt tickers are not represented, biasing results upward.",
        "2. LOOKAHEAD BIAS IN SYMBOL SELECTION: The 7 trading symbols were chosen with "
        "knowledge of their 2024-2026 performance (especially NVDA, SMCI). A truly "
        "prospective test would use only symbols known at the start of each period.",
        "3. BACKFILL BIAS: Some symbols (e.g., SMCI) have limited history pre-2018. "
        "Results for early periods reflect only a subset of the universe.",
        "4. SLIPPAGE MODEL IS SIMPLIFIED: Fixed 2 bps slippage does not capture "
        "market impact for large orders, low-liquidity stocks, or during crashes.",
        "5. COMMISSION MODEL IS SIMPLIFIED: Flat per-share commission does not account "
        "for exchange fees, SEC fees, or payment-for-order-flow.",
        "6. NO SHORT SELLING COSTS: Borrow fees, locate costs, and short squeeze risk "
        "are not modeled.",
        "7. FILL ASSUMPTIONS: All orders are assumed filled at the simulated price. "
        "In reality, limit orders may not fill and market orders may experience "
        "adverse selection.",
        "8. SINGLE COMBO TESTED: Only Combo B is run here. The prior combo selection "
        "step introduces selection bias (partially addressed by DSR).",
        "9. REGIME DEFINITIONS ARE EX-POST: The regime date ranges were defined with "
        "hindsight. A live system would need to detect regimes in real-time.",
        "10. RISK-FREE RATE ASSUMPTION: A constant 4.5% Rf is used across all 10 years. "
        "In reality, the Rf varied from ~0% (2016-2021) to ~5% (2023-2024).",
        "11. DATA QUALITY: 5-minute bar data from IBKR may have gaps, corporate action "
        "artifacts, or timestamp inconsistencies.",
        "12. STRATEGY PARAMETER STATIONARITY: Strategy parameters are held constant for "
        "10 years. Adaptive re-optimization was not performed.",
    ]
    for caveat in caveats:
        lines.append(f"  {caveat}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────
    lines.append(sep())
    lines.append("  END OF 10-YEAR ENTERPRISE BACKTEST REPORT")
    lines.append(sep())
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# Report generation — JSON
# =========================================================================

def generate_json_report(
    full_result: MultiStrategyBacktestResults,
    hard_split: HardSplitResults,
    wf_windows: List[WFWindowResult],
    regime_results: List[RegimeResult],
    benchmark_metrics: Dict[str, float],
    dsr_value: float,
    dsr_pval: float,
    monthly_returns: Dict[Tuple[int, int], float],
    yearly_returns: Dict[int, float],
    yearly_spy: Dict[int, float],
    coverage: Dict[str, Dict],
) -> Dict[str, Any]:
    """Build JSON-serializable report."""
    fr = full_result

    # Sample equity curve daily (take every ~78th point to approximate daily)
    ec = fr.equity_curve
    step = max(1, len(ec) // (len(fr.daily_returns) + 1)) if fr.daily_returns else 1
    sampled_equity = ec[::step]

    # Serialize timestamps
    ts_strings = [t.isoformat() for t in fr.timestamps] if fr.timestamps else []

    report = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "strategies": COMBO_B_STRATEGIES,
            "trading_symbols": TRADING_SYMBOLS,
            "reference_symbols": REFERENCE_SYMBOLS,
            "initial_capital": INITIAL_CAPITAL,
            "risk_free_rate": RISK_FREE_RATE,
        },
        "full_period_metrics": {
            "total_return": fr.total_return,
            "annualized_return": fr.annualized_return,
            "sharpe_ratio": fr.sharpe_ratio,
            "sortino_ratio": fr.sortino_ratio,
            "calmar_ratio": fr.calmar_ratio,
            "max_drawdown": fr.max_drawdown,
            "max_drawdown_duration_days": fr.max_drawdown_duration_days,
            "volatility": fr.volatility,
            "total_trades": fr.total_trades,
            "win_rate": fr.win_rate,
            "profit_factor": fr.profit_factor,
            "expectancy_per_trade": fr.expectancy_per_trade,
            "skewness": fr.skewness,
            "kurtosis": fr.kurtosis,
            "var_95": fr.var_95,
            "cvar_95": fr.cvar_95,
            "annual_turnover": fr.annual_turnover,
        },
        "benchmark_metrics": benchmark_metrics,
        "statistical_validation": {
            "deflated_sharpe": dsr_value,
            "dsr_p_value": dsr_pval,
        },
        "hard_split": {},
        "walk_forward_windows": [],
        "regime_breakdown": [],
        "equity_curve_daily": sampled_equity,
        "daily_returns": fr.daily_returns,
        "timestamps": ts_strings,
        "monthly_returns": {f"{y}-{m:02d}": v for (y, m), v in monthly_returns.items()},
        "yearly_returns": {str(y): v for y, v in yearly_returns.items()},
        "yearly_spy_returns": {str(y): v for y, v in yearly_spy.items()},
        "coverage": {sym: {k: str(v) if isinstance(v, date) else v
                           for k, v in info.items()}
                     for sym, info in coverage.items()},
        "strategy_attribution": {
            name: {"n_signals": attr.n_signals}
            for name, attr in fr.strategy_attribution.items()
        },
    }

    # Hard split
    if hard_split.is_result:
        report["hard_split"]["is"] = {
            "date_range": [str(hard_split.is_date_range[0]), str(hard_split.is_date_range[1])],
            "total_return": hard_split.is_result.total_return,
            "annualized_return": hard_split.is_result.annualized_return,
            "sharpe_ratio": hard_split.is_result.sharpe_ratio,
            "sortino_ratio": hard_split.is_result.sortino_ratio,
            "max_drawdown": hard_split.is_result.max_drawdown,
            "volatility": hard_split.is_result.volatility,
            "total_trades": hard_split.is_result.total_trades,
            "win_rate": hard_split.is_result.win_rate,
            "profit_factor": hard_split.is_result.profit_factor,
        }
    if hard_split.oos_result:
        report["hard_split"]["oos"] = {
            "date_range": [str(hard_split.oos_date_range[0]), str(hard_split.oos_date_range[1])],
            "total_return": hard_split.oos_result.total_return,
            "annualized_return": hard_split.oos_result.annualized_return,
            "sharpe_ratio": hard_split.oos_result.sharpe_ratio,
            "sortino_ratio": hard_split.oos_result.sortino_ratio,
            "max_drawdown": hard_split.oos_result.max_drawdown,
            "volatility": hard_split.oos_result.volatility,
            "total_trades": hard_split.oos_result.total_trades,
            "win_rate": hard_split.oos_result.win_rate,
            "profit_factor": hard_split.oos_result.profit_factor,
        }

    # Walk-forward windows
    for w in wf_windows:
        report["walk_forward_windows"].append({
            "window_idx": w.window_idx,
            "is_start_date": str(w.is_start_date),
            "is_end_date": str(w.is_end_date),
            "oos_start_date": str(w.oos_start_date),
            "oos_end_date": str(w.oos_end_date),
            "is_sharpe": w.is_sharpe,
            "oos_sharpe": w.oos_sharpe,
            "is_return": w.is_return,
            "oos_return": w.oos_return,
            "ratio": w.ratio,
            "error": w.error,
        })

    # Regime breakdown
    for rr in regime_results:
        report["regime_breakdown"].append({
            "name": rr.name,
            "start_date": str(rr.start_date),
            "end_date": str(rr.end_date),
            "sharpe": rr.sharpe,
            "total_return": rr.total_return,
            "annualized_return": rr.annualized_return,
            "max_drawdown": rr.max_drawdown,
            "total_trades": rr.total_trades,
            "volatility": rr.volatility,
            "error": rr.error,
        })

    return report


# =========================================================================
# CSV export
# =========================================================================

def export_equity_curve_csv(
    full_result: MultiStrategyBacktestResults,
    output_path: Path,
) -> None:
    """Export daily equity curve to CSV."""
    ec = full_result.equity_curve
    ts = full_result.timestamps

    # The equity curve has one more entry than timestamps (initial capital)
    # Map equity to dates: use timestamps for the daily snapshots
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "equity"])

        # Write initial
        if ts:
            writer.writerow([ts[0].date(), ec[0]])

        for i, t in enumerate(ts):
            eq_idx = i + 1 if i + 1 < len(ec) else len(ec) - 1
            writer.writerow([t.date(), ec[eq_idx]])


def export_monthly_returns_csv(
    monthly_returns: Dict[Tuple[int, int], float],
    output_path: Path,
) -> None:
    """Export monthly returns to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "month", "return"])
        for (year, month), ret in sorted(monthly_returns.items()):
            writer.writerow([year, month, f"{ret:.6f}"])


def export_walk_forward_csv(
    windows: List[WFWindowResult],
    output_path: Path,
) -> None:
    """Export walk-forward results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window", "is_start", "is_end", "oos_start", "oos_end",
                         "IS_sharpe", "OOS_sharpe", "ratio", "IS_ret", "OOS_ret", "error"])
        for w in windows:
            writer.writerow([
                w.window_idx,
                str(w.is_start_date),
                str(w.is_end_date),
                str(w.oos_start_date),
                str(w.oos_end_date),
                f"{w.is_sharpe:.4f}",
                f"{w.oos_sharpe:.4f}",
                f"{w.ratio:.4f}",
                f"{w.is_return:.6f}",
                f"{w.oos_return:.6f}",
                w.error or "",
            ])


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    t_global_start = time.time()

    print()
    print("#" * 70)
    print("#  10-YEAR ENTERPRISE-GRADE VALIDATED BACKTEST")
    print("#  Combo B: 8-Strategy Ensemble")
    print(f"#  {START_DATE} to {END_DATE}")
    print("#" * 70)
    print()

    # ── Step 1: Load data ─────────────────────────────────────────────
    print(f"[Step 1/7] Loading data ...")
    t0 = time.time()
    bar_data, coverage = load_bar_data()
    print(f"  Data loading took {time.time() - t0:.1f}s\n")

    # Store in module-level global so forked workers can access it
    # without pickling (copy-on-write via fork)
    global _SHARED_BAR_DATA
    _SHARED_BAR_DATA = bar_data

    # ── Step 2: Extract SPY benchmark returns ─────────────────────────
    print(f"[Step 2/7] Extracting SPY daily returns for benchmark ...")
    spy_daily, spy_dates = extract_spy_daily_returns(bar_data)
    print(f"  SPY: {len(spy_daily)} daily returns "
          f"({spy_dates[0] if spy_dates else 'N/A'} to {spy_dates[-1] if spy_dates else 'N/A'})")
    spy_total_ret = total_return_from_daily(spy_daily) if len(spy_daily) > 0 else 0
    spy_ann_ret = annualized_return_from_daily(spy_daily) if len(spy_daily) > 0 else 0
    print(f"  SPY total return: {spy_total_ret * 100:+.2f}%")
    print(f"  SPY annualized return: {spy_ann_ret * 100:+.2f}%\n")

    # ── Step 3: Walk-forward analysis ─────────────────────────────────
    print(f"[Step 3/7] Running walk-forward analysis ...")
    t0 = time.time()
    wf_windows = run_walk_forward(bar_data)
    print(f"  Walk-forward took {time.time() - t0:.1f}s\n")

    # ── Step 4: Hard IS/OOS split ─────────────────────────────────────
    print(f"[Step 4/7] Running hard IS/OOS split ...")
    t0 = time.time()
    hard_split = run_hard_split(bar_data)
    print(f"  Hard split took {time.time() - t0:.1f}s\n")

    # ── Step 5: Full-period backtest ──────────────────────────────────
    print(f"[Step 5/7] Running full-period backtest ...")
    t0 = time.time()
    full_result = run_full_period(bar_data)
    print(f"  Full-period backtest took {time.time() - t0:.1f}s\n")

    # ── Step 6: Market regime breakdown ───────────────────────────────
    print(f"[Step 6/7] Running market regime breakdown ...")
    t0 = time.time()
    regime_results = run_regime_breakdown(bar_data)
    print(f"  Regime breakdown took {time.time() - t0:.1f}s\n")

    # ── Step 7: Compute derived metrics and generate reports ──────────
    print(f"[Step 7/7] Computing derived metrics and generating reports ...")

    # Benchmark metrics (alpha, beta, IR, correlation)
    algo_daily = np.array(full_result.daily_returns)
    benchmark_metrics = compute_benchmark_metrics(algo_daily, spy_daily)

    # Deflated Sharpe Ratio
    # n_trials=1 since we are only testing Combo B here; the prior combo
    # selection from run_validated_backtest.py tested 4 combos.
    dsr_value, dsr_pval = compute_deflated_sharpe(
        observed_sharpe=full_result.sharpe_ratio,
        n_trials=4,  # 4 combos were tested in the selection step
        n_daily_obs=len(full_result.daily_returns),
        skewness=full_result.skewness,
        kurtosis=full_result.kurtosis + 3.0,  # excess -> raw
    )

    # Monthly and yearly returns
    monthly_returns = compute_monthly_returns(full_result.daily_returns, full_result.timestamps)
    yearly_returns = compute_yearly_returns(full_result.daily_returns, full_result.timestamps)
    yearly_spy = compute_yearly_spy_returns(spy_daily, spy_dates)

    total_elapsed = time.time() - t_global_start

    # ── Generate text report ──────────────────────────────────────────
    report_text = generate_text_report(
        full_result=full_result,
        hard_split=hard_split,
        wf_windows=wf_windows,
        regime_results=regime_results,
        spy_daily=spy_daily,
        spy_dates=spy_dates,
        coverage=coverage,
        benchmark_metrics=benchmark_metrics,
        dsr_value=dsr_value,
        dsr_pval=dsr_pval,
        monthly_returns=monthly_returns,
        yearly_returns=yearly_returns,
        yearly_spy=yearly_spy,
        total_elapsed=total_elapsed,
    )

    # ── Generate JSON report ──────────────────────────────────────────
    report_json = generate_json_report(
        full_result=full_result,
        hard_split=hard_split,
        wf_windows=wf_windows,
        regime_results=regime_results,
        benchmark_metrics=benchmark_metrics,
        dsr_value=dsr_value,
        dsr_pval=dsr_pval,
        monthly_returns=monthly_returns,
        yearly_returns=yearly_returns,
        yearly_spy=yearly_spy,
        coverage=coverage,
    )

    # ── Write outputs ─────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Text report
    txt_path = RESULTS_DIR / "10yr_validated_report.txt"
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"  Text report:    {txt_path}")

    # JSON report
    json_path = RESULTS_DIR / "10yr_validated_report.json"
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"  JSON report:    {json_path}")

    # CSV: equity curve
    eq_csv_path = RESULTS_DIR / "10yr_equity_curve.csv"
    export_equity_curve_csv(full_result, eq_csv_path)
    print(f"  Equity CSV:     {eq_csv_path}")

    # CSV: monthly returns
    mr_csv_path = RESULTS_DIR / "10yr_monthly_returns.csv"
    export_monthly_returns_csv(monthly_returns, mr_csv_path)
    print(f"  Monthly CSV:    {mr_csv_path}")

    # CSV: walk-forward
    wf_csv_path = RESULTS_DIR / "10yr_walk_forward.csv"
    export_walk_forward_csv(wf_windows, wf_csv_path)
    print(f"  Walk-fwd CSV:   {wf_csv_path}")

    # ── Print report to stdout ────────────────────────────────────────
    print("\n" + report_text)

    total_elapsed = time.time() - t_global_start
    print(f"\nTotal elapsed time: {total_elapsed/3600:.2f} hours ({total_elapsed:.0f}s)")
    print("Done.")


if __name__ == "__main__":
    main()
