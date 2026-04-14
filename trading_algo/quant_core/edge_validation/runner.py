from __future__ import annotations

import time
from datetime import time as dtime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from trading_algo.quant_core.edge_validation.config import EdgeValidationConfig
from trading_algo.quant_core.edge_validation.types import (
    EdgeValidationReport,
    ExcursionComparison,
    MonteCarloResult,
    PatternEdgeReport,
    SignificanceResult,
    Verdict,
    WalkForwardEdgeResult,
)
from trading_algo.quant_core.edge_validation.data_fetcher import FuturesDataFetcher
from trading_algo.quant_core.edge_validation.patterns import (
    GapFadeDetector,
    ORBDetector,
    PatternDetector,
    VWAPReversionDetector,
)
from trading_algo.quant_core.edge_validation.analysis.excursion import (
    compare_to_random,
    compute_excursions,
)
from trading_algo.quant_core.edge_validation.analysis.significance import (
    run_all_significance,
)
from trading_algo.quant_core.edge_validation.analysis.monte_carlo import (
    bootstrap_confidence_intervals,
    random_entry_permutation_test,
    whites_reality_check,
)
from trading_algo.quant_core.edge_validation.analysis.walk_forward import (
    walk_forward_edge_test,
)
from trading_algo.quant_core.edge_validation.analysis.regime import (
    classify_trading_days,
    regime_conditional_test,
)
from trading_algo.quant_core.edge_validation.report import EdgeValidationReporter


MIN_OCCURRENCES = 20
ENTRY_START = dtime(10, 0)
ENTRY_END = dtime(15, 0)
EOD_HOUR = 16
EOD_MINUTE = 0


class EdgeValidationRunner:
    def __init__(self, config: EdgeValidationConfig | None = None) -> None:
        self._config = config or EdgeValidationConfig()

    def run(self, offline: bool = False) -> EdgeValidationReport:
        t_total = time.perf_counter()
        cfg = self._config
        report = EdgeValidationReport()

        fetcher = FuturesDataFetcher(cfg, offline=offline)
        bars_by_symbol: dict[str, pd.DataFrame] = {}

        for symbol in cfg.symbols:
            t0 = time.perf_counter()
            print(f"Fetching {symbol} data...", flush=True)
            bars_by_symbol[symbol] = fetcher.get_bars(symbol)
            print(f"  {len(bars_by_symbol[symbol])} bars in {time.perf_counter() - t0:.1f}s", flush=True)

        detectors: list[PatternDetector] = []
        for rm in cfg.orb_range_minutes:
            detectors.append(ORBDetector(range_minutes=rm))
        detectors.append(GapFadeDetector(
            min_gap_pct=cfg.gap_fade_min_gap_pct,
            max_gap_pct=cfg.gap_fade_max_gap_pct,
        ))
        detectors.append(VWAPReversionDetector(
            sigma_threshold=cfg.vwap_deviation_sigma,
            min_bars_for_std=cfg.vwap_min_bars_for_std,
        ))

        total_detectors = len(detectors) * len(cfg.symbols)
        all_strategy_returns: dict[str, NDArray[np.float64]] = {}

        for symbol in cfg.symbols:
            bars = bars_by_symbol[symbol]
            all_bar_returns = self._compute_all_bar_returns(bars)
            report.symbol_reports[symbol] = []

            for detector in detectors:
                t0 = time.perf_counter()
                pattern_name = detector.name
                key = f"{symbol}_{pattern_name}"
                print(f"Detecting {pattern_name} on {symbol}...", end=" ", flush=True)

                occurrences = detector.detect(bars)
                print(f"found {len(occurrences)} occurrences ({time.perf_counter() - t0:.1f}s)", flush=True)
                report.total_patterns_tested += 1

                if len(occurrences) < MIN_OCCURRENCES:
                    print(f"  WARNING: <{MIN_OCCURRENCES} occurrences, skipping {key}", flush=True)
                    report.patterns_failed += 1
                    continue

                t0 = time.perf_counter()
                print(f"  Computing excursions...", end=" ", flush=True)
                exc = compare_to_random(occurrences, bars, n_random=10_000, seed=cfg.random_seed)
                print(f"{time.perf_counter() - t0:.1f}s", flush=True)

                returns = exc.pattern.realized_pnl
                if len(returns) == 0:
                    print(f"  WARNING: no valid returns for {key}, skipping", flush=True)
                    report.patterns_failed += 1
                    continue

                all_strategy_returns[key] = returns

                t0 = time.perf_counter()
                print(f"  Running significance tests...", end=" ", flush=True)
                sig = run_all_significance(returns, n_trials=total_detectors, alpha=cfg.significance_level)
                print(f"{time.perf_counter() - t0:.1f}s", flush=True)

                t0 = time.perf_counter()
                print(f"  Running Monte Carlo...", end=" ", flush=True)
                bootstrap_cis = bootstrap_confidence_intervals(
                    returns, n_resamples=cfg.n_bootstrap,
                    confidence=1.0 - cfg.significance_level, seed=cfg.random_seed,
                )
                perm = random_entry_permutation_test(
                    returns, all_bar_returns,
                    n_permutations=cfg.n_permutations, seed=cfg.random_seed,
                    alpha=cfg.significance_level,
                )
                sharpe_ci = bootstrap_cis[0]
                pf_ci = bootstrap_cis[1] if len(bootstrap_cis) > 1 else bootstrap_cis[0]
                mr_ci = bootstrap_cis[2] if len(bootstrap_cis) > 2 else bootstrap_cis[0]
                mc = MonteCarloResult(
                    bootstrap_sharpe=sharpe_ci,
                    bootstrap_profit_factor=pf_ci,
                    bootstrap_mean_return=mr_ci,
                    permutation=perm,
                )
                print(f"{time.perf_counter() - t0:.1f}s", flush=True)

                t0 = time.perf_counter()
                print(f"  Running walk-forward...", end=" ", flush=True)
                wf = walk_forward_edge_test(
                    occurrences, returns,
                    train_days=cfg.wf_train_days, test_days=cfg.wf_test_days,
                    step_days=cfg.wf_step_days,
                )
                print(f"{time.perf_counter() - t0:.1f}s", flush=True)

                t0 = time.perf_counter()
                print(f"  Running regime analysis...", end=" ", flush=True)
                regime_labels = classify_trading_days(bars)
                regime = regime_conditional_test(
                    occurrences, returns, regime_labels, alpha=cfg.significance_level,
                )
                print(f"{time.perf_counter() - t0:.1f}s", flush=True)

                verdict, cond_met, cond_total = self._compute_verdict(sig, mc, wf, exc)

                wins = float(np.sum(returns > 0))
                losses_sum = float(np.abs(returns[returns < 0].sum())) if np.any(returns < 0) else 0.0
                gains_sum = float(returns[returns > 0].sum()) if np.any(returns > 0) else 0.0
                profit_factor = gains_sum / losses_sum if losses_sum > 0 else float("inf")

                pattern_report = PatternEdgeReport(
                    pattern_name=pattern_name,
                    symbol=symbol,
                    n_occurrences=len(returns),
                    win_rate=float(wins / len(returns)),
                    mean_return_points=float(np.mean(returns)),
                    median_return_points=float(np.median(returns)),
                    profit_factor=profit_factor,
                    excursion=exc,
                    significance=sig,
                    monte_carlo=mc,
                    walk_forward=wf,
                    regime=regime,
                    verdict=verdict,
                    conditions_met=cond_met,
                    conditions_total=cond_total,
                )
                report.symbol_reports[symbol].append(pattern_report)

                if verdict == "PASS":
                    report.patterns_passed += 1
                elif verdict == "WEAK":
                    report.patterns_weak += 1
                else:
                    report.patterns_failed += 1

                print(f"  -> {verdict} ({cond_met}/{cond_total})", flush=True)

        if len(all_strategy_returns) > 1:
            t0 = time.perf_counter()
            print(f"\nRunning White's Reality Check across {len(all_strategy_returns)} strategies...", end=" ", flush=True)
            report.whites_rc = whites_reality_check(
                all_strategy_returns,
                n_bootstrap=cfg.n_bootstrap, seed=cfg.random_seed,
                alpha=cfg.significance_level,
            )
            print(f"{time.perf_counter() - t0:.1f}s", flush=True)

        reporter = EdgeValidationReporter()
        point_values = {s: cfg.get_point_value(s) for s in cfg.symbols}
        reporter.render(report, point_values=point_values)

        elapsed = time.perf_counter() - t_total
        print(f"\nTotal time: {elapsed:.1f}s", flush=True)

        return report

    def _compute_all_bar_returns(self, bars: pd.DataFrame) -> NDArray[np.float64]:
        bars = bars.sort_index()
        groups = bars.groupby(bars.index.date)
        returns: list[float] = []

        for day, day_bars in groups:
            times = day_bars.index.time
            eod_mask = times >= dtime(EOD_HOUR, EOD_MINUTE)
            if not eod_mask.any():
                eod_bars = day_bars
                if eod_bars.empty:
                    continue
                eod_close = float(day_bars.iloc[-1]["close"])
            else:
                eod_idx = day_bars.index[eod_mask][0]
                eod_close = float(day_bars.loc[eod_idx, "close"])

            eligible_mask = (times >= ENTRY_START) & (times < ENTRY_END)
            eligible = day_bars.loc[eligible_mask]

            for _, row in eligible.iterrows():
                returns.append(eod_close - float(row["close"]))

        return np.array(returns, dtype=np.float64) if returns else np.array([], dtype=np.float64)

    def _compute_verdict(
        self,
        sig: SignificanceResult,
        mc: MonteCarloResult,
        wf: WalkForwardEdgeResult,
        exc: ExcursionComparison,
    ) -> tuple[Verdict, int, int]:
        conditions_total = 6
        met = 0

        alpha = self._config.significance_level

        if sig.binomial.p_value < alpha:
            met += 1
        if sig.ttest.p_value < alpha:
            met += 1
        if sig.psr.psr > 0.80:
            met += 1
        if wf.wf_efficiency > 0.50:
            met += 1
        if wf.pct_folds_positive >= 0.50:
            met += 1
        if exc.ks_pnl.p_value < alpha:
            met += 1

        if met >= 5:
            verdict: Verdict = "PASS"
        elif met >= 3:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        return verdict, met, conditions_total
