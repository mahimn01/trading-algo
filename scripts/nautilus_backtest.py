#!/usr/bin/env python3
"""
Backtest the repo's strategies on NautilusTrader.

What it does:
    Runs one or more of the repo's daily strategies (unchanged
    ``TradingStrategy`` adapters behind ``MultiStrategyController``) through
    NautilusTrader's event-driven engine. Decisions are made at each session's
    close and fill at the next session's open, priced with the repo's realistic
    cost model. It then compares the strategy with a benchmark (SPY by default)
    using NautilusTrader's benchmark statistics, and with an equal-weight
    buy-and-hold of the same symbols.

Why:
    An independent, event-driven engine checks the repo's own backtester. With
    ``--compare`` the same run is repeated on ``MultiStrategyBacktestRunner``
    with identical inputs, and the two are reconciled trade by trade.

How to run (from the repo root, with the NautilusTrader requirements installed):
    .venv/bin/python scripts/nautilus_backtest.py --strategies PureMomentum \\
        --universe DEV_MEGACAP --allow-dev --start 2016-06-01 --end 2026-03-31 \\
        --benchmark SPY --compare

    Pick symbols with --universe (an index name or a DEV_* universe) or
    --symbols (a comma list of tickers). DEV_* universes and ticker lists are
    survivorship-biased, so they need --allow-dev and are labelled
    exploratory in the output.

What it changes:
    Read-only apart from writing result files to
    backtest_results/nautilus/<run id>/ (summary.json, nav.csv, fills.csv,
    signals.csv, positions.csv, tearsheet.html). It never connects to a broker.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=RuntimeWarning)

import nautilus_trader  # noqa: E402

from trading_algo.data.universe import UniverseResolver  # noqa: E402
from trading_algo.multi_strategy.adapters import (  # noqa: E402
    FlowPressureAdapter,
    MomentumStrategyAdapter,
    ReversalStrategyAdapter,
)
from trading_algo.multi_strategy.controller import MultiStrategyController  # noqa: E402
from trading_algo.multi_strategy.protocol import TradingStrategy  # noqa: E402
from trading_algo.nautilus_backtest import (  # noqa: E402
    NautilusBacktestConfig,
    NautilusBacktestResult,
    ParityReport,
    benchmark_statistics,
    buy_and_hold,
    explicit_controller_config,
    load_daily_bars,
    price_returns,
    run_nautilus_backtest,
    run_parity,
    write_report,
)
from trading_algo.nautilus_backtest.benchmark import PRICE_RETURN_LABEL, simple_stats  # noqa: E402
from trading_algo.nautilus_backtest.data import DEFAULT_DATA_DIR  # noqa: E402


@dataclass(frozen=True)
class StrategySpec:
    """A daily strategy adapter and the fewest symbols it needs."""

    factory: Callable[[], TradingStrategy]
    min_symbols: int


STRATEGIES: dict[str, StrategySpec] = {
    "PureMomentum": StrategySpec(MomentumStrategyAdapter, min_symbols=1),
    "FlowPressure": StrategySpec(FlowPressureAdapter, min_symbols=1),
    "ShortTermReversal": StrategySpec(ReversalStrategyAdapter, min_symbols=3),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--strategies", default="PureMomentum", help=f"comma list from: {', '.join(STRATEGIES)}"
    )
    pick = parser.add_mutually_exclusive_group(required=True)
    pick.add_argument("--universe", help="index name (e.g. SP500) or DEV_* universe")
    pick.add_argument("--symbols", help="comma list of tickers, e.g. AAPL,MSFT")
    parser.add_argument(
        "--allow-dev",
        action="store_true",
        help="allow survivorship-biased DEV_* or literal universes",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--costs", choices=["realistic", "legacy"], default="realistic")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="legacy costs only")
    parser.add_argument(
        "--commission-per-share", type=float, default=0.0035, help="legacy costs only"
    )
    parser.add_argument("--max-position-pct", type=float, default=0.25)
    parser.add_argument("--max-gross-exposure", type=float, default=1.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.045)
    parser.add_argument("--no-vol-management", action="store_true")
    parser.add_argument("--no-risk-controller", action="store_true")
    parser.add_argument("--benchmark", default="SPY", help="benchmark ticker, or 'none'")
    parser.add_argument(
        "--compare", action="store_true", help="also run MultiStrategyBacktestRunner and reconcile"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "backtest_results" / "nautilus")
    parser.add_argument("--no-tearsheet", action="store_true")
    return parser.parse_args(argv)


def resolve_universe(args: argparse.Namespace) -> tuple[list[str], bool]:
    """Resolve symbols through ``UniverseResolver``. Returns (symbols, exploratory)."""
    resolver = UniverseResolver()
    if args.symbols:
        if not args.allow_dev:
            raise SystemExit("A ticker list is survivorship-biased. Pass --allow-dev.")
        return resolver.get_universe(
            [s.strip() for s in args.symbols.split(",") if s.strip()]
        ), True
    symbols = resolver.get_universe(args.universe, args.start, allow_dev=args.allow_dev)
    return symbols, args.universe.startswith("DEV_")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args(argv)
    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies {unknown}. Choose from {list(STRATEGIES)}.")

    symbols, exploratory = resolve_universe(args)
    universe = args.universe or args.symbols
    if not symbols:
        raise SystemExit(f"Universe {universe!r} resolved to no symbols.")
    for name in names:
        if len(symbols) < STRATEGIES[name].min_symbols:
            raise SystemExit(f"{name} needs at least {STRATEGIES[name].min_symbols} symbols.")

    frames = load_daily_bars(symbols, args.data_dir, args.start, args.end)
    controller_config = explicit_controller_config(
        names,
        len(symbols),
        vol_management=not args.no_vol_management,
        risk_controller=not args.no_risk_controller,
    )

    def controller_factory() -> MultiStrategyController:
        controller = MultiStrategyController(controller_config)
        for name in names:
            controller.register(STRATEGIES[name].factory())
        return controller

    config = NautilusBacktestConfig(
        initial_capital=args.capital,
        costs=args.costs,
        slippage_bps=args.slippage_bps,
        commission_per_share=args.commission_per_share,
        max_position_pct=args.max_position_pct,
        max_gross_exposure=args.max_gross_exposure,
        risk_free_rate=args.risk_free_rate,
    )
    result = run_nautilus_backtest(controller_factory(), frames, config)
    reported = result.reported_returns

    benchmark = benchmark_returns = None
    benchmark_label = ""
    if args.benchmark.lower() != "none":
        bench_frame = load_daily_bars([args.benchmark], args.data_dir, args.start, args.end)
        benchmark_returns = price_returns(bench_frame[args.benchmark]["close"])
        benchmark_returns = benchmark_returns.loc[reported.index[0] : reported.index[-1]]
        benchmark_label = f"{args.benchmark} {PRICE_RETURN_LABEL}"
        benchmark = benchmark_statistics(reported, benchmark_returns, args.risk_free_rate)

    start_day = result.nav.index[result.nav.index.get_loc(reported.index[0]) - 1]
    baseline_nav = buy_and_hold(frames, start_day, result.nav[start_day])
    baseline = {
        "name": f"equal-weight buy-and-hold of {', '.join(symbols)} (frictionless, "
        f"{PRICE_RETURN_LABEL})",
        **simple_stats(baseline_nav.loc[: reported.index[-1]]),
    }
    parity = run_parity(controller_factory, frames, config) if args.compare else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + "-".join(names)
    first, last = (
        min(f.index[0] for f in frames.values()),
        max(f.index[-1] for f in frames.values()),
    )
    run_info = {
        "id": run_id,
        "title": f"{' + '.join(names)} on {universe} (NautilusTrader {nautilus_trader.__version__})",
        "engine": f"NautilusTrader {nautilus_trader.__version__}",
        "strategies": names,
        "universe": universe,
        "symbols": symbols,
        "exploratory_universe": exploratory,
        "survivorship_note": (
            "Universe chosen with hindsight (survivorship-biased); results are exploratory."
            if exploratory
            else "Point-in-time universe."
        ),
        "data_dir": str(args.data_dir),
        "data_range": [first.date(), last.date()],
        "reported_range": [reported.index[0], reported.index[-1]],
        "price_adjustment": PRICE_RETURN_LABEL,
    }
    out_dir = write_report(
        result,
        args.out / run_id,
        run_info=run_info,
        controller_config=controller_config,
        benchmark=benchmark,
        benchmark_label=benchmark_label,
        benchmark_returns=benchmark_returns,
        baseline=baseline,
        parity=parity,
        tearsheet=not args.no_tearsheet,
    )
    print_summary(result, benchmark, benchmark_label, baseline, parity, run_info, out_dir)
    return 0 if parity is None or parity.status != "mismatch" else 1


def print_summary(
    result: NautilusBacktestResult,
    benchmark: dict[str, float] | None,
    benchmark_label: str,
    baseline: dict[str, Any],
    parity: ParityReport | None,
    run_info: dict[str, Any],
    out_dir: Path,
) -> None:
    """Print a plain-English summary of the run."""
    m = result.metrics
    reported = result.reported_returns
    print(f"\n{run_info['title']}")
    print(
        f"  Reported period: {reported.index[0]} to {reported.index[-1]} ({len(reported)} sessions)"
    )
    if run_info["exploratory_universe"]:
        print(f"  Note: {run_info['survivorship_note']}")
    print(f"  Final NAV ${result.final_nav:,.2f} from ${result.config.initial_capital:,.2f}")
    print(
        f"  Annual return {m.annualized_return:.2%}, volatility {m.volatility:.2%}, "
        f"max drawdown {m.max_drawdown:.2%}"
    )
    print(
        f"  Sharpe {result.sharpe_rf0:.2f} (rf=0), {m.sharpe_ratio:.2f} "
        f"(rf={result.config.risk_free_rate:.1%})"
    )
    costs = result.cost_breakdown
    print(
        f"  Costs: commission ${costs['commission']:,.2f}, slippage ${costs['slippage']:,.2f}, "
        f"{len(result.fills)} fills"
    )
    if benchmark:
        print(
            f"  vs {benchmark_label}: beta {benchmark['Beta']:.2f}, "
            f"alpha {benchmark['Alpha (252 days)']:.2%}, "
            f"information ratio {benchmark['Information Ratio (252 days)']:.2f}"
        )
    print(
        f"  Buy-and-hold baseline: annual return {baseline['annualized_return']:.2%}, "
        f"Sharpe {baseline['sharpe_rf0']:.2f}"
    )
    if parity is not None:
        print(
            f"  Parity vs MultiStrategyBacktestRunner: {parity.status}"
            + (
                f" (max NAV difference {parity.max_nav_rel_diff:.2e})"
                if parity.sessions_compared
                else ""
            )
            + (f" - {parity.note}" if parity.status == "not_comparable" else "")
        )
    print(f"  Results: {out_dir}\n")


if __name__ == "__main__":
    sys.exit(main())
