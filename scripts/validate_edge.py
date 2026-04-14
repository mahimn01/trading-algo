#!/usr/bin/env python3
"""Validate statistical edge of futures day trading strategies.

Usage:
    python scripts/validate_edge.py                        # Default: NQ, all patterns
    python scripts/validate_edge.py --symbols ES NQ        # Multiple symbols
    python scripts/validate_edge.py --offline              # Cache only, no IBKR
    python scripts/validate_edge.py --patterns orb gap     # Subset of patterns
    python scripts/validate_edge.py --orb-minutes 60       # Only 60-min ORB
    python scripts/validate_edge.py --bootstrap 50000      # More bootstrap samples
    python scripts/validate_edge.py --quick                # Reduced bootstrap/permutation (1000)
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.quant_core.edge_validation.config import EdgeValidationConfig
from trading_algo.quant_core.edge_validation.runner import EdgeValidationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate statistical edge of futures day trading strategies",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to validate (default: NQ)")
    parser.add_argument("--offline", action="store_true", help="Use cached data only, no IBKR connection")
    parser.add_argument("--patterns", nargs="+", default=None, choices=["orb", "gap", "vwap"],
                        help="Subset of patterns to test")
    parser.add_argument("--orb-minutes", nargs="+", type=int, default=None,
                        help="ORB range minutes (default: 15 30 60)")
    parser.add_argument("--bootstrap", type=int, default=None, help="Number of bootstrap resamples")
    parser.add_argument("--permutations", type=int, default=None, help="Number of permutation test iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1000 bootstrap/permutation samples")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--alpha", type=float, default=None, help="Significance level (default: 0.05)")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EdgeValidationConfig:
    kwargs: dict = {}

    if args.symbols:
        kwargs["symbols"] = tuple(args.symbols)

    if args.orb_minutes:
        kwargs["orb_range_minutes"] = tuple(args.orb_minutes)
    elif args.patterns and "orb" not in args.patterns:
        kwargs["orb_range_minutes"] = ()

    if args.patterns and "gap" not in args.patterns:
        kwargs["gap_fade_min_gap_pct"] = float("inf")
    if args.patterns and "vwap" not in args.patterns:
        kwargs["vwap_deviation_sigma"] = float("inf")

    if args.quick:
        kwargs["n_bootstrap"] = 1_000
        kwargs["n_permutations"] = 1_000
    if args.bootstrap:
        kwargs["n_bootstrap"] = args.bootstrap
    if args.permutations:
        kwargs["n_permutations"] = args.permutations
    if args.seed is not None:
        kwargs["random_seed"] = args.seed
    if args.start_date:
        kwargs["start_date"] = args.start_date
    if args.end_date:
        kwargs["end_date"] = args.end_date
    if args.alpha is not None:
        kwargs["significance_level"] = args.alpha

    return EdgeValidationConfig(**kwargs)


def main() -> None:
    args = parse_args()
    config = build_config(args)

    print("=" * 70, flush=True)
    print("  Edge Validation — Futures Day Trading Strategy Analysis", flush=True)
    print("=" * 70, flush=True)
    print(f"Symbols:      {', '.join(config.symbols)}", flush=True)
    print(f"Date range:   {config.start_date} to {config.end_date}", flush=True)
    print(f"ORB minutes:  {config.orb_range_minutes}", flush=True)
    print(f"Bootstrap:    {config.n_bootstrap:,}", flush=True)
    print(f"Permutations: {config.n_permutations:,}", flush=True)
    print(f"Alpha:        {config.significance_level}", flush=True)
    print(f"Seed:         {config.random_seed}", flush=True)
    if args.offline:
        print("Mode:         OFFLINE (cache only)", flush=True)
    if args.patterns:
        print(f"Patterns:     {', '.join(args.patterns)}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    t0 = time.perf_counter()

    try:
        runner = EdgeValidationRunner(config)
        report = runner.run(offline=args.offline)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
        raise

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f}s", flush=True)
    print(f"Results: {report.patterns_passed} PASS / {report.patterns_weak} WEAK / {report.patterns_failed} FAIL",
          flush=True)


if __name__ == "__main__":
    main()
