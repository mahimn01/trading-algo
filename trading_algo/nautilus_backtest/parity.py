"""Cross-check NautilusTrader results against ``MultiStrategyBacktestRunner``.

Both engines get the same controller setup, bars and per-share commission.
For a single symbol their timelines are identical: decide at the close, fill
at the next open. They should then agree on every trade's date and side, and
on NAV up to whole-share rounding.

The check runs at $100M with zero slippage. At that size whole-share rounding
is negligible (the runner trades fractional shares). Zero slippage sidesteps a
runner bug that applies slippage in the trader's favour when selling down a
long. The runner's sub-share rebalances, which NautilusTrader cannot place,
are left out of the trade comparison.

With several symbols the committed runner generates each day's signals as soon
as the first symbol's new bar arrives, so it trades on partly next-day data.
Those runs are reported as not comparable instead of being forced to match.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import pandas as pd

from trading_algo.multi_strategy.backtest_runner import (
    MultiStrategyBacktestConfig,
    MultiStrategyBacktestRunner,
)
from trading_algo.multi_strategy.controller import MultiStrategyController
from trading_algo.nautilus_backtest.data import runner_bars
from trading_algo.nautilus_backtest.engine import (
    NautilusBacktestConfig,
    NautilusBacktestResult,
    run_nautilus_backtest,
)

NAV_TOLERANCE = 1e-4
# A trade can differ by up to two shares: one from truncating the order to
# whole shares and one from the whole-share position it starts from.
QUANTITY_TOLERANCE_SHARES = 2.0
QUANTITY_TOLERANCE_REL = 1e-3
PARITY_CAPITAL = 100_000_000.0


@dataclass
class ParityReport:
    """How closely the two engines agree."""

    status: str
    capital: float = 0.0
    sessions_compared: int = 0
    max_nav_rel_diff: float = 0.0
    final_nav_nautilus: float = 0.0
    final_nav_runner: float = 0.0
    trades_nautilus: int = 0
    trades_runner: int = 0
    runner_sub_share_trades_skipped: int = 0
    trade_mismatches: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "match"


def run_parity(
    controller_factory: Callable[[], MultiStrategyController],
    frames: Mapping[str, pd.DataFrame],
    config: NautilusBacktestConfig,
) -> ParityReport:
    """Run both engines with per-share commission and no slippage, then compare."""
    if len(frames) != 1:
        return ParityReport(
            status="not_comparable",
            note=(
                "The committed runner generates signals when the first symbol's "
                "new-day bar arrives, before the other symbols' bars, so it trades "
                "on partly next-day data. Only single-symbol runs are compared."
            ),
        )
    legacy = replace(
        config,
        costs="legacy",
        slippage_bps=0.0,
        initial_capital=PARITY_CAPITAL,
    )
    nautilus = run_nautilus_backtest(controller_factory(), frames, legacy)
    runner = MultiStrategyBacktestRunner(
        controller_factory(),
        MultiStrategyBacktestConfig(
            initial_capital=legacy.initial_capital,
            symbols=list(frames),
            commission_per_share=legacy.commission_per_share,
            slippage_bps=legacy.slippage_bps,
            risk_free_rate=legacy.risk_free_rate,
            max_position_pct=legacy.max_position_pct,
            max_gross_exposure=legacy.max_gross_exposure,
        ),
    )
    runner_result = runner.run(runner_bars(frames))
    report = compare(nautilus, runner_result.equity_curve, runner_result.timestamps, runner._trades)
    report.capital = PARITY_CAPITAL
    return report


def compare(
    nautilus: NautilusBacktestResult,
    runner_equity: list[float],
    runner_timestamps: list[Any],
    runner_trades: list[dict[str, Any]],
) -> ParityReport:
    """Compare per-session NAV, final NAV and the trade list."""
    mismatches: list[str] = []
    # equity_curve[0] is the starting capital and its last entry is the
    # post-liquidation NAV. Entries in between line up with timestamps[:-1],
    # one per session after the first.
    runner_nav = {
        ts.date(): equity
        for ts, equity in zip(runner_timestamps[:-1], runner_equity[1:-1], strict=True)
    }
    if list(runner_nav) != list(nautilus.nav.index[1:]):
        mismatches.append("session dates differ between the engines")
    diffs = [
        abs(nautilus.nav[day] - equity) / abs(equity)
        for day, equity in runner_nav.items()
        if day in nautilus.nav.index
    ]
    final_runner = runner_equity[-1]
    diffs.append(abs(nautilus.final_nav - final_runner) / abs(final_runner))

    ours = [(row.date, row.side, row.quantity) for row in nautilus.fills.itertuples()]
    whole = [t for t in runner_trades if abs(t["shares"]) >= 1]
    theirs = [
        (
            t["timestamp"].date(),
            "BUY" if t["side"] in ("BUY", "COVER") else "SELL",
            abs(t["shares"]),
        )
        for t in whole
    ]
    for i, (a, b) in enumerate(zip(ours, theirs, strict=False)):
        same_shares = abs(a[2] - b[2]) <= max(
            QUANTITY_TOLERANCE_SHARES, QUANTITY_TOLERANCE_REL * b[2]
        )
        if a[:2] != b[:2] or not same_shares:
            mismatches.append(f"#{i}: nautilus={a} runner={b}")
    if len(ours) != len(theirs):
        mismatches.append(f"trade count: nautilus={len(ours)} runner={len(theirs)}")

    worst = max(diffs)
    status = "match" if worst <= NAV_TOLERANCE and not mismatches else "mismatch"
    return ParityReport(
        status=status,
        sessions_compared=len(diffs) - 1,
        max_nav_rel_diff=worst,
        final_nav_nautilus=nautilus.final_nav,
        final_nav_runner=final_runner,
        trades_nautilus=len(ours),
        trades_runner=len(theirs),
        runner_sub_share_trades_skipped=len(runner_trades) - len(whole),
        trade_mismatches=mismatches[:20],
        note="Runner trades fractional shares, NautilusTrader whole shares.",
    )
