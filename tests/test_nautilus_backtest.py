"""Tests for the NautilusTrader backtest path (trading_algo/nautilus_backtest).

All tests use synthetic bars, except the real-data test, which runs only when
NAUTILUS_BT_DATA_DIR points at the IBKR daily cache (data/atlas_ibkr).
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader")

from trading_algo.backtest_v2.cost_model import CostModelConfig, ibkr_tiered_commission
from trading_algo.multi_strategy.backtest_runner import (
    MultiStrategyBacktestConfig,
    MultiStrategyBacktestRunner,
)
from trading_algo.multi_strategy.controller import MultiStrategyController
from trading_algo.multi_strategy.protocol import (
    StrategySignal,
    StrategyState,
    TradingStrategy,
)
from trading_algo.nautilus_backtest import (
    BacktestIntegrityError,
    NautilusBacktestConfig,
    benchmark_statistics,
    build_sessions,
    explicit_controller_config,
    lagged_cost_inputs,
    load_daily_bars,
    run_nautilus_backtest,
    run_parity,
)
from trading_algo.nautilus_backtest.costs import NO_HISTORY, RealisticCosts
from trading_algo.nautilus_backtest.data import runner_bars
from trading_algo.nautilus_backtest.benchmark import buy_and_hold
from trading_algo.nautilus_backtest.parity import compare
from trading_algo.nautilus_backtest.strategy import ControllerStrategy, Fill, Ledger

FREE = dict(costs="legacy", slippage_bps=0.0, commission_per_share=0.0)


# ── helpers ──────────────────────────────────────────────────────────────


def make_frame(n: int = 60, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    """Random-walk daily OHLCV with 4-decimal prices."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n).as_unit("ns")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    frame = pd.DataFrame({"open": open_, "close": close}, index=index)
    frame["high"] = frame[["open", "close"]].max(axis=1) * (1 + rng.uniform(0.001, 0.01, n))
    frame["low"] = frame[["open", "close"]].min(axis=1) * (1 - rng.uniform(0.001, 0.01, n))
    frame = frame[["open", "high", "low", "close"]].round(4)
    frame["volume"] = 1_000_000.0
    return frame


class Scripted(TradingStrategy):
    """Emits the next weight from a script for every symbol, every session.

    It also records what it was shown, to prove there is no look-ahead.
    """

    def __init__(self, weights: list[float], direction: int = 1) -> None:
        self.weights = weights
        self.direction = direction
        self.calls = 0
        self.updates: list[tuple[str, datetime]] = []
        self.signal_times: list[tuple[datetime, int]] = []

    @property
    def name(self) -> str:
        return "Scripted"

    @property
    def state(self) -> StrategyState:
        return StrategyState.ACTIVE

    def update(self, symbol: str, timestamp: datetime, *bar: float) -> None:
        self.updates.append((symbol, timestamp))

    def generate_signals(self, symbols: list[str], timestamp: datetime) -> list[StrategySignal]:
        self.signal_times.append((timestamp, len(self.updates)))
        weight = self.weights[self.calls % len(self.weights)]
        self.calls += 1
        if weight is None:
            return [StrategySignal("Scripted", s, 0, 0.0, 1.0) for s in symbols]
        return [StrategySignal("Scripted", s, self.direction, weight, 1.0) for s in symbols]


def controller_for(
    strategy: TradingStrategy, n_symbols: int = 1, **flags: bool
) -> MultiStrategyController:
    controller = MultiStrategyController(
        explicit_controller_config(
            [strategy.name],
            n_symbols,
            vol_management=flags.get("vol_management", False),
            risk_controller=flags.get("risk_controller", False),
        ),
    )
    controller.register(strategy)
    return controller


def config(**overrides: Any) -> NautilusBacktestConfig:
    return replace(NautilusBacktestConfig(initial_capital=1_000_000.0, **FREE), **overrides)


# ── 1. sessions ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("day", "open_utc", "close_utc"),
    [
        ("2024-03-08", "14:30", "21:00"),  # EST
        ("2024-03-11", "13:30", "20:00"),  # EDT, first Monday after the change
        ("2024-11-01", "13:30", "20:00"),  # EDT
        ("2024-11-04", "14:30", "21:00"),  # EST again
    ],
)
def test_session_times_follow_new_york_dst(day: str, open_utc: str, close_utc: str) -> None:
    frame = make_frame(1, start=day)
    (session,) = build_sessions({"AAA": frame})
    assert pd.Timestamp(session.open_ns, tz="UTC").strftime("%H:%M") == open_utc
    assert pd.Timestamp(session.close_ns, tz="UTC").strftime("%H:%M") == close_utc
    assert session.timestamp == datetime.fromisoformat(day)


# ── 2-3. timing and look-ahead ───────────────────────────────────────────


def test_fills_happen_at_next_session_open() -> None:
    frame = make_frame(10)
    result = run_nautilus_backtest(
        controller_for(Scripted([0.2, 0.25])),
        {"AAA": frame},
        config(liquidate_at_end=False),
    )

    days = [ts.date() for ts in frame.index]
    first = result.fills.iloc[0]
    # The first signal is generated at session 1's close (none on session 0),
    # so the first fill is at session 2's open, at exactly the open price.
    assert first["date"] == days[2]
    assert first["price"] == frame["open"].iloc[2]
    for fill in result.fills.itertuples():
        assert fill.price == frame.loc[pd.Timestamp(fill.date), "open"]


def test_signals_never_see_future_bars() -> None:
    frames = {s: make_frame(15, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    strategy = Scripted([0.1])
    run_nautilus_backtest(controller_for(strategy, 3), frames, config())

    assert len(strategy.signal_times) == 14
    for timestamp, seen in strategy.signal_times:
        shown = strategy.updates[:seen]
        assert all(ts <= timestamp for _, ts in shown)
        # Every symbol's bar for the session arrived before the decision.
        assert sum(1 for _, ts in shown if ts == timestamp) == 3


# ── 4. sizing and accounts ───────────────────────────────────────────────


def test_position_is_capped_at_max_position_pct() -> None:
    frame = make_frame(8)
    result = run_nautilus_backtest(
        controller_for(Scripted([0.9])),
        {"AAA": frame},
        config(liquidate_at_end=False),
    )
    first = result.fills.iloc[0]
    expected = int(1_000_000.0 * 0.25 / first["price"])
    assert first["side"] == "BUY"
    assert first["quantity"] == expected


def test_gross_exposure_cap_limits_new_exposure() -> None:
    frames = {s: make_frame(8, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    result = run_nautilus_backtest(
        controller_for(Scripted([0.25]), 3),
        frames,
        config(max_gross_exposure=0.5, liquidate_at_end=False),
    )
    day = result.fills["date"].iloc[0]
    opening = result.fills[result.fills["date"] == day]
    gross = (opening["quantity"] * opening["price"]).sum()
    assert gross <= 0.5 * 1_000_000.0 + 1e-6
    assert gross > 0.49 * 1_000_000.0


def test_exit_signal_flattens_and_shorts_work() -> None:
    frame = make_frame(12)
    result = run_nautilus_backtest(
        controller_for(Scripted([0.2, 0.2, None, None, 0.1], direction=-1)),
        {"AAA": frame},
        config(liquidate_at_end=False),
    )
    sides = result.fills["side"].tolist()
    assert sides[0] == "SELL"  # opening a short
    position = 0
    for fill in result.fills.itertuples():
        position += fill.quantity if fill.side == "BUY" else -fill.quantity
        assert position <= 0
    exits = result.fills[result.fills["side"] == "BUY"]
    assert len(exits) > 0  # exits cover the short


def test_one_fill_per_symbol_per_session_and_leverage_one() -> None:
    frames = {s: make_frame(30, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])}
    result = run_nautilus_backtest(
        controller_for(Scripted([0.25, 0.3, 0.2]), 4),
        frames,
        config(max_position_pct=0.25, max_gross_exposure=1.0),
    )
    assert (
        not result.fills.duplicated(["date", "symbol"]).any()
        or (result.fills["strategy"] == "liquidation").any()
    )
    # Fully invested at 1x and never borrowing cash: NAV stays positive and
    # the account never needs more than the capital it has.
    assert (result.nav > 0).all()


# ── 5-6. costs ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("quantity", [1, 50, 100, 2_000, 400_000])
def test_commission_matches_repo_ibkr_schedule(side: str, quantity: int) -> None:
    costs = RealisticCosts(inputs={})
    assert costs.commission(side, quantity, 123.45) == ibkr_tiered_commission(
        side, quantity, 123.45
    )


def test_realistic_run_charges_repo_commission_on_every_fill() -> None:
    frame = make_frame(25)
    result = run_nautilus_backtest(
        controller_for(Scripted([0.2, 0.1])),
        {"AAA": frame},
        config(costs="realistic"),
    )
    assert len(result.fills) > 5
    for fill in result.fills.itertuples():
        expected = ibkr_tiered_commission(fill.side, fill.quantity, fill.price)
        assert fill.commission == pytest.approx(expected, abs=0.006)  # cents


def test_cost_inputs_use_only_prior_sessions() -> None:
    frame = make_frame(30)
    base = lagged_cost_inputs({"AAA": frame})
    shocked = frame.copy()
    shocked.iloc[20, shocked.columns.get_loc("high")] *= 1.2
    shocked.iloc[20, shocked.columns.get_loc("volume")] *= 50
    after = lagged_cost_inputs({"AAA": shocked})

    day = [ts.date() for ts in frame.index]
    assert base[("AAA", day[0])] == NO_HISTORY
    for i in range(21):  # sessions up to and including the shocked one
        assert after[("AAA", day[i])] == base[("AAA", day[i])]
    assert after[("AAA", day[21])] != base[("AAA", day[21])]


def test_realistic_slippage_moves_against_the_order() -> None:
    frame = make_frame(30)
    costs = RealisticCosts(inputs=lagged_cost_inputs({"AAA": frame}), config=CostModelConfig())
    day = frame.index[25].date()
    assert costs.fill_price("AAA", day, "BUY", 1_000, 100.0) > 100.0
    assert costs.fill_price("AAA", day, "SELL", 1_000, 100.0) < 100.0
    # First session: no history, so the default spread applies and impact uses
    # the conservative fallbacks (200 bps vol, participation at its 10% cap).
    first = frame.index[0].date()
    half_spread_bps = CostModelConfig().fallback_spread_bps / 2
    impact_bps = 0.5 * 200.0 * np.sqrt(0.10)
    assert costs.fill_price("AAA", first, "BUY", 1_000, 100.0) == pytest.approx(
        100.0 * (1 + (half_spread_bps + impact_bps) / 1e4),
    )


# ── 7. accounting ────────────────────────────────────────────────────────


def test_nav_matches_an_independent_recompute() -> None:
    frames = {s: make_frame(40, seed=i) for i, s in enumerate(["AAA", "BBB"])}
    result = run_nautilus_backtest(
        controller_for(Scripted([0.2, 0.35, 0.1]), 2),
        frames,
        config(costs="realistic"),
    )
    cash = result.config.initial_capital
    positions = {"AAA": 0, "BBB": 0}
    fills = result.fills.groupby("date")
    for day, nav in result.nav.items():
        if day in fills.groups:
            for fill in fills.get_group(day).itertuples():
                if fill.strategy == "liquidation":
                    continue
                signed = fill.quantity if fill.side == "BUY" else -fill.quantity
                positions[fill.symbol] += signed
                cash -= signed * fill.price + fill.commission
        marks = {s: f.loc[pd.Timestamp(day), "close"] for s, f in frames.items()}
        assert nav == pytest.approx(
            cash + sum(q * marks[s] for s, q in positions.items()), abs=1e-6
        )
    # The engine also reconciled its ledger with NautilusTrader at every close.
    assert result.max_reconcile_diff <= 0.01 * (len(result.fills) + 1)


def test_ledger_flip_and_commission_accounting() -> None:
    ledger = Ledger(cash=100_000.0)

    def fill(side: str, quantity: int, price: float, commission: float) -> Fill:
        return Fill(date(2024, 1, 2), 0, "AAA", side, quantity, price, price, commission, "x")

    ledger.apply(fill("BUY", 10, 100.0, 1.0))
    assert ledger.cost["AAA"] == pytest.approx(100.1)
    flip = fill("SELL", 15, 110.0, 1.5)
    ledger.apply(flip)
    # Closes 10 long shares: (110 - 100.1) * 10 less this fill's fee on them.
    assert flip.realized_pnl == pytest.approx(98.0)
    assert ledger.positions["AAA"] == -5
    assert ledger.cost["AAA"] == pytest.approx(109.9)
    cover = fill("BUY", 5, 105.0, 0.5)
    ledger.apply(cover)
    assert cover.realized_pnl == pytest.approx(24.0)
    assert "AAA" not in ledger.positions
    assert ledger.cash == pytest.approx(100_000 - 1_001 + 1_648.5 - 525.5)


def test_liquidation_fills_at_the_last_close_less_costs() -> None:
    frame = make_frame(10)
    result = run_nautilus_backtest(
        controller_for(Scripted([0.2])),
        {"AAA": frame},
        config(slippage_bps=10.0),
    )
    last = result.fills.iloc[-1]
    assert last["strategy"] == "liquidation"
    assert last["date"] == frame.index[-1].date()
    assert last["price"] == pytest.approx(frame["close"].iloc[-1] * (1 - 0.001), abs=1e-4)
    assert result.final_nav == pytest.approx(
        result.nav.iloc[-1] - result.fills.iloc[-1]["slippage_cost"] - last["commission"], abs=0.02
    )


# ── 8. parity with the repo runner ───────────────────────────────────────


class RecordingController(MultiStrategyController):
    """Logs the order of the per-session controller calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    def generate_signals(
        self,
        symbols: list[str],
        timestamp: datetime,
        equity: float | None = None,
    ) -> list[StrategySignal]:
        self.calls.append("generate_signals")
        return super().generate_signals(symbols, timestamp, equity)

    def update_portfolio_state(
        self,
        equity: float,
        positions: dict[str, float],
        daily_pnl: float = 0.0,
    ) -> None:
        self.calls.append("update_portfolio_state")
        super().update_portfolio_state(equity, positions, daily_pnl)

    def add_return(self, daily_return: float) -> None:
        self.calls.append("add_return")
        super().add_return(daily_return)

    def new_trading_day(self) -> None:
        self.calls.append("new_trading_day")
        super().new_trading_day()


def recording_controller(weights: list[float]) -> RecordingController:
    controller = RecordingController(
        explicit_controller_config(["Scripted"], 1, vol_management=True, risk_controller=True),
    )
    controller.register(Scripted(weights))
    return controller


def test_controller_call_order_matches_runner() -> None:
    frame = make_frame(30)
    ours = recording_controller([0.2])
    run_nautilus_backtest(ours, {"AAA": frame}, config())
    theirs = recording_controller([0.2])
    MultiStrategyBacktestRunner(
        theirs,
        MultiStrategyBacktestConfig(initial_capital=1_000_000.0, symbols=["AAA"]),
    ).run(runner_bars({"AAA": frame}))
    assert ours.calls == theirs.calls
    assert ours.calls[:4] == [
        "generate_signals",
        "update_portfolio_state",
        "add_return",
        "new_trading_day",
    ]


def test_matches_multi_strategy_runner_on_one_symbol() -> None:
    frame = make_frame(260, seed=3)
    weights = [0.10, 0.20, 0.20, 0.15, 0.25, 0.12, 0.18]

    def factory() -> MultiStrategyController:
        controller = MultiStrategyController(
            explicit_controller_config(["Scripted"], 1, vol_management=True, risk_controller=True),
        )
        controller.register(Scripted(weights))
        return controller

    report = run_parity(factory, {"AAA": frame}, NautilusBacktestConfig())
    assert report.status == "match", report.trade_mismatches
    assert report.trades_nautilus == report.trades_runner > 200
    assert report.max_nav_rel_diff < 1e-5


def test_multi_symbol_parity_is_reported_not_comparable() -> None:
    frames = {s: make_frame(10, seed=i) for i, s in enumerate(["AAA", "BBB"])}
    report = run_parity(
        lambda: controller_for(Scripted([0.1]), 2), frames, NautilusBacktestConfig()
    )
    assert report.status == "not_comparable"


# ── 9. benchmark statistics ──────────────────────────────────────────────


def test_benchmark_statistics_match_closed_form() -> None:
    rng = np.random.default_rng(11)
    days = [ts.date() for ts in pd.bdate_range("2023-01-02", periods=300)]
    bench = pd.Series(rng.normal(0.0005, 0.01, len(days)), index=days)
    strat = 1.5 * bench + rng.normal(0.0001, 0.002, len(days))
    rf_annual = 0.04
    stats = benchmark_statistics(strat, bench, rf_annual)

    r, b = strat.to_numpy(), bench.to_numpy()
    beta = np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1)
    rf = (1 + rf_annual) ** (1 / 252) - 1
    alpha = (1 + (r.mean() - rf) - beta * (b.mean() - rf)) ** 252 - 1
    active = r - b
    ir = active.mean() / active.std(ddof=1) * np.sqrt(252)

    assert stats["Beta"] == pytest.approx(beta, rel=1e-9)
    assert stats["Alpha (252 days)"] == pytest.approx(alpha, rel=1e-9)
    assert stats["Information Ratio (252 days)"] == pytest.approx(ir, rel=1e-9)
    assert stats["Sessions"] == 300
    assert 1.4 < stats["Beta"] < 1.6


# ── 10. robustness ───────────────────────────────────────────────────────


def test_missing_bar_for_one_symbol_does_not_stall() -> None:
    frames = {"AAA": make_frame(20, seed=1), "BBB": make_frame(20, seed=2)}
    gap = frames["BBB"].index[10]
    frames["BBB"] = frames["BBB"].drop(index=gap)
    result = run_nautilus_backtest(controller_for(Scripted([0.2, 0.1]), 2), frames, config())

    assert len(result.nav) == 20
    bbb = result.fills[result.fills["symbol"] == "BBB"]
    assert gap.date() not in set(bbb["date"])  # nothing trades without a bar


def test_runs_are_deterministic() -> None:
    frames = {s: make_frame(40, seed=i) for i, s in enumerate(["AAA", "BBB"])}

    def run() -> pd.DataFrame:
        return run_nautilus_backtest(
            controller_for(Scripted([0.2, 0.3]), 2),
            frames,
            config(costs="realistic"),
        ).fills

    pd.testing.assert_frame_equal(run(), run())


class FailingController(MultiStrategyController):
    def generate_signals(
        self,
        symbols: list[str],
        timestamp: datetime,
        equity: float | None = None,
    ) -> list[StrategySignal]:
        raise ZeroDivisionError("controller failed")


def test_callback_exceptions_fail_the_run() -> None:
    # NautilusTrader swallows exceptions raised in strategy callbacks.
    controller = FailingController(explicit_controller_config(["X"], 1))
    with pytest.raises(BacktestIntegrityError, match="controller failed"):
        run_nautilus_backtest(controller, {"AAA": make_frame(6)}, config())


def test_reconciliation_catches_a_missed_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ControllerStrategy.on_order_filled
    seen: list[int] = []

    def drop_first(self: ControllerStrategy, event: Any) -> None:
        seen.append(1)
        if len(seen) > 1:
            original(self, event)

    monkeypatch.setattr(ControllerStrategy, "on_order_filled", drop_first)
    with pytest.raises(BacktestIntegrityError, match="position"):
        run_nautilus_backtest(controller_for(Scripted([0.2])), {"AAA": make_frame(8)}, config())


def test_parity_flags_misaligned_sessions() -> None:
    frame = make_frame(30)
    factory = lambda: controller_for(Scripted([0.2, 0.1]))  # noqa: E731
    ours = run_nautilus_backtest(factory(), {"AAA": frame}, config())
    runner = MultiStrategyBacktestRunner(
        factory(),
        MultiStrategyBacktestConfig(initial_capital=1_000_000.0, slippage_bps=0.0),
    )
    runner_result = runner.run(runner_bars({"AAA": frame}))
    shifted = [ts + pd.Timedelta(days=1) for ts in runner_result.timestamps]
    report = compare(ours, runner_result.equity_curve, shifted, runner._trades)
    assert report.status == "mismatch"
    assert "session dates differ between the engines" in report.trade_mismatches


def test_buy_and_hold_rejects_symbols_without_a_start_price() -> None:
    late = make_frame(10, start="2024-01-10")
    with pytest.raises(ValueError, match="no price"):
        buy_and_hold({"AAA": make_frame(20), "BBB": late}, date(2024, 1, 2), 100_000.0)


# ── 11. real data ────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("NAUTILUS_BT_DATA_DIR"),
    reason="set NAUTILUS_BT_DATA_DIR to the IBKR daily cache (data/atlas_ibkr) to run",
)
def test_pure_momentum_on_real_data() -> None:
    from trading_algo.multi_strategy.adapters import MomentumStrategyAdapter

    data_dir = Path(os.environ["NAUTILUS_BT_DATA_DIR"])
    assert data_dir.is_dir(), f"NAUTILUS_BT_DATA_DIR={data_dir} does not exist"
    frames = load_daily_bars(["AAPL"], data_dir, date(2019, 1, 2), date(2021, 12, 31))

    def factory() -> MultiStrategyController:
        controller = MultiStrategyController(explicit_controller_config(["PureMomentum"], 1))
        controller.register(MomentumStrategyAdapter())
        return controller

    result = run_nautilus_backtest(factory(), frames, NautilusBacktestConfig())
    assert len(result.fills) > 10
    assert result.cost_breakdown["commission"] > 0
    assert run_parity(factory, frames, NautilusBacktestConfig()).status == "match"
