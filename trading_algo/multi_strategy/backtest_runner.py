"""
Multi-Strategy Backtest Runner

Runs all strategies simultaneously on the same historical data to
measure combined portfolio performance and diversification benefit.

Unlike the single-strategy BacktestEngine in backtest_v2, this runner:
  - Feeds data to the MultiStrategyController
  - Tracks per-strategy attribution
  - Measures diversification benefit (combined vs individual Sharpe)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trading_algo.multi_strategy.controller import MultiStrategyController
from trading_algo.multi_strategy.protocol import StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class MultiStrategyBacktestConfig:
    """Configuration for multi-strategy backtesting."""
    initial_capital: float = 100_000
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    commission_per_share: float = 0.0035
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.045  # Annual risk-free rate (current ~4.5%)
    signal_interval_bars: int = 0  # 0 = daily only; 12 = ~hourly on 5-min data
    intraday_vol_threshold: float = 0.0  # Only intraday signals when ann vol > this (0=always)
    max_position_pct: float = 0.25  # Max % of equity per symbol
    max_gross_exposure: float = 1.0  # Max gross exposure


@dataclass
class StrategyAttribution:
    """Per-strategy performance attribution."""
    name: str
    n_signals: int = 0
    n_trades: int = 0
    gross_pnl: float = 0.0
    win_rate: float = 0.0
    avg_weight: float = 0.0


@dataclass
class MultiStrategyBacktestResults:
    """Results from a multi-strategy backtest."""
    # Portfolio-level metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0

    # Institutional-quality metrics
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy_per_trade: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown_duration_days: int = 0
    annual_turnover: float = 0.0

    # Benchmark-relative metrics
    beta: float = 0.0
    alpha_annual: float = 0.0
    information_ratio: float = 0.0
    benchmark_correlation: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Per-strategy attribution
    strategy_attribution: Dict[str, StrategyAttribution] = field(default_factory=dict)

    # Diversification metrics
    diversification_ratio: float = 0.0
    """Ratio of combined Sharpe to average individual Sharpe.
    >1.0 means diversification is adding value."""


class MultiStrategyBacktestRunner:
    """
    Run all strategies in the MultiStrategyController on historical bars.

    Usage::

        runner = MultiStrategyBacktestRunner(controller, config)
        results = runner.run(data)
    """

    def __init__(
        self,
        controller: MultiStrategyController,
        config: Optional[MultiStrategyBacktestConfig] = None,
    ):
        self.controller = controller
        self.config = config or MultiStrategyBacktestConfig()

        # Risk limits (from config)
        self.MAX_POSITION_PCT = self.config.max_position_pct
        self.MAX_GROSS_EXPOSURE = self.config.max_gross_exposure

        # State
        self._equity = self.config.initial_capital
        self._cash = self.config.initial_capital
        self._positions: Dict[str, float] = {}  # symbol -> shares
        self._position_prices: Dict[str, float] = {}  # symbol -> avg entry price
        self._current_prices: Dict[str, float] = {}

        # Tracking
        self._equity_curve: List[float] = [self.config.initial_capital]
        self._daily_returns: List[float] = []
        self._timestamps: List[datetime] = []
        self._trades: List[Dict] = []
        self._signals_by_strategy: Dict[str, int] = {}
        self._winning_trades: int = 0
        self._closed_trades: int = 0

    def run(
        self,
        data: Dict[str, List[Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> MultiStrategyBacktestResults:
        """
        Run the multi-strategy backtest.

        Args:
            data: Dict of symbol -> list of bar objects.
                  Each bar must have: timestamp, open, high, low, close, volume
            progress_callback: Optional (pct, msg) callback for progress.

        Returns:
            MultiStrategyBacktestResults
        """
        symbols = self.config.symbols or list(data.keys())

        # Build a unified timeline from all bars
        # Each symbol's bars are already sorted by timestamp — use merge
        # NOTE: use a helper to avoid closure bug (symbol captured by reference)
        def _tagged(sym, bars):
            return ((bar.timestamp, sym, bar) for bar in bars)

        iterables = [_tagged(sym, bars) for sym, bars in data.items()]
        all_bars_iter = heapq.merge(*iterables, key=lambda x: x[0])
        all_bars = list(all_bars_iter)

        if not all_bars:
            return self._build_results()

        if progress_callback:
            progress_callback(0.05, f"Processing {len(all_bars)} bars across {len(data)} symbols")

        # Process bars
        total = len(all_bars)
        last_day = None
        daily_equity_open = self._equity
        bars_since_signal = 0
        signal_interval = self.config.signal_interval_bars
        day_count = 0

        for i, (ts, symbol, bar) in enumerate(all_bars):
            o = bar.open
            h = bar.high
            l = bar.low
            c = bar.close
            v = bar.volume

            # Feed to controller
            self.controller.update(symbol, ts, o, h, l, c, v)
            self._current_prices[symbol] = c

            # Update position values (only when symbol has an open position)
            if symbol in self._positions:
                self._update_equity()

            # Track daily boundaries
            current_day = ts.date()
            is_new_day = last_day is not None and current_day != last_day

            # Intraday signal generation at configured interval
            bars_since_signal += 1
            if (
                signal_interval > 0
                and bars_since_signal >= signal_interval
                and self._equity > 0
                and not is_new_day
                and self._is_high_vol()
            ):
                signals = self.controller.generate_signals(
                    symbols, ts, self._equity
                )
                self._process_signals(signals, ts)
                self._sync_portfolio_state(daily_equity_open)
                bars_since_signal = 0

            if is_new_day:
                # Generate signals at day boundary for daily strategies
                if self._equity > 0:
                    signals = self.controller.generate_signals(
                        symbols, ts, self._equity
                    )
                    self._process_signals(signals, ts)
                    self._sync_portfolio_state(daily_equity_open)
                    bars_since_signal = 0

                # Record equity and daily return
                self._equity_curve.append(self._equity)
                self._timestamps.append(ts)

                if daily_equity_open > 0:
                    daily_ret = (self._equity / daily_equity_open) - 1
                    self._daily_returns.append(daily_ret)
                    self.controller.add_return(daily_ret)

                # Reset daily counters
                self.controller.new_trading_day()
                daily_equity_open = self._equity

                # Detect regime every 5 trading days
                day_count += 1
                if (
                    day_count % 5 == 0
                    and hasattr(self.controller, 'detect_regime')
                    and self.controller.config.enable_regime_adaptation
                ):
                    self.controller.detect_regime()

            last_day = current_day

            if progress_callback and i % (total // 20 + 1) == 0:
                progress_callback(0.05 + 0.90 * (i / total), f"Bar {i}/{total}")

        # Final day: close all open positions to realise P&L
        if all_bars:
            last_ts = all_bars[-1][0]
            for sym in list(self._positions):
                self._close_position(sym, last_ts)
            self._update_equity()
            self._equity_curve.append(self._equity)
            self._timestamps.append(last_ts)

        if progress_callback:
            progress_callback(0.95, "Computing metrics...")

        return self._build_results()

    def _is_high_vol(self) -> bool:
        """Check if recent volatility exceeds the intraday threshold."""
        threshold = self.config.intraday_vol_threshold
        if threshold <= 0:
            return True  # No vol gating — always use intraday signals
        if len(self._daily_returns) < 20:
            return False  # Not enough data — stay daily-only
        recent_vol = float(np.std(self._daily_returns[-20:]) * np.sqrt(252))
        return recent_vol >= threshold

    def _sync_portfolio_state(self, daily_equity_open: float) -> None:
        """Sync portfolio state back to controller for accurate risk checks."""
        pos_weights: Dict[str, float] = {}
        for sym, shares in self._positions.items():
            px = self._current_prices.get(sym, 0)
            if self._equity > 0 and px > 0:
                pos_weights[sym] = (shares * px) / self._equity
        daily_pnl = (
            (self._equity / daily_equity_open) - 1
            if daily_equity_open > 0 else 0.0
        )
        self.controller.update_portfolio_state(
            equity=self._equity,
            positions=pos_weights,
            daily_pnl=daily_pnl,
        )

    def _process_signals(
        self, signals: List[StrategySignal], timestamp: datetime
    ) -> None:
        """Process signals into simulated trades."""
        for sig in signals:
            if sig.is_exit:
                self._close_position(sig.symbol, timestamp)
            elif sig.is_entry:
                self._open_position(sig, timestamp)

            # Track per-strategy
            strategy = sig.strategy_name.split("+")[0]
            self._signals_by_strategy[strategy] = self._signals_by_strategy.get(strategy, 0) + 1

    def _open_position(self, sig: StrategySignal, timestamp: datetime) -> None:
        """Open or rebalance to a target position with risk limits."""
        price = self._current_prices.get(sig.symbol)
        if price is None or price <= 0 or self._equity <= 0:
            return

        # Apply slippage
        slippage = price * (self.config.slippage_bps / 10000)
        exec_price = price + slippage if sig.direction > 0 else price - slippage

        # Compute desired dollar amount, capped at MAX_POSITION_PCT
        weight = min(abs(sig.target_weight), self.MAX_POSITION_PCT)
        target_dollar = self._equity * weight * sig.direction

        # Current position value for this symbol
        current_shares = self._positions.get(sig.symbol, 0.0)
        current_value = current_shares * price

        # Delta needed to reach target
        delta_dollar = target_dollar - current_value

        # Check gross exposure limit
        gross_exposure = sum(
            abs(s * self._current_prices.get(sym, 0))
            for sym, s in self._positions.items()
        )
        max_new_exposure = self._equity * self.MAX_GROSS_EXPOSURE - gross_exposure
        if abs(delta_dollar) > max_new_exposure and max_new_exposure > 0:
            delta_dollar = np.sign(delta_dollar) * max_new_exposure
        elif max_new_exposure <= 0:
            return  # At exposure limit

        delta_shares = delta_dollar / exec_price
        if abs(delta_shares) < 0.01:
            return

        # Ensure we have enough cash for buys
        cost = delta_shares * exec_price
        commission = abs(delta_shares) * self.config.commission_per_share
        if cost > 0 and cost + commission > self._cash:
            # Scale down to available cash
            max_cost = self._cash * 0.95 - commission
            if max_cost <= 0:
                return
            delta_shares = max_cost / exec_price
            cost = delta_shares * exec_price

        self._cash -= cost + commission

        # Update position
        new_shares = current_shares + delta_shares
        if abs(new_shares) < 0.01:
            self._positions.pop(sig.symbol, None)
            self._position_prices.pop(sig.symbol, None)
        else:
            self._positions[sig.symbol] = new_shares
            self._position_prices[sig.symbol] = exec_price

        self._trades.append({
            "timestamp": timestamp,
            "symbol": sig.symbol,
            "side": "BUY" if delta_shares > 0 else "SELL",
            "shares": delta_shares,
            "price": exec_price,
            "strategy": sig.strategy_name,
        })

    def _close_position(self, symbol: str, timestamp: datetime) -> None:
        """Close an existing position."""
        shares = self._positions.get(symbol, 0)
        if abs(shares) < 0.01:
            return

        price = self._current_prices.get(symbol)
        if price is None:
            return

        slippage = price * (self.config.slippage_bps / 10000)
        exec_price = price - slippage if shares > 0 else price + slippage

        commission = abs(shares) * self.config.commission_per_share
        proceeds = shares * exec_price - commission
        self._cash += proceeds

        # Track win/loss
        entry_price = self._position_prices.get(symbol, exec_price)
        pnl = (exec_price - entry_price) * shares
        self._closed_trades += 1
        if pnl > 0:
            self._winning_trades += 1

        self._trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": "SELL" if shares > 0 else "COVER",
            "shares": -shares,
            "price": exec_price,
            "strategy": "exit",
        })

        self._positions.pop(symbol, None)
        self._position_prices.pop(symbol, None)

    def _update_equity(self) -> None:
        """Recalculate equity from cash + positions."""
        if not self._positions:
            self._equity = self._cash
            return
        pos_value = sum(
            shares * self._current_prices.get(sym, 0)
            for sym, shares in self._positions.items()
        )
        self._equity = self._cash + pos_value

    def _build_results(
        self,
        benchmark_daily_returns: Optional[np.ndarray] = None,
    ) -> MultiStrategyBacktestResults:
        """Compute final metrics and build results.

        Args:
            benchmark_daily_returns: Optional array of benchmark (e.g. SPY)
                daily returns aligned to the same dates as portfolio returns.
                Used for beta, alpha, information ratio, and correlation.
        """
        ec = np.array(self._equity_curve) if self._equity_curve else np.array([self.config.initial_capital])
        dr = np.array(self._daily_returns) if self._daily_returns else np.array([0.0])

        total_return = (ec[-1] / ec[0]) - 1 if ec[0] > 0 else 0
        n_years = max(len(dr) / 252, 1 / 252)
        ann_return = (1 + total_return) ** (1 / n_years) - 1

        # Annualized volatility (ddof=1 for unbiased estimator)
        vol = float(np.std(dr, ddof=1) * np.sqrt(252)) if len(dr) > 1 else 0.15

        # Sharpe ratio: (mean_daily_excess / std_daily) * sqrt(252)
        rf = self.config.risk_free_rate
        daily_rf = (1 + rf) ** (1 / 252) - 1
        excess = dr - daily_rf
        if len(dr) > 1 and np.std(excess, ddof=1) > 1e-10:
            sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino ratio: use correct semi-deviation over ALL observations
        # semi_dev = sqrt(mean(min(r - target, 0)^2))  (daily, not annualized)
        downside_diff = np.minimum(dr - daily_rf, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)))
        if downside_dev > 1e-10:
            sortino = float(np.mean(excess) * np.sqrt(252) / downside_dev)
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(ec)
        dd = (peak - ec) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd))

        # Win rate from closed trades (not len(self._trades) which double-counts)
        total_trades = self._closed_trades
        win_rate = (self._winning_trades / self._closed_trades
                    if self._closed_trades > 0 else 0.0)

        # --- Institutional-quality metrics ---

        # Calmar ratio: annualized return / max drawdown
        calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

        # Max drawdown duration (days in drawdown)
        dd_duration = 0
        max_dd_duration = 0
        for d in dd:
            if d > 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # VaR and CVaR (95%)
        if len(dr) >= 10:
            var_95 = float(-np.percentile(dr, 5))
            tail = dr[dr <= np.percentile(dr, 5)]
            cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else var_95
        else:
            var_95 = 0.0
            cvar_95 = 0.0

        # Skewness and kurtosis
        if len(dr) > 2:
            mean_dr = np.mean(dr)
            std_dr = np.std(dr, ddof=1)
            if std_dr > 1e-10:
                skewness = float(np.mean(((dr - mean_dr) / std_dr) ** 3))
                kurtosis = float(np.mean(((dr - mean_dr) / std_dr) ** 4) - 3)
            else:
                skewness = 0.0
                kurtosis = 0.0
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Profit factor and expectancy from trade PnLs
        trade_pnls = self._compute_trade_pnls()
        if len(trade_pnls) > 0:
            wins = trade_pnls[trade_pnls > 0]
            losses = trade_pnls[trade_pnls < 0]
            total_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
            total_losses = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
            profit_factor = total_wins / total_losses if total_losses > 1e-10 else 0.0
            expectancy = float(np.mean(trade_pnls))
        else:
            profit_factor = 0.0
            expectancy = 0.0

        # Annual turnover: sum(|delta_shares * price|) / avg_equity / n_years
        total_traded_value = sum(
            abs(t["shares"] * t["price"]) for t in self._trades
        )
        avg_equity = float(np.mean(ec)) if len(ec) > 0 else self.config.initial_capital
        annual_turnover = (total_traded_value / avg_equity / n_years) if avg_equity > 0 and n_years > 0 else 0.0

        # --- Benchmark-relative metrics ---
        beta = 0.0
        alpha_annual = 0.0
        information_ratio = 0.0
        benchmark_corr = 0.0

        if benchmark_daily_returns is not None and len(benchmark_daily_returns) >= 10:
            bench = benchmark_daily_returns
            # Align lengths
            min_len = min(len(dr), len(bench))
            algo_r = dr[:min_len]
            bench_r = bench[:min_len]

            # Beta = Cov(algo, bench) / Var(bench)
            cov_matrix = np.cov(algo_r, bench_r)
            var_bench = cov_matrix[1, 1]
            if var_bench > 1e-10:
                beta = float(cov_matrix[0, 1] / var_bench)

            # Jensen's alpha: algo_ann - rf - beta * (bench_ann - rf)
            bench_ann = float(np.mean(bench_r) * 252)
            algo_ann = float(np.mean(algo_r) * 252)
            alpha_annual = algo_ann - rf - beta * (bench_ann - rf)

            # Information ratio: (algo - bench) / tracking_error
            active_returns = algo_r - bench_r
            tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(252))
            if tracking_error > 1e-10:
                information_ratio = float(np.mean(active_returns) * 252 / tracking_error)

            # Correlation
            corr = np.corrcoef(algo_r, bench_r)
            benchmark_corr = float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0

        # Per-strategy attribution
        attribution = {}
        for name, count in self._signals_by_strategy.items():
            attribution[name] = StrategyAttribution(
                name=name,
                n_signals=count,
            )

        return MultiStrategyBacktestResults(
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            volatility=vol,
            total_trades=total_trades,
            win_rate=win_rate,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            expectancy_per_trade=expectancy,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown_duration_days=max_dd_duration,
            annual_turnover=annual_turnover,
            beta=beta,
            alpha_annual=alpha_annual,
            information_ratio=information_ratio,
            benchmark_correlation=benchmark_corr,
            equity_curve=ec.tolist(),
            daily_returns=dr.tolist(),
            timestamps=self._timestamps,
            strategy_attribution=attribution,
        )

    def _compute_trade_pnls(self) -> np.ndarray:
        """Extract per-trade P&L from the trade log.

        Pairs consecutive BUY/SELL events for the same symbol into
        round-trip P&L values.
        """
        # Match open/close pairs by symbol
        open_trades: Dict[str, List[Dict]] = {}
        pnls: List[float] = []

        for t in self._trades:
            sym = t["symbol"]
            if t["side"] in ("BUY", "SHORT"):
                open_trades.setdefault(sym, []).append(t)
            elif t["side"] in ("SELL", "COVER"):
                opens = open_trades.get(sym, [])
                if opens:
                    entry = opens.pop(0)
                    entry_price = entry["price"]
                    exit_price = t["price"]
                    shares = abs(entry["shares"])
                    if entry["side"] == "BUY":
                        pnl = (exit_price - entry_price) * shares
                    else:
                        pnl = (entry_price - exit_price) * shares
                    pnls.append(pnl)

        return np.array(pnls) if pnls else np.array([])
