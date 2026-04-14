"""
Auto-Optimizer — periodic re-validation and parameter tuning for options strategies.

Runs walk-forward re-validation, parameter sensitivity re-checks,
symbol universe scoring, and automated report generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from trading_algo.quant_core.strategies.options.wheel import WheelConfig, WheelStrategy
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_series_from_prices,
    iv_rank as compute_iv_rank,
    realized_volatility,
)
from trading_algo.quant_core.strategies.options.strategy_monitor import (
    StrategyMonitor,
    MonitorAlert,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    config: WheelConfig
    live_sharpe: float
    backtest_sharpe: float
    live_return_pct: float
    backtest_return_pct: float
    degradation_pct: float
    needs_review: bool
    details: dict = field(default_factory=dict)


@dataclass
class SensitivityResult:
    parameter: str
    current_value: float
    optimal_value: float
    current_sharpe: float
    optimal_sharpe: float
    sharpe_improvement: float
    robust: bool
    recommendation: str


@dataclass
class SymbolScore:
    symbol: str
    total_score: float
    iv_rank_score: float
    liquidity_score: float
    trend_stability_score: float
    price_range_score: float
    correlation_score: float
    details: dict = field(default_factory=dict)


@dataclass
class DailyReport:
    date: datetime
    positions: list[dict]
    total_equity: float
    daily_pnl: float
    daily_return_pct: float
    alerts: list[str]
    regime: str

    def render(self) -> str:
        lines = [
            f"=== Daily Report: {self.date:%Y-%m-%d} ===",
            f"Equity: ${self.total_equity:,.2f}  |  P&L: ${self.daily_pnl:+,.2f} ({self.daily_return_pct:+.2f}%)  |  Regime: {self.regime}",
        ]
        if self.positions:
            lines.append("Positions:")
            for p in self.positions:
                lines.append(f"  {p.get('symbol','?')} {p.get('type','?')} K={p.get('strike',0)} exp={p.get('expiry','?')}")
        if self.alerts:
            lines.append("Alerts:")
            for a in self.alerts:
                lines.append(f"  {a}")
        return "\n".join(lines)


@dataclass
class WeeklyReport:
    week_ending: datetime
    start_equity: float
    end_equity: float
    weekly_return_pct: float
    rolling_sharpe_30d: float
    rolling_sharpe_60d: float
    regime: str
    regime_changes: int
    param_health: str
    trades: int
    win_rate: float
    alerts_summary: list[str]

    def render(self) -> str:
        lines = [
            f"=== Weekly Report: week ending {self.week_ending:%Y-%m-%d} ===",
            f"Equity: ${self.start_equity:,.2f} -> ${self.end_equity:,.2f} ({self.weekly_return_pct:+.2f}%)",
            f"30d Sharpe: {self.rolling_sharpe_30d:.2f}  |  60d Sharpe: {self.rolling_sharpe_60d:.2f}",
            f"Regime: {self.regime}  |  Changes this period: {self.regime_changes}",
            f"Trades: {self.trades}  |  Win rate: {self.win_rate:.0%}",
            f"Parameter health: {self.param_health}",
        ]
        if self.alerts_summary:
            lines.append("Alert summary:")
            for a in self.alerts_summary:
                lines.append(f"  {a}")
        return "\n".join(lines)


@dataclass
class MonthlyReport:
    year: int
    month: int
    start_equity: float
    end_equity: float
    monthly_return_pct: float
    sharpe: float
    max_dd_pct: float
    benchmark_return_pct: float
    trades: int
    win_rate: float
    premium_collected: float
    premium_paid: float
    net_premium: float
    regime: str
    recommendations: list[str]

    def render(self) -> str:
        lines = [
            f"=== Monthly Report: {self.year}-{self.month:02d} ===",
            f"Equity: ${self.start_equity:,.2f} -> ${self.end_equity:,.2f} ({self.monthly_return_pct:+.2f}%)",
            f"Sharpe: {self.sharpe:.2f}  |  Max DD: {self.max_dd_pct:.1f}%  |  Benchmark: {self.benchmark_return_pct:+.2f}%",
            f"Trades: {self.trades}  |  Win rate: {self.win_rate:.0%}",
            f"Net premium: ${self.net_premium:,.2f} (collected ${self.premium_collected:,.2f}, paid ${self.premium_paid:,.2f})",
            f"Regime: {self.regime}",
        ]
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoOptimizer
# ---------------------------------------------------------------------------

class AutoOptimizer:
    """
    Periodically re-validates parameters and suggests updates for the Wheel strategy.
    """

    DEGRADATION_THRESHOLD = 0.40
    SHARPE_IMPROVEMENT_THRESHOLD = 0.15

    def __init__(
        self,
        base_config: WheelConfig | None = None,
        risk_free_rate: float = 0.045,
    ):
        self.base_config = base_config or WheelConfig()
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Walk-forward re-validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        prices: np.ndarray,
        dates: list[datetime],
        iv_series: np.ndarray,
        iv_rank_series: np.ndarray,
        split_ratio: float = 0.7,
    ) -> ValidationResult:
        """
        Split data into train/test, run the config on both,
        compare performance to detect degradation.
        """
        n = len(prices)
        split_idx = int(n * split_ratio)
        warmup = 60

        # Train period
        train_sharpe = self._run_period(
            prices[:split_idx], dates[:split_idx],
            iv_series[:split_idx], iv_rank_series[:split_idx], warmup,
        )

        # Test period (simulates "live")
        test_sharpe = self._run_period(
            prices[split_idx:], dates[split_idx:],
            iv_series[split_idx:], iv_rank_series[split_idx:], min(warmup, split_idx // 4),
        )

        train_ret = self._run_return(prices[:split_idx], dates[:split_idx], iv_series[:split_idx], iv_rank_series[:split_idx], warmup)
        test_ret = self._run_return(prices[split_idx:], dates[split_idx:], iv_series[split_idx:], iv_rank_series[split_idx:], min(warmup, split_idx // 4))

        degradation = 0.0
        if train_sharpe > 0:
            degradation = max(0, 1.0 - test_sharpe / train_sharpe)

        return ValidationResult(
            config=self.base_config,
            live_sharpe=round(test_sharpe, 3),
            backtest_sharpe=round(train_sharpe, 3),
            live_return_pct=round(test_ret, 2),
            backtest_return_pct=round(train_ret, 2),
            degradation_pct=round(degradation * 100, 1),
            needs_review=degradation > self.DEGRADATION_THRESHOLD,
            details={"split_idx": split_idx, "train_days": split_idx, "test_days": n - split_idx},
        )

    # ------------------------------------------------------------------
    # Parameter sensitivity re-check
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        prices: np.ndarray,
        dates: list[datetime],
        iv_series: np.ndarray,
        iv_rank_series: np.ndarray,
    ) -> list[SensitivityResult]:
        """
        Re-run sensitivity sweeps on latest data for key parameters.
        Returns recommendations only when improvement is significant and robust.
        """
        warmup = 60
        results: list[SensitivityResult] = []

        # Delta sweep
        delta_values = [0.20, 0.25, 0.30, 0.35, 0.40]
        results.append(self._sweep_param(
            "put_delta", delta_values, prices, dates, iv_series, iv_rank_series, warmup,
        ))

        # DTE sweep
        dte_values = [30, 45, 60, 75, 90]
        results.append(self._sweep_param(
            "target_dte", dte_values, prices, dates, iv_series, iv_rank_series, warmup,
        ))

        # Trend filter SMA sweep
        sma_values = [0, 20, 50, 100, 200]
        results.append(self._sweep_param(
            "trend_sma_period", sma_values, prices, dates, iv_series, iv_rank_series, warmup,
        ))

        # Profit target sweep
        pt_values = [0.0, 0.25, 0.50, 0.75]
        results.append(self._sweep_param(
            "profit_target", pt_values, prices, dates, iv_series, iv_rank_series, warmup,
        ))

        return results

    # ------------------------------------------------------------------
    # Symbol universe scoring
    # ------------------------------------------------------------------

    def score_symbol(
        self,
        symbol: str,
        prices: np.ndarray,
        volume: np.ndarray | None = None,
        options_volume: float | None = None,
        options_oi: float | None = None,
        portfolio_returns: np.ndarray | None = None,
        account_size: float = 10_000.0,
    ) -> SymbolScore:
        """
        Compute Wheel Suitability Score for a candidate symbol.
        Each sub-score is 0-100, total is weighted average.
        """
        # IV rank score (higher = better for premium selling)
        rv = realized_volatility(prices, 30)
        valid_rv = rv[~np.isnan(rv)]
        iv_rank_score = 0.0
        if len(valid_rv) > 20:
            current_rv = valid_rv[-1]
            lo, hi = float(np.min(valid_rv)), float(np.max(valid_rv))
            if hi > lo:
                iv_rank_score = min(100, (current_rv - lo) / (hi - lo) * 100)

        # Liquidity score
        liquidity_score = 50.0
        if options_volume is not None and options_oi is not None and options_oi > 0:
            vol_oi_ratio = options_volume / options_oi
            liquidity_score = min(100, vol_oi_ratio * 100)
        elif volume is not None and len(volume) > 0:
            avg_vol = float(np.mean(volume[-30:]))
            liquidity_score = min(100, avg_vol / 1_000_000 * 10)

        # Trend stability (lower ADX-proxy = more range-bound = better)
        trend_stability_score = 50.0
        if len(prices) > 30:
            log_rets = np.diff(np.log(prices[-60:]))
            if len(log_rets) > 10:
                # Directional ratio: abs(mean return) / std as ADX proxy
                mean_ret = abs(float(np.mean(log_rets)))
                std_ret = float(np.std(log_rets))
                if std_ret > 0:
                    directional_ratio = mean_ret / std_ret
                    trend_stability_score = max(0, min(100, (1 - directional_ratio * 5) * 100))

        # Price range score (affordable for account)
        price_range_score = 0.0
        if len(prices) > 0:
            current_price = float(prices[-1])
            cost_per_contract = current_price * 100
            if cost_per_contract > 0:
                contracts_affordable = account_size * 0.8 / cost_per_contract
                if contracts_affordable >= 2:
                    price_range_score = 100
                elif contracts_affordable >= 1:
                    price_range_score = 70
                elif contracts_affordable >= 0.5:
                    price_range_score = 30
                else:
                    price_range_score = 0

        # Correlation score (lower = better diversification)
        correlation_score = 50.0
        if portfolio_returns is not None and len(prices) > 30:
            sym_rets = np.diff(np.log(prices[-len(portfolio_returns) - 1:]))
            if len(sym_rets) >= len(portfolio_returns):
                sym_rets = sym_rets[-len(portfolio_returns):]
                corr = float(np.corrcoef(sym_rets, portfolio_returns)[0, 1])
                if np.isfinite(corr):
                    correlation_score = max(0, (1 - abs(corr)) * 100)

        # Weighted total
        weights = {
            "iv_rank": 0.25,
            "liquidity": 0.20,
            "trend_stability": 0.15,
            "price_range": 0.25,
            "correlation": 0.15,
        }
        total = (
            iv_rank_score * weights["iv_rank"]
            + liquidity_score * weights["liquidity"]
            + trend_stability_score * weights["trend_stability"]
            + price_range_score * weights["price_range"]
            + correlation_score * weights["correlation"]
        )

        return SymbolScore(
            symbol=symbol,
            total_score=round(total, 1),
            iv_rank_score=round(iv_rank_score, 1),
            liquidity_score=round(liquidity_score, 1),
            trend_stability_score=round(trend_stability_score, 1),
            price_range_score=round(price_range_score, 1),
            correlation_score=round(correlation_score, 1),
            details={"current_price": round(float(prices[-1]), 2) if len(prices) > 0 else 0},
        )

    def rank_symbols(
        self,
        symbol_data: dict[str, dict],
        portfolio_returns: np.ndarray | None = None,
        account_size: float = 10_000.0,
        top_n: int = 10,
    ) -> list[SymbolScore]:
        """
        Score and rank multiple symbols.

        symbol_data: {symbol: {"prices": ndarray, "volume": ndarray | None, ...}}
        """
        scores: list[SymbolScore] = []
        for sym, data in symbol_data.items():
            prices = data.get("prices")
            if prices is None or len(prices) < 60:
                continue
            score = self.score_symbol(
                symbol=sym,
                prices=prices,
                volume=data.get("volume"),
                options_volume=data.get("options_volume"),
                options_oi=data.get("options_oi"),
                portfolio_returns=portfolio_returns,
                account_size=account_size,
            )
            scores.append(score)

        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores[:top_n]

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_daily_report(
        self,
        monitor: StrategyMonitor,
        positions: list[dict] | None = None,
    ) -> DailyReport:
        """One-liner summary of positions, P&L, alerts."""
        equities = monitor._equities
        dates = monitor._dates

        daily_pnl = 0.0
        daily_ret = 0.0
        if len(equities) >= 2:
            daily_pnl = equities[-1] - equities[-2]
            if equities[-2] > 0:
                daily_ret = daily_pnl / equities[-2] * 100

        recent_alerts = [
            str(a) for a in monitor.alerts[-5:]
        ]

        return DailyReport(
            date=dates[-1] if dates else datetime.now(),
            positions=positions or [],
            total_equity=equities[-1] if equities else 0,
            daily_pnl=round(daily_pnl, 2),
            daily_return_pct=round(daily_ret, 2),
            alerts=recent_alerts,
            regime=monitor.regime.current,
        )

    def generate_weekly_report(
        self,
        monitor: StrategyMonitor,
    ) -> WeeklyReport:
        """Rolling performance, regime state, parameter health."""
        equities = monitor._equities
        dates = monitor._dates

        week_start = max(0, len(equities) - 5)
        start_eq = equities[week_start] if week_start < len(equities) else 0
        end_eq = equities[-1] if equities else 0
        weekly_ret = ((end_eq / start_eq - 1) * 100) if start_eq > 0 else 0.0

        # Parameter health
        param_health = "OK"
        if monitor.last_validation_date:
            days_since = (dates[-1] - monitor.last_validation_date).days if dates else 0
            if days_since > 90:
                param_health = f"STALE ({days_since}d since validation)"
            elif days_since > 60:
                param_health = f"AGING ({days_since}d since validation)"

        # Alert summary: count by level
        week_alerts = [a for a in monitor.alerts if dates and a.timestamp >= dates[max(0, len(dates) - 5)]]
        alert_levels: dict[str, int] = {}
        for a in week_alerts:
            alert_levels[a.level] = alert_levels.get(a.level, 0) + 1
        alerts_summary = [f"{level}: {count}" for level, count in sorted(alert_levels.items())]

        # Trade count this week
        week_start_date = dates[week_start] if week_start < len(dates) else datetime.min
        trades = sum(1 for e in monitor._journal if e.date >= week_start_date)

        return WeeklyReport(
            week_ending=dates[-1] if dates else datetime.now(),
            start_equity=round(start_eq, 2),
            end_equity=round(end_eq, 2),
            weekly_return_pct=round(weekly_ret, 2),
            rolling_sharpe_30d=monitor.rolling_sharpe(30),
            rolling_sharpe_60d=monitor.rolling_sharpe(60),
            regime=monitor.regime.current,
            regime_changes=monitor.regime.change_count,
            param_health=param_health,
            trades=trades,
            win_rate=monitor.win_rate(20),
            alerts_summary=alerts_summary,
        )

    def generate_monthly_report(
        self,
        monitor: StrategyMonitor,
        year: int,
        month: int,
        benchmark_prices: np.ndarray | None = None,
    ) -> MonthlyReport:
        """Full performance attribution, benchmark comparison, recommendations."""
        equities = monitor._equities
        dates = monitor._dates

        month_idx = [i for i, d in enumerate(dates) if d.year == year and d.month == month]
        if not month_idx:
            return MonthlyReport(
                year=year, month=month, start_equity=0, end_equity=0,
                monthly_return_pct=0, sharpe=0, max_dd_pct=0,
                benchmark_return_pct=0, trades=0, win_rate=0,
                premium_collected=0, premium_paid=0, net_premium=0,
                regime="unknown", recommendations=[],
            )

        start_eq = equities[month_idx[0]]
        end_eq = equities[month_idx[-1]]
        monthly_ret = ((end_eq / start_eq - 1) * 100) if start_eq > 0 else 0.0

        # Monthly Sharpe
        month_returns = [monitor._daily_returns[i - 1] for i in month_idx if i > 0 and i - 1 < len(monitor._daily_returns)]
        sharpe = 0.0
        if len(month_returns) > 5:
            rets = np.array(month_returns)
            daily_rf = self.risk_free_rate / 252
            excess = rets - daily_rf
            std = float(np.std(excess))
            if std > 0:
                sharpe = float(np.mean(excess) / std * np.sqrt(252))

        # Max DD within month
        month_eq = np.array([equities[i] for i in month_idx])
        max_dd = 0.0
        if len(month_eq) > 1:
            peak = np.maximum.accumulate(month_eq)
            dd = (peak - month_eq) / np.where(peak > 0, peak, 1)
            max_dd = float(np.max(dd) * 100)

        # Benchmark
        bench_ret = 0.0
        if benchmark_prices is not None and len(benchmark_prices) > 1:
            bench_ret = (benchmark_prices[-1] / benchmark_prices[0] - 1) * 100

        # Trades this month
        month_trades = [e for e in monitor._journal if e.date.year == year and e.date.month == month]
        trades = len(month_trades)
        wins = sum(1 for t in month_trades if t.pnl > 0)
        wr = wins / trades if trades > 0 else 0.0
        premium_collected = sum(t.premium * t.contracts * 100 for t in month_trades if t.action.startswith("sell"))
        premium_paid = sum(abs(t.pnl) for t in month_trades if t.pnl < 0)
        net_premium = premium_collected - premium_paid

        # Recommendations
        recs: list[str] = []
        if sharpe < 0:
            recs.append("Negative Sharpe this month. Consider reducing position size.")
        if max_dd > 20:
            recs.append(f"Max DD of {max_dd:.1f}% this month. Review stop-loss settings.")
        if monitor.regime.current == "volatile":
            recs.append("Volatile regime. Consider tightening delta or reducing DTE.")
        if wr < 0.6 and trades >= 5:
            recs.append(f"Win rate {wr:.0%} below 60%. Review entry criteria (IV rank filter, trend filter).")
        if monthly_ret < bench_ret - 2:
            recs.append(f"Underperforming benchmark by {bench_ret - monthly_ret:.1f}%. Review strategy allocation.")

        return MonthlyReport(
            year=year, month=month,
            start_equity=round(start_eq, 2),
            end_equity=round(end_eq, 2),
            monthly_return_pct=round(monthly_ret, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd, 1),
            benchmark_return_pct=round(bench_ret, 2),
            trades=trades,
            win_rate=round(wr, 2),
            premium_collected=round(premium_collected, 2),
            premium_paid=round(premium_paid, 2),
            net_premium=round(net_premium, 2),
            regime=monitor.regime.current,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_period(
        self,
        prices: np.ndarray,
        dates: list[datetime],
        iv_series: np.ndarray,
        iv_rank_series: np.ndarray,
        warmup: int,
    ) -> float:
        """Run strategy on a slice and return Sharpe."""
        if len(prices) < warmup + 10:
            return 0.0

        strategy = WheelStrategy(self.base_config)
        for i in range(warmup, len(prices)):
            current_iv = float(iv_series[i]) if i < len(iv_series) and not np.isnan(iv_series[i]) else 0.25
            current_iv = max(current_iv, 0.05)
            current_rank = float(iv_rank_series[i]) if i < len(iv_rank_series) else 50.0
            strategy.on_bar(dates[i], float(prices[i]), current_iv, current_rank)

        s = strategy.summary()
        return s.get("sharpe_ratio", 0.0)

    def _run_return(
        self,
        prices: np.ndarray,
        dates: list[datetime],
        iv_series: np.ndarray,
        iv_rank_series: np.ndarray,
        warmup: int,
    ) -> float:
        """Run strategy on a slice and return total return %."""
        if len(prices) < warmup + 10:
            return 0.0

        strategy = WheelStrategy(self.base_config)
        for i in range(warmup, len(prices)):
            current_iv = float(iv_series[i]) if i < len(iv_series) and not np.isnan(iv_series[i]) else 0.25
            current_iv = max(current_iv, 0.05)
            current_rank = float(iv_rank_series[i]) if i < len(iv_rank_series) else 50.0
            strategy.on_bar(dates[i], float(prices[i]), current_iv, current_rank)

        s = strategy.summary()
        return s.get("total_return_pct", 0.0)

    def _sweep_param(
        self,
        param_name: str,
        values: list,
        prices: np.ndarray,
        dates: list[datetime],
        iv_series: np.ndarray,
        iv_rank_series: np.ndarray,
        warmup: int,
    ) -> SensitivityResult:
        """Sweep a single parameter and find the optimal value."""
        current_val = getattr(self.base_config, param_name)
        sharpes: dict[float, float] = {}

        for val in values:
            from dataclasses import replace
            cfg = replace(self.base_config, **{param_name: val})
            strategy = WheelStrategy(cfg)
            for i in range(warmup, len(prices)):
                current_iv = float(iv_series[i]) if i < len(iv_series) and not np.isnan(iv_series[i]) else 0.25
                current_iv = max(current_iv, 0.05)
                current_rank = float(iv_rank_series[i]) if i < len(iv_rank_series) else 50.0
                strategy.on_bar(dates[i], float(prices[i]), current_iv, current_rank)
            s = strategy.summary()
            sharpes[val] = s.get("sharpe_ratio", 0.0)

        current_sharpe = sharpes.get(current_val, 0.0)
        optimal_val = max(sharpes, key=lambda v: sharpes[v])
        optimal_sharpe = sharpes[optimal_val]
        improvement = optimal_sharpe - current_sharpe

        # Check robustness: optimal should not be an edge value
        sorted_vals = sorted(values)
        is_edge = optimal_val == sorted_vals[0] or optimal_val == sorted_vals[-1]

        # Also check neighbors are within 0.1 Sharpe
        robust = not is_edge
        if robust:
            idx = sorted_vals.index(optimal_val)
            neighbors = []
            if idx > 0:
                neighbors.append(sharpes[sorted_vals[idx - 1]])
            if idx < len(sorted_vals) - 1:
                neighbors.append(sharpes[sorted_vals[idx + 1]])
            if neighbors:
                robust = all(abs(n - optimal_sharpe) < 0.3 for n in neighbors)

        recommend = "KEEP"
        if improvement > self.SHARPE_IMPROVEMENT_THRESHOLD and robust:
            recommend = f"CHANGE {param_name}: {current_val} -> {optimal_val}"
        elif improvement > self.SHARPE_IMPROVEMENT_THRESHOLD and not is_edge:
            recommend = f"CONSIDER {param_name}: {current_val} -> {optimal_val} (check robustness)"

        return SensitivityResult(
            parameter=param_name,
            current_value=current_val,
            optimal_value=optimal_val,
            current_sharpe=round(current_sharpe, 3),
            optimal_sharpe=round(optimal_sharpe, 3),
            sharpe_improvement=round(improvement, 3),
            robust=robust,
            recommendation=recommend,
        )
