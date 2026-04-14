"""
Strategy Monitor — continuous health tracking for live options strategies.

Tracks rolling performance, detects regime changes, monitors drawdowns,
flags stale parameters, and maintains a detailed trade journal.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

AlertLevel = Literal["INFO", "WARN", "ALERT", "CRITICAL"]


@dataclass
class MonitorAlert:
    timestamp: datetime
    level: AlertLevel
    source: str
    message: str
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.level}] {self.timestamp:%Y-%m-%d} [{self.source}] {self.message}"


# ---------------------------------------------------------------------------
# Trade journal entry
# ---------------------------------------------------------------------------

@dataclass
class TradeJournalEntry:
    date: datetime
    symbol: str
    action: str
    strike: float
    premium: float
    underlying_price: float
    iv: float
    iv_rank: float
    regime: str
    contracts: int = 1
    pnl: float = 0.0
    attribution: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regime state
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    current: Literal["calm", "volatile", "unknown"] = "unknown"
    rv_30d: float = 0.0
    rv_252d_median: float = 0.0
    last_change: datetime | None = None
    change_count: int = 0


# ---------------------------------------------------------------------------
# StrategyMonitor
# ---------------------------------------------------------------------------

class StrategyMonitor:
    """
    Tracks live strategy performance and compares to backtest expectations.

    Feed daily equity/trade data via update() and log_trade().
    Query alerts, regime, drawdown state, and reports at any time.
    """

    REGIME_THRESHOLD = 1.5
    DD_WARN = 0.15
    DD_ALERT = 0.25
    DD_CRITICAL = 0.35
    SHARPE_DEGRADATION_THRESHOLD = 0.50
    WIN_RATE_FLOOR = 0.70
    WIN_RATE_LOOKBACK = 20
    PARAM_STALENESS_DAYS = 90

    def __init__(
        self,
        strategy_name: str,
        backtest_sharpe: float = 0.0,
        backtest_win_rate: float = 0.0,
        last_validation_date: datetime | None = None,
        db_path: str | None = None,
        risk_free_rate: float = 0.045,
    ):
        self.strategy_name = strategy_name
        self.backtest_sharpe = backtest_sharpe
        self.backtest_win_rate = backtest_win_rate
        self.last_validation_date = last_validation_date
        self.risk_free_rate = risk_free_rate

        # Time series
        self._dates: list[datetime] = []
        self._equities: list[float] = []
        self._daily_returns: list[float] = []
        self._prices: list[float] = []

        # Drawdown tracking
        self._peak_equity: float = 0.0
        self._current_dd: float = 0.0
        self._paused: bool = False

        # Trade journal
        self._journal: list[TradeJournalEntry] = []
        self._trade_results: list[float] = []

        # Regime
        self.regime = RegimeState()
        self._rv_history: list[float] = []

        # Alerts
        self.alerts: list[MonitorAlert] = []

        # Optional SQLite persistence
        self._db: sqlite3.Connection | None = None
        if db_path:
            self._db = sqlite3.connect(db_path)
            self._ensure_monitor_schema()

    # ------------------------------------------------------------------
    # Public: daily update
    # ------------------------------------------------------------------

    def update(
        self,
        date: datetime,
        equity: float,
        underlying_price: float,
        rv_30d: float | None = None,
    ) -> list[MonitorAlert]:
        """Feed one day of data. Returns any new alerts."""
        new_alerts: list[MonitorAlert] = []

        self._dates.append(date)
        self._equities.append(equity)
        self._prices.append(underlying_price)

        if len(self._equities) > 1:
            prev = self._equities[-2]
            ret = (equity - prev) / prev if prev > 0 else 0.0
            self._daily_returns.append(ret)

        # Drawdown
        new_alerts.extend(self._update_drawdown(date, equity))

        # Regime detection
        if rv_30d is not None:
            new_alerts.extend(self._update_regime(date, rv_30d))

        # Rolling performance checks
        new_alerts.extend(self._check_rolling_performance(date))

        # Win rate check
        new_alerts.extend(self._check_win_rate(date))

        # Parameter staleness
        new_alerts.extend(self._check_param_staleness(date))

        self.alerts.extend(new_alerts)

        if self._db:
            self._persist_daily(date, equity, underlying_price, rv_30d)
            for a in new_alerts:
                self._persist_alert(a)

        return new_alerts

    # ------------------------------------------------------------------
    # Public: trade logging
    # ------------------------------------------------------------------

    def log_trade(self, entry: TradeJournalEntry) -> None:
        """Record a trade in the journal."""
        self._journal.append(entry)
        if entry.pnl != 0:
            self._trade_results.append(entry.pnl)

        if self._db:
            self._persist_trade(entry)

    # ------------------------------------------------------------------
    # Public: queries
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def current_drawdown(self) -> float:
        return self._current_dd

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    def rolling_sharpe(self, window_days: int = 30) -> float:
        """Annualized Sharpe over the last N trading days."""
        if len(self._daily_returns) < window_days:
            return 0.0
        rets = np.array(self._daily_returns[-window_days:])
        daily_rf = self.risk_free_rate / 252
        excess = rets - daily_rf
        std = float(np.std(excess))
        if std < 1e-10:
            return 0.0
        result = float(np.mean(excess) / std * np.sqrt(252))
        return max(min(result, 100.0), -100.0)

    def rolling_return(self, window_days: int = 30) -> float:
        """Cumulative return over last N trading days (%)."""
        if len(self._equities) < window_days + 1:
            return 0.0
        start = self._equities[-(window_days + 1)]
        end = self._equities[-1]
        if start <= 0:
            return 0.0
        return (end / start - 1) * 100

    def rolling_max_dd(self, window_days: int = 30) -> float:
        """Max drawdown in last N trading days (as fraction, e.g. 0.15 = 15%)."""
        if len(self._equities) < window_days + 1:
            return 0.0
        window = np.array(self._equities[-(window_days + 1):])
        peak = np.maximum.accumulate(window)
        dd = (peak - window) / np.where(peak > 0, peak, 1)
        return float(np.max(dd))

    def win_rate(self, last_n: int | None = None) -> float:
        """Win rate over last N trades (or all trades)."""
        results = self._trade_results if last_n is None else self._trade_results[-last_n:]
        if not results:
            return 0.0
        wins = sum(1 for r in results if r > 0)
        return wins / len(results)

    def monthly_summary(self, year: int, month: int) -> dict:
        """Generate a monthly performance summary."""
        month_entries = [
            e for e in self._journal
            if e.date.year == year and e.date.month == month
        ]
        month_equities = [
            (d, e) for d, e in zip(self._dates, self._equities)
            if d.year == year and d.month == month
        ]

        total_pnl = sum(e.pnl for e in month_entries)
        trades = len(month_entries)
        wins = sum(1 for e in month_entries if e.pnl > 0)
        losses = sum(1 for e in month_entries if e.pnl < 0)

        start_eq = month_equities[0][1] if month_equities else 0
        end_eq = month_equities[-1][1] if month_equities else 0
        ret_pct = ((end_eq / start_eq - 1) * 100) if start_eq > 0 else 0.0

        premium_income = sum(
            e.pnl for e in month_entries
            if e.attribution.get("type") == "premium"
        )
        stock_pnl = sum(
            e.attribution.get("stock_pnl", 0) for e in month_entries
        )

        return {
            "year": year,
            "month": month,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(ret_pct, 2),
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / trades * 100, 1) if trades else 0.0,
            "premium_income": round(premium_income, 2),
            "stock_pnl": round(stock_pnl, 2),
            "regime": self.regime.current,
            "start_equity": round(start_eq, 2),
            "end_equity": round(end_eq, 2),
        }

    # ------------------------------------------------------------------
    # Internal: drawdown
    # ------------------------------------------------------------------

    def _update_drawdown(self, date: datetime, equity: float) -> list[MonitorAlert]:
        alerts: list[MonitorAlert] = []

        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            self._current_dd = (self._peak_equity - equity) / self._peak_equity
        else:
            self._current_dd = 0.0

        if self._current_dd >= self.DD_CRITICAL and not self._paused:
            self._paused = True
            alerts.append(MonitorAlert(
                timestamp=date, level="CRITICAL", source="drawdown",
                message=f"Drawdown {self._current_dd:.1%} hit CRITICAL. Strategy PAUSED (circuit breaker).",
                data={"drawdown": round(self._current_dd, 4), "peak": round(self._peak_equity, 2), "equity": round(equity, 2)},
            ))
        elif self._current_dd >= self.DD_ALERT:
            alerts.append(MonitorAlert(
                timestamp=date, level="ALERT", source="drawdown",
                message=f"Drawdown {self._current_dd:.1%} exceeds 25% threshold.",
                data={"drawdown": round(self._current_dd, 4)},
            ))
        elif self._current_dd >= self.DD_WARN:
            alerts.append(MonitorAlert(
                timestamp=date, level="WARN", source="drawdown",
                message=f"Drawdown {self._current_dd:.1%} exceeds 15% warning level.",
                data={"drawdown": round(self._current_dd, 4)},
            ))

        # Auto-unpause if drawdown recovers below WARN
        if self._paused and self._current_dd < self.DD_WARN:
            self._paused = False
            alerts.append(MonitorAlert(
                timestamp=date, level="INFO", source="drawdown",
                message="Drawdown recovered below 15%. Strategy UNPAUSED.",
                data={"drawdown": round(self._current_dd, 4)},
            ))

        return alerts

    # ------------------------------------------------------------------
    # Internal: regime detection
    # ------------------------------------------------------------------

    def _update_regime(self, date: datetime, rv_30d: float) -> list[MonitorAlert]:
        alerts: list[MonitorAlert] = []
        self._rv_history.append(rv_30d)
        self.regime.rv_30d = rv_30d

        if len(self._rv_history) < 60:
            return alerts

        # 252d median of rv_30d, or as much history as we have
        lookback = min(len(self._rv_history), 252)
        rv_252d_median = float(np.median(self._rv_history[-lookback:]))
        self.regime.rv_252d_median = rv_252d_median

        if rv_252d_median <= 0:
            return alerts

        ratio = rv_30d / rv_252d_median
        new_regime: Literal["calm", "volatile", "unknown"]

        if ratio >= self.REGIME_THRESHOLD:
            new_regime = "volatile"
        elif ratio <= 1.0 / self.REGIME_THRESHOLD:
            new_regime = "calm"
        else:
            new_regime = self.regime.current if self.regime.current != "unknown" else "calm"

        if new_regime != self.regime.current and self.regime.current != "unknown":
            old = self.regime.current
            self.regime.current = new_regime
            self.regime.last_change = date
            self.regime.change_count += 1
            alerts.append(MonitorAlert(
                timestamp=date, level="ALERT", source="regime",
                message=f"Regime change: {old} -> {new_regime} (30d RV={rv_30d:.1%}, ratio={ratio:.2f}x median).",
                data={"old": old, "new": new_regime, "rv_30d": round(rv_30d, 4), "rv_252d_median": round(rv_252d_median, 4), "ratio": round(ratio, 2)},
            ))
        elif self.regime.current == "unknown":
            self.regime.current = new_regime

        return alerts

    # ------------------------------------------------------------------
    # Internal: rolling performance check
    # ------------------------------------------------------------------

    def _check_rolling_performance(self, date: datetime) -> list[MonitorAlert]:
        alerts: list[MonitorAlert] = []

        if self.backtest_sharpe <= 0 or len(self._daily_returns) < 30:
            return alerts

        for window in [30, 60, 90]:
            if len(self._daily_returns) < window:
                continue
            live_sharpe = self.rolling_sharpe(window)
            threshold = self.backtest_sharpe * self.SHARPE_DEGRADATION_THRESHOLD
            if live_sharpe < threshold:
                alerts.append(MonitorAlert(
                    timestamp=date, level="ALERT", source="performance",
                    message=f"{window}d rolling Sharpe ({live_sharpe:.2f}) < 50% of backtest ({self.backtest_sharpe:.2f}).",
                    data={"window": window, "live_sharpe": round(live_sharpe, 3), "backtest_sharpe": round(self.backtest_sharpe, 3)},
                ))
                break  # one alert per update is enough

        return alerts

    # ------------------------------------------------------------------
    # Internal: win rate check
    # ------------------------------------------------------------------

    def _check_win_rate(self, date: datetime) -> list[MonitorAlert]:
        alerts: list[MonitorAlert] = []
        if len(self._trade_results) < self.WIN_RATE_LOOKBACK:
            return alerts

        recent_wr = self.win_rate(self.WIN_RATE_LOOKBACK)
        if recent_wr < self.WIN_RATE_FLOOR:
            alerts.append(MonitorAlert(
                timestamp=date, level="ALERT", source="win_rate",
                message=f"Win rate over last {self.WIN_RATE_LOOKBACK} trades: {recent_wr:.0%} (below {self.WIN_RATE_FLOOR:.0%} floor).",
                data={"win_rate": round(recent_wr, 3), "lookback": self.WIN_RATE_LOOKBACK},
            ))

        return alerts

    # ------------------------------------------------------------------
    # Internal: parameter staleness
    # ------------------------------------------------------------------

    def _check_param_staleness(self, date: datetime) -> list[MonitorAlert]:
        alerts: list[MonitorAlert] = []

        if self.last_validation_date is None:
            return alerts

        days_since = (date - self.last_validation_date).days
        if days_since > self.PARAM_STALENESS_DAYS:
            # Only alert once per week
            if len(self._dates) % 5 == 0:
                alerts.append(MonitorAlert(
                    timestamp=date, level="WARN", source="staleness",
                    message=f"Parameters last validated {days_since}d ago (>{self.PARAM_STALENESS_DAYS}d). Suggest re-running walk-forward.",
                    data={"days_since_validation": days_since, "last_validation": self.last_validation_date.isoformat()},
                ))

        if self.regime.last_change and self.last_validation_date:
            if self.regime.last_change > self.last_validation_date:
                if len(self._dates) % 5 == 0:
                    alerts.append(MonitorAlert(
                        timestamp=date, level="WARN", source="staleness",
                        message="Regime changed since last parameter validation. Suggest re-optimization.",
                        data={"regime_change": self.regime.last_change.isoformat(), "last_validation": self.last_validation_date.isoformat()},
                    ))

        return alerts

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _ensure_monitor_schema(self) -> None:
        if not self._db:
            return
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS monitor_daily(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                date TEXT NOT NULL,
                equity REAL NOT NULL,
                underlying_price REAL,
                rv_30d REAL,
                drawdown REAL,
                regime TEXT
            );
            CREATE TABLE IF NOT EXISTS monitor_alerts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                data_json TEXT
            );
            CREATE TABLE IF NOT EXISTS monitor_trades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                strike REAL,
                premium REAL,
                underlying_price REAL,
                iv REAL,
                iv_rank REAL,
                regime TEXT,
                contracts INTEGER,
                pnl REAL,
                attribution_json TEXT
            );
        """)
        self._db.commit()

    def _persist_daily(self, date: datetime, equity: float, price: float, rv: float | None) -> None:
        if not self._db:
            return
        self._db.execute(
            "INSERT INTO monitor_daily(strategy, date, equity, underlying_price, rv_30d, drawdown, regime) VALUES(?,?,?,?,?,?,?)",
            (self.strategy_name, date.isoformat(), equity, price, rv, self._current_dd, self.regime.current),
        )
        self._db.commit()

    def _persist_alert(self, alert: MonitorAlert) -> None:
        if not self._db:
            return
        self._db.execute(
            "INSERT INTO monitor_alerts(strategy, timestamp, level, source, message, data_json) VALUES(?,?,?,?,?,?)",
            (self.strategy_name, alert.timestamp.isoformat(), alert.level, alert.source, alert.message, json.dumps(alert.data)),
        )
        self._db.commit()

    def _persist_trade(self, entry: TradeJournalEntry) -> None:
        if not self._db:
            return
        self._db.execute(
            "INSERT INTO monitor_trades(strategy, date, symbol, action, strike, premium, underlying_price, iv, iv_rank, regime, contracts, pnl, attribution_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.strategy_name, entry.date.isoformat(), entry.symbol, entry.action, entry.strike, entry.premium,
             entry.underlying_price, entry.iv, entry.iv_rank, entry.regime, entry.contracts, entry.pnl, json.dumps(entry.attribution)),
        )
        self._db.commit()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
