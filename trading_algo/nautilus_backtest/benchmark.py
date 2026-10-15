"""Benchmark-relative statistics and passive baselines.

Statistics come from NautilusTrader's analysis module (beta, alpha, tracking
error, information ratio, Treynor, up/down capture, Sharpe, Sortino). They are
computed on trading-session returns only. NautilusTrader's own backtest returns
include zero rows for weekends, which would understate volatility.

The bundled IBKR bars are split-adjusted but not dividend-adjusted. Benchmark
and baseline returns are therefore price returns, and they are labelled that
way.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from nautilus_trader.analysis import (
    Alpha,
    BetaRatio,
    DownCaptureRatio,
    InformationRatio,
    PortfolioAnalyzer,
    SharpeRatio,
    SortinoRatio,
    TrackingError,
    TreynorRatio,
    UpCaptureRatio,
)

TRADING_DAYS = 252
PRICE_RETURN_LABEL = "price return (split-adjusted, dividends excluded)"


def daily_risk_free(annual_rate: float) -> float:
    """Per-session risk-free rate, compounded over 252 sessions."""
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0


def _keys(days: Iterable[date]) -> list[int]:
    """UTC-midnight nanosecond keys, the format NautilusTrader statistics take."""
    return [pd.Timestamp(day).tz_localize("UTC").as_unit("ns").value for day in days]


def benchmark_statistics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Strategy statistics against a benchmark, on the sessions both share.

    ``risk_free_rate`` is annual. Alpha and Treynor need it per session, so it
    is converted here.
    """
    joined = pd.concat({"strategy": returns, "benchmark": benchmark_returns}, axis=1).dropna()
    if len(joined) < 3:
        raise ValueError("need at least 3 shared sessions for benchmark statistics")
    rf = daily_risk_free(risk_free_rate)

    analyzer = PortfolioAnalyzer()
    for key, value in zip(_keys(joined.index), joined["strategy"], strict=True):
        analyzer.add_return(key, float(value))
    for statistic in (
        BetaRatio(),
        Alpha(period=TRADING_DAYS, risk_free_rate=rf),
        InformationRatio(period=TRADING_DAYS),
        TrackingError(period=TRADING_DAYS),
        TreynorRatio(period=TRADING_DAYS, risk_free_rate=rf),
        UpCaptureRatio(period=TRADING_DAYS),
        DownCaptureRatio(period=TRADING_DAYS),
        SharpeRatio(period=TRADING_DAYS),
        SortinoRatio(period=TRADING_DAYS),
    ):
        analyzer.register_statistic(statistic)

    benchmark = dict(zip(_keys(joined.index), joined["benchmark"].astype(float), strict=True))
    stats = {
        **analyzer.get_performance_stats_returns(),
        **analyzer.get_performance_stats_returns_vs_benchmark(benchmark),
        "Correlation": float(np.corrcoef(joined["strategy"], joined["benchmark"])[0, 1]),
        "Sessions": float(len(joined)),
    }
    return dict(sorted(stats.items()))


def price_returns(close: pd.Series) -> pd.Series:
    """Session-to-session price returns indexed by session date."""
    returns = close.pct_change().dropna()
    returns.index = [ts.date() for ts in returns.index]
    return returns


def buy_and_hold(
    frames: Mapping[str, pd.DataFrame],
    start: date,
    initial_capital: float,
) -> pd.Series:
    """Equal-weight buy-and-hold NAV from the close of ``start``, no rebalancing.

    A frictionless reference point, not a tradable result.
    """
    closes = pd.concat({s: f["close"] for s, f in frames.items()}, axis=1).ffill()
    closes.index = [ts.date() for ts in closes.index]
    closes = closes.loc[start:]
    first = closes.iloc[0]
    unpriced = sorted(first[first.isna()].index)
    if unpriced:
        raise ValueError(f"no price on {start} for {unpriced}, cannot buy them at the start")
    shares = (initial_capital / len(first)) / first
    return (closes * shares).sum(axis=1).rename("buy_and_hold_nav")


def simple_stats(nav: pd.Series) -> dict[str, float]:
    """Total and annualized return, volatility, Sharpe (rf=0) and max drawdown."""
    returns = nav.pct_change().dropna()
    years = len(returns) / TRADING_DAYS
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    drawdown = float((1.0 - nav / nav.cummax()).max())
    return {
        "total_return": total,
        "annualized_return": (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else 0.0,
        "volatility": vol,
        "sharpe_rf0": float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": drawdown,
    }
