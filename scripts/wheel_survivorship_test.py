"""
Wheel Strategy Survivorship Bias Test
Tests on 27 diverse stocks across 7 categories to check if the strategy
works broadly or only on cherry-picked winners.
"""

import sys
import time
sys.path.insert(0, "/Users/mahimnpatel/Documents/dev/randomThings")

import yfinance as yf
import numpy as np
from datetime import datetime

from trading_algo.broker.base import Bar
from trading_algo.quant_core.strategies.options.wheel import WheelStrategy, WheelConfig
from trading_algo.quant_core.strategies.options.options_backtester import run_options_backtest


SYMBOLS = {
    "Tech Winners": ["AAPL", "MSFT", "NVDA", "GOOG"],
    "Tech Losers/Flat": ["INTC", "SNAP", "PINS", "PYPL"],
    "Meme/Volatile": ["GME", "AMC", "MARA"],
    "Financials": ["JPM", "GS", "C", "WFC"],
    "Consumer": ["KO", "PG", "MCD", "WMT"],
    "Energy": ["XOM", "CVX"],
    "REITs/Other": ["O", "SCHW"],
    "ETFs": ["SPY", "QQQ", "IWM", "XLF"],
}

BEST_CONFIG = WheelConfig(
    initial_capital=100_000.0,
    put_delta=0.30,
    call_delta=0.30,
    target_dte=60,
    profit_target=0.0,       # no profit target — let expire
    roll_dte=0,              # let expire / allow assignment
    stop_loss=0.0,           # no stop loss
    trend_sma_period=50,     # SMA50 trend filter
    min_iv_rank=25.0,
    min_premium_pct=0.005,
    cash_reserve_pct=0.20,
    risk_free_rate=0.045,
    commission_per_contract=0.90,
    commission_per_share=0.005,
    bid_ask_slip_per_share=0.05,
)


def yf_to_bars(ticker: str, period: str = "2y") -> list[Bar]:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return []
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    bars = []
    for idx, row in df.iterrows():
        try:
            o = float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"])
            h = float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"])
            l = float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"])
            c = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
            v = float(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else float(row["Volume"])
        except Exception:
            continue
        if np.isnan(c):
            continue
        bars.append(Bar(timestamp_epoch_s=idx.timestamp(), open=o, high=h, low=l, close=c, volume=v))
    return bars


def run_single(symbol: str, category: str) -> dict | None:
    bars = yf_to_bars(symbol)
    if len(bars) < 120:
        print(f"  {symbol}: SKIP — only {len(bars)} bars")
        return None

    strategy = WheelStrategy(BEST_CONFIG)
    try:
        report = run_options_backtest(strategy, bars, symbol)
    except Exception as e:
        print(f"  {symbol}: ERROR — {e}")
        return None

    s = report.summary
    return {
        "symbol": symbol,
        "category": category,
        "strat_ret": s["total_return_pct"],
        "bnh_ret": report.benchmark_return_pct,
        "vs_bnh": s["total_return_pct"] - report.benchmark_return_pct,
        "sharpe": s["sharpe_ratio"],
        "max_dd": s["max_drawdown_pct"],
        "win_rate": s["win_rate"],
        "trades": s["total_trades"],
        "cycles": s["wheel_cycles"],
        "final_eq": s["final_equity"],
        "bnh_sharpe": report.benchmark_sharpe,
        "bnh_dd": report.benchmark_max_dd_pct,
        "ann_ret": report.annualized_return(),
        "ann_bnh": report.benchmark_annualized(),
        "days": s["days"],
    }


def main():
    print("=" * 120)
    print("  WHEEL STRATEGY — SURVIVORSHIP BIAS TEST (27 STOCKS, 7 CATEGORIES)")
    print("  Config: delta=0.30, DTE=60, no profit target, let expire, no stop loss, SMA50, min_iv_rank=25")
    print("  Capital: $100,000 per symbol | Period: 2Y daily via yfinance")
    print("=" * 120)
    print()

    all_results: list[dict] = []
    for category, tickers in SYMBOLS.items():
        print(f"[{category}]")
        for sym in tickers:
            t0 = time.time()
            result = run_single(sym, category)
            elapsed = time.time() - t0
            if result:
                all_results.append(result)
                arrow = "+" if result["strat_ret"] > 0 else ""
                print(f"  {sym:<6} {arrow}{result['strat_ret']:>7.2f}% (BnH: {result['bnh_ret']:>+7.2f}%)  [{elapsed:.1f}s]")
            else:
                print(f"  {sym:<6} FAILED [{elapsed:.1f}s]")
        print()

    if not all_results:
        print("No results to display!")
        return

    # ── Main results table ──
    print()
    print("=" * 140)
    print(f"{'Symbol':<8} {'Category':<18} {'Strat Ret%':>10} {'BnH Ret%':>10} {'vs BnH':>8} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7} {'Cycles':>7}")
    print("-" * 140)

    for r in sorted(all_results, key=lambda x: x["strat_ret"], reverse=True):
        vs = r["vs_bnh"]
        vs_str = f"{vs:>+7.2f}"
        print(f"{r['symbol']:<8} {r['category']:<18} {r['strat_ret']:>10.2f} {r['bnh_ret']:>10.2f} {vs_str:>8} {r['sharpe']:>8.3f} {r['max_dd']:>8.2f} {r['win_rate']:>9.1f} {r['trades']:>7d} {r['cycles']:>7d}")
    print("-" * 140)

    # ── Aggregate stats ──
    n = len(all_results)
    avg_strat = np.mean([r["strat_ret"] for r in all_results])
    avg_bnh = np.mean([r["bnh_ret"] for r in all_results])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    avg_dd = np.mean([r["max_dd"] for r in all_results])
    avg_wr = np.mean([r["win_rate"] for r in all_results])
    med_strat = np.median([r["strat_ret"] for r in all_results])

    print()
    print(f"{'OVERALL AVERAGE':<27} {avg_strat:>10.2f} {avg_bnh:>10.2f} {avg_strat - avg_bnh:>+8.2f} {avg_sharpe:>8.3f} {avg_dd:>8.2f} {avg_wr:>9.1f}")
    print(f"{'MEDIAN RETURN':<27} {med_strat:>10.2f}")
    print()

    # ── By category ──
    print("=" * 100)
    print(f"{'Category':<20} {'Avg Strat%':>11} {'Avg BnH%':>11} {'vs BnH':>9} {'Avg Sharpe':>11} {'Avg MaxDD%':>11}")
    print("-" * 100)
    for category in SYMBOLS:
        cat_results = [r for r in all_results if r["category"] == category]
        if not cat_results:
            continue
        cs = np.mean([r["strat_ret"] for r in cat_results])
        cb = np.mean([r["bnh_ret"] for r in cat_results])
        csh = np.mean([r["sharpe"] for r in cat_results])
        cdd = np.mean([r["max_dd"] for r in cat_results])
        print(f"{category:<20} {cs:>11.2f} {cb:>11.2f} {cs - cb:>+9.2f} {csh:>11.3f} {cdd:>11.2f}")
    print("-" * 100)

    # ── Key counts ──
    profitable = sum(1 for r in all_results if r["strat_ret"] > 0)
    beat_rf = sum(1 for r in all_results if r["strat_ret"] > 8.0)
    beat_bnh = sum(1 for r in all_results if r["strat_ret"] > r["bnh_ret"])
    positive_sharpe = sum(1 for r in all_results if r["sharpe"] > 0)

    print()
    print("KEY METRICS:")
    print(f"  Profitable (>0%):           {profitable}/{n} ({profitable / n * 100:.0f}%)")
    print(f"  Beat risk-free (>8%):       {beat_rf}/{n} ({beat_rf / n * 100:.0f}%)")
    print(f"  Beat buy-and-hold:          {beat_bnh}/{n} ({beat_bnh / n * 100:.0f}%)")
    print(f"  Positive Sharpe:            {positive_sharpe}/{n} ({positive_sharpe / n * 100:.0f}%)")
    print()

    # ── Best / Worst ──
    best = max(all_results, key=lambda r: r["strat_ret"])
    worst = min(all_results, key=lambda r: r["strat_ret"])

    print(f"BEST:  {best['symbol']} ({best['category']})")
    print(f"  Return: {best['strat_ret']:>+.2f}% | BnH: {best['bnh_ret']:>+.2f}% | Sharpe: {best['sharpe']:.3f} | MaxDD: {best['max_dd']:.2f}% | WinRate: {best['win_rate']:.1f}%")
    if best["strat_ret"] > best["bnh_ret"]:
        print(f"  Beat BnH by {best['strat_ret'] - best['bnh_ret']:.2f}pp")
    print()

    print(f"WORST: {worst['symbol']} ({worst['category']})")
    print(f"  Return: {worst['strat_ret']:>+.2f}% | BnH: {worst['bnh_ret']:>+.2f}% | Sharpe: {worst['sharpe']:.3f} | MaxDD: {worst['max_dd']:.2f}% | WinRate: {worst['win_rate']:.1f}%")
    if worst["bnh_ret"] > worst["strat_ret"]:
        print(f"  Underperformed BnH by {worst['bnh_ret'] - worst['strat_ret']:.2f}pp")
    print()

    # ── Risk-adjusted analysis ──
    best_sharpe = max(all_results, key=lambda r: r["sharpe"])
    worst_dd = max(all_results, key=lambda r: r["max_dd"])
    print("RISK-ADJUSTED:")
    print(f"  Best Sharpe:   {best_sharpe['symbol']} ({best_sharpe['sharpe']:.3f})")
    print(f"  Worst MaxDD:   {worst_dd['symbol']} ({worst_dd['max_dd']:.2f}%)")
    print()

    # ── Verdict ──
    print("=" * 80)
    print("VERDICT:")
    if profitable / n >= 0.7 and avg_sharpe > 0.3:
        print(f"  PASS — Strategy is profitable on {profitable}/{n} symbols with avg Sharpe {avg_sharpe:.3f}")
        if beat_bnh / n < 0.5:
            print(f"  CAVEAT — Only beats buy-and-hold on {beat_bnh}/{n} symbols")
            print(f"           Premium selling adds income but caps upside in strong trends")
    elif profitable / n >= 0.5:
        print(f"  MIXED — Profitable on {profitable}/{n} symbols but inconsistent")
        print(f"          Avg return: {avg_strat:.2f}% | Median: {med_strat:.2f}%")
    else:
        print(f"  FAIL — Only profitable on {profitable}/{n} symbols")
        print(f"         Strategy does NOT generalize beyond cherry-picked winners")
    print("=" * 80)


if __name__ == "__main__":
    main()
