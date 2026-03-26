"""
Portfolio-Level Wheel Backtest Runner

Pulls 5Y daily data for 15 symbols, runs PortfolioWheel with shared $50K
capital, and compares to SPY buy-and-hold.
"""

import sys
import time
sys.path.insert(0, "/Users/mahimnpatel/Documents/dev/randomThings")

import numpy as np
import yfinance as yf
from datetime import datetime

from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_rank as compute_iv_rank,
    iv_series_from_prices,
)
from trading_algo.quant_core.strategies.options.portfolio_wheel import (
    PortfolioWheel,
    PortfolioWheelConfig,
)

SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "JPM", "BAC",
    "KO", "PG", "XOM", "JNJ", "T",
    "F", "SOFI", "PLTR", "NIO", "SPY",
]

WARMUP = 60
RV_WINDOW = 30
IV_LOOKBACK = 252


def download_data(symbols: list[str], period: str = "5y") -> dict[str, tuple[np.ndarray, list[datetime]]]:
    """Download daily close prices for all symbols. Returns {sym: (prices_array, dates_list)}."""
    print(f"Downloading {period} daily data for {len(symbols)} symbols...")
    t0 = time.time()

    df = yf.download(symbols, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned empty dataframe")

    # Handle multi-level columns
    if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "levels"):
        try:
            close_df = df["Close"]
        except KeyError:
            close_df = df
    else:
        close_df = df

    # If single symbol, close_df is a Series
    if len(symbols) == 1:
        close_df = close_df.to_frame(name=symbols[0])

    result: dict[str, tuple[np.ndarray, list[datetime]]] = {}
    for sym in symbols:
        if sym not in close_df.columns:
            print(f"  {sym}: no data, skipping")
            continue
        col = close_df[sym].dropna()
        if len(col) < WARMUP + 50:
            print(f"  {sym}: only {len(col)} bars, skipping")
            continue
        prices = col.values.astype(float)
        dates = [datetime(d.year, d.month, d.day) for d in col.index]
        result[sym] = (prices, dates)
        print(f"  {sym}: {len(prices)} bars ({dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')})")

    elapsed = time.time() - t0
    print(f"Download complete in {elapsed:.1f}s — {len(result)}/{len(symbols)} symbols loaded\n")
    return result


def build_aligned_series(
    data: dict[str, tuple[np.ndarray, list[datetime]]],
) -> tuple[list[datetime], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Align all symbols to a common date index.
    Returns (common_dates, prices_dict, iv_dict, iv_rank_dict) where each dict
    maps sym -> array aligned to common_dates.
    """
    # Find common date range
    all_date_sets = [set(dates) for _, (_, dates) in data.items()]
    common_dates_set = all_date_sets[0]
    for ds in all_date_sets[1:]:
        common_dates_set &= ds
    common_dates = sorted(common_dates_set)
    print(f"Common date range: {common_dates[0].strftime('%Y-%m-%d')} -> {common_dates[-1].strftime('%Y-%m-%d')} ({len(common_dates)} days)")

    prices_dict: dict[str, np.ndarray] = {}
    iv_dict: dict[str, np.ndarray] = {}
    ivr_dict: dict[str, np.ndarray] = {}

    for sym, (all_prices, all_dates) in data.items():
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        indices = [date_to_idx[d] for d in common_dates]
        aligned_prices = all_prices[indices]
        prices_dict[sym] = aligned_prices

        # Compute IV series and IV rank
        iv_series = iv_series_from_prices(aligned_prices, RV_WINDOW, 1.20, dynamic=True)
        iv_dict[sym] = iv_series

        ivr = np.zeros(len(common_dates))
        for i in range(len(common_dates)):
            ivr[i] = compute_iv_rank(iv_series, i, IV_LOOKBACK)
        ivr_dict[sym] = ivr

    return common_dates, prices_dict, iv_dict, ivr_dict


def compute_spy_benchmark(
    dates: list[datetime],
    prices: np.ndarray,
    warmup: int,
    risk_free_rate: float = 0.045,
) -> dict:
    bench_prices = prices[warmup:]
    if len(bench_prices) < 2:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}
    ret = (bench_prices[-1] / bench_prices[0] - 1) * 100
    daily_returns = np.diff(bench_prices) / bench_prices[:-1]
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    sharpe = 0.0
    if np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))
    peak = np.maximum.accumulate(bench_prices)
    dd = (peak - bench_prices) / peak
    max_dd = float(np.max(dd) * 100)
    days = len(bench_prices)
    total_ret_dec = ret / 100
    ann_ret = ((1 + total_ret_dec) ** (365 / days) - 1) * 100 if total_ret_dec > -1 else 0.0
    return {
        "return_pct": round(ret, 2),
        "annualized_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
    }


def print_results(
    strat_summary: dict,
    spy_bench: dict,
    pw: PortfolioWheel,
    common_dates: list[datetime],
) -> None:
    s = strat_summary
    days = s["days"]
    total_ret_dec = s["total_return_pct"] / 100
    ann_ret = ((1 + total_ret_dec) ** (365 / days) - 1) * 100 if days > 0 and total_ret_dec > -1 else 0.0

    print("=" * 80)
    print("  PORTFOLIO WHEEL BACKTEST RESULTS")
    print(f"  {s['start_date']} -> {s['end_date']}  ({days} trading days)")
    print("=" * 80)
    print()
    print(f"                         {'Portfolio':>12}   {'SPY B&H':>12}")
    print(f"                         {'--------':>12}   {'--------':>12}")
    print(f"  Total Return:          {s['total_return_pct']:>11.2f} %   {spy_bench['return_pct']:>11.2f} %")
    print(f"  Annualized Return:     {ann_ret:>11.2f} %   {spy_bench['annualized_pct']:>11.2f} %")
    print(f"  Sharpe Ratio:          {s['sharpe_ratio']:>11.3f}     {spy_bench['sharpe']:>11.3f}")
    print(f"  Max Drawdown:          {s['max_drawdown_pct']:>11.2f} %   {spy_bench['max_dd_pct']:>11.2f} %")
    print()
    print(f"  Initial Capital:     ${s['initial_capital']:>12,.2f}")
    print(f"  Final Equity:        ${s['final_equity']:>12,.2f}")
    print()
    print(f"  Total Trades:         {s['total_trades']:>11d}")
    print(f"  Wins / Losses:        {s['wins']:>5d} / {s['losses']:<5d}")
    print(f"  Win Rate:             {s['win_rate']:>11.1f} %")
    print(f"  Wheel Cycles:         {s['wheel_cycles']:>11d}")
    print()
    print(f"  Net Premium:         ${s['net_premium']:>12,.2f}")
    print(f"  Premium Collected:   ${s['total_premium_collected']:>12,.2f}")
    print(f"  Premium Paid:        ${s['total_premium_paid']:>12,.2f}")
    print(f"  Total Commissions:   ${s['total_commissions']:>12,.2f}")
    print()
    print("  -- Portfolio Risk Metrics --")
    print(f"  Margin Breaches:      {s['margin_breaches']:>11d}")
    print(f"  Peak Simul. Assigns:  {s['peak_simultaneous_assignments']:>11d}")
    print(f"  Avg Margin Util:      {s['avg_margin_utilization']:>11.1f} %")
    print(f"  Max Margin Util:      {s['max_margin_utilization']:>11.1f} %")

    # Simultaneous assignment events
    if pw.simultaneous_assignments:
        print()
        multi = [(d, n) for d, n in pw.simultaneous_assignments if n >= 2]
        print(f"  Multi-symbol assignment days: {len(multi)}")
        for d, n in sorted(multi, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {d.strftime('%Y-%m-%d')}: {n} symbols assigned simultaneously")

    # Per-symbol status at end
    print()
    print("  -- Per-Symbol Final State --")
    print(f"  {'Symbol':<8} {'Phase':<5} {'Stock':>6} {'Avg Cost':>10} {'Assigns':>8}")
    for sym in sorted(pw.symbols.keys()):
        st = pw.symbols[sym]
        print(f"  {sym:<8} {st.phase:<5} {st.stock_qty:>6} {st.stock_avg_cost:>10.2f} {len(st.assignment_dates):>8}")

    # Correlation matrix
    corr_syms, corr_mat = pw.correlation_matrix()
    if len(corr_syms) >= 2:
        print()
        print("  -- Return Correlation Matrix (top pairs) --")
        pairs = []
        for i in range(len(corr_syms)):
            for j in range(i + 1, len(corr_syms)):
                pairs.append((corr_syms[i], corr_syms[j], corr_mat[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for a, b, c in pairs[:10]:
            print(f"    {a:>5} / {b:<5}  {c:>+.3f}")
        avg_corr = float(np.mean([c for _, _, c in pairs]))
        print(f"  Average pairwise correlation: {avg_corr:+.3f}")

    # Last 20 events
    trade_events = [
        e for e in pw.events
        if e.pnl != 0 or "sell" in e.event_type or "assign" in e.event_type
    ]
    if trade_events:
        print()
        n_show = min(20, len(trade_events))
        print(f"  Last {n_show} events:")
        print(f"  {'Date':<12} {'Event':<30} {'PnL':>10} {'Details'}")
        print(f"  {'-' * 85}")
        for ev in trade_events[-n_show:]:
            d = ev.details
            detail_str = d.get("symbol", "")
            if "strike" in d:
                detail_str += f" K={d['strike']}"
            if "premium" in d:
                detail_str += f" prem=${d['premium']:.2f}"
            if "underlying" in d:
                detail_str += f" @{d['underlying']}"
            if "contracts" in d:
                detail_str += f" x{d['contracts']}"
            pnl_str = f"${ev.pnl:>+9.2f}" if ev.pnl != 0 else ""
            print(f"  {ev.date.strftime('%Y-%m-%d'):<12} {ev.event_type:<30} {pnl_str:>10} {detail_str}")

    print()
    print("  -- Simulation Assumptions --")
    print("  IV estimation:       Dynamic IV/RV ratio (regime-dependent)")
    print("  Bid-ask slippage:    Fixed $0.05/share")
    print("  Commission+exchange: $0.90/contract")
    print("  Margin model:        max(prem + 0.20*S - OTM, prem + 0.10*K)")
    print("  Assignment:          At expiry only (no early assignment)")
    print("  Vol model:           BSM flat vol (no skew/smile)")
    print("  Capital:             Shared pool across all symbols")
    print()
    print("=" * 80)


def main() -> None:
    data = download_data(SYMBOLS, period="5y")
    if len(data) < 3:
        print("Too few symbols loaded, aborting")
        return

    common_dates, prices_dict, iv_dict, ivr_dict = build_aligned_series(data)

    cfg = PortfolioWheelConfig(
        initial_capital=50_000.0,
        max_symbols=5,
        max_allocation_per_symbol=0.25,
        max_portfolio_delta=200.0,
        max_margin_utilization=0.70,
        put_delta=0.30,
        call_delta=0.30,
        target_dte=45,
        profit_target=0.50,
        roll_dte=14,
        stop_loss=0.0,
        min_iv_rank=25.0,
        min_premium_pct=0.005,
        trend_sma_period=50,
        risk_free_rate=0.045,
        commission_per_contract=0.90,
        commission_per_share=0.005,
        bid_ask_slip_per_share=0.05,
    )
    pw = PortfolioWheel(cfg)

    print(f"\nRunning portfolio wheel on {len(prices_dict)} symbols with ${cfg.initial_capital:,.0f} capital...")
    print(f"Max {cfg.max_symbols} concurrent positions, {cfg.max_allocation_per_symbol*100:.0f}% max per symbol")
    print(f"Margin cap: {cfg.max_margin_utilization*100:.0f}%, Delta cap: {cfg.max_portfolio_delta:.0f}")
    print()

    t0 = time.time()
    for i in range(WARMUP, len(common_dates)):
        date = common_dates[i]
        day_prices = {sym: float(prices_dict[sym][i]) for sym in prices_dict}
        day_ivs = {
            sym: max(float(iv_dict[sym][i]), 0.05) if not np.isnan(iv_dict[sym][i]) else 0.25
            for sym in iv_dict
        }
        day_ivr = {sym: float(ivr_dict[sym][i]) for sym in ivr_dict}
        pw.on_day(date, day_prices, day_ivs, day_ivr)

    elapsed = time.time() - t0
    print(f"Backtest complete in {elapsed:.1f}s ({len(common_dates) - WARMUP} trading days)\n")

    strat_summary = pw.summary()

    spy_bench = {"return_pct": 0.0, "annualized_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}
    if "SPY" in prices_dict:
        spy_bench = compute_spy_benchmark(
            common_dates, prices_dict["SPY"], WARMUP, cfg.risk_free_rate,
        )

    print_results(strat_summary, spy_bench, pw, common_dates)


if __name__ == "__main__":
    main()
