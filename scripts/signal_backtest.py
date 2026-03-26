#!/usr/bin/env python3
"""
Enhanced Wheel Signal Backtest

Tests each of the 8 signals individually (all others off) to measure
marginal lift, then tests the best combination. Uses 5Y yfinance data
across 15 symbols.

Usage:
    .venv/bin/python scripts/signal_backtest.py
    .venv/bin/python scripts/signal_backtest.py --symbols SPY AAPL MSFT
    .venv/bin/python scripts/signal_backtest.py --period 3y
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf

from trading_algo.broker.base import Bar
from trading_algo.quant_core.strategies.options.iv_rank import (
    iv_rank,
    iv_series_from_prices,
)
from trading_algo.quant_core.strategies.options.wheel import WheelConfig, WheelStrategy
from trading_algo.quant_core.strategies.options.enhanced_wheel import (
    EnhancedWheel,
    EnhancedWheelConfig,
    PortfolioCorrelationTracker,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
    "GOOG", "JPM", "XOM", "KO", "WMT",
    "INTC", "PYPL", "IWM", "GS", "PG",
]

BASE_WHEEL = WheelConfig(
    initial_capital=100_000.0,
    put_delta=0.30,
    call_delta=0.30,
    target_dte=45,
    profit_target=0.50,
    roll_dte=14,
    stop_loss=0.0,
    trend_sma_period=50,
    min_iv_rank=25.0,
    min_premium_pct=0.005,
    cash_reserve_pct=0.20,
    risk_free_rate=0.045,
    commission_per_contract=0.90,
    commission_per_share=0.005,
    bid_ask_slip_per_share=0.05,
)


def yf_to_bars(ticker: str, period: str = "5y") -> list[Bar]:
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
        bars.append(Bar(
            timestamp_epoch_s=idx.timestamp(),
            open=o, high=h, low=l, close=c, volume=v,
        ))
    return bars


# ---------------------------------------------------------------------------
# Run a single strategy instance on bars
# ---------------------------------------------------------------------------

WARMUP = 60


def _run_base_wheel(bars: list[Bar], symbol: str) -> dict:
    prices = np.array([b.close for b in bars], dtype=float)
    timestamps = [datetime.fromtimestamp(b.timestamp_epoch_s) for b in bars]
    iv_series = iv_series_from_prices(prices, 30, dynamic=True)

    strategy = WheelStrategy(BASE_WHEEL)
    for i in range(WARMUP, len(bars)):
        date = timestamps[i]
        price = float(prices[i])
        current_iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        current_iv = max(current_iv, 0.05)
        current_rank = iv_rank(iv_series, i, 252)
        strategy.on_bar(date, price, current_iv, current_rank)

    return strategy.summary()


def _run_enhanced_wheel(
    bars: list[Bar], symbol: str, cfg: EnhancedWheelConfig,
) -> dict:
    prices = np.array([b.close for b in bars], dtype=float)
    volumes = np.array([b.volume or 0.0 for b in bars], dtype=float)
    timestamps = [datetime.fromtimestamp(b.timestamp_epoch_s) for b in bars]
    iv_series = iv_series_from_prices(prices, 30, dynamic=True)

    strategy = EnhancedWheel(cfg, symbol=symbol)
    for i in range(WARMUP, len(bars)):
        date = timestamps[i]
        price = float(prices[i])
        vol = float(volumes[i])
        current_iv = float(iv_series[i]) if not np.isnan(iv_series[i]) else 0.25
        current_iv = max(current_iv, 0.05)
        current_rank = iv_rank(iv_series, i, 252)
        strategy.on_bar(date, price, current_iv, current_rank, volume=vol)

    return strategy.summary()


def _buy_and_hold(bars: list[Bar]) -> dict:
    prices = np.array([b.close for b in bars], dtype=float)
    p = prices[WARMUP:]
    if len(p) < 2:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
    ret = (p[-1] / p[0] - 1) * 100
    daily_ret = np.diff(p) / p[:-1]
    daily_rf = 0.045 / 252
    excess = daily_ret - daily_rf
    sharpe = 0.0
    if np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))
    peak = np.maximum.accumulate(p)
    dd = (peak - p) / peak
    max_dd = float(np.max(dd) * 100)
    return {
        "total_return_pct": round(ret, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Signal configs — each signal isolated
# ---------------------------------------------------------------------------

def _all_signals_off() -> EnhancedWheelConfig:
    return EnhancedWheelConfig(
        base=BASE_WHEEL,
        vix_overlay=False,
        earnings_avoidance=False,
        rsi_filter=False,
        momentum_delta_adjust=False,
        term_structure_filter=False,
        volume_filter=False,
        adaptive_dte=False,
        sector_limit=0,
    )


SIGNAL_CONFIGS: dict[str, EnhancedWheelConfig] = {}

# Signal 1: VIX Regime Overlay only
_cfg = _all_signals_off()
SIGNAL_CONFIGS["S1_VIX_Regime"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=True,
    earnings_avoidance=False,
    rsi_filter=False,
    momentum_delta_adjust=False,
    term_structure_filter=False,
    volume_filter=False,
    adaptive_dte=False,
    sector_limit=0,
)

# Signal 2: Earnings Avoidance only
SIGNAL_CONFIGS["S2_Earnings"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=False,
    earnings_avoidance=True,
    rsi_filter=False,
    momentum_delta_adjust=False,
    term_structure_filter=False,
    volume_filter=False,
    adaptive_dte=False,
    sector_limit=0,
)

# Signal 3: RSI Filter only
SIGNAL_CONFIGS["S3_RSI"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=False,
    earnings_avoidance=False,
    rsi_filter=True,
    momentum_delta_adjust=False,
    term_structure_filter=False,
    volume_filter=False,
    adaptive_dte=False,
    sector_limit=0,
)

# Signal 4: Momentum Delta only
SIGNAL_CONFIGS["S4_Momentum"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=False,
    earnings_avoidance=False,
    rsi_filter=False,
    momentum_delta_adjust=True,
    term_structure_filter=False,
    volume_filter=False,
    adaptive_dte=False,
    sector_limit=0,
)

# Signal 5: Term Structure only
SIGNAL_CONFIGS["S5_TermStructure"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=False,
    earnings_avoidance=False,
    rsi_filter=False,
    momentum_delta_adjust=False,
    term_structure_filter=True,
    volume_filter=False,
    adaptive_dte=False,
    sector_limit=0,
)

# Signal 6: Volume Confirmation only
SIGNAL_CONFIGS["S6_Volume"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=False,
    earnings_avoidance=False,
    rsi_filter=False,
    momentum_delta_adjust=False,
    term_structure_filter=False,
    volume_filter=True,
    adaptive_dte=False,
    sector_limit=0,
)

# Signal 7: Adaptive DTE only
SIGNAL_CONFIGS["S7_AdaptiveDTE"] = EnhancedWheelConfig(
    base=_cfg.base,
    vix_overlay=False,
    earnings_avoidance=False,
    rsi_filter=False,
    momentum_delta_adjust=False,
    term_structure_filter=False,
    volume_filter=False,
    adaptive_dte=True,
    sector_limit=0,
)

# Signal 8 is portfolio-level, tested separately

# Combined: all signals on
SIGNAL_CONFIGS["ALL_SIGNALS"] = EnhancedWheelConfig(base=BASE_WHEEL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Wheel Signal Backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    symbols = args.symbols
    period = args.period

    print(f"\n{'=' * 80}")
    print(f"  ENHANCED WHEEL SIGNAL BACKTEST")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period:  {period}")
    print(f"{'=' * 80}\n")

    # --- Phase 1: Download data ---
    print("Downloading data...")
    all_bars: dict[str, list[Bar]] = {}
    for sym in symbols:
        t0 = time.time()
        bars = yf_to_bars(sym, period)
        dt = time.time() - t0
        if len(bars) < WARMUP + 50:
            print(f"  {sym}: SKIP (only {len(bars)} bars)")
            continue
        all_bars[sym] = bars
        print(f"  {sym}: {len(bars)} bars ({dt:.1f}s)")
    print()

    if not all_bars:
        print("No usable data. Exiting.")
        return

    # --- Phase 2: Baseline ---
    print("Running baseline (vanilla Wheel + buy-and-hold)...")
    baseline_results: dict[str, dict] = {}
    bnh_results: dict[str, dict] = {}
    for sym, bars in all_bars.items():
        baseline_results[sym] = _run_base_wheel(bars, sym)
        bnh_results[sym] = _buy_and_hold(bars)
    print("  Done.\n")

    # --- Phase 3: Per-signal tests ---
    signal_results: dict[str, dict[str, dict]] = {}  # signal_name -> {sym -> summary}

    for sig_name, sig_cfg in SIGNAL_CONFIGS.items():
        print(f"Testing {sig_name}...")
        signal_results[sig_name] = {}
        for sym, bars in all_bars.items():
            signal_results[sig_name][sym] = _run_enhanced_wheel(bars, sym, sig_cfg)
        print(f"  Done.")
    print()

    # --- Phase 4: Results ---
    print(f"\n{'=' * 80}")
    print(f"  RESULTS: Per-Signal Marginal Lift vs Base Wheel")
    print(f"{'=' * 80}\n")

    # Header
    header = f"  {'Signal':<22} {'Avg Return':>12} {'Lift':>8} {'Avg Sharpe':>12} {'Lift':>8} {'Avg MaxDD':>10} {'Trades':>8}"
    print(header)
    print(f"  {'-' * 84}")

    # Buy-and-hold line
    bnh_rets = [bnh_results[s]["total_return_pct"] for s in all_bars]
    bnh_sharpes = [bnh_results[s]["sharpe_ratio"] for s in all_bars]
    bnh_dds = [bnh_results[s]["max_drawdown_pct"] for s in all_bars]
    print(f"  {'Buy & Hold':<22} {np.mean(bnh_rets):>11.2f}% {'':>8} {np.mean(bnh_sharpes):>12.3f} {'':>8} {np.mean(bnh_dds):>9.2f}% {'N/A':>8}")

    # Base wheel line
    base_rets = [baseline_results[s]["total_return_pct"] for s in all_bars]
    base_sharpes = [baseline_results[s]["sharpe_ratio"] for s in all_bars]
    base_dds = [baseline_results[s]["max_drawdown_pct"] for s in all_bars]
    base_trades = [baseline_results[s]["total_trades"] for s in all_bars]
    avg_base_ret = np.mean(base_rets)
    avg_base_sharpe = np.mean(base_sharpes)
    avg_base_dd = np.mean(base_dds)
    print(f"  {'Base Wheel':<22} {avg_base_ret:>11.2f}% {'(base)':>8} {avg_base_sharpe:>12.3f} {'(base)':>8} {avg_base_dd:>9.2f}% {int(np.mean(base_trades)):>8}")
    print(f"  {'-' * 84}")

    # Per-signal lines
    signal_lifts: dict[str, tuple[float, float]] = {}  # name -> (ret_lift, sharpe_lift)
    for sig_name in SIGNAL_CONFIGS:
        rets = [signal_results[sig_name][s]["total_return_pct"] for s in all_bars]
        sharpes = [signal_results[sig_name][s]["sharpe_ratio"] for s in all_bars]
        dds = [signal_results[sig_name][s]["max_drawdown_pct"] for s in all_bars]
        trades = [signal_results[sig_name][s]["total_trades"] for s in all_bars]

        avg_ret = np.mean(rets)
        avg_sharpe = np.mean(sharpes)
        avg_dd = np.mean(dds)
        avg_trades = int(np.mean(trades))
        ret_lift = avg_ret - avg_base_ret
        sharpe_lift = avg_sharpe - avg_base_sharpe

        signal_lifts[sig_name] = (ret_lift, sharpe_lift)

        ret_lift_str = f"{ret_lift:>+7.2f}%"
        sharpe_lift_str = f"{sharpe_lift:>+8.3f}"
        print(f"  {sig_name:<22} {avg_ret:>11.2f}% {ret_lift_str:>8} {avg_sharpe:>12.3f} {sharpe_lift_str:>8} {avg_dd:>9.2f}% {avg_trades:>8}")

    # --- Phase 5: Per-symbol detail ---
    print(f"\n{'=' * 80}")
    print(f"  PER-SYMBOL DETAIL: Base Wheel vs ALL_SIGNALS vs Buy-and-Hold")
    print(f"{'=' * 80}\n")

    sym_header = f"  {'Symbol':<8} {'B&H Ret':>10} {'Base Ret':>10} {'All Ret':>10} {'B&H Sharpe':>11} {'Base Sharpe':>12} {'All Sharpe':>11} {'Base Trades':>12} {'All Trades':>11}"
    print(sym_header)
    print(f"  {'-' * 108}")

    for sym in all_bars:
        bnh = bnh_results[sym]
        base = baseline_results[sym]
        enhanced = signal_results["ALL_SIGNALS"][sym]
        print(
            f"  {sym:<8} "
            f"{bnh['total_return_pct']:>9.2f}% "
            f"{base['total_return_pct']:>9.2f}% "
            f"{enhanced['total_return_pct']:>9.2f}% "
            f"{bnh['sharpe_ratio']:>11.3f} "
            f"{base['sharpe_ratio']:>12.3f} "
            f"{enhanced['sharpe_ratio']:>11.3f} "
            f"{base['total_trades']:>12d} "
            f"{enhanced['total_trades']:>11d}"
        )

    # --- Phase 6: Best combination ---
    print(f"\n{'=' * 80}")
    print(f"  SIGNAL RANKING (by Sharpe lift)")
    print(f"{'=' * 80}\n")

    ranked = sorted(
        [(name, lifts[1], lifts[0]) for name, lifts in signal_lifts.items() if name != "ALL_SIGNALS"],
        key=lambda x: x[1],
        reverse=True,
    )
    for i, (name, sharpe_lift, ret_lift) in enumerate(ranked, 1):
        marker = " <-- positive" if sharpe_lift > 0 else ""
        print(f"  {i}. {name:<22} Sharpe lift: {sharpe_lift:>+.3f}   Return lift: {ret_lift:>+.2f}%{marker}")

    positive_signals = [name for name, sharpe_lift, _ in ranked if sharpe_lift > 0]

    if positive_signals:
        print(f"\n  Positive-lift signals: {', '.join(positive_signals)}")
        print(f"  Recommendation: Enable these {len(positive_signals)} signals for production use.")

        # Build config with only positive signals
        best_cfg = EnhancedWheelConfig(
            base=BASE_WHEEL,
            vix_overlay="S1_VIX_Regime" in positive_signals,
            earnings_avoidance="S2_Earnings" in positive_signals,
            rsi_filter="S3_RSI" in positive_signals,
            momentum_delta_adjust="S4_Momentum" in positive_signals,
            term_structure_filter="S5_TermStructure" in positive_signals,
            volume_filter="S6_Volume" in positive_signals,
            adaptive_dte="S7_AdaptiveDTE" in positive_signals,
            sector_limit=0,
        )

        print(f"\n  Running best-combination backtest...")
        best_results: dict[str, dict] = {}
        for sym, bars in all_bars.items():
            best_results[sym] = _run_enhanced_wheel(bars, sym, best_cfg)

        best_rets = [best_results[s]["total_return_pct"] for s in all_bars]
        best_sharpes = [best_results[s]["sharpe_ratio"] for s in all_bars]
        best_dds = [best_results[s]["max_drawdown_pct"] for s in all_bars]
        best_trades = [best_results[s]["total_trades"] for s in all_bars]

        print(f"\n  {'Strategy':<22} {'Avg Return':>12} {'Avg Sharpe':>12} {'Avg MaxDD':>10} {'Avg Trades':>11}")
        print(f"  {'-' * 70}")
        print(f"  {'Buy & Hold':<22} {np.mean(bnh_rets):>11.2f}% {np.mean(bnh_sharpes):>12.3f} {np.mean(bnh_dds):>9.2f}% {'N/A':>11}")
        print(f"  {'Base Wheel':<22} {avg_base_ret:>11.2f}% {avg_base_sharpe:>12.3f} {avg_base_dd:>9.2f}% {int(np.mean(base_trades)):>11}")
        print(f"  {'Best Combo':<22} {np.mean(best_rets):>11.2f}% {np.mean(best_sharpes):>12.3f} {np.mean(best_dds):>9.2f}% {int(np.mean(best_trades)):>11}")
        print(f"  {'All Signals':<22} {np.mean([signal_results['ALL_SIGNALS'][s]['total_return_pct'] for s in all_bars]):>11.2f}% {np.mean([signal_results['ALL_SIGNALS'][s]['sharpe_ratio'] for s in all_bars]):>12.3f} {np.mean([signal_results['ALL_SIGNALS'][s]['max_drawdown_pct'] for s in all_bars]):>9.2f}% {int(np.mean([signal_results['ALL_SIGNALS'][s]['total_trades'] for s in all_bars])):>11}")

        # Final lift
        best_combo_sharpe_lift = np.mean(best_sharpes) - avg_base_sharpe
        best_combo_ret_lift = np.mean(best_rets) - avg_base_ret
        print(f"\n  Best combo lift over base: Return {best_combo_ret_lift:>+.2f}%, Sharpe {best_combo_sharpe_lift:>+.3f}")
    else:
        print("\n  No signals showed positive Sharpe lift. The base Wheel may be optimal for this universe/period.")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
