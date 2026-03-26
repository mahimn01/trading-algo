"""
Wheel Strategy: Sensitivity Analysis & Stress Tests
"""

import sys
import os
import warnings
import copy

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yfinance as yf
from datetime import datetime
from trading_algo.quant_core.strategies.options.wheel import WheelConfig, WheelStrategy

# ─── Configuration ───────────────────────────────────────────────────────────

SYMBOLS = ["SOFI", "F", "PLTR", "T", "BAC", "NIO", "RIVN"]

BEST_CONFIG = dict(
    initial_capital=10_000.0,
    put_delta=0.30,
    call_delta=0.30,
    target_dte=60,
    profit_target=0.0,      # disabled
    roll_dte=0,              # let expire
    stop_loss=0.0,           # disabled
    min_iv_rank=25.0,
    trend_sma_period=50,
    risk_free_rate=0.045,
    commission_per_contract=0.90,
    bid_ask_slip_per_share=0.05,
)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(symbol: str) -> dict | None:
    """Download 2y daily data, compute IV proxy and IV rank."""
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        # Flatten multi-level columns
        if hasattr(df.columns, "get_level_values") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])

        prices = df["Close"].values.astype(float)
        dates = [datetime.combine(d.date(), datetime.min.time()) if hasattr(d, "date") else d for d in df.index]

        # IV proxy: 20-day realized vol, annualized
        log_ret = np.diff(np.log(prices))
        iv_series = np.full(len(prices), 0.30)
        for i in range(20, len(log_ret)):
            iv_series[i + 1] = float(np.std(log_ret[i - 19 : i + 1]) * np.sqrt(252))
        iv_series[:21] = iv_series[21] if len(iv_series) > 21 else 0.30

        # IV rank: percentile over trailing 252 days
        iv_rank_series = np.full(len(prices), 50.0)
        for i in range(252, len(prices)):
            window = iv_series[i - 252 : i]
            iv_min, iv_max = np.min(window), np.max(window)
            if iv_max > iv_min:
                iv_rank_series[i] = (iv_series[i] - iv_min) / (iv_max - iv_min) * 100
            else:
                iv_rank_series[i] = 50.0

        return {
            "prices": prices,
            "dates": dates,
            "iv": iv_series,
            "iv_rank": iv_rank_series,
        }
    except Exception as e:
        print(f"  [WARN] Failed to load {symbol}: {e}")
        return None


def run_backtest(
    data: dict,
    config_overrides: dict | None = None,
    crash_at_mid: bool = False,
) -> dict | None:
    """Run the Wheel on preloaded data, return summary dict."""
    cfg_dict = copy.deepcopy(BEST_CONFIG)
    if config_overrides:
        cfg_dict.update(config_overrides)
    cfg = WheelConfig(**cfg_dict)

    prices = data["prices"].copy()
    dates = list(data["dates"])
    iv = data["iv"].copy()
    iv_rank = data["iv_rank"].copy()

    if crash_at_mid:
        mid = len(prices) // 2
        prices[mid:] *= 0.70
        # Spike IV during crash
        iv[mid : min(mid + 30, len(iv))] *= 1.5

    strat = WheelStrategy(cfg)
    for i in range(len(prices)):
        strat.on_bar(dates[i], float(prices[i]), float(iv[i]), float(iv_rank[i]))

    s = strat.summary()
    return s


# ─── Formatting helpers ─────────────────────────────────────────────────────

def fmt(v, width=10, decimals=2):
    if isinstance(v, float):
        return f"{v:>{width}.{decimals}f}"
    return f"{str(v):>{width}}"


def print_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None):
    if col_widths is None:
        col_widths = [max(len(str(h)), 10) for h in headers]
    header_line = " | ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep)
    for row in rows:
        cells = []
        for val, w in zip(row, col_widths):
            if isinstance(val, float):
                cells.append(f"{val:>{w}.2f}")
            else:
                cells.append(f"{str(val):>{w}}")
        print(" | ".join(cells))


# ─── PART A: Sensitivity Analysis ───────────────────────────────────────────

def run_sensitivity():
    print("=" * 80)
    print("PART A: SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    print("Best config baseline: delta=0.30, DTE=60, no profit target, let expire,")
    print("no stop loss, SMA50, min_iv_rank=25, slip=$0.05, comm=$0.90")
    print()

    # Load all data upfront
    all_data = {}
    for sym in SYMBOLS:
        print(f"  Loading {sym}...", end=" ", flush=True)
        d = load_data(sym)
        if d is not None:
            all_data[sym] = d
            print(f"OK ({len(d['prices'])} bars)")
        else:
            print("FAILED")

    active_symbols = list(all_data.keys())
    n_sym = len(active_symbols)
    print(f"\n  Active symbols: {active_symbols} ({n_sym})")
    print()

    sweeps = {
        "put_delta": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "target_dte": [21, 30, 45, 60, 90],
        "trend_sma_period": [0, 20, 50, 100, 200],
        "min_iv_rank": [0, 15, 25, 35, 50],
        "bid_ask_slip_per_share": [0.03, 0.05, 0.08, 0.10, 0.15],
        "commission_per_contract": [0.65, 0.90, 1.50, 2.00],
    }

    for param_name, values in sweeps.items():
        print(f"--- Sweep: {param_name} ---")
        headers = ["Value", "Avg Sharpe", "Avg Ret%", "Avg MaxDD%", "Profitable"]
        widths = [12, 12, 12, 12, 12]
        rows = []

        for val in values:
            sharpes, rets, dds = [], [], []
            profitable = 0

            for sym in active_symbols:
                s = run_backtest(all_data[sym], {param_name: val})
                if s is None:
                    continue
                sharpes.append(s["sharpe_ratio"])
                rets.append(s["total_return_pct"])
                dds.append(s["max_drawdown_pct"])
                if s["total_return_pct"] > 0:
                    profitable += 1

            if not sharpes:
                rows.append([val, 0.0, 0.0, 0.0, f"0/{n_sym}"])
                continue

            rows.append([
                val,
                np.mean(sharpes),
                np.mean(rets),
                np.mean(dds),
                f"{profitable}/{n_sym}",
            ])

        print_table(headers, rows, widths)
        print()


# ─── PART B: Crash Stress Test ──────────────────────────────────────────────

def run_crash_test():
    print("=" * 80)
    print("PART B: STRESS TEST — 30% SYNTHETIC CRASH AT MIDPOINT")
    print("=" * 80)
    print()
    print("Inject 30% gap-down at midpoint of data. IV spikes 1.5x for 30 days.")
    print("Running best config on stressed data.")
    print()

    all_data = {}
    for sym in SYMBOLS:
        d = load_data(sym)
        if d is not None:
            all_data[sym] = d

    headers = ["Symbol", "Return%", "Sharpe", "MaxDD%", "Trades", "WinRate%", "Cycles"]
    widths = [8, 10, 10, 10, 8, 10, 8]
    rows = []

    for sym in all_data:
        s = run_backtest(all_data[sym], crash_at_mid=True)
        if s is None:
            continue
        rows.append([
            sym,
            s["total_return_pct"],
            s["sharpe_ratio"],
            s["max_drawdown_pct"],
            s["total_trades"],
            s["win_rate"],
            s["wheel_cycles"],
        ])

    print_table(headers, rows, widths)

    # Summary stats
    if rows:
        avg_ret = np.mean([r[1] for r in rows])
        avg_sharpe = np.mean([r[2] for r in rows])
        avg_dd = np.mean([r[3] for r in rows])
        survived = sum(1 for r in rows if r[1] > -50)
        print(f"\n  Avg Return: {avg_ret:.2f}%  |  Avg Sharpe: {avg_sharpe:.3f}  |  Avg MaxDD: {avg_dd:.2f}%")
        print(f"  Survived (return > -50%): {survived}/{len(rows)}")
    print()


# ─── PART C: High Cost Regime ───────────────────────────────────────────────

def run_high_cost_test():
    print("=" * 80)
    print("PART C: STRESS TEST — HIGH COST REGIME")
    print("=" * 80)
    print()
    print("Punishing costs: slip=$0.12/share, commission=$1.50/contract")
    print("vs. normal costs: slip=$0.05/share, commission=$0.90/contract")
    print()

    all_data = {}
    for sym in SYMBOLS:
        d = load_data(sym)
        if d is not None:
            all_data[sym] = d

    headers = ["Symbol", "Norm Ret%", "Norm Sharpe", "High Ret%", "High Sharpe", "Ret Drag%"]
    widths = [8, 12, 12, 12, 12, 12]
    rows = []

    for sym in all_data:
        s_normal = run_backtest(all_data[sym])
        s_high = run_backtest(all_data[sym], {
            "bid_ask_slip_per_share": 0.12,
            "commission_per_contract": 1.50,
        })
        if s_normal is None or s_high is None:
            continue
        drag = s_normal["total_return_pct"] - s_high["total_return_pct"]
        rows.append([
            sym,
            s_normal["total_return_pct"],
            s_normal["sharpe_ratio"],
            s_high["total_return_pct"],
            s_high["sharpe_ratio"],
            drag,
        ])

    print_table(headers, rows, widths)

    if rows:
        avg_drag = np.mean([r[5] for r in rows])
        high_profitable = sum(1 for r in rows if r[3] > 0)
        print(f"\n  Avg Return Drag from costs: {avg_drag:.2f}%")
        print(f"  Still profitable under high costs: {high_profitable}/{len(rows)}")
    print()


# ─── PART D: Trend Filter Value ─────────────────────────────────────────────

def run_trend_filter_test():
    print("=" * 80)
    print("PART D: STRESS TEST — TREND FILTER VALUE")
    print("=" * 80)
    print()
    print("Comparing SMA50 trend filter (ON) vs. no filter (OFF)")
    print()

    all_data = {}
    for sym in SYMBOLS:
        d = load_data(sym)
        if d is not None:
            all_data[sym] = d

    headers = ["Symbol", "ON Ret%", "ON Sharpe", "ON MaxDD%", "OFF Ret%", "OFF Sharpe", "OFF MaxDD%", "Sharpe Lift"]
    widths = [8, 10, 10, 10, 10, 10, 10, 12]
    rows = []

    for sym in all_data:
        s_on = run_backtest(all_data[sym])  # SMA50 = best config
        s_off = run_backtest(all_data[sym], {"trend_sma_period": 0})
        if s_on is None or s_off is None:
            continue
        sharpe_lift = s_on["sharpe_ratio"] - s_off["sharpe_ratio"]
        rows.append([
            sym,
            s_on["total_return_pct"],
            s_on["sharpe_ratio"],
            s_on["max_drawdown_pct"],
            s_off["total_return_pct"],
            s_off["sharpe_ratio"],
            s_off["max_drawdown_pct"],
            sharpe_lift,
        ])

    print_table(headers, rows, widths)

    if rows:
        avg_sharpe_on = np.mean([r[2] for r in rows])
        avg_sharpe_off = np.mean([r[5] for r in rows])
        avg_dd_on = np.mean([r[3] for r in rows])
        avg_dd_off = np.mean([r[6] for r in rows])
        avg_ret_on = np.mean([r[1] for r in rows])
        avg_ret_off = np.mean([r[4] for r in rows])

        print(f"\n  --- Averages ---")
        print(f"  Filter ON  :  Return={avg_ret_on:+.2f}%  Sharpe={avg_sharpe_on:.3f}  MaxDD={avg_dd_on:.2f}%")
        print(f"  Filter OFF :  Return={avg_ret_off:+.2f}%  Sharpe={avg_sharpe_off:.3f}  MaxDD={avg_dd_off:.2f}%")
        print(f"  Sharpe improvement from filter: {avg_sharpe_on - avg_sharpe_off:+.3f}")
        print(f"  MaxDD improvement from filter:  {avg_dd_off - avg_dd_on:+.2f}% (positive = filter reduces DD)")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("*" * 80)
    print("  WHEEL STRATEGY — SENSITIVITY ANALYSIS & STRESS TESTS")
    print("  Symbols:", ", ".join(SYMBOLS))
    print("  Date:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("*" * 80)
    print()

    run_sensitivity()
    run_crash_test()
    run_high_cost_test()
    run_trend_filter_test()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
