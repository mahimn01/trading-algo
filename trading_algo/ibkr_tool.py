"""
Comprehensive IBKR data + operations CLI.

Covers: accounts, positions, PnL streaming, quotes, chains, greeks, depth,
real-time bars, tick-by-tick, historical bars/ticks, fundamentals, news,
scanner, contract search, executions, open/completed orders, what-if preview,
combo orders, global cancel, WSH events, FX, time, market rules.

Usage: python -m trading_algo.ibkr_tool <command> [args]
Defaults to IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID from .env.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

try:
    from ib_async import (
        IB,
        Bag,
        ComboLeg,
        Contract,
        ExecutionFilter,
        Forex,
        Future,
        Index,
        Option,
        Order,
        ScannerSubscription,
        Stock,
        TagValue,
        Ticker,
        util,
    )
except Exception as exc:  # pragma: no cover
    print(f"ERROR: ib_async not installed: {exc}", file=sys.stderr)
    print("Install: .venv/bin/pip install ib_async", file=sys.stderr)
    sys.exit(2)


# ============================================================
# .env loader + connection helpers
# ============================================================

def _load_dotenv() -> None:
    path = ".env"
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if os.getenv(k) in (None, ""):
                os.environ[k] = v


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


DEFAULT_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
DEFAULT_PORT = _env_int("IBKR_PORT", 4001)
# Tool uses a distinct client id so it doesn't collide with the trading engine (which uses IBKR_CLIENT_ID from .env).
DEFAULT_CLIENT_ID = _env_int("IBKR_TOOL_CLIENT_ID", 177)


def _connect(args: argparse.Namespace) -> IB:
    ib = IB()
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    client_id = args.client_id if args.client_id is not None else DEFAULT_CLIENT_ID
    ib.connect(host, port, clientId=client_id, timeout=args.timeout)
    if args.market_data_type:
        ib.reqMarketDataType(args.market_data_type)
    return ib


# ============================================================
# Output helpers
# ============================================================

def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _emit(data: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(_to_jsonable(data), indent=2, default=str))
        return
    if fmt == "csv":
        rows = data if isinstance(data, list) else [data]
        rows = [_to_jsonable(r) for r in rows if r is not None]
        rows = [r if isinstance(r, dict) else {"value": r} for r in rows]
        if not rows:
            return
        keys = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(sys.stdout, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
        return
    # table
    rows = data if isinstance(data, list) else [data]
    rows = [_to_jsonable(r) for r in rows if r is not None]
    if not rows:
        print("(no rows)")
        return
    if isinstance(rows[0], dict):
        keys = sorted({k for r in rows for k in r.keys()})
        widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
        header = "  ".join(k.ljust(widths[k]) for k in keys)
        print(header)
        print("  ".join("-" * widths[k] for k in keys))
        for r in rows:
            print("  ".join(str(r.get(k, "") if r.get(k) is not None else "").ljust(widths[k]) for k in keys))
    else:
        for r in rows:
            print(r)


# ============================================================
# Contract builders
# ============================================================

def _build_contract(args: argparse.Namespace) -> Contract:
    kind = args.kind.upper()
    sym = args.symbol
    exch = args.exchange or ("SMART" if kind in ("STK", "OPT") else "")
    ccy = args.currency or "USD"

    if kind == "STK":
        return Stock(sym, exch, ccy, primaryExchange=args.primary or "")
    if kind == "OPT":
        if not (args.expiry and args.right and args.strike):
            raise SystemExit("OPT requires --expiry YYYYMMDD --right C|P --strike N")
        return Option(sym, args.expiry, float(args.strike), args.right, exch or "SMART", multiplier=args.multiplier or "100", currency=ccy)
    if kind == "FUT":
        if not args.expiry:
            raise SystemExit("FUT requires --expiry YYYYMM or YYYYMMDD")
        return Future(sym, args.expiry, exch or "CME", currency=ccy, multiplier=args.multiplier or "")
    if kind == "FX":
        # symbol like "USDJPY" or pair
        return Forex(sym)
    if kind == "IND":
        return Index(sym, exch or "CBOE", ccy)
    raise SystemExit(f"Unknown kind: {kind}")


def _add_contract_args(p: argparse.ArgumentParser, default_kind: str = "STK") -> None:
    p.add_argument("--kind", choices=["STK", "OPT", "FUT", "FX", "IND"], default=default_kind)
    p.add_argument("--symbol", required=True)
    p.add_argument("--exchange", default=None)
    p.add_argument("--primary", default=None, help="STK primaryExchange (e.g. NASDAQ, NYSE)")
    p.add_argument("--currency", default=None)
    p.add_argument("--expiry", default=None, help="OPT: YYYYMMDD; FUT: YYYYMM or YYYYMMDD")
    p.add_argument("--right", choices=["C", "P"], default=None)
    p.add_argument("--strike", default=None)
    p.add_argument("--multiplier", default=None)


# ============================================================
# Commands — connection / meta
# ============================================================

def cmd_connect(args: argparse.Namespace) -> int:
    ib = _connect(args)
    info = {
        "connected": ib.isConnected(),
        "client_id": ib.client.clientId,
        "server_version": ib.client.serverVersion(),
        "server_time": str(ib.reqCurrentTime()),
        "managed_accounts": ib.managedAccounts(),
    }
    _emit(info, args.format)
    ib.disconnect()
    return 0


def cmd_time(args: argparse.Namespace) -> int:
    ib = _connect(args)
    t = ib.reqCurrentTime()
    _emit({"server_time": str(t), "local_time": datetime.now(timezone.utc).isoformat()}, args.format)
    ib.disconnect()
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    ib = _connect(args)
    _emit(list(ib.managedAccounts()), args.format)
    ib.disconnect()
    return 0


def cmd_user_info(args: argparse.Namespace) -> int:
    ib = _connect(args)
    try:
        info = ib.reqUserInfo()
    except Exception as exc:
        info = {"error": str(exc)}
    _emit(info, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Account — summary, values, positions, PnL
# ============================================================

def cmd_summary(args: argparse.Namespace) -> int:
    ib = _connect(args)
    tags = args.tags or "NetLiquidation,TotalCashValue,SettledCash,AccruedCash,BuyingPower,EquityWithLoanValue,GrossPositionValue,InitMarginReq,MaintMarginReq,AvailableFunds,ExcessLiquidity,Cushion,FullInitMarginReq,FullMaintMarginReq,FullAvailableFunds,FullExcessLiquidity,LookAheadNextChange,LookAheadInitMarginReq,LookAheadMaintMarginReq,LookAheadAvailableFunds,LookAheadExcessLiquidity,HighestSeverity,DayTradesRemaining,Leverage,$LEDGER:ALL"
    rows = ib.accountSummary(args.account or "")
    out = []
    wanted = set(t.strip() for t in tags.split(","))
    for s in rows:
        if args.account and s.account != args.account:
            continue
        if "$LEDGER:ALL" not in wanted and s.tag not in wanted and s.tag != "$LEDGER":
            continue
        out.append({"account": s.account, "tag": s.tag, "value": s.value, "currency": s.currency})
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_values(args: argparse.Namespace) -> int:
    ib = _connect(args)
    ib.reqAccountUpdates(True, args.account or "")
    ib.sleep(2.0)
    values = [
        {"account": v.account, "tag": v.tag, "value": v.value, "currency": v.currency, "modelCode": v.modelCode}
        for v in ib.accountValues(args.account or "")
    ]
    if args.tag:
        values = [v for v in values if args.tag.lower() in v["tag"].lower()]
    _emit(values, args.format)
    ib.reqAccountUpdates(False, args.account or "")
    ib.disconnect()
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    ib = _connect(args)
    positions = ib.positions(args.account or "")
    out = []
    for p in positions:
        c = p.contract
        out.append({
            "account": p.account,
            "conId": c.conId,
            "secType": c.secType,
            "symbol": c.symbol,
            "localSymbol": c.localSymbol,
            "currency": c.currency,
            "exchange": c.exchange,
            "expiry": getattr(c, "lastTradeDateOrContractMonth", "") or "",
            "right": getattr(c, "right", "") or "",
            "strike": getattr(c, "strike", 0.0) or 0.0,
            "multiplier": getattr(c, "multiplier", "") or "",
            "position": p.position,
            "avgCost": p.avgCost,
            "marketValue": p.position * p.avgCost,
        })
    if args.symbol:
        out = [r for r in out if r["symbol"] == args.symbol]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    ib = _connect(args)
    ib.reqAccountUpdates(True, args.account or "")
    ib.sleep(2.0)
    items = ib.portfolio(args.account or "")
    out = []
    for p in items:
        c = p.contract
        out.append({
            "account": p.account,
            "secType": c.secType,
            "symbol": c.symbol,
            "localSymbol": c.localSymbol,
            "position": p.position,
            "marketPrice": p.marketPrice,
            "marketValue": p.marketValue,
            "avgCost": p.averageCost,
            "unrealizedPNL": p.unrealizedPNL,
            "realizedPNL": p.realizedPNL,
        })
    _emit(out, args.format)
    ib.reqAccountUpdates(False, args.account or "")
    ib.disconnect()
    return 0


def cmd_pnl(args: argparse.Namespace) -> int:
    ib = _connect(args)
    accounts = [args.account] if args.account else ib.managedAccounts()
    results = []
    for acct in accounts:
        pnl = ib.reqPnL(acct, "")
        ib.sleep(2.0)
        results.append({
            "account": acct,
            "dailyPnL": pnl.dailyPnL,
            "unrealizedPnL": pnl.unrealizedPnL,
            "realizedPnL": pnl.realizedPnL,
        })
        ib.cancelPnL(acct, "")
    _emit(results, args.format)
    ib.disconnect()
    return 0


def cmd_pnl_single(args: argparse.Namespace) -> int:
    ib = _connect(args)
    pnl = ib.reqPnLSingle(args.account, "", args.con_id)
    ib.sleep(2.0)
    result = {
        "account": args.account,
        "conId": args.con_id,
        "position": pnl.position,
        "dailyPnL": pnl.dailyPnL,
        "unrealizedPnL": pnl.unrealizedPnL,
        "realizedPnL": pnl.realizedPnL,
        "value": pnl.value,
    }
    ib.cancelPnLSingle(args.account, "", args.con_id)
    _emit(result, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Quotes / market data
# ============================================================

def _ticker_to_dict(t: Ticker) -> dict:
    g = t.modelGreeks or t.lastGreeks or t.askGreeks or t.bidGreeks
    return {
        "symbol": t.contract.symbol,
        "localSymbol": t.contract.localSymbol or "",
        "secType": t.contract.secType,
        "bid": t.bid, "bidSize": t.bidSize,
        "ask": t.ask, "askSize": t.askSize,
        "last": t.last, "lastSize": t.lastSize,
        "close": t.close, "open": t.open,
        "high": t.high, "low": t.low,
        "volume": t.volume, "vwap": t.vwap,
        "halted": t.halted,
        "delta": getattr(g, "delta", None) if g else None,
        "gamma": getattr(g, "gamma", None) if g else None,
        "vega": getattr(g, "vega", None) if g else None,
        "theta": getattr(g, "theta", None) if g else None,
        "impliedVol": getattr(g, "impliedVol", None) if g else None,
        "undPrice": getattr(g, "undPrice", None) if g else None,
    }


def cmd_quote(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        ib.sleep(0.2)
        if (t.bid and t.ask) or t.last:
            if args.kind == "OPT" and not t.modelGreeks and time.time() < deadline - 0.5:
                continue
            break
    _emit(_ticker_to_dict(t), args.format)
    ib.cancelMktData(c)
    ib.disconnect()
    return 0


def cmd_quotes(args: argparse.Namespace) -> int:
    """Batch snapshot for multiple symbols using reqTickers (one-shot)."""
    ib = _connect(args)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    contracts = [Stock(s, "SMART", args.currency or "USD") for s in symbols]
    ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*contracts, regulatorySnapshot=False)
    _emit([_ticker_to_dict(t) for t in tickers], args.format)
    ib.disconnect()
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    end = time.time() + args.duration
    print(f"streaming {c.symbol} for {args.duration}s (Ctrl+C to stop)", file=sys.stderr)
    try:
        while time.time() < end:
            ib.sleep(max(0.1, args.interval))
            row = _ticker_to_dict(t)
            row["ts"] = datetime.now(timezone.utc).isoformat()
            print(json.dumps(_to_jsonable(row), default=str))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    ib.cancelMktData(c)
    ib.disconnect()
    return 0


def cmd_depth(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    t = ib.reqMktDepth(c, numRows=args.rows, isSmartDepth=args.smart)
    ib.sleep(args.wait)
    bids = [{"side": "BID", "pos": b.position, "price": b.price, "size": b.size, "mm": b.marketMaker, "exch": b.exchange} for b in t.domBids]
    asks = [{"side": "ASK", "pos": b.position, "price": b.price, "size": b.size, "mm": b.marketMaker, "exch": b.exchange} for b in t.domAsks]
    _emit(bids + asks, args.format)
    ib.cancelMktDepth(c, isSmartDepth=args.smart)
    ib.disconnect()
    return 0


def cmd_depth_exchanges(args: argparse.Namespace) -> int:
    ib = _connect(args)
    out = [
        {"exchange": d.exchange, "secType": d.secType, "listingExch": d.listingExch, "serviceDataType": d.serviceDataType, "aggGroup": d.aggGroup}
        for d in ib.reqMktDepthExchanges()
    ]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_realtime_bars(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    bars = ib.reqRealTimeBars(c, barSize=5, whatToShow=args.what_to_show, useRTH=args.rth)
    end = time.time() + args.duration
    print(f"realtime bars for {c.symbol} ({args.duration}s)...", file=sys.stderr)
    seen = 0
    while time.time() < end:
        ib.sleep(1.0)
        while seen < len(bars):
            b = bars[seen]
            seen += 1
            row = {"time": str(b.time), "open": b.open_, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume, "wap": b.wap, "count": b.count}
            print(json.dumps(_to_jsonable(row), default=str))
            sys.stdout.flush()
    ib.cancelRealTimeBars(bars)
    ib.disconnect()
    return 0


def cmd_ticks(args: argparse.Namespace) -> int:
    """Tick-by-tick: Last, AllLast, BidAsk, MidPoint. Reads ticker.tickByTicks."""
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    tt = ib.reqTickByTickData(c, tickType=args.tick_type, numberOfTicks=0, ignoreSize=False)
    end = time.time() + args.duration
    print(f"tick-by-tick {args.tick_type} on {c.symbol} ({args.duration}s)...", file=sys.stderr)
    seen = 0
    while time.time() < end:
        ib.sleep(0.25)
        ticks = tt.tickByTicks
        while seen < len(ticks):
            x = ticks[seen]
            seen += 1
            fields = getattr(x, "_fields", None)
            row: dict
            if fields:
                row = {k: getattr(x, k) for k in fields if k != "tickAttribBidAsk" and k != "tickAttribLast"}
            else:
                row = {"tick": repr(x)}
            print(json.dumps(_to_jsonable(row), default=str))
            sys.stdout.flush()
    ib.cancelTickByTickData(c, args.tick_type)
    ib.disconnect()
    return 0


# ============================================================
# Historical
# ============================================================

def cmd_history(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    bars = ib.reqHistoricalData(
        c,
        endDateTime=args.end or "",
        durationStr=args.duration,
        barSizeSetting=args.bar_size,
        whatToShow=args.what_to_show,
        useRTH=args.rth,
        formatDate=1,
        keepUpToDate=False,
    )
    out = [{"time": str(b.date), "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume, "wap": b.average, "count": b.barCount} for b in bars]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_history_ticks(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    ticks = ib.reqHistoricalTicks(
        c,
        startDateTime=args.start or "",
        endDateTime=args.end or "",
        numberOfTicks=args.count,
        whatToShow=args.what_to_show,
        useRth=args.rth,
    )
    out = []
    for x in ticks:
        row = {"time": str(x.time)}
        for k in ("price", "size", "priceBid", "sizeBid", "priceAsk", "sizeAsk", "exchange"):
            if hasattr(x, k):
                row[k] = getattr(x, k)
        out.append(row)
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_head_timestamp(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    ts = ib.reqHeadTimeStamp(c, whatToShow=args.what_to_show, useRTH=args.rth, formatDate=1)
    _emit({"symbol": c.symbol, "head_timestamp": str(ts)}, args.format)
    ib.disconnect()
    return 0


def cmd_histogram(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    hist = ib.reqHistogramData(c, useRTH=args.rth, period=args.period)
    _emit([{"price": h.price, "count": h.count} for h in hist], args.format)
    ib.disconnect()
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    sched = ib.reqHistoricalSchedule(c, endDateTime=args.end or "", durationStr=args.duration, useRTH=args.rth)
    out = [{"start": str(s.startDateTime), "end": str(s.endDateTime), "refDate": str(s.refDate), "timezone": s.timeZone} for s in sched.sessions]
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Options
# ============================================================

def cmd_chain(args: argparse.Namespace) -> int:
    ib = _connect(args)
    underlying = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    [u] = ib.qualifyContracts(underlying)
    params = ib.reqSecDefOptParams(u.symbol, "", u.secType, u.conId)
    out = []
    for p in params:
        out.append({
            "exchange": p.exchange,
            "underlyingConId": p.underlyingConId,
            "tradingClass": p.tradingClass,
            "multiplier": p.multiplier,
            "expirations": sorted(p.expirations),
            "strikes": sorted(p.strikes),
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_chain_quote(args: argparse.Namespace) -> int:
    """Snapshot all strikes for a given expiry, both rights."""
    ib = _connect(args)
    underlying = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    [u] = ib.qualifyContracts(underlying)
    params = ib.reqSecDefOptParams(u.symbol, "", u.secType, u.conId)
    if not params:
        _emit({"error": "no chain params"}, args.format)
        ib.disconnect()
        return 1
    # pick SMART exchange param if possible
    pp = next((p for p in params if p.exchange == "SMART"), params[0])
    strikes = sorted(pp.strikes)
    if args.min_strike is not None:
        strikes = [s for s in strikes if s >= args.min_strike]
    if args.max_strike is not None:
        strikes = [s for s in strikes if s <= args.max_strike]
    if args.expiry not in pp.expirations:
        print(f"WARN: expiry {args.expiry} not in chain {sorted(pp.expirations)[:5]}...", file=sys.stderr)

    rights = ["C", "P"] if args.rights == "both" else [args.rights]
    opts = []
    for s in strikes:
        for r in rights:
            opts.append(Option(args.symbol, args.expiry, s, r, "SMART", tradingClass=pp.tradingClass, multiplier=pp.multiplier, currency=args.currency or "USD"))
    ib.qualifyContracts(*opts)
    # Request market data for all
    tickers = [ib.reqMktData(o, "", False, False) for o in opts if o.conId]
    ib.sleep(args.wait)
    out = []
    for t in tickers:
        d = _ticker_to_dict(t)
        d["strike"] = t.contract.strike
        d["right"] = t.contract.right
        d["expiry"] = t.contract.lastTradeDateOrContractMonth
        out.append(d)
    for o in opts:
        if o.conId:
            ib.cancelMktData(o)
    out.sort(key=lambda r: (r.get("right", ""), r.get("strike", 0)))
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_calc_iv(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    iv = ib.calculateImpliedVolatility(c, optionPrice=args.option_price, underPrice=args.under_price)
    _emit(_ticker_to_dict(iv) if iv else None, args.format)
    ib.disconnect()
    return 0


def cmd_calc_price(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    p = ib.calculateOptionPrice(c, volatility=args.vol, underPrice=args.under_price)
    _emit(_ticker_to_dict(p) if p else None, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Discovery / contract details
# ============================================================

def cmd_search(args: argparse.Namespace) -> int:
    ib = _connect(args)
    matches = ib.reqMatchingSymbols(args.query)
    out = []
    for m in matches:
        c = m.contract
        out.append({
            "conId": c.conId,
            "symbol": c.symbol,
            "secType": c.secType,
            "primaryExchange": c.primaryExchange,
            "currency": c.currency,
            "description": getattr(m, "description", ""),
            "derivativeSecTypes": ",".join(m.derivativeSecTypes or []),
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_contract(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    details = ib.reqContractDetails(c)
    out = []
    for d in details:
        out.append({
            "conId": d.contract.conId,
            "symbol": d.contract.symbol,
            "localSymbol": d.contract.localSymbol,
            "secType": d.contract.secType,
            "exchange": d.contract.exchange,
            "primaryExchange": d.contract.primaryExchange,
            "currency": d.contract.currency,
            "longName": d.longName,
            "industry": d.industry,
            "category": d.category,
            "subcategory": d.subcategory,
            "timeZoneId": d.timeZoneId,
            "tradingHours": d.tradingHours,
            "liquidHours": d.liquidHours,
            "minTick": d.minTick,
            "orderTypes": d.orderTypes,
            "validExchanges": d.validExchanges,
            "priceMagnifier": d.priceMagnifier,
            "underConId": d.underConId,
            "underSymbol": d.underSymbol,
            "underSecType": d.underSecType,
            "marketRuleIds": d.marketRuleIds,
            "secIdList": [(t.tag, t.value) for t in (d.secIdList or [])],
            "stockType": d.stockType,
            "contractMonth": d.contractMonth,
            "lastTradeTime": d.lastTradeTime,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_smart_components(args: argparse.Namespace) -> int:
    ib = _connect(args)
    comps = ib.reqSmartComponents(args.bbo_exchange)
    out = [{"bitNumber": k, "exchange": v.exchange, "exchangeLetter": v.exchangeLetter} for k, v in comps.items()]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_market_rule(args: argparse.Namespace) -> int:
    ib = _connect(args)
    rule = ib.reqMarketRule(args.rule_id)
    out = [{"lowEdge": inc.lowEdge, "increment": inc.increment} for inc in (rule or [])]
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Fundamentals
# ============================================================

def cmd_fundamentals(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    ib.qualifyContracts(c)
    data = ib.reqFundamentalData(c, args.report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(data or "")
        print(f"wrote {args.out} ({len(data or '')} bytes)")
    else:
        print(data or "")
    ib.disconnect()
    return 0


# ============================================================
# News
# ============================================================

def cmd_news_providers(args: argparse.Namespace) -> int:
    ib = _connect(args)
    provs = ib.reqNewsProviders()
    _emit([{"code": p.code, "name": p.name} for p in provs], args.format)
    ib.disconnect()
    return 0


def cmd_news(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = Stock(args.symbol, args.exchange or "SMART", args.currency or "USD")
    [c] = ib.qualifyContracts(c)
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S.0")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.0")
    providers = args.providers or "BRFG+BRFUPDN+DJNL+DJ-RT"
    items = ib.reqHistoricalNews(c.conId, providers, start, end, args.count)
    out = []
    for h in items:
        out.append({
            "time": str(h.time),
            "providerCode": h.providerCode,
            "articleId": h.articleId,
            "headline": h.headline,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_article(args: argparse.Namespace) -> int:
    ib = _connect(args)
    art = ib.reqNewsArticle(args.provider, args.article_id)
    out = {"articleType": art.articleType, "articleText": art.articleText}
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_news_bulletins(args: argparse.Namespace) -> int:
    ib = _connect(args)
    ib.reqNewsBulletins(allMessages=args.all_messages)
    end = time.time() + args.duration
    print(f"news bulletins for {args.duration}s...", file=sys.stderr)
    seen = 0
    while time.time() < end:
        ib.sleep(1.0)
        bulletins = ib.newsBulletins()
        while seen < len(bulletins):
            b = bulletins[seen]
            seen += 1
            print(json.dumps(_to_jsonable({
                "msgId": b.msgId, "msgType": b.msgType, "message": b.message, "origExchange": b.origExchange,
            }), default=str))
            sys.stdout.flush()
    ib.cancelNewsBulletins()
    ib.disconnect()
    return 0


# ============================================================
# Scanner
# ============================================================

def cmd_scanner_params(args: argparse.Namespace) -> int:
    ib = _connect(args)
    xml = ib.reqScannerParameters()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(xml)
        print(f"wrote {args.out} ({len(xml)} bytes)")
    else:
        print(xml)
    ib.disconnect()
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    ib = _connect(args)
    sub = ScannerSubscription(
        instrument=args.instrument,
        locationCode=args.location,
        scanCode=args.scan_code,
        numberOfRows=args.count,
    )
    if args.above_price is not None:
        sub.abovePrice = args.above_price
    if args.below_price is not None:
        sub.belowPrice = args.below_price
    if args.above_volume is not None:
        sub.aboveVolume = args.above_volume
    if args.market_cap_above is not None:
        sub.marketCapAbove = args.market_cap_above
    if args.market_cap_below is not None:
        sub.marketCapBelow = args.market_cap_below
    filters = []
    if args.filters:
        for kv in args.filters.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                filters.append(TagValue(k.strip(), v.strip()))
    results = ib.reqScannerData(sub, [], filters)
    out = []
    for r in results:
        c = r.contractDetails.contract
        out.append({
            "rank": r.rank,
            "symbol": c.symbol,
            "conId": c.conId,
            "secType": c.secType,
            "primaryExchange": c.primaryExchange,
            "currency": c.currency,
            "distance": r.distance,
            "benchmark": r.benchmark,
            "projection": r.projection,
            "legsStr": r.legsStr,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Orders — list / what-if / place / cancel
# ============================================================

def _order_dict(trade: Any) -> dict:
    o = trade.order
    c = trade.contract
    s = trade.orderStatus
    return {
        "orderId": o.orderId,
        "permId": o.permId,
        "account": o.account,
        "action": o.action,
        "totalQuantity": o.totalQuantity,
        "orderType": o.orderType,
        "lmtPrice": o.lmtPrice,
        "auxPrice": o.auxPrice,
        "tif": o.tif,
        "symbol": c.symbol,
        "secType": c.secType,
        "localSymbol": c.localSymbol,
        "expiry": getattr(c, "lastTradeDateOrContractMonth", ""),
        "right": getattr(c, "right", ""),
        "strike": getattr(c, "strike", 0.0),
        "status": s.status,
        "filled": s.filled,
        "remaining": s.remaining,
        "avgFillPrice": s.avgFillPrice,
        "whyHeld": s.whyHeld,
    }


def cmd_open_orders(args: argparse.Namespace) -> int:
    ib = _connect(args)
    if args.all:
        trades = ib.reqAllOpenOrders()
    else:
        trades = ib.reqOpenOrders()
    ib.sleep(1.0)
    out = [_order_dict(t) for t in trades]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_completed_orders(args: argparse.Namespace) -> int:
    ib = _connect(args)
    trades = ib.reqCompletedOrders(apiOnly=args.api_only)
    ib.sleep(1.0)
    out = [_order_dict(t) for t in trades]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_executions(args: argparse.Namespace) -> int:
    ib = _connect(args)
    f = ExecutionFilter()
    if args.account:
        f.acctCode = args.account
    if args.client_id_filter is not None:
        f.clientId = args.client_id_filter
    if args.symbol:
        f.symbol = args.symbol
    if args.sec_type:
        f.secType = args.sec_type
    if args.exchange_filter:
        f.exchange = args.exchange_filter
    if args.side:
        f.side = args.side
    if args.time:
        f.time = args.time
    fills = ib.reqExecutions(f)
    out = []
    for x in fills:
        c = x.contract
        e = x.execution
        cr = x.commissionReport
        out.append({
            "time": str(e.time),
            "account": e.acctNumber,
            "symbol": c.symbol,
            "secType": c.secType,
            "localSymbol": c.localSymbol,
            "side": e.side,
            "shares": e.shares,
            "price": e.price,
            "exchange": e.exchange,
            "orderId": e.orderId,
            "permId": e.permId,
            "execId": e.execId,
            "liquidation": e.liquidation,
            "cumQty": e.cumQty,
            "avgPrice": e.avgPrice,
            "commission": cr.commission if cr else None,
            "commissionCurrency": cr.currency if cr else None,
            "realizedPNL": cr.realizedPNL if cr else None,
        })
    _emit(out, args.format)
    ib.disconnect()
    return 0


def _build_order(args: argparse.Namespace) -> Order:
    o = Order()
    o.action = args.side
    o.totalQuantity = float(args.qty)
    o.orderType = args.type
    if args.limit_price is not None:
        o.lmtPrice = float(args.limit_price)
    if args.stop_price is not None:
        o.auxPrice = float(args.stop_price)
    o.tif = args.tif
    if args.account:
        o.account = args.account
    if args.order_ref:
        o.orderRef = args.order_ref
    if args.oca_group:
        o.ocaGroup = args.oca_group
    if args.outside_rth:
        o.outsideRth = True
    o.transmit = not args.no_transmit
    return o


def cmd_whatif(args: argparse.Namespace) -> int:
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    o = _build_order(args)
    if not o.account:
        accts = ib.managedAccounts()
        if not accts:
            raise SystemExit("No account specified and no managed accounts visible.")
        o.account = accts[0]
    st = ib.whatIfOrder(c, o)
    out = {
        "account": o.account,
        "status": st.status,
        "initMarginBefore": st.initMarginBefore,
        "initMarginChange": st.initMarginChange,
        "initMarginAfter": st.initMarginAfter,
        "maintMarginBefore": st.maintMarginBefore,
        "maintMarginChange": st.maintMarginChange,
        "maintMarginAfter": st.maintMarginAfter,
        "equityWithLoanBefore": st.equityWithLoanBefore,
        "equityWithLoanChange": st.equityWithLoanChange,
        "equityWithLoanAfter": st.equityWithLoanAfter,
        "commission": st.commission,
        "minCommission": st.minCommission,
        "maxCommission": st.maxCommission,
        "commissionCurrency": st.commissionCurrency,
        "warningText": st.warningText,
    }
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_place(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit("Refusing to place live order without --yes confirmation flag.")
    ib = _connect(args)
    c = _build_contract(args)
    ib.qualifyContracts(c)
    o = _build_order(args)
    trade = ib.placeOrder(c, o)
    ib.sleep(2.0)
    _emit(_order_dict(trade), args.format)
    ib.disconnect()
    return 0


def cmd_combo(args: argparse.Namespace) -> int:
    """Place a multi-leg BAG order. Legs: --legs 'BUY:conId:ratio,SELL:conId:ratio,...'"""
    if not args.yes:
        raise SystemExit("Refusing to place live combo order without --yes confirmation flag.")
    ib = _connect(args)
    legs = []
    for leg_spec in args.legs.split(","):
        parts = leg_spec.strip().split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad leg spec '{leg_spec}' (want ACTION:conId:ratio)")
        action, con_id, ratio = parts
        legs.append(ComboLeg(conId=int(con_id), ratio=int(ratio), action=action.upper(), exchange=args.exchange or "SMART"))
    bag = Bag(symbol=args.symbol, currency=args.currency or "USD", exchange=args.exchange or "SMART")
    bag.comboLegs = legs
    o = Order()
    o.action = args.side
    o.totalQuantity = float(args.qty)
    o.orderType = args.type
    if args.limit_price is not None:
        o.lmtPrice = float(args.limit_price)
    o.tif = args.tif
    if args.account:
        o.account = args.account
    o.transmit = not args.no_transmit
    trade = ib.placeOrder(bag, o)
    ib.sleep(2.0)
    out = _order_dict(trade)
    out["legs"] = [(l.action, l.conId, l.ratio) for l in legs]
    _emit(out, args.format)
    ib.disconnect()
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    ib = _connect(args)
    trades = ib.reqOpenOrders()
    ib.sleep(0.5)
    target = None
    for t in trades:
        if t.order.orderId == args.order_id:
            target = t
            break
    if not target:
        _emit({"error": f"order {args.order_id} not found"}, args.format)
        ib.disconnect()
        return 1
    ib.cancelOrder(target.order)
    ib.sleep(1.0)
    _emit(_order_dict(target), args.format)
    ib.disconnect()
    return 0


def cmd_cancel_all(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit("Refusing to global-cancel without --yes")
    ib = _connect(args)
    ib.reqGlobalCancel()
    ib.sleep(1.5)
    _emit({"global_cancel": "sent"}, args.format)
    ib.disconnect()
    return 0


# ============================================================
# WSH / FX / misc
# ============================================================

def cmd_wsh_meta(args: argparse.Namespace) -> int:
    ib = _connect(args)
    try:
        meta = ib.reqWshMetaData()
        ib.sleep(2.0)
    except Exception as exc:
        meta = f"error: {exc}"
    _emit({"meta": str(meta)[:10000]}, args.format)
    ib.disconnect()
    return 0


def cmd_fx(args: argparse.Namespace) -> int:
    ib = _connect(args)
    pair = args.pair.upper().replace("/", "").replace(".", "")
    c = Forex(pair)
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    ib.sleep(args.wait)
    out = _ticker_to_dict(t)
    out["pair"] = pair
    if args.amount and out.get("bid") and out.get("ask"):
        mid = (out["bid"] + out["ask"]) / 2
        out["converted"] = args.amount * mid
    ib.cancelMktData(c)
    _emit(out, args.format)
    ib.disconnect()
    return 0


# ============================================================
# Argparse wiring
# ============================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--client-id", type=int, default=None)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--market-data-type", type=int, default=None, help="1=Live 2=Frozen 3=Delayed 4=DelayedFrozen")
    p.add_argument("--format", choices=["json", "csv", "table"], default="table")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trading_algo.ibkr_tool",
        description="Comprehensive IBKR data + operations CLI (ib_async based).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], int], help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        _add_common(sp)
        sp.set_defaults(func=fn)
        return sp

    # --- Meta ---
    add("connect", cmd_connect, "Test connection and show server info")
    add("time", cmd_time, "Get IBKR server time")
    add("accounts", cmd_accounts, "List managed account codes")
    add("user-info", cmd_user_info, "Get user info (white-brand)")

    # --- Account ---
    s = add("summary", cmd_summary, "Account summary tags")
    s.add_argument("--account", default=None)
    s.add_argument("--tags", default=None, help="Comma list or leave blank for defaults")

    s = add("values", cmd_values, "Full account values via reqAccountUpdates")
    s.add_argument("--account", default=None)
    s.add_argument("--tag", default=None, help="Substring filter on tag name")

    s = add("positions", cmd_positions, "List all positions")
    s.add_argument("--account", default=None)
    s.add_argument("--symbol", default=None)

    s = add("portfolio", cmd_portfolio, "Portfolio items with MTM (via reqAccountUpdates)")
    s.add_argument("--account", default=None)

    s = add("pnl", cmd_pnl, "Account-level daily/realized/unrealized PnL")
    s.add_argument("--account", default=None)

    s = add("pnl-single", cmd_pnl_single, "Per-position PnL")
    s.add_argument("--account", required=True)
    s.add_argument("--con-id", type=int, required=True)

    # --- Quotes / live data ---
    s = add("quote", cmd_quote, "Snap quote for one contract (includes greeks for OPT)")
    _add_contract_args(s)
    s.add_argument("--wait", type=float, default=3.0)

    s = add("quotes", cmd_quotes, "Batch snap for multiple STK symbols via reqTickers")
    s.add_argument("--symbols", required=True, help="Comma list")
    s.add_argument("--currency", default=None)

    s = add("stream", cmd_stream, "Stream ticks for a contract")
    _add_contract_args(s)
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--interval", type=float, default=1.0)

    s = add("depth", cmd_depth, "Market depth (DOM) ladder")
    _add_contract_args(s)
    s.add_argument("--rows", type=int, default=10)
    s.add_argument("--smart", action="store_true")
    s.add_argument("--wait", type=float, default=2.0)

    s = add("depth-exchanges", cmd_depth_exchanges, "List market depth exchanges")

    s = add("realtime-bars", cmd_realtime_bars, "5-sec realtime bars stream")
    _add_contract_args(s)
    s.add_argument("--what-to-show", default="TRADES")
    s.add_argument("--rth", action="store_true")
    s.add_argument("--duration", type=float, default=30.0)

    s = add("ticks", cmd_ticks, "Tick-by-tick data stream")
    _add_contract_args(s)
    s.add_argument("--tick-type", choices=["Last", "AllLast", "BidAsk", "MidPoint"], default="Last")
    s.add_argument("--duration", type=float, default=30.0)

    # --- Historical ---
    s = add("history", cmd_history, "Historical bars (reqHistoricalData)")
    _add_contract_args(s)
    s.add_argument("--duration", default="1 D")
    s.add_argument("--bar-size", default="5 mins")
    s.add_argument("--what-to-show", default="TRADES")
    s.add_argument("--rth", action="store_true")
    s.add_argument("--end", default=None)

    s = add("history-ticks", cmd_history_ticks, "Historical ticks")
    _add_contract_args(s)
    s.add_argument("--start", default=None)
    s.add_argument("--end", default=None)
    s.add_argument("--count", type=int, default=1000)
    s.add_argument("--what-to-show", choices=["TRADES", "BID_ASK", "MIDPOINT"], default="TRADES")
    s.add_argument("--rth", action="store_true")

    s = add("head-timestamp", cmd_head_timestamp, "Earliest available data timestamp")
    _add_contract_args(s)
    s.add_argument("--what-to-show", default="TRADES")
    s.add_argument("--rth", action="store_true")

    s = add("histogram", cmd_histogram, "Price histogram (reqHistogramData)")
    _add_contract_args(s)
    s.add_argument("--rth", action="store_true")
    s.add_argument("--period", default="20 days")

    s = add("schedule", cmd_schedule, "Historical trading schedule/sessions")
    _add_contract_args(s)
    s.add_argument("--duration", default="1 M")
    s.add_argument("--end", default=None)
    s.add_argument("--rth", action="store_true")

    # --- Options ---
    s = add("chain", cmd_chain, "Option chain metadata (reqSecDefOptParams)")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)

    s = add("chain-quote", cmd_chain_quote, "Snap all strikes at one expiry (both rights)")
    s.add_argument("--symbol", required=True)
    s.add_argument("--expiry", required=True)
    s.add_argument("--rights", choices=["C", "P", "both"], default="both")
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--min-strike", type=float, default=None)
    s.add_argument("--max-strike", type=float, default=None)
    s.add_argument("--wait", type=float, default=4.0)

    s = add("calc-iv", cmd_calc_iv, "Calculate implied volatility for an option")
    _add_contract_args(s, default_kind="OPT")
    s.add_argument("--option-price", type=float, required=True)
    s.add_argument("--under-price", type=float, required=True)

    s = add("calc-price", cmd_calc_price, "Calculate theoretical option price")
    _add_contract_args(s, default_kind="OPT")
    s.add_argument("--vol", type=float, required=True)
    s.add_argument("--under-price", type=float, required=True)

    # --- Discovery ---
    s = add("search", cmd_search, "Search matching symbols (reqMatchingSymbols)")
    s.add_argument("--query", required=True)

    s = add("contract", cmd_contract, "Full contract details")
    _add_contract_args(s)

    s = add("smart-components", cmd_smart_components, "SMART routing components")
    s.add_argument("--bbo-exchange", required=True)

    s = add("market-rule", cmd_market_rule, "Price increment rules")
    s.add_argument("--rule-id", type=int, required=True)

    # --- Fundamentals ---
    s = add("fundamentals", cmd_fundamentals, "Fundamental data reports")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--report", choices=[
        "ReportsFinSummary", "ReportSnapshot", "ReportRatios",
        "ReportsFinStatements", "ReportsOwnership", "RESC", "CalendarReport",
    ], default="ReportSnapshot")
    s.add_argument("--out", default=None)

    # --- News ---
    add("news-providers", cmd_news_providers, "List news providers")

    s = add("news", cmd_news, "Historical news headlines")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--providers", default=None, help="Default: BRFG+BRFUPDN+DJNL+DJ-RT")
    s.add_argument("--start", default=None, help="YYYY-MM-DD HH:MM:SS.0")
    s.add_argument("--end", default=None)
    s.add_argument("--count", type=int, default=20)

    s = add("article", cmd_article, "Fetch news article body")
    s.add_argument("--provider", required=True)
    s.add_argument("--article-id", required=True)

    s = add("news-bulletins", cmd_news_bulletins, "Stream TWS news bulletins")
    s.add_argument("--duration", type=float, default=30.0)
    s.add_argument("--all-messages", action="store_true")

    # --- Scanner ---
    s = add("scanner-params", cmd_scanner_params, "Dump scanner parameters XML")
    s.add_argument("--out", default=None)

    s = add("scan", cmd_scan, "Run a scanner subscription")
    s.add_argument("--scan-code", default="TOP_PERC_GAIN")
    s.add_argument("--instrument", default="STK")
    s.add_argument("--location", default="STK.US.MAJOR")
    s.add_argument("--count", type=int, default=25)
    s.add_argument("--above-price", type=float, default=None)
    s.add_argument("--below-price", type=float, default=None)
    s.add_argument("--above-volume", type=int, default=None)
    s.add_argument("--market-cap-above", type=float, default=None)
    s.add_argument("--market-cap-below", type=float, default=None)
    s.add_argument("--filters", default=None, help="Comma list k=v")

    # --- Orders ---
    s = add("open-orders", cmd_open_orders, "List open orders")
    s.add_argument("--all", action="store_true", help="All clients (reqAllOpenOrders)")

    s = add("completed-orders", cmd_completed_orders, "List completed orders")
    s.add_argument("--api-only", action="store_true")

    s = add("executions", cmd_executions, "List executions (fills) with commission reports")
    s.add_argument("--account", default=None)
    s.add_argument("--client-id-filter", type=int, default=None)
    s.add_argument("--symbol", default=None)
    s.add_argument("--sec-type", default=None)
    s.add_argument("--exchange-filter", default=None)
    s.add_argument("--side", default=None)
    s.add_argument("--time", default=None, help="yyyymmdd-hh:mm:ss UTC")

    def _add_order_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--side", choices=["BUY", "SELL"], required=True)
        sp.add_argument("--qty", required=True)
        sp.add_argument("--type", choices=["MKT", "LMT", "STP", "STPLMT", "TRAIL", "MOC", "LOC"], default="LMT")
        sp.add_argument("--limit-price", type=float, default=None)
        sp.add_argument("--stop-price", type=float, default=None)
        sp.add_argument("--tif", default="DAY")
        sp.add_argument("--account", default=None)
        sp.add_argument("--order-ref", default=None)
        sp.add_argument("--oca-group", default=None)
        sp.add_argument("--outside-rth", action="store_true")
        sp.add_argument("--no-transmit", action="store_true")

    s = add("whatif", cmd_whatif, "Preview order (whatIf) — margin impact, commission")
    _add_contract_args(s)
    _add_order_args(s)

    s = add("place", cmd_place, "Place a single-leg order (requires --yes)")
    _add_contract_args(s)
    _add_order_args(s)
    s.add_argument("--yes", action="store_true", help="Required live confirmation flag")

    s = add("combo", cmd_combo, "Place multi-leg BAG combo order")
    s.add_argument("--symbol", required=True)
    s.add_argument("--exchange", default=None)
    s.add_argument("--currency", default=None)
    s.add_argument("--legs", required=True, help="ACTION:conId:ratio,ACTION:conId:ratio,...")
    _add_order_args(s)
    s.add_argument("--yes", action="store_true")

    s = add("cancel", cmd_cancel, "Cancel one order by orderId")
    s.add_argument("--order-id", type=int, required=True)

    s = add("cancel-all", cmd_cancel_all, "Global cancel (all orders, all clients)")
    s.add_argument("--yes", action="store_true")

    # --- WSH / FX ---
    add("wsh-meta", cmd_wsh_meta, "Wall Street Horizon metadata")

    s = add("fx", cmd_fx, "Forex quote + optional conversion")
    s.add_argument("--pair", required=True, help="e.g. USDCAD, EURUSD")
    s.add_argument("--amount", type=float, default=None)
    s.add_argument("--wait", type=float, default=2.0)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
