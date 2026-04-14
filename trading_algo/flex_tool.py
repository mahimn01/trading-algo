"""
Comprehensive IBKR Flex Web Service CLI.

Covers: send + poll + cache, then parse every section of a FlexQueryResponse:
trades, open positions, prior period positions, cash report, cash transactions,
change in NAV, change in position values, complex positions, conversion rates,
corporate actions, securities info, statement of funds, transfers, SLB activity,
dividends/interest/fees, PnL aggregation (by symbol / account / asset class).

Usage:
    python -m trading_algo.flex_tool <command> [args]

Environment (from .env):
    IBKR_FLEX_TOKEN       — Flex Web Service token
    IBKR_FLEX_QUERY_ACTIVITY — default Activity Flex Query ID (used if --query-id omitted)

All commands either fetch fresh (--fetch) or read cached XML (default: latest).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


# ============================================================
# .env loader + config
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

CACHE_DIR = Path("data/flex")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FLEX_VERSION = "3"
SEND_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
GET_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"

FLEX_QUERIES = {
    "activity": os.getenv("IBKR_FLEX_QUERY_ACTIVITY"),
    "trades": os.getenv("IBKR_FLEX_QUERY_TRADES"),
    "custom": os.getenv("IBKR_FLEX_QUERY_CUSTOM"),
}


# ============================================================
# Output helpers
# ============================================================

def _fmt_number(v: Any) -> Any:
    if v in (None, ""):
        return None
    try:
        f = float(v)
        if f == int(f):
            return int(f)
        return f
    except (ValueError, TypeError):
        return v


def _emit(data: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(data, indent=2, default=str))
        return
    if fmt == "csv":
        rows = data if isinstance(data, list) else [data]
        rows = [r if isinstance(r, dict) else {"value": r} for r in rows if r is not None]
        if not rows:
            return
        keys = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(sys.stdout, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) in (None, "") else r.get(k)) for k in keys})
        return
    # table
    rows = data if isinstance(data, list) else [data]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("(no rows)")
        return
    if isinstance(rows[0], dict):
        keys = sorted({k for r in rows for k in r.keys()})
        # prioritize common cols first
        priority = ["accountId", "tradeDate", "dateTime", "symbol", "assetCategory", "quantity", "tradePrice",
                    "proceeds", "commission", "realizedPnl", "fifoPnlRealized", "mtmPnl", "description",
                    "currency", "type", "amount", "total", "pnl"]
        sorted_keys = [k for k in priority if k in keys] + [k for k in keys if k not in priority]
        widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in sorted_keys}
        for k in sorted_keys:
            widths[k] = min(widths[k], 40)
        print("  ".join(k.ljust(widths[k]) for k in sorted_keys))
        print("  ".join("-" * widths[k] for k in sorted_keys))
        for r in rows:
            print("  ".join(str(r.get(k, "") if r.get(k) is not None else "")[: widths[k]].ljust(widths[k]) for k in sorted_keys))
    else:
        for r in rows:
            print(r)


# ============================================================
# Flex Web Service — send + poll
# ============================================================

def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "flex-tool/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_send_response(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    status = root.findtext("Status") or root.findtext("status") or ""
    ref = root.findtext("ReferenceCode") or root.findtext("referenceCode") or ""
    url = root.findtext("Url") or root.findtext("url") or ""
    err_code = root.findtext("ErrorCode") or root.findtext("errorCode") or ""
    err_msg = root.findtext("ErrorMessage") or root.findtext("errorMessage") or ""
    return {"status": status, "referenceCode": ref, "url": url, "errorCode": err_code, "errorMessage": err_msg}


def _send_and_poll(token: str, query_id: str, *, max_wait: float = 120.0, poll_interval: float = 3.0) -> str:
    send = f"{SEND_URL}?t={urllib.parse.quote(token)}&q={urllib.parse.quote(query_id)}&v={FLEX_VERSION}"
    print(f"POST {SEND_URL}?t=***&q={query_id}&v={FLEX_VERSION}", file=sys.stderr)
    send_body = _http_get(send)
    parsed = _parse_send_response(send_body)
    if parsed["status"] != "Success":
        raise SystemExit(f"Flex SendRequest failed: {parsed}")
    ref = parsed["referenceCode"]
    get_url = parsed["url"] or GET_URL
    print(f"  -> reference code {ref}, polling...", file=sys.stderr)

    deadline = time.time() + max_wait
    last = None
    while time.time() < deadline:
        time.sleep(poll_interval)
        fetch = f"{get_url}?t={urllib.parse.quote(token)}&q={urllib.parse.quote(ref)}&v={FLEX_VERSION}"
        body = _http_get(fetch)
        last = body
        if body.lstrip().startswith("<FlexQueryResponse"):
            return body
        # check for in-progress warning
        try:
            root = ET.fromstring(body)
            status = root.findtext("Status") or ""
            err_code = root.findtext("ErrorCode") or ""
            err_msg = root.findtext("ErrorMessage") or ""
            if status == "Warn" and err_code in ("1019", "1020"):
                print(f"  ... {err_msg}", file=sys.stderr)
                continue
            if status != "Success":
                raise SystemExit(f"Flex GetStatement failed: status={status} code={err_code} msg={err_msg}")
        except ET.ParseError:
            pass
    raise SystemExit(f"Flex timeout after {max_wait}s; last body head: {(last or '')[:500]}")


def _resolve_query_id(name_or_id: str | None) -> str:
    if not name_or_id:
        q = FLEX_QUERIES.get("activity")
        if not q:
            raise SystemExit("No query id provided and IBKR_FLEX_QUERY_ACTIVITY not set.")
        return q
    if name_or_id in FLEX_QUERIES and FLEX_QUERIES[name_or_id]:
        return FLEX_QUERIES[name_or_id]
    return name_or_id


def _resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("IBKR_FLEX_TOKEN") or ""
    if not token:
        raise SystemExit("No IBKR_FLEX_TOKEN set. Put it in .env or pass --token.")
    return token


# ============================================================
# Cache
# ============================================================

def _latest_cached(pattern: str = "*.xml") -> Path | None:
    files = sorted(CACHE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _cache_save(xml_text: str, label: str = "flex") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CACHE_DIR / f"{label}_{ts}.xml"
    path.write_text(xml_text, encoding="utf-8")
    return path


def _load_xml(args: argparse.Namespace) -> ET.Element:
    if getattr(args, "file", None):
        return ET.parse(args.file).getroot()
    latest = _latest_cached()
    if not latest:
        raise SystemExit("No cached Flex XML found. Run `flex_tool send` first or pass --file.")
    return ET.parse(latest).getroot()


# ============================================================
# XML → rows helpers
# ============================================================

def _attrs_to_row(el: ET.Element, numeric_keys: Iterable[str] = ()) -> dict:
    row = dict(el.attrib)
    for k in numeric_keys:
        if k in row:
            row[k] = _fmt_number(row[k])
    return row


def _iter_section(root: ET.Element, section_tag: str, leaf_tag: str | None = None) -> list[ET.Element]:
    """Collect leaf elements inside a section across all FlexStatements."""
    out = []
    for stmt in root.iter("FlexStatement"):
        section = stmt.find(section_tag)
        if section is None:
            continue
        if leaf_tag:
            out.extend(section.findall(leaf_tag))
        else:
            out.extend(list(section))
    return out


def _filter_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    acct = getattr(args, "account", None)
    sym = getattr(args, "symbol", None)
    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)
    asset = getattr(args, "asset_category", None)
    currency = getattr(args, "currency", None)

    def _date_key(r: dict) -> str:
        for k in ("tradeDate", "reportDate", "dateTime", "date", "settleDateTarget", "actionDate"):
            if r.get(k):
                return str(r[k])[:8]
        return ""

    out = []
    for r in rows:
        if acct and r.get("accountId") != acct:
            continue
        if sym:
            sym_u = sym.upper()
            fields = [r.get("symbol", ""), r.get("underlyingSymbol", ""), r.get("description", "")]
            if not any(sym_u in (f or "").upper() for f in fields):
                continue
        if asset and r.get("assetCategory") != asset:
            continue
        if currency and r.get("currency") != currency:
            continue
        if date_from or date_to:
            d = _date_key(r)
            if date_from and d and d < date_from.replace("-", ""):
                continue
            if date_to and d and d > date_to.replace("-", ""):
                continue
        out.append(r)
    return out


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", default=None, help="Path to Flex XML (default: latest cached)")
    p.add_argument("--account", default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--date-from", default=None, help="YYYY-MM-DD")
    p.add_argument("--date-to", default=None, help="YYYY-MM-DD")
    p.add_argument("--asset-category", default=None, help="STK, OPT, FUT, FX, CASH, BOND")
    p.add_argument("--currency", default=None)
    p.add_argument("--limit", type=int, default=None)


def _apply_limit(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    limit = getattr(args, "limit", None)
    if limit:
        return rows[-limit:]
    return rows


# ============================================================
# Meta commands — send / parse / cached
# ============================================================

def cmd_list_queries(args: argparse.Namespace) -> int:
    out = [{"name": k, "queryId": v or "(unset)"} for k, v in FLEX_QUERIES.items()]
    out.append({"name": "token_set", "queryId": "yes" if os.getenv("IBKR_FLEX_TOKEN") else "no"})
    _emit(out, args.format)
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    token = _resolve_token(args.token)
    qid = _resolve_query_id(args.query_id)
    xml = _send_and_poll(token, qid, max_wait=args.max_wait, poll_interval=args.poll_interval)
    path = _cache_save(xml, label=args.label)
    print(f"saved {path} ({len(xml)} bytes)")
    if args.print:
        print(xml)
    return 0


def cmd_cached(args: argparse.Namespace) -> int:
    files = sorted(CACHE_DIR.glob("*.xml"), key=lambda p: p.stat().st_mtime)
    out = [{
        "file": str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p),
        "size_kb": round(p.stat().st_size / 1024, 1),
        "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
    } for p in files]
    _emit(out, args.format)
    return 0


def cmd_latest(args: argparse.Namespace) -> int:
    p = _latest_cached()
    if not p:
        print("(no cached XML)")
        return 1
    print(str(p))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    stmts = root.findall("FlexStatements/FlexStatement") or root.findall(".//FlexStatement")
    info = {
        "queryName": root.get("queryName"),
        "type": root.get("type"),
        "statementCount": len(stmts),
        "statements": [
            {
                "accountId": s.get("accountId"),
                "fromDate": s.get("fromDate"),
                "toDate": s.get("toDate"),
                "period": s.get("period"),
                "whenGenerated": s.get("whenGenerated"),
                "sections": sorted({c.tag for c in s}),
            } for s in stmts
        ],
    }
    _emit(info, args.format)
    return 0


# ============================================================
# Sections
# ============================================================

_TRADE_NUMERIC = [
    "quantity", "tradePrice", "tradeMoney", "proceeds", "taxes", "ibCommission",
    "netCash", "closePrice", "cost", "fifoPnlRealized", "capitalGainsPnl", "fxPnl",
    "mtmPnl", "strike", "multiplier", "ibCommissionCurrency",
]


def cmd_trades(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "Trades", "Trade")
    rows = [_attrs_to_row(e, _TRADE_NUMERIC) for e in elems]
    rows = _filter_rows(rows, args)
    rows = _apply_limit(rows, args)
    if args.columns:
        wanted = set(args.columns.split(","))
        rows = [{k: v for k, v in r.items() if k in wanted} for r in rows]
    else:
        keep = ["accountId", "tradeDate", "dateTime", "assetCategory", "symbol", "description",
                "buySell", "quantity", "tradePrice", "proceeds", "ibCommission",
                "netCash", "fifoPnlRealized", "mtmPnl", "currency", "expiry", "strike", "putCall"]
        rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    _emit(rows, args.format)
    return 0


def cmd_open_positions(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "OpenPositions", "OpenPosition")
    rows = [_attrs_to_row(e, ["position", "markPrice", "positionValue", "openPrice", "costBasisMoney",
                               "percentOfNAV", "fifoPnlUnrealized", "unrealizedCapitalGainsPnl",
                               "unrealizedlFxPnl", "strike", "multiplier"]) for e in elems]
    rows = _filter_rows(rows, args)
    keep = ["accountId", "symbol", "description", "assetCategory", "position", "markPrice",
            "positionValue", "openPrice", "costBasisMoney", "percentOfNAV",
            "fifoPnlUnrealized", "currency", "expiry", "strike", "putCall"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    _emit(rows, args.format)
    return 0


def cmd_prior_positions(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "PriorPeriodPositions", "PriorPeriodPosition")
    rows = [_attrs_to_row(e, ["priorMtmPnl", "price", "cost"]) for e in elems]
    rows = _filter_rows(rows, args)
    _emit(rows, args.format)
    return 0


def cmd_cash_report(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "CashReport", "CashReportCurrency")
    rows = [_attrs_to_row(e, ["startingCash", "clientFees", "commissions", "depositWithdrawals",
                              "dividends", "brokerInterest", "bondInterest", "feesReceivables",
                              "endingCash", "endingSettledCash", "netTradesSales", "netTradesPurchases",
                              "withholdingTax", "otherFees", "commissionReceivables"]) for e in elems]
    if args.account:
        rows = [r for r in rows if r.get("accountId") == args.account]
    if args.currency:
        rows = [r for r in rows if r.get("currency") == args.currency]
    if getattr(args, "base_only", False):
        rows = [r for r in rows if r.get("levelOfDetail") == "BaseCurrency"]
    keep = ["accountId", "currency", "startingCash", "depositWithdrawals", "commissions",
            "dividends", "brokerInterest", "withholdingTax", "netTradesSales", "netTradesPurchases",
            "endingCash", "endingSettledCash"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    _emit(rows, args.format)
    return 0


def cmd_cash_transactions(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "CashTransactions", "CashTransaction")
    rows = [_attrs_to_row(e, ["amount"]) for e in elems]
    rows = _filter_rows(rows, args)
    if args.type:
        rows = [r for r in rows if args.type.lower() in (r.get("type") or "").lower()]
    keep = ["accountId", "dateTime", "settleDate", "type", "description", "symbol", "amount", "currency"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    rows = _apply_limit(rows, args)
    _emit(rows, args.format)
    return 0


def cmd_change_in_nav(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "ChangeInNAV")
    if not elems:
        elems = [c for stmt in root.iter("FlexStatement") for c in stmt.findall("ChangeInNAV")]
    if not elems:
        elems = list(root.iter("ChangeInNAV"))
    rows = [_attrs_to_row(e) for e in elems]
    for r in rows:
        for k in list(r.keys()):
            if k not in ("accountId", "acctAlias", "model", "currency", "fromDate", "toDate"):
                r[k] = _fmt_number(r[k])
    if args.account:
        rows = [r for r in rows if r.get("accountId") == args.account]
    if args.drop_zero:
        for r in rows:
            for k in list(r.keys()):
                if isinstance(r[k], (int, float)) and r[k] == 0:
                    del r[k]
    _emit(rows, args.format)
    return 0


def cmd_change_in_position_values(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "ChangeInPositionValues", "ChangeInPositionValue")
    rows = [_attrs_to_row(e, ["priorPeriodValue", "transactions", "mtmPriorPeriodPositions",
                              "mtmTransactions", "corporateActions", "other",
                              "accountTransfers", "linkingAdjustments", "fxTranslationPnl",
                              "futurePriceAdjustments", "settledCash", "endOfPeriodValue"]) for e in elems]
    rows = _filter_rows(rows, args)
    _emit(rows, args.format)
    return 0


def cmd_complex_positions(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "ComplexPositions", "ComplexPosition")
    rows = [_attrs_to_row(e) for e in elems]
    rows = _filter_rows(rows, args)
    _emit(rows, args.format)
    return 0


def cmd_conversion_rates(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "ConversionRates", "ConversionRate")
    rows = [_attrs_to_row(e, ["rate"]) for e in elems]
    if args.currency:
        rows = [r for r in rows if r.get("fromCurrency") == args.currency or r.get("toCurrency") == args.currency]
    _emit(rows, args.format)
    return 0


def cmd_corporate_actions(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "CorporateActions", "CorporateAction")
    rows = [_attrs_to_row(e, ["quantity", "proceeds", "value", "fifoPnlRealized", "mtmPnl"]) for e in elems]
    rows = _filter_rows(rows, args)
    _emit(rows, args.format)
    return 0


def cmd_securities_info(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "SecuritiesInfo", "SecurityInfo")
    rows = [_attrs_to_row(e) for e in elems]
    if args.symbol:
        rows = [r for r in rows if r.get("symbol") == args.symbol]
    if args.asset_category:
        rows = [r for r in rows if r.get("assetCategory") == args.asset_category]
    _emit(rows, args.format)
    return 0


def cmd_stmt_funds(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "StmtFunds", "StatementOfFundsLine")
    rows = [_attrs_to_row(e, ["amount", "balance", "debit", "credit", "tradeQuantity", "tradePrice",
                              "tradeGross", "tradeCommission", "tradeTax"]) for e in elems]
    rows = _filter_rows(rows, args)
    rows = _apply_limit(rows, args)
    keep = ["accountId", "date", "activityDescription", "symbol", "amount", "balance", "currency"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    _emit(rows, args.format)
    return 0


def cmd_transfers(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "Transfers", "Transfer")
    rows = [_attrs_to_row(e, ["quantity", "positionAmount", "positionAmountInBase", "cashTransfer"]) for e in elems]
    rows = _filter_rows(rows, args)
    _emit(rows, args.format)
    return 0


def cmd_slb_activities(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "SLBActivities", "SLBActivity")
    rows = [_attrs_to_row(e) for e in elems]
    rows = _filter_rows(rows, args)
    _emit(rows, args.format)
    return 0


# ============================================================
# Analytical commands
# ============================================================

def cmd_pnl_by_symbol(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "Trades", "Trade")
    rows = [_attrs_to_row(e, _TRADE_NUMERIC) for e in elems]
    rows = _filter_rows(rows, args)

    agg = defaultdict(lambda: {"symbol": "", "accountId": "", "assetCategory": "", "trades": 0,
                                "gross": 0.0, "commissions": 0.0, "realized": 0.0, "mtm": 0.0,
                                "proceeds": 0.0, "currency": ""})
    for r in rows:
        key = (r.get("accountId", ""), r.get("symbol", ""), r.get("assetCategory", ""), r.get("currency", ""))
        a = agg[key]
        a["accountId"] = key[0]
        a["symbol"] = key[1]
        a["assetCategory"] = key[2]
        a["currency"] = key[3]
        a["trades"] += 1
        a["gross"] += float(r.get("tradeMoney") or 0)
        a["commissions"] += float(r.get("ibCommission") or 0)
        a["realized"] += float(r.get("fifoPnlRealized") or 0)
        a["mtm"] += float(r.get("mtmPnl") or 0)
        a["proceeds"] += float(r.get("proceeds") or 0)

    out = [dict(v, pnl=round(v["realized"] + v["commissions"], 2),
                realized=round(v["realized"], 2),
                commissions=round(v["commissions"], 2),
                mtm=round(v["mtm"], 2),
                gross=round(v["gross"], 2),
                proceeds=round(v["proceeds"], 2)) for v in agg.values()]
    out.sort(key=lambda r: r["realized"], reverse=True)
    if args.top:
        out = out[:args.top]
    _emit(out, args.format)
    return 0


def cmd_pnl_by_account(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "Trades", "Trade")
    rows = [_attrs_to_row(e, _TRADE_NUMERIC) for e in elems]

    agg = defaultdict(lambda: {"accountId": "", "currency": "", "trades": 0, "realized": 0.0,
                                "commissions": 0.0, "mtm": 0.0})
    for r in rows:
        key = (r.get("accountId", ""), r.get("currency", ""))
        a = agg[key]
        a["accountId"] = key[0]
        a["currency"] = key[1]
        a["trades"] += 1
        a["realized"] += float(r.get("fifoPnlRealized") or 0)
        a["commissions"] += float(r.get("ibCommission") or 0)
        a["mtm"] += float(r.get("mtmPnl") or 0)
    out = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in agg.values()]
    _emit(out, args.format)
    return 0


def cmd_commissions_total(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "Trades", "Trade")
    rows = [_attrs_to_row(e, _TRADE_NUMERIC) for e in elems]
    rows = _filter_rows(rows, args)
    agg = defaultdict(lambda: {"currency": "", "trades": 0, "commission_abs": 0.0})
    for r in rows:
        a = agg[r.get("currency", "")]
        a["currency"] = r.get("currency", "")
        a["trades"] += 1
        a["commission_abs"] += abs(float(r.get("ibCommission") or 0))
    _emit([dict(v, commission_abs=round(v["commission_abs"], 2)) for v in agg.values()], args.format)
    return 0


def cmd_dividends(args: argparse.Namespace) -> int:
    """Dividends aggregated from CashTransactions where type contains 'Dividend'."""
    root = _load_xml(args)
    elems = _iter_section(root, "CashTransactions", "CashTransaction")
    rows = [_attrs_to_row(e, ["amount"]) for e in elems
            if "dividend" in (e.get("type") or "").lower() or "div" in (e.get("description") or "").lower()]
    rows = _filter_rows(rows, args)
    keep = ["accountId", "dateTime", "settleDate", "symbol", "description", "type", "amount", "currency"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    total_by_ccy = defaultdict(float)
    for r in rows:
        total_by_ccy[r.get("currency", "")] += float(r.get("amount") or 0)
    if args.totals_only:
        out = [{"currency": c, "total": round(v, 2)} for c, v in total_by_ccy.items()]
    else:
        out = rows + [{"currency": c, "total": round(v, 2)} for c, v in total_by_ccy.items()]
    _emit(out, args.format)
    return 0


def cmd_interest(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "CashTransactions", "CashTransaction")
    rows = [_attrs_to_row(e, ["amount"]) for e in elems
            if "interest" in (e.get("type") or "").lower() or "interest" in (e.get("description") or "").lower()]
    rows = _filter_rows(rows, args)
    keep = ["accountId", "dateTime", "settleDate", "description", "type", "amount", "currency"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    _emit(rows, args.format)
    return 0


def cmd_fees(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    elems = _iter_section(root, "CashTransactions", "CashTransaction")
    rows = [_attrs_to_row(e, ["amount"]) for e in elems
            if any(term in (e.get("type") or "").lower() or term in (e.get("description") or "").lower()
                   for term in ("fee", "commission", "tax"))]
    rows = _filter_rows(rows, args)
    keep = ["accountId", "dateTime", "description", "type", "amount", "currency"]
    rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
    _emit(rows, args.format)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """One-shot overview: NAV, P&L, position count, trade count, commissions."""
    root = _load_xml(args)
    stmts = root.findall(".//FlexStatement")

    out = []
    for stmt in stmts:
        acct = stmt.get("accountId")
        cinv = stmt.find("ChangeInNAV")
        trades = stmt.findall("Trades/Trade")
        positions = stmt.findall("OpenPositions/OpenPosition")

        realized = sum(float(t.get("fifoPnlRealized") or 0) for t in trades)
        mtm = sum(float(t.get("mtmPnl") or 0) for t in trades)
        commissions = sum(abs(float(t.get("ibCommission") or 0)) for t in trades)
        unrealized = sum(float(p.get("fifoPnlUnrealized") or 0) for p in positions)

        def _cget(key: str) -> float | None:
            if cinv is None or cinv.get(key) in (None, ""):
                return None
            try:
                return float(cinv.get(key))
            except (TypeError, ValueError):
                return None

        out.append({
            "accountId": acct,
            "currency": cinv.get("currency") if cinv is not None else None,
            "fromDate": stmt.get("fromDate"),
            "toDate": stmt.get("toDate"),
            "navStart": round(_cget("startingValue"), 2) if _cget("startingValue") is not None else None,
            "navEnd": round(_cget("endingValue"), 2) if _cget("endingValue") is not None else None,
            "twr_pct": round(_cget("twr"), 4) if _cget("twr") is not None else None,
            "depositsWithdrawals": round(_cget("depositsWithdrawals"), 2) if _cget("depositsWithdrawals") is not None else None,
            "realizedPnl": round(realized, 2),
            "unrealizedPnl": round(unrealized, 2),
            "mtmPnl": round(mtm, 2),
            "commissions": round(commissions, 2),
            "dividends": round(_cget("dividends"), 2) if _cget("dividends") is not None else None,
            "interest": round(_cget("interest"), 2) if _cget("interest") is not None else None,
            "brokerFees": round(_cget("brokerFees"), 2) if _cget("brokerFees") is not None else None,
            "tradeCount": len(trades),
            "openPositionCount": len(positions),
        })
    _emit(out, args.format)
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """send + summary in one shot."""
    token = _resolve_token(args.token)
    qid = _resolve_query_id(args.query_id)
    xml = _send_and_poll(token, qid, max_wait=args.max_wait, poll_interval=args.poll_interval)
    path = _cache_save(xml, label=args.label)
    print(f"saved {path}", file=sys.stderr)
    # Now render summary
    args_ns = argparse.Namespace(file=str(path), format=args.format, account=None)
    return cmd_summary(args_ns)


def cmd_symbols(args: argparse.Namespace) -> int:
    """List all unique symbols traded in the Flex XML."""
    root = _load_xml(args)
    elems = _iter_section(root, "Trades", "Trade")
    seen = defaultdict(lambda: {"count": 0, "asset": "", "currency": ""})
    for e in elems:
        s = e.get("symbol")
        if not s:
            continue
        rec = seen[s]
        rec["count"] += 1
        rec["asset"] = e.get("assetCategory")
        rec["currency"] = e.get("currency")
    out = [{"symbol": k, **v} for k, v in seen.items()]
    out.sort(key=lambda r: r["count"], reverse=True)
    _emit(out, args.format)
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    root = _load_xml(args)
    out = [{"accountId": s.get("accountId"), "fromDate": s.get("fromDate"),
            "toDate": s.get("toDate"), "period": s.get("period")}
           for s in root.iter("FlexStatement")]
    _emit(out, args.format)
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    """Ad-hoc regex search within all attributes of all elements."""
    root = _load_xml(args)
    pattern = re.compile(args.pattern, re.IGNORECASE)
    out = []
    for stmt in root.iter("FlexStatement"):
        for sect in stmt:
            for leaf in sect.iter():
                if leaf is sect:
                    continue
                if any(pattern.search(v or "") for v in leaf.attrib.values()):
                    row = {"_section": sect.tag, "_tag": leaf.tag}
                    row.update(leaf.attrib)
                    out.append(row)
                    if args.limit and len(out) >= args.limit:
                        _emit(out, args.format)
                        return 0
    _emit(out, args.format)
    return 0


# ============================================================
# Argparse
# ============================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=["json", "csv", "table"], default="table")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trading_algo.flex_tool",
        description="IBKR Flex Web Service CLI — send, cache, parse, aggregate.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_text):
        sp = sub.add_parser(name, help=help_text)
        _add_common(sp)
        sp.set_defaults(func=fn)
        return sp

    # --- meta ---
    add("list-queries", cmd_list_queries, "Show configured Flex queries from env")

    s = add("send", cmd_send, "Send a Flex query, poll, and cache XML")
    s.add_argument("--query-id", default=None, help="Query id or alias (activity/trades/custom)")
    s.add_argument("--token", default=None)
    s.add_argument("--label", default="flex")
    s.add_argument("--max-wait", type=float, default=180.0)
    s.add_argument("--poll-interval", type=float, default=3.0)
    s.add_argument("--print", action="store_true")

    add("cached", cmd_cached, "List cached Flex XMLs")
    add("latest", cmd_latest, "Print path to latest cached XML")

    s = add("info", cmd_info, "Show metadata for cached XML (statements, sections)")
    s.add_argument("--file", default=None)

    s = add("accounts", cmd_accounts, "List accounts in cached XML")
    s.add_argument("--file", default=None)

    # --- sections ---
    s = add("trades", cmd_trades, "Trades section (with filters)")
    _add_filter_args(s)
    s.add_argument("--columns", default=None, help="Comma-sep subset of columns")

    s = add("open-positions", cmd_open_positions, "OpenPositions section")
    _add_filter_args(s)

    s = add("prior-positions", cmd_prior_positions, "PriorPeriodPositions section")
    _add_filter_args(s)

    s = add("cash-report", cmd_cash_report, "Cash report by currency")
    _add_filter_args(s)
    s.add_argument("--base-only", action="store_true")

    s = add("cash", cmd_cash_transactions, "CashTransactions (deposits, divs, etc.)")
    _add_filter_args(s)
    s.add_argument("--type", default=None, help="Substring filter on type")

    s = add("change-in-nav", cmd_change_in_nav, "ChangeInNAV per statement")
    s.add_argument("--file", default=None)
    s.add_argument("--account", default=None)
    s.add_argument("--drop-zero", action="store_true")

    s = add("change-in-position-values", cmd_change_in_position_values, "ChangeInPositionValues")
    _add_filter_args(s)

    s = add("complex-positions", cmd_complex_positions, "ComplexPositions")
    _add_filter_args(s)

    s = add("conversion-rates", cmd_conversion_rates, "FX conversion rates")
    s.add_argument("--file", default=None)
    s.add_argument("--currency", default=None)

    s = add("corporate-actions", cmd_corporate_actions, "Corporate actions")
    _add_filter_args(s)

    s = add("securities-info", cmd_securities_info, "SecuritiesInfo reference data")
    s.add_argument("--file", default=None)
    s.add_argument("--symbol", default=None)
    s.add_argument("--asset-category", default=None)

    s = add("stmt-funds", cmd_stmt_funds, "Statement of Funds")
    _add_filter_args(s)

    s = add("transfers", cmd_transfers, "Account/position transfers")
    _add_filter_args(s)

    s = add("slb-activities", cmd_slb_activities, "Stock loan borrow activities")
    _add_filter_args(s)

    # --- analytics ---
    s = add("pnl-by-symbol", cmd_pnl_by_symbol, "Aggregate P&L by symbol")
    _add_filter_args(s)
    s.add_argument("--top", type=int, default=None)

    s = add("pnl-by-account", cmd_pnl_by_account, "Aggregate P&L by account")
    s.add_argument("--file", default=None)

    s = add("commissions-total", cmd_commissions_total, "Total commissions paid")
    _add_filter_args(s)

    s = add("dividends", cmd_dividends, "Dividends from cash transactions")
    _add_filter_args(s)
    s.add_argument("--totals-only", action="store_true")

    s = add("interest", cmd_interest, "Interest from cash transactions")
    _add_filter_args(s)

    s = add("fees", cmd_fees, "Fees/commissions/tax from cash transactions")
    _add_filter_args(s)

    s = add("summary", cmd_summary, "One-shot per-account summary")
    s.add_argument("--file", default=None)
    s.add_argument("--account", default=None)

    s = add("symbols", cmd_symbols, "List all symbols traded")
    s.add_argument("--file", default=None)

    s = add("auto", cmd_auto, "Send query + cache + render summary in one shot")
    s.add_argument("--query-id", default=None)
    s.add_argument("--token", default=None)
    s.add_argument("--label", default="flex")
    s.add_argument("--max-wait", type=float, default=180.0)
    s.add_argument("--poll-interval", type=float, default=3.0)

    s = add("grep", cmd_grep, "Regex search across all attributes")
    s.add_argument("--file", default=None)
    s.add_argument("--pattern", required=True)
    s.add_argument("--limit", type=int, default=200)

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
