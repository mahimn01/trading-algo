from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Literal

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.broker.sim import SimBroker
from trading_algo.config import TradingConfig
from trading_algo.engine import Engine, default_risk_manager
from trading_algo.broker.base import OrderRequest
from trading_algo.instruments import InstrumentSpec, validate_instrument
from trading_algo.logging_setup import configure_logging
from trading_algo.orders import TradeIntent
from trading_algo.persistence import SqliteStore
from trading_algo.strategy.example import ExampleStrategy
from trading_algo.oms import OrderManager
from trading_algo.backtest.data import load_bars_csv
from trading_algo.backtest.runner import BacktestConfig, run_backtest
from trading_algo.backtest.export import ExportConfig, export_historical_bars
from trading_algo.backtest.validate import validate_bars


def _load_dotenv_if_present() -> None:
    # Minimal .env loader to avoid extra dependencies.
    if not os.path.exists(".env"):
        return
    with open(".env", "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Set only if missing/empty so shell overrides still work, but blanks get filled.
            if os.getenv(k) is None or os.getenv(k) == "":
                os.environ[k] = v


def _live_confirm_prompt(message: str) -> bool:
    """Interactive confirmation prompt for live account operations."""
    try:
        response = input(message).strip()
        return response == "YES"
    except (EOFError, KeyboardInterrupt):
        return False


def _make_broker(kind: Literal["ibkr", "sim"], cfg: TradingConfig):
    if kind == "sim":
        return SimBroker()
    if kind == "ibkr":
        return IBKRBroker(
            cfg.ibkr,
            require_paper=cfg.require_paper,
            allow_live=cfg.allow_live,
            live_confirm_callback=_live_confirm_prompt if cfg.allow_live else None,
        )
    raise ValueError(f"Unsupported broker: {kind}")


def _apply_cli_overrides(cfg: TradingConfig, args: argparse.Namespace) -> TradingConfig:
    ibkr = cfg.ibkr
    if args.ibkr_host is not None or args.ibkr_port is not None or args.ibkr_client_id is not None:
        ibkr = type(cfg.ibkr)(
            host=args.ibkr_host or cfg.ibkr.host,
            port=int(args.ibkr_port or cfg.ibkr.port),
            client_id=int(args.ibkr_client_id or cfg.ibkr.client_id),
        )

    dry_run = cfg.dry_run
    if getattr(args, "dry_run", False):
        dry_run = True
    if getattr(args, "no_dry_run", False):
        dry_run = False

    allow_live = cfg.allow_live or getattr(args, "allow_live", False)

    return TradingConfig(
        broker=cfg.broker,
        live_enabled=cfg.live_enabled,
        require_paper=not allow_live,
        allow_live=allow_live,
        dry_run=dry_run,
        order_token=cfg.order_token,
        confirm_token_required=cfg.confirm_token_required,
        db_path=cfg.db_path,
        poll_seconds=cfg.poll_seconds,
        ibkr=ibkr,
    )


def _assert_ibkr_order_authorized(cfg: TradingConfig, confirm_token: str | None) -> None:
    """
    Second safety gate for any IBKR order submission.
    """
    if cfg.dry_run:
        return
    if not cfg.live_enabled:
        raise SystemExit("Refusing to place IBKR orders with TRADING_LIVE_ENABLED=false (set it true explicitly).")
    if cfg.confirm_token_required:
        if not cfg.order_token:
            raise SystemExit("Refusing to place IBKR orders without TRADING_ORDER_TOKEN set (second confirmation gate).")
        if confirm_token != cfg.order_token:
            raise SystemExit("Refusing to place IBKR orders: --confirm-token does not match TRADING_ORDER_TOKEN.")


def _cmd_place_order(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        intent = TradeIntent(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type=args.type,
            limit_price=float(args.limit_price) if args.limit_price is not None else None,
            stop_price=float(args.stop_price) if args.stop_price is not None else None,
            tif=args.tif,
        )
        if cfg.dry_run:
            print(f"DRY RUN: would place {intent}")
            if store and run_id is not None:
                store.log_decision(run_id, strategy="cli.place-order", intent=intent, accepted=False, reason="dry_run")
            return 0
        # This command is intentionally "direct": it does not use Engine risk gating.
        req = OrderRequest(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type=args.type,
            limit_price=float(args.limit_price) if args.limit_price is not None else None,
            stop_price=float(args.stop_price) if args.stop_price is not None else None,
            tif=args.tif,
            outside_rth=bool(args.outside_rth),
            good_till_date=args.good_till_date,
            account=args.account,
            order_ref=args.order_ref,
            oca_group=args.oca_group,
            transmit=not bool(args.no_transmit),
        )
        result = broker.place_order(req)
        if store and run_id is not None:
            store.log_order(run_id, broker=args.broker, order_id=result.order_id, request=req, status=result.status)
            try:
                st = broker.get_order_status(result.order_id)
                store.log_order_status_event(run_id, args.broker, st)
            except Exception as exc:
                store.log_error(run_id, where="cli.place-order.status", message=str(exc))
        print(f"orderId={result.order_id} status={result.status}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        snap = broker.get_market_data_snapshot(instrument)
        print(
            f"{snap.instrument.kind} {snap.instrument.symbol} bid={snap.bid} ask={snap.ask} last={snap.last} "
            f"close={snap.close} volume={snap.volume} ts={snap.timestamp_epoch_s}"
        )
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_history(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        bars = broker.get_historical_bars(
            instrument,
            duration=args.duration,
            bar_size=args.bar_size,
            what_to_show=args.what_to_show,
            use_rth=bool(args.use_rth),
        )
        print(f"bars={len(bars)}")
        for b in bars[: min(len(bars), 5)]:
            print(f"ts={b.timestamp_epoch_s} o={b.open} h={b.high} l={b.low} c={b.close} v={b.volume}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    cfg = TradingConfig(
        broker=args.broker,
        live_enabled=cfg.live_enabled,
        require_paper=True,
        dry_run=cfg.dry_run,
        order_token=cfg.order_token,
        db_path=cfg.db_path,
        poll_seconds=args.poll_seconds or cfg.poll_seconds,
        ibkr=cfg.ibkr,
    )
    broker = _make_broker(args.broker, cfg)
    strategy = ExampleStrategy(symbol=args.symbol)
    engine = Engine(
        broker=broker,
        config=cfg,
        strategy=strategy,
        risk=default_risk_manager(),
        confirm_token=args.confirm_token,
    )

    if args.once:
        engine.run_once()
    else:
        engine.run_forever()
    return 0


def _cmd_order_status(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        st = broker.get_order_status(args.order_id)
        if store and run_id is not None:
            store.log_order_status_event(run_id, args.broker, st)
        print(f"orderId={st.order_id} status={st.status} filled={st.filled} remaining={st.remaining} avgFill={st.avg_fill_price}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_cancel_order(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        broker.cancel_order(args.order_id)
        if store and run_id is not None:
            try:
                st = broker.get_order_status(args.order_id)
                store.log_order_status_event(run_id, args.broker, st)
            except Exception as exc:
                store.log_error(run_id, where="cli.cancel-order.status", message=str(exc))
        print(f"cancelled orderId={args.order_id}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_modify_order(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)
    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        req = OrderRequest(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type=args.type,
            limit_price=float(args.limit_price) if args.limit_price is not None else None,
            stop_price=float(args.stop_price) if args.stop_price is not None else None,
            tif=args.tif,
            outside_rth=bool(args.outside_rth),
            good_till_date=args.good_till_date,
            account=args.account,
            order_ref=args.order_ref,
            oca_group=args.oca_group,
            transmit=not bool(args.no_transmit),
        )
        if cfg.dry_run:
            print(f"DRY RUN: would modify orderId={args.order_id} -> {req}")
            return 0

        res = broker.modify_order(args.order_id, req)
        if store and run_id is not None:
            store.log_order(run_id, broker=args.broker, order_id=res.order_id, request=req, status=res.status)
            try:
                st = broker.get_order_status(res.order_id)
                store.log_order_status_event(run_id, args.broker, st)
            except Exception as exc:
                store.log_error(run_id, where="cli.modify-order.status", message=str(exc))
        print(f"orderId={res.order_id} status={res.status}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_place_bracket(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker == "ibkr":
        _assert_ibkr_order_authorized(cfg, args.confirm_token)

    broker = _make_broker(args.broker, cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        if cfg.dry_run:
            print(
                "DRY RUN: would place bracket "
                f"{instrument.kind} {instrument.symbol} side={args.side} qty={args.qty} "
                f"entry={args.entry_limit} tp={args.take_profit} sl={args.stop_loss}"
            )
            if store and run_id is not None:
                store.log_error(run_id, where="cli.place-bracket", message="dry_run")
            return 0

        from trading_algo.broker.base import BracketOrderRequest

        req = BracketOrderRequest(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            entry_limit_price=float(args.entry_limit),
            take_profit_limit_price=float(args.take_profit),
            stop_loss_stop_price=float(args.stop_loss),
            tif=args.tif,
        )
        res = broker.place_bracket_order(req)
        if store and run_id is not None:
            store.log_error(run_id, where="cli.place-bracket", message=f"placed parent={res.parent_order_id}")
        print(
            f"parent={res.parent_order_id} takeProfit={res.take_profit_order_id} stopLoss={res.stop_loss_order_id}"
        )
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_paper_smoke(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker != "ibkr":
        raise SystemExit("paper-smoke is only supported with --broker ibkr")

    broker = _make_broker("ibkr", cfg)
    store = SqliteStore(cfg.db_path) if cfg.db_path else None
    run_id = store.start_run(cfg) if store else None
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        snap = broker.get_market_data_snapshot(instrument)
        print(
            f"OK: connected paper account, snapshot {snap.instrument.kind} {snap.instrument.symbol} "
            f"bid={snap.bid} ask={snap.ask} last={snap.last} ts={snap.timestamp_epoch_s}"
        )

        if not args.order_test:
            return 0

        _assert_ibkr_order_authorized(cfg, args.confirm_token)

        px = snap.last or snap.close
        if px is None or px <= 0:
            raise SystemExit("Cannot run order-test without a usable last/close price from snapshot")

        # Place a limit order far from market and cancel shortly after.
        if args.side == "BUY":
            limit_price = max(0.01, float(px) * 0.5)
        else:
            limit_price = float(px) * 1.5

        intent = TradeIntent(
            instrument=instrument,
            side=args.side,
            quantity=float(args.qty),
            order_type="LMT",
            limit_price=limit_price,
        )
        if cfg.dry_run:
            print(f"DRY RUN: would place+cancel smoke-test order {intent}")
            return 0

        res = broker.place_order(intent.to_order_request())
        print(f"Placed smoke-test order orderId={res.order_id} status={res.status}; cancelling...")
        broker.cancel_order(res.order_id)
        st = broker.get_order_status(res.order_id)
        print(f"After cancel: orderId={st.order_id} status={st.status}")
        return 0
    finally:
        broker.disconnect()
        if store and run_id is not None:
            store.end_run(run_id)
        if store:
            store.close()


def _cmd_oms_reconcile(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("oms-reconcile requires TRADING_DB_PATH to be set")
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        oms = OrderManager(broker, cfg, confirm_token=args.confirm_token)
        try:
            res = oms.reconcile()
            print(f"reconciled={len(res)}")
            for oid, st in res.items():
                print(f"orderId={oid} status={st}")
        finally:
            oms.close()
        return 0
    finally:
        broker.disconnect()


def _cmd_oms_track(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if not cfg.db_path:
        raise SystemExit("oms-track requires TRADING_DB_PATH to be set")
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        oms = OrderManager(broker, cfg, confirm_token=args.confirm_token)
        try:
            oms.reconcile()
            oms.track_open_orders(poll_seconds=float(args.poll_seconds), timeout_seconds=float(args.timeout_seconds) if args.timeout_seconds else None)
            print("ok")
        finally:
            oms.close()
        return 0
    finally:
        broker.disconnect()


def _cmd_backtest(args: argparse.Namespace) -> int:
    instrument = validate_instrument(
        InstrumentSpec(
            kind=args.kind,
            symbol=args.symbol,
            exchange=args.exchange,
            currency=args.currency,
            expiry=args.expiry,
            right=getattr(args, "right", None),
            strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
            multiplier=getattr(args, "multiplier", None),
        )
    )
    series = load_bars_csv(args.csv, instrument)
    cfg = BacktestConfig(
        initial_cash=float(args.initial_cash),
        commission_per_order=float(args.commission_per_order),
        slippage_bps=float(args.slippage_bps),
        spread=float(args.spread),
        db_path=args.db_path,
    )
    res = run_backtest(ExampleStrategy(symbol=instrument.symbol), instrument, series.bars, cfg)
    print(f"start={res.start_equity} end={res.end_equity} returnPct={res.return_pct}")
    return 0


def _cmd_export_history(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    if args.broker != "ibkr":
        raise SystemExit("export-history currently supports only --broker ibkr")
    import os

    if os.path.exists(args.out_csv) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.out_csv} (use --overwrite)")
    broker = _make_broker("ibkr", cfg)
    broker.connect()
    try:
        instrument = validate_instrument(
            InstrumentSpec(
                kind=args.kind,
                symbol=args.symbol,
                exchange=args.exchange,
                currency=args.currency,
                expiry=args.expiry,
                right=getattr(args, "right", None),
                strike=(float(getattr(args, "strike", 0.0)) if getattr(args, "strike", None) is not None else None),
                multiplier=getattr(args, "multiplier", None),
            )
        )
        export_cfg = ExportConfig(
            duration_per_call=args.duration_per_call,
            bar_size=args.bar_size,
            what_to_show=args.what_to_show,
            use_rth=bool(args.use_rth),
            pacing_sleep_seconds=float(args.pacing_sleep_seconds),
            max_calls=int(args.max_calls),
        )
        bars = export_historical_bars(
            broker,
            instrument,
            out_csv_path=args.out_csv,
            cfg=export_cfg,
            end_datetime=args.end_datetime,
        )
        if args.validate:
            issues = validate_bars(bars)
            errors = [i for i in issues if i.level == "error"]
            for i in issues:
                print(f"{i.level}: {i.message}")
            if errors:
                raise SystemExit("bar validation failed")
        print(f"wrote={args.out_csv} bars={len(bars)}")
        return 0
    finally:
        broker.disconnect()


def _cmd_llm_run(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)

    from trading_algo.llm.config import LLMConfig
    from trading_algo.llm.gemini import GeminiClient
    from trading_algo.llm.trader import LLMTrader
    from trading_algo.risk import RiskLimits, RiskManager

    llm_cfg = LLMConfig.from_env()
    if llm_cfg.provider != "gemini":
        raise SystemExit("LLM_PROVIDER must be 'gemini' for llm-run")
    if not llm_cfg.enabled:
        raise SystemExit("LLM_ENABLED must be true for llm-run")
    if not llm_cfg.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY must be set for llm-run")
    if not str(llm_cfg.gemini_model).startswith("gemini-3"):
        raise SystemExit(
            f"Refusing to run with GEMINI_MODEL={llm_cfg.gemini_model!r}; set GEMINI_MODEL to a Gemini 3 model id "
            "(e.g. gemini-3-pro-preview or gemini-3-flash-preview)."
        )
    if not llm_cfg.allowed_symbols():
        raise SystemExit("LLM_ALLOWED_SYMBOLS must be set (comma-separated)")

    if args.broker == "ibkr":
        # Orders go through OMS gates too, but keep explicit CLI auth for clarity.
        _assert_ibkr_order_authorized(cfg, args.confirm_token)

    broker = _make_broker(args.broker, cfg)
    trader = LLMTrader(
        broker=broker,
        trading=cfg,
        llm=llm_cfg,
        client=GeminiClient(api_key=llm_cfg.gemini_api_key, model=llm_cfg.normalized_gemini_model()),
        risk=RiskManager(RiskLimits()),
        confirm_token=args.confirm_token,
        sleep_seconds=float(args.sleep_seconds),
        max_ticks=(int(args.max_ticks) if args.max_ticks is not None else None),
    )
    if args.once:
        trader.run_once()
    else:
        trader.run()
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(TradingConfig.from_env(), args)
    broker = _make_broker(args.broker, cfg)
    broker.connect()
    try:
        results = broker.scan_market(
            args.scan_type,
            instrument_type=args.instrument_type,
            location=args.location,
            num_rows=int(args.max_results),
            above_price=float(args.min_price) if args.min_price is not None else None,
            below_price=float(args.max_price) if args.max_price is not None else None,
            above_volume=int(args.min_volume) if args.min_volume is not None else None,
            market_cap_above=float(args.min_market_cap) if args.min_market_cap is not None else None,
            market_cap_below=float(args.max_market_cap) if args.max_market_cap is not None else None,
        )
        print(f"results={len(results)} scan={args.scan_type}")
        for r in results:
            extra_str = ""
            if r.extra:
                name = r.extra.get("longName", "")
                industry = r.extra.get("industry", "")
                if name:
                    extra_str += f" name={name}"
                if industry:
                    extra_str += f" industry={industry}"
            print(f"  #{r.rank}: {r.instrument.symbol} ({r.instrument.kind}){extra_str}")
        return 0
    finally:
        broker.disconnect()


def _cmd_chat(args: argparse.Namespace) -> int:
    from trading_algo.llm.chat import main as chat_main

    argv: list[str] = []
    if args.broker is not None:
        argv += ["--broker", str(args.broker)]
    if args.confirm_token is not None:
        argv += ["--confirm-token", str(args.confirm_token)]
    if args.ibkr_host is not None:
        argv += ["--ibkr-host", str(args.ibkr_host)]
    if args.ibkr_port is not None:
        argv += ["--ibkr-port", str(args.ibkr_port)]
    if args.ibkr_client_id is not None:
        argv += ["--ibkr-client-id", str(args.ibkr_client_id)]
    if bool(args.no_stream):
        argv += ["--no-stream"]
    if bool(args.show_raw):
        argv += ["--show-raw"]
    if bool(args.no_color):
        argv += ["--no-color"]
    if bool(getattr(args, "quiet_ibkr_logs", False)):
        argv += ["--quiet-ibkr-logs"]
    if getattr(args, "ui", None):
        argv += ["--ui", str(args.ui)]
    return int(chat_main(argv))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading-algo", description="IBKR paper trading algo skeleton")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--ibkr-host", default=None, help="Override IBKR host (default from env/.env)")
    p.add_argument("--ibkr-port", default=None, help="Override IBKR port (default from env/.env)")
    p.add_argument("--ibkr-client-id", default=None, help="Override IBKR clientId (default from env/.env)")
    p.add_argument(
        "--confirm-token",
        default=None,
        help="Must match TRADING_ORDER_TOKEN if TRADING_CONFIRM_TOKEN_REQUIRED=true",
    )
    p.add_argument("--dry-run", action="store_true", help="Stage orders only (no sends), overrides TRADING_DRY_RUN")
    p.add_argument("--no-dry-run", action="store_true", help="Allow sending orders, overrides TRADING_DRY_RUN")
    p.add_argument(
        "--allow-live", action="store_true",
        help="Allow connecting to LIVE (non-paper) IBKR accounts. "
             "All order operations will require interactive YES confirmation.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    place = sub.add_parser("place-order", help="Place a single test order")
    place.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    place.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    place.add_argument("--symbol", required=True)
    place.add_argument("--exchange", default=None)
    place.add_argument("--currency", default=None)
    place.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    place.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    place.add_argument("--strike", default=None, help="OPT only: strike price")
    place.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    place.add_argument("--side", choices=["BUY", "SELL"], required=True)
    place.add_argument("--qty", required=True)
    place.add_argument("--type", choices=["MKT", "LMT", "STP", "STPLMT"], default="MKT")
    place.add_argument("--limit-price", default=None)
    place.add_argument("--stop-price", default=None)
    place.add_argument("--tif", default="DAY", help="DAY|GTC|GTD (if GTD, set --good-till-date)")
    place.add_argument("--good-till-date", default=None, help="IBKR GTD time string (e.g. 20260116 09:30:00)")
    place.add_argument("--outside-rth", action="store_true", help="Allow execution outside regular trading hours")
    place.add_argument("--account", default=None)
    place.add_argument("--order-ref", default=None)
    place.add_argument("--oca-group", default=None)
    place.add_argument("--no-transmit", action="store_true", help="Create order with transmit=false (advanced)")
    place.set_defaults(func=_cmd_place_order)

    snap = sub.add_parser("snapshot", help="Fetch a market data snapshot")
    snap.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    snap.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    snap.add_argument("--symbol", required=True)
    snap.add_argument("--exchange", default=None)
    snap.add_argument("--currency", default=None)
    snap.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    snap.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    snap.add_argument("--strike", default=None, help="OPT only: strike price")
    snap.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    snap.set_defaults(func=_cmd_snapshot)

    hist = sub.add_parser("history", help="Fetch historical bars (IBKR reqHistoricalData)")
    hist.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    hist.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    hist.add_argument("--symbol", required=True)
    hist.add_argument("--exchange", default=None)
    hist.add_argument("--currency", default=None)
    hist.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    hist.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    hist.add_argument("--strike", default=None, help="OPT only: strike price")
    hist.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    hist.add_argument("--duration", default="1 D", help="IBKR durationStr (e.g. '1 D', '2 W')")
    hist.add_argument("--bar-size", default="5 mins", help="IBKR barSizeSetting (e.g. '1 min', '5 mins')")
    hist.add_argument("--what-to-show", default="TRADES")
    hist.add_argument("--use-rth", action="store_true")
    hist.set_defaults(func=_cmd_history)

    run = sub.add_parser("run", help="Run example strategy loop")
    run.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    run.add_argument("--symbol", default="AAPL")
    run.add_argument("--poll-seconds", type=int, default=None)
    run.add_argument("--once", action="store_true")
    run.set_defaults(func=_cmd_run)

    status = sub.add_parser("order-status", help="Get order status by orderId")
    status.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    status.add_argument("--order-id", required=True)
    status.set_defaults(func=_cmd_order_status)

    cancel = sub.add_parser("cancel-order", help="Cancel order by orderId")
    cancel.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    cancel.add_argument("--order-id", required=True)
    cancel.set_defaults(func=_cmd_cancel_order)

    mod = sub.add_parser("modify-order", help="Modify an existing order by orderId")
    mod.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    mod.add_argument("--order-id", required=True)
    mod.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    mod.add_argument("--symbol", required=True)
    mod.add_argument("--exchange", default=None)
    mod.add_argument("--currency", default=None)
    mod.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    mod.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    mod.add_argument("--strike", default=None, help="OPT only: strike price")
    mod.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    mod.add_argument("--side", choices=["BUY", "SELL"], required=True)
    mod.add_argument("--qty", required=True)
    mod.add_argument("--type", choices=["MKT", "LMT", "STP", "STPLMT"], default="LMT")
    mod.add_argument("--limit-price", default=None)
    mod.add_argument("--stop-price", default=None)
    mod.add_argument("--tif", default="DAY")
    mod.add_argument("--good-till-date", default=None)
    mod.add_argument("--outside-rth", action="store_true")
    mod.add_argument("--account", default=None)
    mod.add_argument("--order-ref", default=None)
    mod.add_argument("--oca-group", default=None)
    mod.add_argument("--no-transmit", action="store_true")
    mod.set_defaults(func=_cmd_modify_order)

    bracket = sub.add_parser("place-bracket", help="Place a bracket order (LMT entry + TP LMT + SL STP)")
    bracket.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    bracket.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    bracket.add_argument("--symbol", required=True)
    bracket.add_argument("--exchange", default=None)
    bracket.add_argument("--currency", default=None)
    bracket.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    bracket.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    bracket.add_argument("--strike", default=None, help="OPT only: strike price")
    bracket.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    bracket.add_argument("--side", choices=["BUY", "SELL"], required=True)
    bracket.add_argument("--qty", required=True)
    bracket.add_argument("--entry-limit", required=True)
    bracket.add_argument("--take-profit", required=True)
    bracket.add_argument("--stop-loss", required=True)
    bracket.add_argument("--tif", default="DAY")
    bracket.set_defaults(func=_cmd_place_bracket)

    smoke = sub.add_parser("paper-smoke", help="Paper connectivity smoke test (connect + verify paper + snapshot; optional place+cancel)")
    smoke.add_argument("--broker", choices=["ibkr"], default="ibkr")
    smoke.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    smoke.add_argument("--symbol", default="AAPL")
    smoke.add_argument("--exchange", default=None)
    smoke.add_argument("--currency", default=None)
    smoke.add_argument("--expiry", default=None, help="FUT: YYYYMM or YYYYMMDD, OPT: YYYYMMDD")
    smoke.add_argument("--right", choices=["C", "P"], default=None, help="OPT only: C or P")
    smoke.add_argument("--strike", default=None, help="OPT only: strike price")
    smoke.add_argument("--multiplier", default=None, help="OPT only: contract multiplier (default 100)")
    smoke.add_argument("--order-test", action="store_true", help="Place a tiny LMT order and cancel it (requires TRADING_LIVE_ENABLED + token)")
    smoke.add_argument("--side", choices=["BUY", "SELL"], default="BUY")
    smoke.add_argument("--qty", default="1")
    smoke.set_defaults(func=_cmd_paper_smoke)

    rec = sub.add_parser("oms-reconcile", help="Reconcile open orders from TRADING_DB_PATH with broker open orders")
    rec.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    rec.set_defaults(func=_cmd_oms_reconcile)

    track = sub.add_parser("oms-track", help="Poll and persist order status transitions until terminal/timeout")
    track.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    track.add_argument("--poll-seconds", default="1.0")
    track.add_argument("--timeout-seconds", default=None)
    track.set_defaults(func=_cmd_oms_track)

    bt = sub.add_parser("backtest", help="Run a deterministic historical backtest from a CSV file")
    bt.add_argument("--csv", required=True, help="CSV with columns: timestamp,open,high,low,close[,volume]")
    bt.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    bt.add_argument("--symbol", required=True)
    bt.add_argument("--exchange", default=None)
    bt.add_argument("--currency", default=None)
    bt.add_argument("--expiry", default=None)
    bt.add_argument("--right", choices=["C", "P"], default=None)
    bt.add_argument("--strike", default=None)
    bt.add_argument("--multiplier", default=None)
    bt.add_argument("--initial-cash", type=float, default=100000.0)
    bt.add_argument("--commission-per-order", type=float, default=0.0)
    bt.add_argument("--slippage-bps", type=float, default=0.0)
    bt.add_argument("--spread", type=float, default=0.0)
    bt.add_argument("--db-path", default=None)
    bt.set_defaults(func=_cmd_backtest)

    exp = sub.add_parser("export-history", help="Export IBKR historical bars to a backtest CSV")
    exp.add_argument("--broker", choices=["ibkr"], default="ibkr")
    exp.add_argument("--kind", choices=["STK", "FUT", "FX", "OPT"], default="STK")
    exp.add_argument("--symbol", required=True)
    exp.add_argument("--exchange", default=None)
    exp.add_argument("--currency", default=None)
    exp.add_argument("--expiry", default=None)
    exp.add_argument("--right", choices=["C", "P"], default=None)
    exp.add_argument("--strike", default=None)
    exp.add_argument("--multiplier", default=None)
    exp.add_argument("--out-csv", required=True)
    exp.add_argument("--overwrite", action="store_true")
    exp.add_argument("--bar-size", default="5 mins")
    exp.add_argument("--duration-per-call", default="30 D")
    exp.add_argument("--what-to-show", default="TRADES")
    exp.add_argument("--use-rth", action="store_true")
    exp.add_argument("--end-datetime", default=None, help="IBKR endDateTime; empty means now. Epoch/ISO are accepted.")
    exp.add_argument("--pacing-sleep-seconds", default="0.25")
    exp.add_argument("--max-calls", default="500")
    exp.add_argument("--validate", action="store_true")
    exp.set_defaults(func=_cmd_export_history)

    llm_run = sub.add_parser("llm-run", help="Run the LLM trader loop (paper-only enforced)")
    llm_run.add_argument("--broker", choices=["ibkr", "sim"], default="sim")
    llm_run.add_argument("--sleep-seconds", type=float, default=5.0)
    llm_run.add_argument("--max-ticks", type=int, default=None)
    llm_run.add_argument("--once", action="store_true", help="Run exactly one LLM tick")
    llm_run.set_defaults(func=_cmd_llm_run)

    scan = sub.add_parser("scan", help="Run IBKR market-wide scanner (e.g. top gainers, most active, high options volume)")
    scan.add_argument("--broker", choices=["ibkr", "sim"], default="ibkr")
    scan.add_argument("--scan-type", default="MOST_ACTIVE",
        help="IBKR scan code: MOST_ACTIVE, TOP_PERC_GAIN, TOP_PERC_LOSE, "
             "HOT_BY_VOLUME, HIGH_OPT_IMP_VOLAT, HOT_BY_OPT_VOLUME, "
             "HIGH_DIVIDEND_YIELD_IB, TOP_OPEN_PERC_GAIN, TOP_OPEN_PERC_LOSE, "
             "HIGH_VS_13W_HL, LOW_VS_13W_HL, MOST_ACTIVE_USD, HIGH_SYNTH_BID_REV_NAT")
    scan.add_argument("--instrument-type", default="STK", help="STK, FUT, IND, etc.")
    scan.add_argument("--location", default="STK.US.MAJOR",
        help="Scanner location: STK.US.MAJOR, STK.US, STK.NASDAQ, STK.NYSE, STK.AMEX")
    scan.add_argument("--max-results", type=int, default=25, help="Max results 1-50")
    scan.add_argument("--min-price", type=float, default=None, help="Min stock price filter")
    scan.add_argument("--max-price", type=float, default=None, help="Max stock price filter")
    scan.add_argument("--min-volume", type=int, default=None, help="Min volume filter")
    scan.add_argument("--min-market-cap", type=float, default=None, help="Min market cap in USD (e.g. 1000000000 for $1B)")
    scan.add_argument("--max-market-cap", type=float, default=None, help="Max market cap in USD")
    scan.set_defaults(func=_cmd_scan)

    chat = sub.add_parser("chat", help="Interactive terminal chat (Gemini + OMS tools)")
    chat.add_argument("--broker", choices=["ibkr", "sim"], default=None)
    chat.add_argument("--no-stream", action="store_true")
    chat.add_argument("--show-raw", action="store_true")
    chat.add_argument("--no-color", action="store_true")
    chat.add_argument("--quiet-ibkr-logs", action="store_true", dest="quiet_ibkr_logs")
    chat.add_argument("--ui", choices=["auto", "plain", "rich", "tui"], default="auto")
    chat.set_defaults(func=_cmd_chat)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_present()
    cfg = TradingConfig.from_env()

    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)

    # prompt_toolkit runs in full-screen mode for `chat --ui tui`; any stdout/stderr writes from
    # logging will corrupt the terminal UI. Route logs to a file instead.
    tui_mode = bool(getattr(args, "ui", None) == "tui" and getattr(args, "func", None) == _cmd_chat)
    if tui_mode:
        log_path = os.getenv("TUI_LOG_PATH", "logs/tui.log")
        configure_logging(level=log_level, log_file=log_path, console=False)
    else:
        configure_logging(level=log_level)
    logging.getLogger(__name__).debug("Loaded config: %s", cfg)

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
