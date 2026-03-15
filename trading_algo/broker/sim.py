from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from trading_algo.broker.base import (
    AccountSnapshot,
    Bar,
    BracketOrderRequest,
    BracketOrderResult,
    MarketDataSnapshot,
    NewsArticle,
    NewsHeadline,
    NewsProvider,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    ScannerResult,
    validate_order_request,
)
from trading_algo.instruments import InstrumentSpec, validate_instrument

log = logging.getLogger(__name__)


@dataclass
class SimBroker:
    """
    Deterministic, in-memory broker for tests and local development.

    - Orders are immediately "Filled"
    - Market data is provided via `set_market_data(...)`
    """

    connected: bool = False
    orders: list[OrderRequest] = field(default_factory=list)
    market_data: dict[InstrumentSpec, MarketDataSnapshot] = field(default_factory=dict)
    historical_bars: dict[InstrumentSpec, list[Bar]] = field(default_factory=dict)
    account: str = "SIM"
    _positions: list[Position] = field(default_factory=list, repr=False)
    _account_values: dict[str, float] = field(
        default_factory=lambda: {"NetLiquidation": 100_000.0, "GrossPositionValue": 0.0, "AvailableFunds": 100_000.0},
        repr=False,
    )
    _statuses: dict[str, OrderStatus] = field(default_factory=dict, repr=False)

    def connect(self) -> None:
        self.connected = True
        log.info("SimBroker connected")

    def disconnect(self) -> None:
        self.connected = False
        log.info("SimBroker disconnected")

    def set_market_data(
        self,
        instrument: InstrumentSpec,
        *,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
        close: float | None = None,
        volume: float | None = None,
        timestamp_epoch_s: float | None = None,
    ) -> None:
        instrument = validate_instrument(instrument)
        self.market_data[instrument] = MarketDataSnapshot(
            instrument=instrument,
            bid=bid,
            ask=ask,
            last=last,
            close=close,
            volume=volume,
            timestamp_epoch_s=time.time() if timestamp_epoch_s is None else float(timestamp_epoch_s),
        )

    def set_historical_bars(self, instrument: InstrumentSpec, bars: list[Bar]) -> None:
        instrument = validate_instrument(instrument)
        self.historical_bars[instrument] = list(bars)

    def set_positions(self, positions: list[Position]) -> None:
        self._positions = list(positions)

    def set_account_values(self, values: dict[str, float]) -> None:
        self._account_values = dict(values)

    def get_market_data_snapshot(self, instrument: InstrumentSpec) -> MarketDataSnapshot:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        instrument = validate_instrument(instrument)
        if instrument not in self.market_data:
            raise KeyError(f"No market data set for {instrument}")
        return self.market_data[instrument]

    def get_positions(self) -> list[Position]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return list(self._positions)

    def get_account_snapshot(self) -> AccountSnapshot:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return AccountSnapshot(account=self.account, values=dict(self._account_values), timestamp_epoch_s=time.time())

    def place_order(self, req: OrderRequest) -> OrderResult:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        req = validate_order_request(req)
        self.orders.append(req)
        order_id = f"sim-{uuid.uuid4()}"
        log.info(
            "SIM order filled kind=%s symbol=%s side=%s qty=%s type=%s",
            req.instrument.kind,
            req.instrument.symbol,
            req.side,
            req.quantity,
            req.order_type,
        )
        status = OrderStatus(order_id=order_id, status="Filled", filled=req.quantity, remaining=0.0, avg_fill_price=None)
        self._statuses[order_id] = status
        return OrderResult(order_id=order_id, status=status.status)

    def modify_order(self, order_id: str, new_req: OrderRequest) -> OrderResult:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        if order_id not in self._statuses:
            raise KeyError(f"Unknown order_id: {order_id}")
        new_req = validate_order_request(new_req)
        self.orders.append(new_req)
        # For sim, treat modify as "Submitted" unless already Filled/Cancelled.
        cur = self._statuses[order_id]
        if cur.status in {"Filled", "Cancelled"}:
            return OrderResult(order_id=order_id, status=cur.status)
        self._statuses[order_id] = OrderStatus(order_id=order_id, status="Submitted", filled=cur.filled, remaining=cur.remaining, avg_fill_price=cur.avg_fill_price)
        return OrderResult(order_id=order_id, status="Submitted")

    def cancel_order(self, order_id: str) -> None:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        status = self._statuses.get(order_id)
        if status is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        if status.status in {"Filled", "Cancelled"}:
            return
        self._statuses[order_id] = OrderStatus(
            order_id=order_id,
            status="Cancelled",
            filled=status.filled,
            remaining=status.remaining,
            avg_fill_price=status.avg_fill_price,
        )

    def get_order_status(self, order_id: str) -> OrderStatus:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        status = self._statuses.get(order_id)
        if status is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        return status

    def list_open_order_statuses(self) -> list[OrderStatus]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return [st for st in self._statuses.values() if st.status not in {"Filled", "Cancelled"}]

    def _inject_order_status(self, status: OrderStatus) -> None:
        # Test helper (not part of Broker protocol).
        self._statuses[status.order_id] = status

    def place_bracket_order(self, req: BracketOrderRequest) -> BracketOrderResult:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        parent_id = f"sim-{uuid.uuid4()}"
        tp_id = f"sim-{uuid.uuid4()}"
        sl_id = f"sim-{uuid.uuid4()}"
        self._statuses[parent_id] = OrderStatus(parent_id, "Submitted", None, None, None)
        self._statuses[tp_id] = OrderStatus(tp_id, "Submitted", None, None, None)
        self._statuses[sl_id] = OrderStatus(sl_id, "Submitted", None, None, None)
        return BracketOrderResult(parent_order_id=parent_id, take_profit_order_id=tp_id, stop_loss_order_id=sl_id)

    def get_historical_bars(
        self,
        instrument: InstrumentSpec,
        *,
        end_datetime: str | None = None,
        duration: str,
        bar_size: str,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[Bar]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        instrument = validate_instrument(instrument)
        return list(self.historical_bars.get(instrument, []))

    def list_news_providers(self) -> list[NewsProvider]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return []

    def get_historical_news(
        self,
        instrument: InstrumentSpec,
        *,
        provider_codes: list[str] | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        max_results: int = 25,
    ) -> list[NewsHeadline]:
        _ = (instrument, provider_codes, start_datetime, end_datetime, max_results)
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return []

    def get_news_article(self, *, provider_code: str, article_id: str, format: str = "TEXT") -> NewsArticle:
        _ = (format,)
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return NewsArticle(provider_code=str(provider_code), article_id=str(article_id), text="")

    def scan_market(
        self,
        scan_code: str,
        *,
        instrument_type: str = "STK",
        location: str = "STK.US.MAJOR",
        num_rows: int = 25,
        above_price: float | None = None,
        below_price: float | None = None,
        above_volume: int | None = None,
        market_cap_above: float | None = None,
        market_cap_below: float | None = None,
    ) -> list[ScannerResult]:
        if not self.connected:
            raise RuntimeError("Broker is not connected")
        return []
