from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading_algo.instruments import InstrumentSpec, validate_instrument


@dataclass(frozen=True)
class OrderRequest:
    instrument: InstrumentSpec
    side: str = "BUY"  # BUY|SELL
    quantity: float = 1.0
    order_type: str = "MKT"  # MKT|LMT|STP|STPLMT
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "DAY"
    outside_rth: bool = False
    good_till_date: str | None = None  # IBKR GTD format string
    account: str | None = None
    order_ref: str | None = None
    oca_group: str | None = None
    transmit: bool = True

    def normalized(self) -> "OrderRequest":
        instrument = validate_instrument(self.instrument)
        side = self.side.strip().upper()
        order_type = self.order_type.strip().upper()
        tif = self.tif.strip().upper()
        return OrderRequest(
            instrument=instrument,
            side=side,
            quantity=float(self.quantity),
            order_type=order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            tif=tif,
            outside_rth=bool(self.outside_rth),
            good_till_date=(self.good_till_date.strip() if self.good_till_date else None),
            account=(self.account.strip() if self.account else None),
            order_ref=(self.order_ref.strip() if self.order_ref else None),
            oca_group=(self.oca_group.strip() if self.oca_group else None),
            transmit=bool(self.transmit),
        )


def validate_order_request(req: OrderRequest) -> OrderRequest:
    req = req.normalized()
    if req.side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if req.quantity <= 0:
        raise ValueError("quantity must be positive")
    if req.order_type not in {"MKT", "LMT"}:
        if req.order_type not in {"STP", "STPLMT"}:
            raise ValueError("order_type must be MKT, LMT, STP, or STPLMT")
    if req.order_type == "LMT":
        if req.limit_price is None:
            raise ValueError("limit_price is required for LMT orders")
        if req.limit_price <= 0:
            raise ValueError("limit_price must be positive")
    if req.order_type == "STP":
        if req.stop_price is None or req.stop_price <= 0:
            raise ValueError("stop_price is required and must be positive for STP orders")
    if req.order_type == "STPLMT":
        if req.stop_price is None or req.stop_price <= 0:
            raise ValueError("stop_price is required and must be positive for STPLMT orders")
        if req.limit_price is None or req.limit_price <= 0:
            raise ValueError("limit_price is required and must be positive for STPLMT orders")
    if not req.tif:
        raise ValueError("tif is required")
    if req.tif == "GTD" and not req.good_till_date:
        raise ValueError("good_till_date is required for GTD tif")
    return req


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: str


class Broker(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def place_order(self, req: OrderRequest) -> OrderResult: ...
    def get_market_data_snapshot(self, instrument: InstrumentSpec) -> "MarketDataSnapshot": ...
    def get_positions(self) -> list["Position"]: ...
    def get_account_snapshot(self) -> "AccountSnapshot": ...
    def cancel_order(self, order_id: str) -> None: ...
    def get_order_status(self, order_id: str) -> "OrderStatus": ...
    def place_bracket_order(self, req: "BracketOrderRequest") -> "BracketOrderResult": ...
    def modify_order(self, order_id: str, new_req: OrderRequest) -> OrderResult: ...
    def list_open_order_statuses(self) -> list["OrderStatus"]: ...
    def get_historical_bars(
        self,
        instrument: InstrumentSpec,
        *,
        end_datetime: str | None = None,
        duration: str,
        bar_size: str,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list["Bar"]: ...

    # News / Research (optional, broker-dependent).
    def list_news_providers(self) -> list["NewsProvider"]: ...
    def get_historical_news(
        self,
        instrument: InstrumentSpec,
        *,
        provider_codes: list[str] | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        max_results: int = 25,
    ) -> list["NewsHeadline"]: ...
    def get_news_article(self, *, provider_code: str, article_id: str, format: str = "TEXT") -> "NewsArticle": ...

    # Market Scanner (optional, broker-dependent).
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
    ) -> list["ScannerResult"]: ...


@dataclass(frozen=True)
class MarketDataSnapshot:
    instrument: InstrumentSpec
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None
    volume: float | None
    timestamp_epoch_s: float


@dataclass(frozen=True)
class Position:
    account: str
    instrument: InstrumentSpec
    quantity: float
    avg_cost: float | None
    timestamp_epoch_s: float


@dataclass(frozen=True)
class AccountSnapshot:
    account: str
    # Common tags include: NetLiquidation, GrossPositionValue, AvailableFunds,
    # MaintMarginReq, InitMarginReq, UnrealizedPnL, RealizedPnL.
    values: dict[str, float]
    timestamp_epoch_s: float


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    status: str
    filled: float | None
    remaining: float | None
    avg_fill_price: float | None


@dataclass(frozen=True)
class BracketOrderRequest:
    instrument: InstrumentSpec
    side: str  # BUY|SELL (entry side)
    quantity: float
    entry_limit_price: float
    take_profit_limit_price: float
    stop_loss_stop_price: float
    tif: str = "DAY"


@dataclass(frozen=True)
class BracketOrderResult:
    parent_order_id: str
    take_profit_order_id: str
    stop_loss_order_id: str


@dataclass(frozen=True)
class Bar:
    timestamp_epoch_s: float
    open: float
    high: float
    low: float
    close: float
    volume: float | None


@dataclass(frozen=True)
class NewsProvider:
    code: str
    name: str


@dataclass(frozen=True)
class NewsHeadline:
    timestamp: str
    provider_code: str
    article_id: str
    headline: str


@dataclass(frozen=True)
class NewsArticle:
    provider_code: str
    article_id: str
    text: str


@dataclass(frozen=True)
class ScannerResult:
    """A single result from an IBKR market scanner."""
    instrument: InstrumentSpec
    rank: int
    scan_code: str
    close: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    extra: dict[str, str] | None = None
