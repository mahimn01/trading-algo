from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from trading_algo.broker.base import Bar, MarketDataSnapshot
from trading_algo.instruments import InstrumentSpec
from trading_algo.orders import TradeIntent
from trading_algo.quant_core.strategies.options.wheel import (
    ShortOptionLeg,
    TradeEvent,
    WheelConfig,
    WheelStrategy,
)
from trading_algo.strategy.base import Strategy, StrategyContext

log = logging.getLogger(__name__)


def next_monthly_expiry(from_date: datetime, target_dte: int) -> str:
    """Find the 3rd Friday of the month closest to from_date + target_dte days.

    Returns YYYYMMDD string for InstrumentSpec.expiry.
    """
    target = from_date + timedelta(days=target_dte)

    def _third_friday(year: int, month: int) -> datetime:
        first_day = datetime(year, month, 1, tzinfo=target.tzinfo)
        dow = first_day.weekday()  # 0=Mon ... 4=Fri
        first_friday = first_day + timedelta(days=(4 - dow) % 7)
        return first_friday + timedelta(weeks=2)

    candidates: list[datetime] = []
    for month_offset in range(-1, 3):
        y = target.year
        m = target.month + month_offset
        if m < 1:
            y -= 1
            m += 12
        elif m > 12:
            y += 1
            m -= 12
        fri = _third_friday(y, m)
        if fri >= from_date:
            candidates.append(fri)

    if not candidates:
        candidates.append(_third_friday(target.year, target.month))

    best = min(candidates, key=lambda d: abs((d - target).days))
    return best.strftime("%Y%m%d")


@dataclass
class LiveWheelConfig:
    symbol: str
    wheel_config: WheelConfig = field(default_factory=WheelConfig)
    use_lmt_orders: bool = True
    lmt_offset_pct: float = 0.02
    price_history_bars: int = 300
    iv_rv_window: int = 30


class LiveWheelAdapter:
    """Adapter that makes the Wheel strategy executable via Engine."""

    name: str = "live_wheel"

    def __init__(self, cfg: LiveWheelConfig) -> None:
        self.cfg = cfg
        self.wheel = WheelStrategy(cfg.wheel_config)
        self._stock_instrument = InstrumentSpec(
            kind="STK", symbol=cfg.symbol.upper(), exchange="SMART", currency="USD",
        )
        self._price_history: deque[float] = deque(maxlen=cfg.price_history_bars)
        self._iv_series: deque[float] = deque(maxlen=cfg.price_history_bars)
        self._initialized = False
        self._current_short_spec: InstrumentSpec | None = None
        self._last_tick_date: str | None = None

    def on_tick(self, ctx: StrategyContext) -> list[TradeIntent]:
        now = datetime.fromtimestamp(ctx.now_epoch_s, tz=timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        if today_str == self._last_tick_date:
            return []
        self._last_tick_date = today_str

        try:
            snap = ctx.get_snapshot(self._stock_instrument)
        except Exception as exc:
            log.warning("Wheel: failed to get snapshot for %s: %s", self.cfg.symbol, exc)
            return []

        price = snap.last or snap.close
        if price is None or price <= 0:
            log.warning("Wheel: no usable price for %s (snap=%s)", self.cfg.symbol, snap)
            return []

        self._price_history.append(price)

        if len(self._price_history) < max(self.cfg.iv_rv_window, 2):
            log.info("Wheel: warming up price history (%d/%d)", len(self._price_history), self.cfg.iv_rv_window)
            return []

        iv = self._compute_iv()
        self._iv_series.append(iv)
        iv_rank = self._compute_iv_rank()

        events = self.wheel.on_bar(now, price, iv, iv_rank)
        intents = self._events_to_intents(events, now, snap)

        for intent in intents:
            log.info(
                "Wheel intent: %s %s %s qty=%s lmt=%s",
                intent.side, intent.instrument.kind, intent.instrument.symbol,
                intent.quantity, intent.limit_price,
            )

        return intents

    def _compute_iv(self) -> float:
        """Estimate annualized realized volatility from recent closes as IV proxy."""
        prices = list(self._price_history)
        window = min(self.cfg.iv_rv_window, len(prices) - 1)
        if window < 2:
            return 0.30

        recent = prices[-(window + 1):]
        log_returns = [
            math.log(recent[i + 1] / recent[i])
            for i in range(len(recent) - 1)
            if recent[i] > 0 and recent[i + 1] > 0
        ]
        if len(log_returns) < 2:
            return 0.30

        rv = float(np.std(log_returns)) * math.sqrt(252)
        return max(rv, 0.05)

    def _compute_iv_rank(self) -> float:
        """IV rank: percentile of current IV within historical IV series."""
        if len(self._iv_series) < 2:
            return 50.0

        current = self._iv_series[-1]
        iv_min = min(self._iv_series)
        iv_max = max(self._iv_series)
        iv_range = iv_max - iv_min
        if iv_range <= 0:
            return 50.0

        return max(0.0, min(100.0, (current - iv_min) / iv_range * 100.0))

    def _events_to_intents(
        self,
        events: list[TradeEvent],
        now: datetime,
        snap: MarketDataSnapshot,
    ) -> list[TradeIntent]:
        intents: list[TradeIntent] = []

        for ev in events:
            if ev.event_type == "sell_put":
                spec = self._build_option_spec(ev, now, "P")
                self._current_short_spec = spec
                premium = ev.details.get("premium", 0.0)
                intents.append(self._make_intent(spec, "SELL", ev.details.get("contracts", 1), premium, snap))

            elif ev.event_type == "sell_call":
                spec = self._build_option_spec(ev, now, "C")
                self._current_short_spec = spec
                premium = ev.details.get("premium", 0.0)
                intents.append(self._make_intent(spec, "SELL", ev.details.get("contracts", 1), premium, snap))

            elif ev.event_type.startswith("close_put") or ev.event_type.startswith("close_call"):
                if self._current_short_spec is not None:
                    buyback = ev.details.get("buyback", 0.0)
                    contracts = self.wheel.short_option.contracts if self.wheel.short_option else 1
                    intents.append(self._make_intent(self._current_short_spec, "BUY", contracts, buyback, snap))
                    self._current_short_spec = None
                else:
                    log.warning("Wheel: close event but no tracked short spec: %s", ev.event_type)

            elif ev.event_type in ("put_assigned", "called_away"):
                log.info("Wheel: %s — broker handles assignment automatically", ev.event_type)
                self._current_short_spec = None

            elif ev.event_type in ("put_expired_otm", "call_expired_otm"):
                log.info("Wheel: %s — option expired worthless", ev.event_type)
                self._current_short_spec = None

            else:
                log.debug("Wheel: unhandled event type %s", ev.event_type)

        return intents

    def _build_option_spec(self, ev: TradeEvent, now: datetime, right: str) -> InstrumentSpec:
        strike = ev.details.get("strike", 0.0)
        dte = ev.details.get("dte", self.cfg.wheel_config.target_dte)
        expiry = next_monthly_expiry(now, dte)

        return InstrumentSpec(
            kind="OPT",
            symbol=self.cfg.symbol.upper(),
            exchange="SMART",
            currency="USD",
            expiry=expiry,
            right=right,
            strike=float(strike),
            multiplier="100",
        )

    def _make_intent(
        self,
        spec: InstrumentSpec,
        side: str,
        contracts: int,
        reference_price: float,
        snap: MarketDataSnapshot,
    ) -> TradeIntent:
        if self.cfg.use_lmt_orders and reference_price > 0:
            if side == "SELL":
                limit_price = round(reference_price * (1 - self.cfg.lmt_offset_pct), 2)
            else:
                limit_price = round(reference_price * (1 + self.cfg.lmt_offset_pct), 2)
            limit_price = max(limit_price, 0.01)
            return TradeIntent(
                instrument=spec,
                side=side,
                quantity=float(contracts),
                order_type="LMT",
                limit_price=limit_price,
                tif="DAY",
            )

        return TradeIntent(
            instrument=spec,
            side=side,
            quantity=float(contracts),
            order_type="MKT",
            tif="DAY",
        )
