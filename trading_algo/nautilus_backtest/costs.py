"""Transaction costs for NautilusTrader backtests.

Two cost models share one interface (``ExecutionCosts``) so the fill model, the
fee model and the strategy's pre-trade sizing all price a trade the same way:

* ``RealisticCosts`` is the repo's cost model from
  ``trading_algo/backtest_v2/cost_model.py``. Fill price moves by half the
  estimated spread plus square-root market impact, and commission is the IBKR
  tiered schedule. Spread, average daily volume and volatility come only from
  sessions that finished before the trade (no look-ahead).
* ``LegacyCosts`` mirrors the repo's ``MultiStrategyBacktestRunner``: flat
  basis-point slippage and a flat per-share commission. It exists for parity
  checks against that runner.

``CostFeeModel`` and ``CostFillModel`` plug an ``ExecutionCosts`` into
NautilusTrader. Commission stays a separate cash charge. It is never baked into
the fill price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

import numpy as np
import pandas as pd
from nautilus_trader.execution import FeeModel, FillModel
from nautilus_trader.model import BookOrder, BookType, Money, OrderBook, OrderSide, Quantity

from trading_algo.backtest_v2.cost_model import (
    CostModelConfig,
    adjust_fill_price,
    compute_fill_cost,
    daily_effective_spread,
    ibkr_tiered_commission,
)
from trading_algo.nautilus_backtest.data import NEW_YORK

LOOKBACK_SESSIONS = 20


@dataclass(frozen=True)
class LaggedInputs:
    """Cost-model inputs known before a session opens."""

    spread_fraction: float
    adv: float
    daily_vol_bps: float


# Same conservative fallbacks as the repo's BacktestEngine: with no history,
# volume is treated as tiny (impact hits its participation cap) and daily vol
# as 200 bps. The spread falls back to the cost model's default.
NO_HISTORY = LaggedInputs(spread_fraction=0.0, adv=1.0, daily_vol_bps=200.0)
FALLBACK_DAILY_VOL_BPS = 200.0


def lagged_cost_inputs(
    frames: Mapping[str, pd.DataFrame],
    lookback: int = LOOKBACK_SESSIONS,
) -> dict[tuple[str, date], LaggedInputs]:
    """Spread, ADV and daily vol for each (symbol, session), from prior sessions only.

    Volatility uses log returns, as the repo's ``BacktestEngine`` does. A
    symbol's first session gets ``NO_HISTORY``, and volatility falls back to
    200 bps until two prior returns exist.
    """
    inputs: dict[tuple[str, date], LaggedInputs] = {}
    for symbol, frame in frames.items():
        ohlc = list(
            zip(frame["open"], frame["high"], frame["low"], frame["close"], strict=True),
        )
        volume = frame["volume"].to_numpy()
        returns = np.log(frame["close"]).diff().to_numpy()
        for i, ts in enumerate(frame.index):
            if i == 0:
                inputs[(symbol, ts.date())] = NO_HISTORY
                continue
            lo = max(0, i - lookback)
            prior_returns = returns[max(1, lo) : i]
            vol_bps = (
                float(np.std(prior_returns, ddof=1) * 1e4)
                if len(prior_returns) > 1
                else FALLBACK_DAILY_VOL_BPS
            )
            inputs[(symbol, ts.date())] = LaggedInputs(
                spread_fraction=daily_effective_spread(ohlc[max(0, i - 21) : i]),
                adv=float(np.mean(volume[lo:i])),
                daily_vol_bps=vol_bps,
            )
    return inputs


class ExecutionCosts:
    """Fill price and commission for one US-equity fill."""

    name = "base"

    def fill_price(
        self,
        symbol: str,
        day: date,
        side: str,
        quantity: float,
        reference_price: float,
    ) -> float:
        raise NotImplementedError

    def commission(self, side: str, quantity: float, price: float) -> float:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass
class RealisticCosts(ExecutionCosts):
    """The repo's spread + impact + IBKR tiered commission model."""

    inputs: Mapping[tuple[str, date], LaggedInputs]
    config: CostModelConfig = field(default_factory=CostModelConfig)
    name = "realistic"

    def fill_price(
        self,
        symbol: str,
        day: date,
        side: str,
        quantity: float,
        reference_price: float,
    ) -> float:
        lagged = self.inputs.get((symbol, day), NO_HISTORY)
        price_config = CostModelConfig(
            enable_spread=self.config.enable_spread,
            enable_impact=self.config.enable_impact,
            enable_commission=False,
            Y_impact=self.config.Y_impact,
            fees=self.config.fees,
            fallback_spread_bps=self.config.fallback_spread_bps,
            spread_cap_bps=self.config.spread_cap_bps,
        )
        cost = compute_fill_cost(
            side=side,
            quantity=quantity,
            fill_price=reference_price,
            spread_fraction=lagged.spread_fraction,
            adv=lagged.adv,
            daily_vol_bps=lagged.daily_vol_bps,
            config=price_config,
        )
        return adjust_fill_price(side, reference_price, cost, quantity)

    def commission(self, side: str, quantity: float, price: float) -> float:
        if not self.config.enable_commission:
            return 0.0
        return ibkr_tiered_commission(side, quantity, price, fees=self.config.fees)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enable_spread": self.config.enable_spread,
            "enable_impact": self.config.enable_impact,
            "enable_commission": self.config.enable_commission,
            "impact_Y": self.config.Y_impact,
            "fallback_spread_bps": self.config.fallback_spread_bps,
            "spread_cap_bps": self.config.spread_cap_bps,
            "lookback_sessions": LOOKBACK_SESSIONS,
        }


@dataclass
class LegacyCosts(ExecutionCosts):
    """Flat bps slippage and per-share commission, as in the repo's runner.

    Unlike the runner, slippage always moves against the order's actual side.
    """

    slippage_bps: float = 2.0
    commission_per_share: float = 0.0035
    name = "legacy"

    def fill_price(
        self,
        symbol: str,
        day: date,
        side: str,
        quantity: float,
        reference_price: float,
    ) -> float:
        sign = 1.0 if side == "BUY" else -1.0
        return reference_price * (1.0 + sign * self.slippage_bps / 1e4)

    def commission(self, side: str, quantity: float, price: float) -> float:
        return quantity * self.commission_per_share

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slippage_bps": self.slippage_bps,
            "commission_per_share": self.commission_per_share,
        }


def order_side(order: Any) -> str:
    return "BUY" if order.side == OrderSide.BUY else "SELL"


def session_day(ts_ns: int) -> date:
    """New York calendar day of a UTC nanosecond timestamp."""
    return pd.Timestamp(ts_ns, tz="UTC").tz_convert(NEW_YORK).date()


class CostFeeModel(FeeModel):
    """NautilusTrader fee model charging ``ExecutionCosts.commission``."""

    def __new__(cls, *args: Any, **kwargs: Any) -> CostFeeModel:
        return super().__new__(cls)

    def __init__(self, costs: ExecutionCosts) -> None:
        self.costs = costs

    def get_commission(self, order: Any, fill_quantity: Any, fill_px: Any, instrument: Any) -> Any:
        amount = self.costs.commission(order_side(order), float(fill_quantity), float(fill_px))
        return Money(amount, instrument.quote_currency)


class CostFillModel(FillModel):
    """NautilusTrader fill model pricing each order with ``ExecutionCosts.fill_price``.

    It hands the matching engine a one-level book at the cost-adjusted price,
    deep enough to fill the whole order at that single price.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> CostFillModel:
        return super().__new__(cls)

    def __init__(self, costs: ExecutionCosts) -> None:
        self.costs = costs

    def get_orderbook_for_fill_simulation(
        self,
        instrument: Any,
        order: Any,
        best_bid: Any,
        best_ask: Any,
    ) -> Any:
        side = order_side(order)
        reference = best_ask if side == "BUY" else best_bid
        if reference is None:
            return None
        price = self.costs.fill_price(
            instrument.id.symbol.value,
            session_day(order.ts_init),
            side,
            float(order.quantity),
            float(reference),
        )
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid simulated fill price {price} for {instrument.id}")
        book = OrderBook(instrument.id, BookType.L2_MBP)
        resting_side = OrderSide.SELL if side == "BUY" else OrderSide.BUY
        depth = Quantity.from_int(int(order.quantity) * 10 + 1)
        book.add(BookOrder(resting_side, instrument.make_price(price), depth, 1), 0, 0, 0)
        return book
