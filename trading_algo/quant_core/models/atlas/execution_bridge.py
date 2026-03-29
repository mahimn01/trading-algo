"""Maps ATLAS continuous action vector to concrete trade decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TradeDecision:
    strategy: str  # "wheel_csp" | "wheel_cc" | "pmcc" | "put_spread" | "cash" | "hold_stock"
    delta: float
    dte: int
    leverage: float
    profit_target: float
    direction: float
    position_size_dollars: float
    reason: str


def compute_position_size(
    leverage: float,
    capital: float,
    current_vol: float,
    vol_target: float = 0.15,
    max_position_pct: float = 0.25,
) -> float:
    """Volatility-targeted position size following X-Trend."""
    raw = leverage * capital
    if current_vol > 0.01:
        vol_adjusted = raw * (vol_target / current_vol)
    else:
        vol_adjusted = raw
    return min(vol_adjusted, capital * max_position_pct)


def action_to_trade(
    action: np.ndarray,
    current_price: float,
    capital: float,
    position_state: float,
    iv_rank: float,
    current_vol: float = 0.25,
) -> TradeDecision:
    """
    Map continuous 5D action vector to a concrete trade decision.

    Hard-coded risk limits (NEVER overridden by model):
    - Max position: 25% of capital
    - No selling naked calls
    - Must have sufficient cash for assignment
    """
    delta = float(action[0])
    direction = float(action[1])
    leverage = float(action[2])
    dte = int(round(float(action[3])))
    profit_target = float(action[4])

    # Effective no-trade signals
    if delta < 0.05 or leverage < 0.05:
        return TradeDecision(
            strategy="cash", delta=0, dte=0, leverage=0,
            profit_target=0, direction=direction, position_size_dollars=0,
            reason="delta or leverage below minimum threshold",
        )

    position_size = compute_position_size(leverage, capital, current_vol)

    if direction < -0.3:
        # Premium selling regime
        if position_state == 0:
            strategy = "wheel_csp"
            reason = f"direction={direction:.2f} < -0.3 → sell put"
        elif position_state > 0:
            strategy = "wheel_cc"
            reason = f"holding stock + direction={direction:.2f} → sell call"
        else:
            strategy = "wheel_csp"
            reason = f"direction={direction:.2f} → default CSP"

    elif direction > 0.3:
        # Directional regime
        strategy = "pmcc"
        reason = f"direction={direction:.2f} > 0.3 → PMCC (directional)"

    else:
        # Neutral regime
        if iv_rank > 50:
            strategy = "put_spread"
            reason = f"neutral + IV rank {iv_rank:.0f} > 50 → put spread"
        elif leverage < 0.15:
            strategy = "cash"
            position_size = 0
            reason = f"neutral + low leverage {leverage:.2f} → cash"
        else:
            strategy = "hold_stock"
            reason = f"neutral + moderate leverage → hold"

    # Enforce hard risk limits
    position_size = min(position_size, capital * 0.25)

    # Assignment check for CSP
    if strategy == "wheel_csp":
        strike_est = current_price * (1 - delta)
        max_contracts = int(capital * 0.25 / (strike_est * 100)) if strike_est > 0 else 0
        if max_contracts <= 0:
            return TradeDecision(
                strategy="cash", delta=0, dte=0, leverage=0,
                profit_target=0, direction=direction, position_size_dollars=0,
                reason="insufficient capital for CSP assignment",
            )

    return TradeDecision(
        strategy=strategy,
        delta=delta,
        dte=dte,
        leverage=leverage,
        profit_target=profit_target,
        direction=direction,
        position_size_dollars=round(position_size, 2),
        reason=reason,
    )
