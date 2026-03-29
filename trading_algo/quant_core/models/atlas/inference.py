"""Live inference adapter for ATLAS."""

from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np
import torch

from trading_algo.quant_core.models.atlas.config import ATLASConfig
from trading_algo.quant_core.models.atlas.model import ATLASModel
from trading_algo.quant_core.models.atlas.features import ATLASFeatureComputer, RollingNormalizer
from trading_algo.quant_core.models.atlas.execution_bridge import TradeDecision, action_to_trade


class ATLASInference:
    """
    Load trained model, maintain rolling 90-day buffer, generate daily predictions.

    Usage:
        atlas = ATLASInference.from_checkpoint('checkpoints/atlas/best.pt')
        decision = atlas.predict(
            date=datetime.now(), price=245.73, high=247.10, low=244.50,
            volume=1_200_000, position_state=0, position_pnl=0.0,
            days_in_trade=0, cash_pct=1.0,
        )
    """

    def __init__(self, model: ATLASModel, config: ATLASConfig, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.feature_computer = ATLASFeatureComputer()
        self.normalizer = RollingNormalizer(lookback=252)

        self._closes: deque[float] = deque(maxlen=400)
        self._highs: deque[float] = deque(maxlen=400)
        self._lows: deque[float] = deque(maxlen=400)
        self._volumes: deque[float] = deque(maxlen=400)
        self._dates: deque[datetime] = deque(maxlen=400)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ATLASInference":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get("config", ATLASConfig())
        if isinstance(config, dict):
            config = ATLASConfig(**config)
        model = ATLASModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, config, device)

    def update_buffer(self, date: datetime, price: float, high: float, low: float, volume: float) -> None:
        self._dates.append(date)
        self._closes.append(price)
        self._highs.append(high)
        self._lows.append(low)
        self._volumes.append(volume)

    def predict(
        self,
        date: datetime,
        price: float,
        high: float,
        low: float,
        volume: float,
        position_state: float = 0.0,
        position_pnl: float = 0.0,
        days_in_trade: float = 0.0,
        cash_pct: float = 1.0,
        target_sharpe: float = 1.0,
        capital: float = 10_000.0,
    ) -> TradeDecision:
        """Generate a trade decision for today."""
        self.update_buffer(date, price, high, low, volume)

        n = len(self._closes)
        if n < self.config.context_len + 252:
            return TradeDecision(
                strategy="cash", delta=0, dte=0, leverage=0,
                profit_target=0, direction=0, position_size_dollars=0,
                reason=f"insufficient history ({n} bars, need {self.config.context_len + 252})",
            )

        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        volumes = np.array(self._volumes)

        # Compute features
        raw_features = self.feature_computer.compute_features(closes, highs, lows, volumes)

        # Add position state features
        pos_features = np.zeros((len(raw_features), 4))
        pos_features[-1] = [position_state, position_pnl, days_in_trade, cash_pct]
        full_features = np.concatenate([raw_features, pos_features], axis=1)  # (T, 16)

        # Normalize
        normed, mu_arr, sigma_arr = self.normalizer.normalize(full_features)

        # Extract 90-day window (last 90 days)
        L = self.config.context_len
        window = normed[-L:]  # (90, 16)
        mu_window = mu_arr[-L:]
        sigma_window = sigma_arr[-L:]

        # Build tensors
        features_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 90, 16)
        ts_t = torch.tensor([d.timestamp() for d in list(self._dates)[-L:]], dtype=torch.float32).unsqueeze(0)
        dates_list = list(self._dates)[-L:]
        dow_t = torch.tensor([d.weekday() for d in dates_list], dtype=torch.long).unsqueeze(0)
        mo_t = torch.tensor([d.month - 1 for d in dates_list], dtype=torch.long).unsqueeze(0)
        opex_t = torch.zeros(1, L)
        qtr_t = torch.zeros(1, L)
        mu_t = torch.tensor(mu_window, dtype=torch.float32).unsqueeze(0)
        sigma_t = torch.tensor(sigma_window, dtype=torch.float32).unsqueeze(0)
        rtg_t = torch.tensor([target_sharpe], dtype=torch.float32)

        # Forward
        with torch.no_grad():
            action = self.model(
                features_t.to(self.device), ts_t.to(self.device),
                dow_t.to(self.device), mo_t.to(self.device),
                opex_t.to(self.device), qtr_t.to(self.device),
                mu_t.to(self.device), sigma_t.to(self.device),
                rtg_t.to(self.device),
            )

        action_np = action.squeeze(0).cpu().numpy()

        # Estimate current vol for position sizing
        if len(closes) >= 30:
            log_rets = np.diff(np.log(closes[-31:]))
            current_vol = float(np.std(log_rets) * np.sqrt(252))
        else:
            current_vol = 0.25

        # Estimate IV rank
        iv_rank = 50.0  # default; could compute from feature
        if window.shape[0] >= 1 and window.shape[1] >= 7:
            iv_rank = float(window[-1, 6]) * 50 + 50  # denormalize

        return action_to_trade(
            action=action_np,
            current_price=price,
            capital=capital,
            position_state=position_state,
            iv_rank=iv_rank,
            current_vol=current_vol,
        )
