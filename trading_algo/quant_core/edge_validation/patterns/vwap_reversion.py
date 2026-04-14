from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

from trading_algo.quant_core.edge_validation.types import PatternOccurrence


SIGNAL_START = dtime(10, 30)
SIGNAL_END = dtime(15, 0)


class VWAPReversionDetector:
    def __init__(self, sigma_threshold: float = 2.0, min_bars_for_std: int = 30) -> None:
        self._sigma_threshold = sigma_threshold
        self._min_bars_for_std = min_bars_for_std

    @property
    def name(self) -> str:
        return "vwap_reversion"

    def detect(self, bars: pd.DataFrame) -> list[PatternOccurrence]:
        if bars.empty:
            return []

        occurrences: list[PatternOccurrence] = []
        bars = bars.sort_index()
        groups = bars.groupby(bars.index.date)

        for day, day_bars in groups:
            if len(day_bars) < self._min_bars_for_std:
                continue

            typical = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3.0
            vol = day_bars["volume"].fillna(0).astype(float)
            cum_tp_vol = (typical * vol).cumsum()
            cum_vol = vol.cumsum()

            # Avoid division by zero
            valid_vol = cum_vol.replace(0, np.nan)
            vwap = cum_tp_vol / valid_vol

            deviation = day_bars["close"] - vwap
            rolling_std = deviation.expanding(min_periods=self._min_bars_for_std).std()

            found_long = False
            found_short = False

            for i in range(self._min_bars_for_std, len(day_bars)):
                idx = day_bars.index[i]
                t = idx.time()

                if t < SIGNAL_START or t >= SIGNAL_END:
                    continue

                std_val = rolling_std.iloc[i]
                if std_val is None or np.isnan(std_val) or std_val == 0:
                    continue

                dev = deviation.iloc[i]
                sigma = dev / std_val

                if not found_short and sigma > self._sigma_threshold:
                    found_short = True
                    occurrences.append(PatternOccurrence(
                        timestamp=idx.timestamp(),
                        pattern=self.name,
                        direction="short",
                        entry_price=day_bars["close"].iloc[i],
                        day=day,
                        metadata={
                            "vwap_price": float(vwap.iloc[i]),
                            "deviation_sigma": float(sigma),
                            "deviation_points": float(dev),
                            "time_of_signal": float(idx.hour * 60 + idx.minute),
                        },
                    ))

                if not found_long and sigma < -self._sigma_threshold:
                    found_long = True
                    occurrences.append(PatternOccurrence(
                        timestamp=idx.timestamp(),
                        pattern=self.name,
                        direction="long",
                        entry_price=day_bars["close"].iloc[i],
                        day=day,
                        metadata={
                            "vwap_price": float(vwap.iloc[i]),
                            "deviation_sigma": float(sigma),
                            "deviation_points": float(dev),
                            "time_of_signal": float(idx.hour * 60 + idx.minute),
                        },
                    ))

                if found_long and found_short:
                    break

        return occurrences
