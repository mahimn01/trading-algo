from __future__ import annotations

from datetime import time as dtime

import pandas as pd

from trading_algo.quant_core.edge_validation.types import PatternOccurrence


MARKET_OPEN = dtime(9, 30)


class ORBDetector:
    def __init__(self, range_minutes: int = 60, min_range_points: float = 1.0) -> None:
        self._range_minutes = range_minutes
        self._min_range_points = min_range_points

    @property
    def name(self) -> str:
        return f"orb_{self._range_minutes}min"

    def detect(self, bars: pd.DataFrame) -> list[PatternOccurrence]:
        if bars.empty:
            return []

        occurrences: list[PatternOccurrence] = []
        bars = bars.sort_index()
        groups = bars.groupby(bars.index.date)

        range_end = dtime(
            MARKET_OPEN.hour + (MARKET_OPEN.minute + self._range_minutes) // 60,
            (MARKET_OPEN.minute + self._range_minutes) % 60,
        )

        for day, day_bars in groups:
            times = day_bars.index.time
            range_mask = (times >= MARKET_OPEN) & (times < range_end)
            range_bars = day_bars.loc[range_mask]

            if range_bars.empty:
                continue

            range_high = range_bars["high"].max()
            range_low = range_bars["low"].min()
            range_size = range_high - range_low

            if range_size < self._min_range_points:
                continue

            post_range_mask = times >= range_end
            post_bars = day_bars.loc[post_range_mask]

            if post_bars.empty:
                continue

            found_long = False
            found_short = False

            for idx, row in post_bars.iterrows():
                if not found_long and row["high"] > range_high:
                    found_long = True
                    bar_index = day_bars.index.get_loc(idx)
                    occurrences.append(PatternOccurrence(
                        timestamp=idx.timestamp(),
                        pattern=self.name,
                        direction="long",
                        entry_price=range_high,
                        day=day,
                        metadata={
                            "range_size": range_size,
                            "range_minutes": float(self._range_minutes),
                            "breakout_bar_index": float(bar_index),
                            "time_of_breakout": float(idx.hour * 60 + idx.minute),
                        },
                    ))

                if not found_short and row["low"] < range_low:
                    found_short = True
                    bar_index = day_bars.index.get_loc(idx)
                    occurrences.append(PatternOccurrence(
                        timestamp=idx.timestamp(),
                        pattern=self.name,
                        direction="short",
                        entry_price=range_low,
                        day=day,
                        metadata={
                            "range_size": range_size,
                            "range_minutes": float(self._range_minutes),
                            "breakout_bar_index": float(bar_index),
                            "time_of_breakout": float(idx.hour * 60 + idx.minute),
                        },
                    ))

                if found_long and found_short:
                    break

        return occurrences
