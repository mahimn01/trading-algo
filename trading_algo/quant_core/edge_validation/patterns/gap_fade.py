from __future__ import annotations

import pandas as pd

from trading_algo.quant_core.edge_validation.types import PatternOccurrence


class GapFadeDetector:
    def __init__(self, min_gap_pct: float = 0.001, max_gap_pct: float = 0.02) -> None:
        self._min_gap_pct = min_gap_pct
        self._max_gap_pct = max_gap_pct

    @property
    def name(self) -> str:
        return "gap_fade"

    def detect(self, bars: pd.DataFrame) -> list[PatternOccurrence]:
        if bars.empty:
            return []

        occurrences: list[PatternOccurrence] = []
        bars = bars.sort_index()
        groups = bars.groupby(bars.index.date)

        day_keys = sorted(groups.groups.keys())

        for i in range(1, len(day_keys)):
            prev_day_bars = groups.get_group(day_keys[i - 1])
            curr_day_bars = groups.get_group(day_keys[i])

            if prev_day_bars.empty or curr_day_bars.empty:
                continue

            prev_close = prev_day_bars.iloc[-1]["close"]
            curr_open = curr_day_bars.iloc[0]["open"]

            if prev_close == 0:
                continue

            gap_pct = (curr_open - prev_close) / prev_close
            abs_gap_pct = abs(gap_pct)

            if abs_gap_pct < self._min_gap_pct or abs_gap_pct > self._max_gap_pct:
                continue

            # Fade direction: gap up -> short, gap down -> long
            direction: str = "short" if gap_pct > 0 else "long"
            gap_points = curr_open - prev_close

            first_bar_idx = curr_day_bars.index[0]
            occurrences.append(PatternOccurrence(
                timestamp=first_bar_idx.timestamp(),
                pattern=self.name,
                direction=direction,
                entry_price=curr_open,
                day=day_keys[i],
                metadata={
                    "gap_pct": gap_pct,
                    "gap_points": gap_points,
                    "abs_gap_pct": abs_gap_pct,
                    "prev_close": prev_close,
                },
            ))

        return occurrences
