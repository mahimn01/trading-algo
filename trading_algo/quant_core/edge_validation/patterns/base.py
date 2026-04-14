from __future__ import annotations

from typing import Protocol

import pandas as pd

from trading_algo.quant_core.edge_validation.types import PatternOccurrence


class PatternDetector(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, bars: pd.DataFrame) -> list[PatternOccurrence]:
        """Scan bars and return every occurrence of this pattern.

        Args:
            bars: DataFrame with DatetimeIndex (ET), columns [open, high, low, close, volume].
                  Index must be timezone-aware US/Eastern.

        Returns:
            List of PatternOccurrence for each detected pattern instance.
        """
        ...
