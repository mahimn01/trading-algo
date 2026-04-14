from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeValidationConfig:
    symbols: tuple[str, ...] = ("NQ",)
    exchange: str = "CME"
    currency: str = "USD"

    start_date: str = "2024-04-01"
    end_date: str = "2026-04-10"
    bar_size: str = "1 min"
    data_cache_dir: str = "data/edge_validation"
    use_rth_only: bool = True

    orb_range_minutes: tuple[int, ...] = (15, 30, 60)
    gap_fade_min_gap_pct: float = 0.001
    gap_fade_max_gap_pct: float = 0.02
    gap_fade_buckets: tuple[float, ...] = (0.001, 0.0035, 0.005, 0.0075, 0.01)
    vwap_deviation_sigma: float = 2.0
    vwap_min_bars_for_std: int = 30

    significance_level: float = 0.05
    n_bootstrap: int = 10_000
    n_permutations: int = 10_000
    random_seed: int = 42

    wf_train_days: int = 126
    wf_test_days: int = 42
    wf_step_days: int = 42

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002
    ibkr_client_id: int = 20

    point_values: dict[str, float] | None = None

    def get_point_value(self, symbol: str) -> float:
        defaults = {"ES": 50.0, "NQ": 20.0, "MES": 5.0, "MNQ": 2.0}
        if self.point_values and symbol in self.point_values:
            return self.point_values[symbol]
        return defaults.get(symbol, 1.0)

    @classmethod
    def from_env(cls) -> EdgeValidationConfig:
        return cls(
            ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
            ibkr_port=int(os.environ.get("IBKR_PORT", "4002")),
            ibkr_client_id=int(os.environ.get("IBKR_CLIENT_ID", "20")),
        )
