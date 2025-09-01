from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Paths
    dx_path: str   = "data/DX_4h.csv"
    eur_path: str  = "data/6E_4h.csv"
    spot_path: str = "data/EURUSD_4h.csv"
    out_dir: str   = "outputs"

    # DX sweeps
    swing_lookback: int = 20
    confirm_next: bool  = True

    # Pivots
    pivot_left: int     = 2
    pivot_right: int    = 2
    pivot_lookback: int = 60

    # Stops & management
    atr_len: int        = 14
    atr_cap_k: float    = 1.5
    ignore_atr: bool    = True      # ATR OFF by default
    min_stop_dist: float = 0.0001   # >= 1 pip
    min_usd_risk_per_contract: Optional[float] = None
    pip_size: float = 0.0001
    usd_per_pip_per_contract: float = 12.5
    trail_step_R: float = 1.0
    hard_tp_R: float    = 4.0
    ambiguous_policy: str = "stop_first"  # {"stop_first","tp_first"}

    # Modeling
    n_splits: int = 5
