from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # -------------------------
    # File paths
    # -------------------------
    dx_path: str   = "data/DX_4h.csv"     # DX (Dollar Index) OHLCV, 4H bars (signal source)
    eur_path: str  = "data/6E_4h.csv"     # 6E futures OHLCV, 4H bars (execution instrument)
    spot_path: str = "data/EURUSD_4h.csv" # EURUSD spot OHLCV, 4H bars (alternative execution instrument)
    out_dir: str   = "outputs"            # Where outputs (trades, reports, plots) are saved

    # -------------------------
    # DX sweep detection
    # -------------------------
    swing_lookback: int = 20   # How many bars back to check for prior high/low sweeps :20
    confirm_next: bool  = True # Require confirmation from the next bar after sweep? (stricter signals)

    # -------------------------
    # Pivot detection (for stop placement)
    # -------------------------
    pivot_left: int     = 2    # Bars to the left that must be higher/lower (classic swing definition) :2
    pivot_right: int    = 2    # Bars to the right that must be higher/lower:2
    pivot_lookback: int = 60   # How far back to search for the last valid swing high/low:60

    # -------------------------
    # Stop & risk management
    # -------------------------
    atr_len: int        = 14   # ATR length (used only if ATR stop capping enabled):14
    atr_cap_k: float    = 1.5  # Maximum multiple of ATR allowed for stops (ignored if ATR disabled):1.5
    ignore_atr: bool    = True # ATR disabled by default (stops use swing + floor instead)

    min_stop_dist: float = 0.0001   # Minimum stop size in price units (1 pip for 6E/EURUSD): 0.0001
    min_usd_risk_per_contract: Optional[float] = None
    # If set (e.g., 100), stop distance is forced large enough so each trade risks at least this much USD per 6E contract.
    # Default None = no USD floor.

    pip_size: float = 0.0001         # One pip increment for EURUSD/6E
    usd_per_pip_per_contract: float = 12.5  # 1 pip = $12.5 per 6E contract

    trail_step_R: float = 1.0   # Trail stop every +1R (move stop to breakeven at +1R, +1R at +2R, etc.)
    hard_tp_R: float    = 4.0   # Hard take-profit (trade closes automatically at +4R)
    ambiguous_policy: str = "stop_first"
    # If both stop & TP hit on same bar: "stop_first" closes at stop, "tp_first" favors TP.

    # -------------------------
    # Probability model
    # -------------------------
    n_splits: int = 5   # Number of splits for TimeSeriesSplit CV (more splits = more robust but slower) : 5
