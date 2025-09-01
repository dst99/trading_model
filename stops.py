import numpy as np
import pandas as pd
from config import Config
from indicators import atr as atr_func

class StopPolicy:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def ensure_atr(self, inst: pd.DataFrame) -> pd.DataFrame:
        if "atr" not in inst.columns:
            inst = inst.copy()
            inst["atr"] = atr_func(inst, self.cfg.atr_len)
        return inst

    def compute_stop_dist(self, inst: pd.DataFrame, ts: pd.Timestamp, direction: int, pivot: float):
        close_px = float(inst.loc[ts, "close"])
        swing_dist = (close_px - pivot) if direction == 1 else (pivot - close_px)
        swing_dist = max(swing_dist, 1e-8)

        if self.cfg.ignore_atr:
            stop_dist = swing_dist
            atr_here = np.nan
        else:
            inst = self.ensure_atr(inst)
            atr_here = float(inst.loc[ts, "atr"])
            stop_dist = min(swing_dist, self.cfg.atr_cap_k * atr_here)

        before = stop_dist
        if self.cfg.min_stop_dist is not None:
            stop_dist = max(stop_dist, self.cfg.min_stop_dist)
        if self.cfg.min_usd_risk_per_contract:
            min_from_usd = (self.cfg.min_usd_risk_per_contract / self.cfg.usd_per_pip_per_contract) * self.cfg.pip_size
            stop_dist = max(stop_dist, min_from_usd)
        clamped = stop_dist > before + 1e-15
        return stop_dist, (np.nan if self.cfg.ignore_atr else atr_here), clamped
