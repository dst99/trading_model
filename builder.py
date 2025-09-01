import numpy as np
import pandas as pd
from config import Config
from pivots import PivotFinder
from stops import StopPolicy
from simulator import TradeSimulator

class TradeBuilder:
    def __init__(self, cfg: Config, pivot_finder: PivotFinder, stop_policy: StopPolicy, simulator: TradeSimulator):
        self.cfg = cfg
        self.pivot_finder = pivot_finder
        self.stop_policy = stop_policy
        self.sim = simulator

    @staticmethod
    def dx_to_inst_direction(dx_sig_aligned: pd.DataFrame, inst_index: pd.DatetimeIndex) -> pd.Series:
        direction = pd.Series(0, index=inst_index, dtype=int)
        direction[dx_sig_aligned["dx_up_sweep_reject"]   == 1] = -1
        direction[dx_sig_aligned["dx_down_sweep_reject"] == 1] =  1
        return direction

    def build(self, dx_sig: pd.DataFrame, inst: pd.DataFrame, label: str) -> pd.DataFrame:
        direction = self.dx_to_inst_direction(dx_sig, inst.index)
        trades = []
        clamped_count = 0

        for ts, dir_ in direction.items():
            if dir_ == 0:
                continue
            pivot = self.pivot_finder.find_last(inst, ts, dir_)
            if pivot is None:
                continue

            stop_dist, atr_here, clamped = self.stop_policy.compute_stop_dist(inst, ts, dir_, pivot)
            if clamped: clamped_count += 1

            r_out, exit_ts, reason, entry_px, exit_px = self.sim.simulate_trade(inst, ts, dir_, stop_dist)

            try:
                loc = inst.index.get_loc(ts)
                if isinstance(loc, slice): loc = loc.stop - 1
                entry_time = inst.index[loc + 1] if loc < len(inst.index) - 1 else pd.NaT
            except Exception:
                entry_time = pd.NaT

            trades.append({
                "instrument": label,
                "signal_time": ts,
                "entry_time": entry_time,
                "direction": int(dir_),
                "entry_price": entry_px,
                "exit_time": exit_ts,
                "exit_price": exit_px,
                "r_outcome": float(r_out),
                "reason": reason,
                "stop_dist": float(stop_dist),
                "atr": float(atr_here) if not np.isnan(atr_here) else np.nan,
            })

        df = pd.DataFrame(trades).sort_values("signal_time")
        df["win1"] = (df["r_outcome"] >= 1.0).astype(int)
        if clamped_count > 0:
            print(f"[{label}] Stop floor clamped {clamped_count} trade(s).")
        return df
