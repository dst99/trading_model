import pandas as pd
from typing import Tuple
from config import Config

class TradeSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def simulate_trade(self, inst: pd.DataFrame, entry_idx: pd.Timestamp, direction: int,
                       stop_dist: float) -> Tuple[float, pd.Timestamp, str, float, float]:
        loc = inst.index.get_loc(entry_idx)
        if isinstance(loc, slice): loc = loc.stop - 1
        if loc >= len(inst) - 1:
            return 0.0, entry_idx, "no_data", float("nan"), float("nan")

        entry_bar = inst.iloc[loc + 1]
        entry_px  = float(entry_bar["open"])
        R         = stop_dist

        stop    = entry_px - R if direction == 1 else entry_px + R
        next_tp = entry_px + R if direction == 1 else entry_px - R
        hard_tp = entry_px + self.cfg.hard_tp_R * R if direction == 1 else entry_px - self.cfg.hard_tp_R * R
        locked_R = 0.0

        def realized_R_at_stop(stop_price: float) -> float:
            return ((stop_price - entry_px) / R) if direction == 1 else ((entry_px - stop_price) / R)

        for ts, row in inst.iloc[loc + 1:].iterrows():
            hi, lo = float(row["high"]), float(row["low"])

            if (direction == 1 and hi >= hard_tp) or (direction == -1 and lo <= hard_tp):
                return self.cfg.hard_tp_R, ts, f"hard_tp{int(self.cfg.hard_tp_R)}R", entry_px, hard_tp

            hit_stop = (lo <= stop) if direction == 1 else (hi >= stop)
            hit_tp   = (hi >= next_tp) if direction == 1 else (lo <= next_tp)

            if hit_stop and hit_tp:
                if self.cfg.ambiguous_policy == "stop_first":
                    r_out = realized_R_at_stop(stop)
                    return r_out, ts, "ambig_stop", entry_px, stop
                else:
                    locked_R += self.cfg.trail_step_R
                    stop    = (entry_px + (locked_R - self.cfg.trail_step_R) * R) if direction == 1 else \
                              (entry_px - (locked_R - self.cfg.trail_step_R) * R)
                    next_tp = (entry_px + (locked_R + self.cfg.trail_step_R) * R) if direction == 1 else \
                              (entry_px - (locked_R + self.cfg.trail_step_R) * R)
                    continue

            if hit_stop:
                r_out = realized_R_at_stop(stop)
                return r_out, ts, "stop", entry_px, stop

            if hit_tp:
                locked_R += self.cfg.trail_step_R
                stop    = (entry_px + (locked_R - self.cfg.trail_step_R) * R) if direction == 1 else \
                          (entry_px - (locked_R - self.cfg.trail_step_R) * R)
                next_tp = (entry_px + (locked_R + self.cfg.trail_step_R) * R) if direction == 1 else \
                          (entry_px - (locked_R + self.cfg.trail_step_R) * R)
                continue

        last_close = float(inst["close"].iloc[-1])
        r_mtm = ((last_close - entry_px) / R) if direction == 1 else ((entry_px - last_close) / R)
        return float(r_mtm), inst.index[-1], "mtm", entry_px, last_close
