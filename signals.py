import pandas as pd
from indicators import rolling_extrema

class DXSweepSignalDetector:
    def __init__(self, lookback: int, confirm_next: bool):
        self.lookback = lookback
        self.confirm_next = confirm_next

    def detect(self, dx: pd.DataFrame) -> pd.DataFrame:
        prev_max = rolling_extrema(dx["high"], self.lookback, "max")
        prev_min = rolling_extrema(dx["low"],  self.lookback, "min")

        up_break   = dx["high"] > prev_max
        down_break = dx["low"]  < prev_min

        up_reject   = up_break   & (dx["close"] < prev_max)
        down_reject = down_break & (dx["close"] > prev_min)

        if self.confirm_next:
            up_reject   &= dx["close"].shift(-1) < dx["close"]
            down_reject &= dx["close"].shift(-1) > dx["close"]

        out = pd.DataFrame(index=dx.index)
        out["dx_up_sweep_reject"]   = up_reject.astype(int)
        out["dx_down_sweep_reject"] = down_reject.astype(int)
        return out
