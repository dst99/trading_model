import pandas as pd
from typing import Optional

class PivotFinder:
    def __init__(self, left: int, right: int, lookback: int):
        self.left, self.right, self.lookback = left, right, lookback

    def find_last(self, inst: pd.DataFrame, ts: pd.Timestamp, direction: int) -> Optional[float]:
        try:
            start_idx = inst.index.get_indexer_for([ts])[0]
        except Exception:
            return None
        start = max(0, start_idx - self.lookback)
        window = inst.iloc[start:start_idx]
        if window.empty: return None

        if direction == 1:  # long -> pivot low
            lows = window["low"]; piv=[]
            for i in range(self.left, len(window) - self.right):
                seg = lows.iloc[i-self.left:i+self.right+1]
                if lows.iloc[i] == seg.min(): piv.append(lows.iloc[i])
            return piv[-1] if piv else None
        else:               # short -> pivot high
            highs = window["high"]; piv=[]
            for i in range(self.left, len(window) - self.right):
                seg = highs.iloc[i-self.left:i+self.right+1]
                if highs.iloc[i] == seg.max(): piv.append(highs.iloc[i])
            return piv[-1] if piv else None
