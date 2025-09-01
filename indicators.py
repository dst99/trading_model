import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    return tr.ewm(alpha=1/n, adjust=False).mean()

def rolling_extrema(s: pd.Series, n: int, mode: str) -> pd.Series:
    return (s.rolling(n).max() if mode == "max" else s.rolling(n).min()).shift(1)
