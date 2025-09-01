import pandas as pd
from typing import List, Tuple

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    assert {"time","open","high","low","close"}.issubset(df.columns), \
        "CSV must have columns: time,open,high,low,close"
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def align_common_range(*dfs: pd.DataFrame) -> Tuple[pd.DatetimeIndex, List[pd.DataFrame]]:
    latest_start = max(df.index.min() for df in dfs)
    earliest_end = min(df.index.max() for df in dfs)
    out = [df[(df.index >= latest_start) & (df.index <= earliest_end)] for df in dfs]
    return pd.date_range(latest_start, earliest_end, freq="h"), out
