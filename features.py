import pandas as pd
from config import Config
from indicators import atr as atr_func

class FeatureEngineer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def make_features(self, trades: pd.DataFrame, inst: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        inst_feat = inst[["close"]].copy()
        if not self.cfg.ignore_atr:
            inst_feat["atr"] = inst.get("atr", atr_func(inst, self.cfg.atr_len))
        inst_feat["ret_24"] = inst_feat["close"].pct_change(6)  # ~24h on 4H bars

        feats = trades.join(inst_feat, on="entry_time", how="left", rsuffix="_inst")
        feat_cols = ["stop_dist", "ret_24"]
        if not self.cfg.ignore_atr:
            feat_cols.insert(1, "atr")
        return feats[feat_cols].fillna(0.0), feat_cols
