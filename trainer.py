import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from features import FeatureEngineer
from config import Config

class ProbabilityTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def train(self, trades: pd.DataFrame, inst: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        X, _ = FeatureEngineer(self.cfg).make_features(trades, inst)
        y = trades["win1"].values

        tscv = TimeSeriesSplit(n_splits=self.cfg.n_splits)
        aucs, briers, logs, prob_frames = [], [], [], []

        for tr_idx, te_idx in tscv.split(X.values):
            clf = GradientBoostingClassifier(random_state=42)
            clf.fit(X.values[tr_idx], y[tr_idx])
            p = clf.predict_proba(X.values[te_idx])[:, 1]
            prob_frames.append(pd.Series(p, index=trades.index[te_idx]))
            aucs.append(roc_auc_score(y[te_idx], p))
            briers.append(brier_score_loss(y[te_idx], p))
            logs.append(log_loss(y[te_idx], p))

        probs = pd.concat(prob_frames).sort_index()
        report = pd.DataFrame({"AUC":[np.mean(aucs)], "Brier":[np.mean(briers)], "LogLoss":[np.mean(logs)]})
        return probs, report
