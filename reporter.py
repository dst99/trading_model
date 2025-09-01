import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from pathlib import Path

class Reporter:
    @staticmethod
    def save_calibration(y_true: pd.Series, p: pd.Series, out_path: Path):
        fig = plt.figure(figsize=(6,5))
        CalibrationDisplay.from_predictions(y_true, p, n_bins=10, strategy="uniform")
        plt.title("Calibration (Reliability)")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    @staticmethod
    def bucket_expectancy(trades: pd.DataFrame, probs: pd.Series, out_csv: Path) -> pd.DataFrame:
        df = trades.loc[probs.index].copy()
        df["p"] = probs.values
        df["bucket"] = pd.qcut(df["p"], 5, labels=False, duplicates="drop")
        summary = df.groupby("bucket").agg(
            n=("r_outcome","size"),
            mean_p=("p","mean"),
            win_rate=("win1","mean"),
            avg_R=("r_outcome","mean"),
            avg_entry=("entry_price","mean"),
            avg_exit=("exit_price","mean"),
        )
        summary.to_csv(out_csv)
        return summary

    @staticmethod
    def r_distribution(trades: pd.DataFrame, out_path: Path):
        if trades.empty: return
        plt.figure(figsize=(6,5))
        plt.hist(trades["r_outcome"].dropna(), bins=40)
        plt.xlabel("R outcome"); plt.ylabel("Frequency"); plt.title("Distribution of R Outcomes")
        plt.tight_layout(); plt.savefig(out_path); plt.close()
