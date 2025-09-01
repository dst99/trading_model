from pathlib import Path
import pandas as pd
from config import Config
from ohlcv_loader import load_csv, align_common_range
from signals import DXSweepSignalDetector
from pivots import PivotFinder
from stops import StopPolicy
from simulator import TradeSimulator
from builder import TradeBuilder
from trainer import ProbabilityTrainer
from reporter import Reporter

class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.signals  = DXSweepSignalDetector(cfg.swing_lookback, cfg.confirm_next)
        self.pivots   = PivotFinder(cfg.pivot_left, cfg.pivot_right, cfg.pivot_lookback)
        self.stopper  = StopPolicy(cfg)
        self.sim      = TradeSimulator(cfg)
        self.builder  = TradeBuilder(cfg, self.pivots, self.stopper, self.sim)
        self.trainer  = ProbabilityTrainer(cfg)
        self.reporter = Reporter()

    def run_for_instrument(self, label: str, dx: pd.DataFrame, inst: pd.DataFrame, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        dx_sig = self.signals.detect(dx)
        dx_sig_aligned = dx_sig.reindex(inst.index, method="ffill").fillna(0)
        dx_sig_aligned = dx_sig_aligned[["dx_up_sweep_reject","dx_down_sweep_reject"]]

        trades = self.builder.build(dx_sig_aligned, inst, label)
        trades.to_csv(out_dir / "trades_labeled.csv", index=False)
        if trades.empty:
            print(f"[{label}] No trades produced."); return

        probs, cv = self.trainer.train(trades, inst)
        cv.to_csv(out_dir / "cv_report.csv", index=False)

        aligned = trades.loc[probs.index]
        self.reporter.save_calibration(aligned["win1"], probs, out_dir / "calibration.png")
        self.reporter.bucket_expectancy(aligned, probs, out_dir / "bucket_expectancy.csv")
        self.reporter.r_distribution(trades, out_dir / "r_outcome_hist.png")

        print(f"[{label}] Done → {out_dir.resolve()}")

    def run(self):
        out_root = Path(self.cfg.out_dir); out_root.mkdir(parents=True, exist_ok=True)
        dx   = load_csv(self.cfg.dx_path)
        eur  = load_csv(self.cfg.eur_path)
        spot = load_csv(self.cfg.spot_path)

        print("\n=== DATA COVERAGE ===")
        print("DX   :", dx.index.min(),  "→", dx.index.max(),   "tz:", dx.index.tz)
        print("6E   :", eur.index.min(), "→", eur.index.max(),  "tz:", eur.index.tz)
        print("SPOT :", spot.index.min(),"→", spot.index.max(), "tz:", spot.index.tz)

        _, (dx_c, eur_c, spot_c) = align_common_range(dx, eur, spot)

        self.run_for_instrument("6E",   dx_c, eur_c,  out_root / "6E")
        self.run_for_instrument("SPOT", dx_c, spot_c, out_root / "SPOT")

        print("\nPer-instrument artifacts written. For consolidated analysis, run:  python summary_report_plus.py --root outputs")
