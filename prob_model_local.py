# prob_model_local.py
# DX(4H) sweeps → entries on 6E 4H and EURUSD spot 4H
# - Swing-based stop (optionally capped by ATR k; or ignore ATR entirely)
# - +1R trailing
# - Hard TP at +R (default 4R)
# - Compares 6E vs SPOT starting from latest common date across DX/6E/SPOT
# - Outputs per instrument + side-by-side summary + R outcome histogram per instrument

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibrationDisplay


# -------------------- CONFIG --------------------
@dataclass
class Config:
    dx_path: str   = "data/DX_4h.csv"
    eur_path: str  = "data/6E_4h.csv"          # 6E futures (B-ADJ if possible)
    spot_path: str = "data/EURUSD_4h.csv"      # EURUSD spot
    out_dir: str   = "outputs"

    # DX sweep detection (4H)
    swing_lookback: int = 20
    confirm_next: bool  = True

    # Stops & trailing
    atr_len: int        = 14
    atr_cap_k: float    = 1.5    # cap swing stop at k*ATR (set high or --ignore-atr to disable)
    ignore_atr: bool    = False  # if True, completely ignore ATR (pure swing stop)
    trail_step_R: float = 1.0    # trail every +1R
    hard_tp_R: float    = 4.0    # hard TP in R

    # Swing pivot detection
    pivot_left: int     = 2
    pivot_right: int    = 2
    pivot_lookback: int = 60     # bars to scan back for last pivot

    # Modeling
    n_splits: int       = 5
    ambiguous_policy: str = "stop_first"  # {"stop_first","tp_first"}


# -------------------- IO --------------------
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    assert {"time","open","high","low","close"}.issubset(df.columns), \
        "CSV must have columns: time, open, high, low, close"
    df["time"] = pd.to_datetime(df["time"], utc=True)  # handles tz offsets
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# -------------------- TECH --------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    return tr.ewm(alpha=1/n, adjust=False).mean()

def ensure_atr(inst: pd.DataFrame, n: int) -> pd.DataFrame:
    if "atr" not in inst.columns:
        inst = inst.copy()
        inst["atr"] = atr(inst, n)
    return inst

def rolling_extrema(s: pd.Series, n: int, mode: str) -> pd.Series:
    return (s.rolling(n).max() if mode == "max" else s.rolling(n).min()).shift(1)


# -------------------- DX SWEEP DETECTION (4H) --------------------
def detect_dx_sweeps(dx: pd.DataFrame, lookback: int, confirm_next: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=dx.index)
    prev_max = rolling_extrema(dx["high"], lookback, "max")
    prev_min = rolling_extrema(dx["low"],  lookback, "min")

    up_break   = dx["high"] > prev_max
    down_break = dx["low"]  < prev_min

    up_reject   = up_break   & (dx["close"] < prev_max)
    down_reject = down_break & (dx["close"] > prev_min)

    if confirm_next:
        up_reject   &= dx["close"].shift(-1) < dx["close"]
        down_reject &= dx["close"].shift(-1) > dx["close"]

    out["dx_up_sweep_reject"]   = up_reject.astype(int)
    out["dx_down_sweep_reject"] = down_reject.astype(int)
    return out


# -------------------- SWING PIVOTS --------------------
def find_last_pivot(inst: pd.DataFrame, ts: pd.Timestamp, direction: int,
                    left: int, right: int, lookback: int) -> Optional[float]:
    """Most recent pivot low (long) or pivot high (short) strictly before ts."""
    try:
        start_idx = inst.index.get_indexer_for([ts])[0]
    except Exception:
        return None
    start  = max(0, start_idx - lookback)
    window = inst.iloc[start:start_idx]
    if window.empty:
        return None

    if direction == 1:  # long -> pivot low
        lows = window["low"]; piv = []
        for i in range(left, len(window) - right):
            seg = lows.iloc[i-left:i+right+1]
            if lows.iloc[i] == seg.min():
                piv.append(lows.iloc[i])
        return piv[-1] if piv else None
    else:               # short -> pivot high
        highs = window["high"]; piv = []
        for i in range(left, len(window) - right):
            seg = highs.iloc[i-left:i+right+1]
            if highs.iloc[i] == seg.max():
                piv.append(highs.iloc[i])
        return piv[-1] if piv else None


# -------------------- TRADE SIM (+1R TRAIL & HARD TP) --------------------
def simulate_trade(inst: pd.DataFrame, entry_idx: pd.Timestamp, direction: int,
                   stop_dist: float, trail_step_R: float, ambiguous_policy: str,
                   hard_tp_R: float) -> Tuple[float, pd.Timestamp, str, float, float]:
    """
    Entry = next bar open. Trail every +1R; hard close at +hard_tp_R.
    Returns: (R_outcome, exit_time, exit_reason, entry_price, exit_price)
    """
    loc = inst.index.get_loc(entry_idx)
    if isinstance(loc, slice): loc = loc.stop - 1
    if loc >= len(inst) - 1:
        return 0.0, entry_idx, "no_data", float("nan"), float("nan")

    entry_bar = inst.iloc[loc + 1]
    entry_px  = float(entry_bar["open"])
    R         = stop_dist

    stop    = entry_px - R if direction == 1 else entry_px + R
    next_tp = entry_px + R if direction == 1 else entry_px - R
    hard_tp = entry_px + hard_tp_R * R if direction == 1 else entry_px - hard_tp_R * R
    locked_R = 0.0  # increases by trail_step_R each time a TP step is reached

    def realized_R_at_stop(stop_price: float) -> float:
        if direction == 1:
            return (stop_price - entry_px) / R
        else:
            return (entry_px - stop_price) / R

    for ts, row in inst.iloc[loc + 1:].iterrows():
        hi, lo = float(row["high"]), float(row["low"])

        # 1) Hard TP check FIRST
        if (direction == 1 and hi >= hard_tp) or (direction == -1 and lo <= hard_tp):
            return hard_tp_R, ts, f"hard_tp{int(hard_tp_R)}R", entry_px, hard_tp

        # 2) Normal stop / tp checks
        hit_stop = (lo <= stop) if direction == 1 else (hi >= stop)
        hit_tp   = (hi >= next_tp) if direction == 1 else (lo <= next_tp)

        if hit_stop and hit_tp:
            if ambiguous_policy == "stop_first":
                r_out = realized_R_at_stop(stop)  # can be -1.0, 0.0, +1.0, ...
                return r_out, ts, "ambig_stop", entry_px, stop
            else:
                # TP-first: lock profit and continue
                locked_R += trail_step_R
                stop    = (entry_px + (locked_R - trail_step_R) * R) if direction == 1 else \
                          (entry_px - (locked_R - trail_step_R) * R)
                next_tp = (entry_px + (locked_R + trail_step_R) * R) if direction == 1 else \
                          (entry_px - (locked_R + trail_step_R) * R)
                continue

        if hit_stop:
            r_out = realized_R_at_stop(stop)      # -1.0 if first stop, 0.0 after first TP, +1.0 after second, etc.
            return r_out, ts, "stop", entry_px, stop

        if hit_tp:
            locked_R += trail_step_R
            # trail one step behind current locked_R
            stop    = (entry_px + (locked_R - trail_step_R) * R) if direction == 1 else \
                      (entry_px - (locked_R - trail_step_R) * R)
            next_tp = (entry_px + (locked_R + trail_step_R) * R) if direction == 1 else \
                      (entry_px - (locked_R + trail_step_R) * R)
            continue

    # 3) Neither stop nor TP hit → mark-to-market at last close
    last_close = float(inst["close"].iloc[-1])
    r_mtm = ((last_close - entry_px) / R) if direction == 1 else ((entry_px - last_close) / R)
    return float(r_mtm), inst.index[-1], "mtm", entry_px, last_close


# -------------------- PIPELINE (generic instrument) --------------------
def build_trades(dx_sig_aligned: pd.DataFrame, inst: pd.DataFrame, cfg: Config, label: str) -> pd.DataFrame:
    # Direction from precomputed signals
    direction = pd.Series(0, index=inst.index, dtype=int)
    direction[dx_sig_aligned["dx_up_sweep_reject"]   == 1] = 1
    direction[dx_sig_aligned["dx_down_sweep_reject"] == 1] = -1

    if not cfg.ignore_atr:
        inst = ensure_atr(inst, cfg.atr_len)

    trades = []
    for ts, dir_ in direction.items():
        if dir_ == 0:
            continue

        pivot = find_last_pivot(inst, ts, dir_, cfg.pivot_left, cfg.pivot_right, cfg.pivot_lookback)
        if pivot is None:
            continue

        close_px   = float(inst.loc[ts, "close"])
        swing_dist = (close_px - pivot) if dir_ == 1 else (pivot - close_px)
        swing_dist = max(swing_dist, 1e-8)

        if cfg.ignore_atr:
            stop_dist = swing_dist
            atr_here  = np.nan
        else:
            atr_here  = float(inst.loc[ts, "atr"])
            stop_dist = min(swing_dist, cfg.atr_cap_k * atr_here)

        r_out, exit_ts, reason, entry_px, exit_px = simulate_trade(
            inst, ts, dir_, stop_dist, cfg.trail_step_R, cfg.ambiguous_policy, cfg.hard_tp_R
        )

        # entry_time = next bar
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
            "atr": atr_here,
        })

    df = pd.DataFrame(trades).sort_values("signal_time")
    df["win1"] = (df["r_outcome"] >= 1.0).astype(int)
    return df


# -------------------- MODEL --------------------
def train_model(trades: pd.DataFrame, inst: pd.DataFrame, cfg: Config):
    if not cfg.ignore_atr:
        inst = ensure_atr(inst, cfg.atr_len)

    feats = trades.copy()
    inst_feat = inst[["close"]].copy()
    if not cfg.ignore_atr:
        inst_feat["atr"] = inst["atr"]
    inst_feat["ret_24"] = inst_feat["close"].pct_change(6)  # ~24h on 4H bars

    feats = feats.join(inst_feat, on="entry_time", how="left", rsuffix="_inst")

    feat_cols = ["stop_dist", "ret_24"]
    if not cfg.ignore_atr:
        feat_cols.insert(1, "atr")

    X = feats[feat_cols].fillna(0.0).values
    y = feats["win1"].values

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    aucs, briers, logs, prob_frames = [], [], [], []
    for tr_idx, te_idx in tscv.split(X):
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X[tr_idx], y[tr_idx])
        p = clf.predict_proba(X[te_idx])[:,1]
        prob_frames.append(pd.Series(p, index=feats.index[te_idx]))
        aucs.append(roc_auc_score(y[te_idx], p))
        briers.append(brier_score_loss(y[te_idx], p))
        logs.append(log_loss(y[te_idx], p))

    probs  = pd.concat(prob_frames).sort_index()
    report = pd.DataFrame({"AUC":[np.mean(aucs)], "Brier":[np.mean(briers)], "LogLoss":[np.mean(logs)]})
    return probs, report


# -------------------- EVAL --------------------
def save_calibration_plot(y_true: pd.Series, p: pd.Series, out_path: Path):
    fig = plt.figure(figsize=(6,5))
    CalibrationDisplay.from_predictions(y_true, p, n_bins=10, strategy="uniform")
    plt.title("Calibration (Reliability)")
    fig.tight_layout()
    fig.savefig(out_path)

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

def save_r_distribution(trades: pd.DataFrame, out_path: Path):
    """Save a histogram of R outcomes for quick sanity (should include negatives if stops occur)."""
    if trades.empty:
        return
    plt.figure(figsize=(6,5))
    plt.hist(trades["r_outcome"].dropna(), bins=40)
    plt.xlabel("R outcome")
    plt.ylabel("Frequency")
    plt.title("Distribution of R Outcomes")
    plt.tight_layout()
    plt.savefig(out_path)


# -------------------- ALIGNMENT --------------------
def align_common_range(dx: pd.DataFrame, eur: pd.DataFrame, spot: pd.DataFrame):
    latest_start  = max(dx.index.min(), eur.index.min(), spot.index.min())
    earliest_end  = min(dx.index.max(), eur.index.max(), spot.index.max())
    print("\n=== COMMON RANGE (all three) ===")
    print("Latest common START:", latest_start)
    print("Earliest common END:", earliest_end)
    mask_dx   = (dx.index   >= latest_start) & (dx.index   <= earliest_end)
    mask_eur  = (eur.index  >= latest_start) & (eur.index  <= earliest_end)
    mask_spot = (spot.index >= latest_start) & (spot.index <= earliest_end)
    return dx.loc[mask_dx], eur.loc[mask_eur], spot.loc[mask_spot]


# -------------------- REPORTING HELPERS --------------------
def summarize_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n":0, "win_rate":np.nan, "avg_R":np.nan, "p_stop":np.nan, "p_tp4R":np.nan, "p_mtm":np.nan}
    win_rate = trades["win1"].mean()
    avg_R    = trades["r_outcome"].mean()
    rc = trades["reason"].value_counts(normalize=True)
    p_stop = float(rc.get("stop", 0.0))
    p_mtm  = float(rc.get("mtm", 0.0))
    p_hard = float(rc[[c for c in rc.index if str(c).startswith("hard_tp")]].sum()) if len(rc) else 0.0
    return {"n":int(n), "win_rate":float(win_rate), "avg_R":float(avg_R), "p_stop":p_stop, "p_tp4R":p_hard, "p_mtm":p_mtm}

def write_markdown_summary(path: Path, rows: pd.DataFrame):
    md = []
    md.append("# Strategy Probability Model — Summary Report\n")
    md.append("This report compares **6E futures** vs **EURUSD spot** over the latest common period among DX/6E/SPOT.\n")
    md.append("## Overview\n")
    # Prefer Markdown table if tabulate (pandas to_markdown) is available
    try:
        md.append(rows.to_markdown(index=False))
    except Exception:
        md.append("```\n" + rows.to_string(index=False) + "\n```")
    md.append("\n\n**Columns:**\n")
    md.append("- `Instrument`: 6E futures vs EURUSD spot\n")
    md.append("- `AUC`, `Brier`, `LogLoss`: cross-validated predictive metrics\n")
    md.append("- `Trades`: number of signals executed\n")
    md.append("- `Win%`: fraction of trades reaching ≥ +1R before stopping out\n")
    md.append("- `Avg R`: mean R-multiple across trades\n")
    md.append("- `%Stop`, `%TP(hard)`, `%MtM`: exit reason distribution\n")
    md.append("\n\n*Generated by prob_model_local.py*\n")
    path.write_text("\n".join(md), encoding="utf-8")


# -------------------- RUN BOTH INSTRUMENTS --------------------
def run_pair(cfg: Config):
    out_root = Path(cfg.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    dx   = load_ohlcv(cfg.dx_path)
    eur  = load_ohlcv(cfg.eur_path)
    spot = load_ohlcv(cfg.spot_path)

    print("\n=== DATA COVERAGE ===")
    print("DX   :", dx.index.min(),  "→", dx.index.max(),   "tz:", dx.index.tz)
    print("6E   :", eur.index.min(), "→", eur.index.max(),  "tz:", eur.index.tz)
    print("SPOT :", spot.index.min(),"→", spot.index.max(), "tz:", spot.index.tz)

    # Trim all to common overlapping window (start at latest available)
    dx_c, eur_c, spot_c = align_common_range(dx, eur, spot)

    # DX sweep signals once on the common grid
    dx_sig_base = detect_dx_sweeps(dx_c, cfg.swing_lookback, cfg.confirm_next)

    # Collect per-instrument summaries to build side-by-side report
    side_by_side = []

    for label, inst in [("6E", eur_c), ("SPOT", spot_c)]:
        print(f"\n=== Running for {label} ===")
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Align DX signals to instrument grid
        dx_sig = dx_sig_base.reindex(inst.index, method="ffill").fillna(0)
        dx_sig = dx_sig[["dx_up_sweep_reject","dx_down_sweep_reject"]]

        # Build trades
        trades = build_trades(dx_sig, inst, cfg, label)
        trades.to_csv(out_dir / "trades_labeled.csv", index=False)

        if trades.empty:
            print("No trades produced for", label)
            continue

        # Train model
        probs, report = train_model(trades, inst, cfg)
        report.to_csv(out_dir / "cv_report.csv", index=False)

        # Plots & bucket stats
        aligned = trades.loc[probs.index]
        save_calibration_plot(aligned["win1"], probs, out_dir / "calibration.png")
        bucket_expectancy(aligned, probs, out_dir / "bucket_expectancy.csv")
        save_r_distribution(trades, out_dir / "r_outcome_hist.png")

        # Build quick summary row
        rep = report.iloc[0].to_dict()
        tr_sum = summarize_trades(trades)
        row = {
            "Instrument": label,
            "AUC": round(rep["AUC"], 4),
            "Brier": round(rep["Brier"], 4),
            "LogLoss": round(rep["LogLoss"], 4),
            "Trades": tr_sum["n"],
            "Win%": round(tr_sum["win_rate"], 4) if pd.notna(tr_sum["win_rate"]) else np.nan,
            "Avg R": round(tr_sum["avg_R"], 4) if pd.notna(tr_sum["avg_R"]) else np.nan,
            "%Stop": round(tr_sum["p_stop"], 4),
            "%TP(hard)": round(tr_sum["p_tp4R"], 4),
            "%MtM": round(tr_sum["p_mtm"], 4),
        }
        side_by_side.append(row)

        print(f"{label} outputs saved → {out_dir.resolve()}")

    # Write side-by-side comparison artifacts
    if side_by_side:
        overview = pd.DataFrame(side_by_side)
        overview_path_csv = out_root / "summary_overview.csv"
        overview.to_csv(overview_path_csv, index=False)

        overview_md_path = out_root / "summary_report.md"
        write_markdown_summary(overview_md_path, overview)

        print("\n=== SIDE-BY-SIDE SUMMARY ===")
        print(overview.to_string(index=False))
        print(f"\nSaved: {overview_path_csv}")
        print(f"Saved: {overview_md_path}")
    else:
        print("\nNo outputs produced (no trades). Check signal parameters or data coverage.")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    CFG = Config()  # defaults

    parser = argparse.ArgumentParser()
    parser.add_argument("--dx",   type=str, default=CFG.dx_path,   help="Path to DX 4h CSV")
    parser.add_argument("--eur",  type=str, default=CFG.eur_path,  help="Path to 6E 4h CSV")
    parser.add_argument("--spot", type=str, default=CFG.spot_path, help="Path to EURUSD spot 4h CSV")
    parser.add_argument("--out",  type=str, default=CFG.out_dir,   help="Output folder root")

    parser.add_argument("--atr-cap",   type=float, default=CFG.atr_cap_k, help="ATR cap multiple (default 1.5)")
    parser.add_argument("--hard-tp",   type=float, default=CFG.hard_tp_R, help="Hard TP in R (default 4)")
    parser.add_argument("--ignore-atr", action="store_true", help="Ignore ATR entirely (pure swing stop)")
    args = parser.parse_args()

    CFG.dx_path    = args.dx
    CFG.eur_path   = args.eur
    CFG.spot_path  = args.spot
    CFG.out_dir    = args.out

    CFG.atr_cap_k  = args.atr_cap
    CFG.hard_tp_R  = args.hard_tp
    CFG.ignore_atr = args.ignore_atr

    run_pair(CFG)