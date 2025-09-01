# summary_report_plus.py
# Enhanced summary for probability model outputs.
# Reads .../trades_labeled.csv under instrument folders and computes:
# %BE, Total RR, longest win/loss streaks, Sharpe (per-trade) & annualized,
# Max Drawdown (R), and Time to Recover. Also saves annotated plots:
# - equity_curve.png (per instrument)
# - exit_reason_bar.png (per instrument)
# - equity_curves_overlay.png (root overlay)
#
# Usage:
#   python summary_report_plus.py --root outputs
#   python summary_report_plus.py --folders outputs/6E outputs/SPOT

from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("entry_time", "signal_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # Sort by entry order for equity curve & streaks
    if "entry_time" in df.columns and df["entry_time"].notna().any():
        df = df.sort_values("entry_time")
    elif "signal_time" in df.columns and df["signal_time"].notna().any():
        df = df.sort_values("signal_time")
    return df.reset_index(drop=True)

def streak_lengths(r: pd.Series) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak) by entry order.
    Win: r>0 ; Loss: r<0 ; BE (==0) breaks both streaks."""
    arr = r.values
    max_w = max_l = cur_w = cur_l = 0
    for x in arr:
        if x > 0:
            cur_w += 1; max_w = max(max_w, cur_w)
            cur_l = 0
        elif x < 0:
            cur_l += 1; max_l = max(max_l, cur_l)
            cur_w = 0
        else:
            cur_w = 0; cur_l = 0
    return max_w, max_l

def sharpe_stats(r: pd.Series, entry_times: pd.Series | None) -> tuple[float, float, float]:
    """Return (per_trade_sharpe, annualized_sharpe, trades_per_year)."""
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return (float("nan"), float("nan"), float("nan"))
    mu = r.mean()
    sd = r.std(ddof=1)
    per_trade = float(mu / sd) if sd > 0 else float("nan")

    trades_per_year = float("nan")
    ann = float("nan")
    if entry_times is not None and entry_times.notna().any():
        t = pd.to_datetime(entry_times, utc=True, errors="coerce").dropna()
        if len(t) >= 2:
            years = (t.max() - t.min()).total_seconds() / (365.25 * 24 * 3600)
            if years > 0:
                trades_per_year = len(r) / years
                ann = per_trade * math.sqrt(trades_per_year) if not math.isnan(per_trade) else float("nan")
    return per_trade, ann, trades_per_year

def max_drawdown(equity: pd.Series) -> tuple[float, int, int, int]:
    """Max drawdown on cumulative R equity curve.
    Returns (max_dd, peak_idx, trough_idx, recovery_idx) where indices are trade indices.
    If no recovery, recovery_idx = -1."""
    roll_max = equity.cummax()
    dd = equity - roll_max
    trough_idx = int(dd.idxmin()) if not dd.empty else -1
    max_dd = float(dd.min()) if not dd.empty else 0.0
    if trough_idx == -1:
        return 0.0, -1, -1, -1
    # find the peak that preceded the trough
    peak_idx = int((equity[:trough_idx+1]).idxmax())
    # find recovery point where equity surpasses old peak after trough
    recovery_idx = -1
    if peak_idx >= 0:
        post = equity[trough_idx+1:]
        rec = post[post >= equity[peak_idx]]
        if not rec.empty:
            recovery_idx = int(rec.index[0])
    return max_dd, peak_idx, trough_idx, recovery_idx

def pct(x: float) -> float:
    return float(x)

def plot_equity_curve(tr: pd.DataFrame, out_path: Path, title: str):
    """Plot cumulative R with annotations: max DD, peak/trough/recovery."""
    if tr.empty:
        return
    equity = tr["r_outcome"].cumsum()
    max_dd, peak, trough, rec = max_drawdown(equity)

    plt.figure(figsize=(8,5))
    plt.plot(equity.values)
    plt.title(title)
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative R")

    # annotate peak/trough/recovery if available
    if peak >= 0:
        plt.scatter([peak], [equity.iloc[peak]], marker="o")
        plt.annotate("Peak", (peak, equity.iloc[peak]), xytext=(5,5), textcoords="offset points", fontsize=8)
    if trough >= 0:
        plt.scatter([trough], [equity.iloc[trough]], marker="x")
        plt.annotate(f"Trough\nDD={max_dd:.2f} R", (trough, equity.iloc[trough]),
                     xytext=(5,-20), textcoords="offset points", fontsize=8)
    if rec >= 0:
        plt.scatter([rec], [equity.iloc[rec]], marker="^")
        plt.annotate("Recovery", (rec, equity.iloc[rec]), xytext=(5,5), textcoords="offset points", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)

def plot_exit_reasons(tr: pd.DataFrame, out_path: Path, title: str):
    """Bar chart of exit reason distribution."""
    if tr.empty or "reason" not in tr.columns:
        return
    counts = tr["reason"].value_counts(normalize=True).sort_values(ascending=False)
    plt.figure(figsize=(7,4))
    counts.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Fraction of Trades")
    plt.tight_layout()
    plt.savefig(out_path)

def summarize_instrument(folder: Path) -> dict:
    csv_path = folder / "trades_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing trades_labeled.csv in {folder}")

    tr = load_trades(csv_path)

    # Save plots
    plot_equity_curve(tr, folder / "equity_curve.png", f"{folder.name} — Equity Curve (Cumulative R)")
    plot_exit_reasons(tr, folder / "exit_reason_bar.png", f"{folder.name} — Exit Reason Distribution")

    n = len(tr)
    if n == 0:
        return {
            "Instrument": folder.name, "Trades": 0,
            "Win%": np.nan, "Avg R": np.nan, "Total RR": 0.0,
            "%Stop": np.nan, "%TP(hard)": np.nan, "%MtM": np.nan, "%BE": np.nan,
            "Max Win Streak": 0, "Max Loss Streak": 0,
            "Max DD (R)": np.nan, "Time to Recover (trades)": np.nan,
            "Sharpe (per-trade)": np.nan, "Sharpe (annual)": np.nan, "Trades/Year": np.nan
        }

    # basic rates
    win = (tr["r_outcome"] > 0).mean()
    avgR = tr["r_outcome"].mean()
    totR = tr["r_outcome"].sum()

    rc = tr["reason"].value_counts(normalize=True)
    p_stop = float(rc.get("stop", 0.0))
    p_mtm  = float(rc.get("mtm", 0.0))
    p_hard = float(rc[[c for c in rc.index if str(c).startswith("hard_tp")]].sum()) if len(rc) else 0.0

    # exactly 0R (breakeven) — allow tiny tolerance
    be = (tr["r_outcome"].abs() <= 1e-10).mean()

    # streaks
    max_w, max_l = streak_lengths(tr["r_outcome"])

    # sharpe
    per_trade_sharpe, ann_sharpe, tpy = sharpe_stats(tr["r_outcome"], tr.get("entry_time"))

    # drawdown on cumulative R
    eq = tr["r_outcome"].cumsum()
    dd, peak, trough, rec = max_drawdown(eq)
    time_to_recover = (rec - trough) if (rec >= 0 and trough >= 0) else np.nan

    return {
        "Instrument": folder.name,
        "Trades": int(n),
        "Win%": round(pct(win), 4),
        "Avg R": round(avgR, 4),
        "Total RR": round(totR, 2),
        "%Stop": round(p_stop, 4),
        "%TP(hard)": round(p_hard, 4),
        "%MtM": round(p_mtm, 4),
        "%BE": round(pct(be), 4),
        "Max Win Streak": int(max_w),
        "Max Loss Streak": int(max_l),
        "Max DD (R)": round(dd, 2) if not np.isnan(dd) else np.nan,
        "Time to Recover (trades)": int(time_to_recover) if not np.isnan(time_to_recover) else np.nan,
        "Sharpe (per-trade)": round(per_trade_sharpe, 4) if not np.isnan(per_trade_sharpe) else np.nan,
        "Sharpe (annual)": round(ann_sharpe, 4) if not np.isnan(ann_sharpe) else np.nan,
        "Trades/Year": round(tpy, 2) if not np.isnan(tpy) else np.nan,
    }

def write_markdown(df: pd.DataFrame, path: Path):
    lines = ["# Enhanced Summary Report\n"]
    lines.append("This report consolidates **key portfolio-quality metrics** computed from `trades_labeled.csv` per instrument.\n")
    lines.append("It includes **%BE**, **Total RR**, **Max Drawdown (R)**, **Time to Recover**, **Streaks**, and **Sharpe ratios**.\n")
    lines.append("## Overview (Table)\n")
    try:
        lines.append(df.to_markdown(index=False))
    except Exception:
        lines.append("```\n" + df.to_string(index=False) + "\n```")

    lines.append("\n## How to read these metrics\n")
    lines.append("- **Trades**: number of executed signals.\n"
                 "- **Win%**: fraction of trades with `r_outcome > 0`.\n"
                 "- **Avg R**: average R multiple per trade.\n"
                 "- **Total RR**: sum of R across all trades (e.g., if 1R = 1% risk, this is % return for the test period).\n"
                 "- **%Stop / %TP(hard) / %MtM**: exit reason distribution (hard TP is the +R cap you set; MtM = mark-to-market at end if used).\n"
                 "- **%BE**: percent of trades at exactly 0R (within 1e-10 tolerance).\n"
                 "- **Max Win/Loss Streak**: longest consecutive winning/losing sequences (0R breaks both streaks).\n"
                 "- **Max DD (R)**: worst peak-to-trough decline on the cumulative R equity curve.\n"
                 "- **Time to Recover**: trades needed to recover from the max drawdown to prior equity peak.\n"
                 "- **Sharpe (per-trade)**: mean(R)/std(R) using trade outcomes.\n"
                 "- **Sharpe (annual)**: per-trade Sharpe × √(trades/year). Trades/year inferred from entry-time span.\n")

    lines.append("\n## Plots generated per instrument\n"
                 "- `equity_curve.png`: cumulative R with **Peak**/**Trough**/**Recovery** annotated; DD label at trough.\n"
                 "- `exit_reason_bar.png`: exit reason distribution (fractions).\n")

    lines.append("\n## Overlay plot\n"
                 "- At the root, `equity_curves_overlay.png` shows all instruments' equity curves together for quick comparison.\n")
    lines.append("\n*Generated by summary_report_plus.py*\n")
    path.write_text("\n".join(lines), encoding="utf-8")

def overlay_equity_curves(folders: list[Path], out_path: Path):
    plt.figure(figsize=(8,5))
    plotted = False
    for f in folders:
        csv_path = f / "trades_labeled.csv"
        if not csv_path.exists():
            continue
        tr = load_trades(csv_path)
        if tr.empty:
            continue
        eq = tr["r_outcome"].cumsum()
        plt.plot(eq.values, label=f.name)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title("Equity Curves Overlay (Cumulative R)")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative R")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs",
                    help="Root folder containing instrument subfolders (default: outputs)")
    ap.add_argument("--folders", type=str, nargs="*", default=None,
                    help="Explicit instrument folders (e.g., outputs/6E outputs/SPOT)")
    args = ap.parse_args()

    if args.folders:
        folders = [Path(p) for p in args.folders]
    else:
        root = Path(args.root)
        # autodiscover immediate subfolders with trades_labeled.csv
        folders = [p for p in root.iterdir() if p.is_dir() and (p / "trades_labeled.csv").exists()]
        if not folders:
            raise SystemExit(f"No instrument folders with trades_labeled.csv found under {root}")

    rows = []
    for f in folders:
        try:
            rows.append(summarize_instrument(f))
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    if not rows:
        raise SystemExit("No summaries generated.")

    df = pd.DataFrame(rows)
    out_root = Path(args.root)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "summary_overview_plus.csv"
    md_path  = out_root / "summary_report_plus.md"

    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)

    # Overlay plot
    overlay_equity_curves(folders, out_root / "equity_curves_overlay.png")

    print("\n=== Enhanced Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {out_root / 'equity_curves_overlay.png'}")

if __name__ == "__main__":
    main()
