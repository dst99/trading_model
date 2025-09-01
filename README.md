# Trading Probability Model — DX Sweeps → EURUSD / 6E

Research framework to backtest and **learn win probabilities** for a DX-sweep strategy, executing on **EURUSD spot** and **6E futures**.  
It simulates entries, stop management and exits, then trains a classifier to estimate the **probability a trade reaches ≥ +1R**.

> ⚠️ Research only — not investment advice. Validate out-of-sample before any live use.


---

## Features
- **Signal**: DX (Dollar Index) **liquidity sweep rejection** (4H/daily configurable).
- **Entry**: Next 4H bar open on EURUSD/6E after DX signal (inverse direction to DX).
- **Stops**:
  - Swing high/low–based.
  - **Minimum stop floor** enforced (default **1 pip = 0.0001**).
  - Optional **minimum USD risk** per 6E contract (via pip value).
  - ATR cap exists but **disabled by default** to avoid micro-stops.
- **Management**:
  - Trail stop **every +1R**.
  - **Hard TP** at **+4R**.
- **Outputs**:
  - `trades_labeled.csv` (entries, exits, R outcomes).
  - Calibration plot, bucketed expectancy, equity curves, exit reason bars.
  - Enhanced summary with Sharpe, streaks, drawdowns, and **R-bucket distribution** (−1R, 0R, +1R…+4R).
- **Probability model**:
  - Gradient Boosting (scikit-learn) with **time-series cross-validation**.
  - Reports AUC, Brier, LogLoss + reliability (calibration) curve.

---

## Quick Start
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put your CSVs in ./data (see format below)

# 4) Run model (writes per-instrument outputs under ./outputs)
python prob_model.py --dx data/DX_4h.csv --eur data/6E_4h.csv --spot data/EURUSD_4h.csv --out outputs

# 5) Generate consolidated reports & plots
python summary_report_plus.py --root outputs

--hard-tp 4            # hard take-profit in R (default 4)
--atr-cap 1.5          # ATR multiple if you enable ATR capping
--ignore-atr           # keep ATR disabled (default True in code)

python summary_report_plus.py --root outputs


