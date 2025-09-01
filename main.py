import argparse
from config import Config
from pipeline import Pipeline

def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx",   type=str, default=cfg.dx_path)
    ap.add_argument("--eur",  type=str, default=cfg.eur_path)
    ap.add_argument("--spot", type=str, default=cfg.spot_path)
    ap.add_argument("--out",  type=str, default=cfg.out_dir)

    ap.add_argument("--atr-cap",   type=float, default=cfg.atr_cap_k)
    ap.add_argument("--hard-tp",   type=float, default=cfg.hard_tp_R)
    ap.add_argument("--ignore-atr", action="store_true", help="(Keep) ATR disabled (default)")
    ap.add_argument("--min-stop",  type=float, default=cfg.min_stop_dist)
    ap.add_argument("--min-usd",   type=float, default=cfg.min_usd_risk_per_contract or 0.0)

    args = ap.parse_args()

    cfg.dx_path  = args.dx
    cfg.eur_path = args.eur
    cfg.spot_path= args.spot
    cfg.out_dir  = args.out

    cfg.atr_cap_k  = args.atr_cap
    cfg.hard_tp_R  = args.hard_tp
    cfg.ignore_atr = True or args.ignore_atr  # ATR OFF by default

    cfg.min_stop_dist = args.min_stop
    cfg.min_usd_risk_per_contract = args.min_usd if args.min_usd > 0 else None

    Pipeline(cfg).run()

if __name__ == "__main__":
    main()
