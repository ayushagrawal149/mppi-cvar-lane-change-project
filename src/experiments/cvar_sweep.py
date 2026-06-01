"""Sweep α and risk_budget; report the safety-vs-performance frontier."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

from src.experiments.compare_controllers import run_one


class _Args:
    horizon = 30; n_samples = 256; max_steps = 80
    n_disturb = 24; penalty_weight = 80.0
    alpha = 0.9; risk_budget = 4.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.7, 0.8, 0.9, 0.95, 0.99])
    p.add_argument("--budgets", nargs="+", type=float,
                   default=[2.0, 4.0, 8.0, 16.0])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--scenario", default="hard_brake")
    p.add_argument("--out", default="sim_output/cvar_sweep.csv")
    args = p.parse_args()

    cfg = _Args()
    rows = []
    for alpha in args.alphas:
        for budget in args.budgets:
            cfg.alpha = alpha; cfg.risk_budget = budget
            crash = 0; nm = 0; tlc = []; n = 0
            for s in range(args.seeds):
                r = run_one("cvar_mppi", args.scenario, s, cfg)
                if r is None: continue
                n += 1; crash += r["crashed"]; nm += r["near_miss"]
                tlc.append(r["t_lc"])
            rows.append({
                "alpha": alpha, "budget": budget, "N": n,
                "crash_rate": crash / max(n, 1),
                "near_miss_rate": nm / max(n, 1),
                "t_lc_mean": float(np.nanmean(tlc)) if tlc else float("nan"),
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
