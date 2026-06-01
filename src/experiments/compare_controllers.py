"""Paired head-to-head: MPPI vs CVaR-MPPI on the same seeds × scenarios.

Produces a CSV of per-run metrics and a printed summary table:

    scenario   controller   N   collisions  near_misses  T_LC(mean)  a_y_peak
    ─────────  ───────────  ──  ──────────  ───────────  ──────────  ────────
    hard_brake mppi         20  6/20=30%    11/20=55%    4.8s        2.4
    hard_brake cvar_mppi    20  1/20=5%     3/20=15%     5.0s        1.9
    cutin      mppi         20  4/20=20%    9/20=45%     N/A         N/A
    cutin      cvar_mppi    20  0/20=0%     2/20=10%     N/A         N/A
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path

import gymnasium as gym
import numpy as np

import src.envs.cvar_lane_change_env  # noqa: F401
from src.controllers.mppi import MPPI, MPPIConfig
from src.controllers.cvar_mppi import CVaRMPPI, CVaRMPPIConfig
from src.experiments.scenarios import (
    install_cutin, install_hard_brake_leader, min_traffic_distance,
)
from src.utils.observation import (
    ego_state, normalize_action, target_lane_y, traffic_state,
)
from src.utils.run_logger import RunLogger
from src.utils.metrics import aggregate_runs


CTRL_FACTORIES = {
    "mppi": lambda dt, target_y, seed, args: MPPI(MPPIConfig(
        horizon=args.horizon, n_samples=args.n_samples, dt=dt,
        target_y=target_y, seed=seed,
    )),
    "cvar_mppi": lambda dt, target_y, seed, args: CVaRMPPI(CVaRMPPIConfig(
        horizon=args.horizon, n_samples=args.n_samples, dt=dt,
        target_y=target_y, seed=seed,
        n_disturb=args.n_disturb, alpha=args.alpha,
        risk_budget=args.risk_budget, penalty_weight=args.penalty_weight,
        idm_sigma_a=args.idm_sigma_a,
    )),
}


def run_one(controller_name, scenario, seed, args):
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=seed)
    setup_extra = {}
    if scenario == "hard_brake":
        if install_hard_brake_leader(env, 8.0, -10.0) is None:
            env.close(); return None
    elif scenario == "cutin":
        if install_cutin(env, 6.0, 4.0, 4.0) is None:
            env.close(); return None
    # lane_change: nothing to install — natural traffic

    ego = env.unwrapped.vehicle
    origin_lane = ego.lane_index[2]
    target_lane = origin_lane if scenario != "lane_change" else (origin_lane + 1) % 3
    target_y = target_lane_y(env, target_lane)
    origin_y = target_lane_y(env, origin_lane)
    dt = 1.0 / float(env.unwrapped.config["policy_frequency"])
    ctrl = CTRL_FACTORIES[controller_name](dt, target_y, seed, args)

    logger = RunLogger(
        scenario=f"{scenario}_{controller_name}", dt=dt,
        target_y=target_y, origin_y=origin_y, seed=seed,
        extra={"controller": controller_name, **setup_extra},
    )
    min_d = float("inf"); crashed = False
    for k in range(args.max_steps):
        x0 = ego_state(env); xt = traffic_state(env)
        u, info = ctrl.step(x0, xt)
        env.step(normalize_action(env, u))
        crashed = bool(env.unwrapped.vehicle.crashed)
        logger.log(t=k*dt, ego_state=x0, action_cmd=u, traffic_state=xt,
                   rollout_info=info, crashed=crashed)
        min_d = min(min_d, min_traffic_distance(env))
        if crashed:
            break
    env.close()
    m = logger.summary()
    if getattr(args, "save_each", False):
        logger.save()
    return {
        "scenario": scenario, "controller": controller_name, "seed": seed,
        "crashed": int(crashed), "min_distance": float(min_d),
        "ttc_min": m["ttc_min"], "a_y_peak": m["a_y_peak"],
        "j_y_peak": m["j_y_peak"], "t_lc": m["t_lc"],
        "near_miss": int(m["ttc_min"] < 1.5),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+",
                   default=["hard_brake", "cutin", "lane_change"])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=80)
    # CVaR knobs:
    p.add_argument("--n-disturb", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--risk-budget", type=float, default=4.0)
    p.add_argument("--penalty-weight", type=float, default=80.0)
    p.add_argument("--idm-sigma-a", type=float, default=1.5,
                   help="σ_a for the disturbed-traffic predictor F̃ "
                        "(raise to widen CVaR's tail-event coverage).")
    p.add_argument("--save-each", action="store_true",
                   help="Persist each run's npz/plots under sim_output/.")
    p.add_argument("--out", type=str, default="sim_output/compare.csv")
    args = p.parse_args()

    rows = []
    for scenario in args.scenarios:
        for controller in ("mppi", "cvar_mppi"):
            for s in range(args.seed_start, args.seed_start + args.seeds):
                r = run_one(controller, scenario, s, args)
                if r is not None:
                    rows.append(r)
    if not rows:
        print("No runs completed — nothing to write.")
        return
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        for r in rows: w.writerow(r)

    # console summary
    print(f"\nResults written to {out}")
    print(f"{'scenario':12s} {'controller':10s} {'N':>3s} {'crash%':>7s} "
          f"{'nearmiss%':>10s} {'T_LC':>6s} {'a_y':>5s}")
    by = {}
    for r in rows:
        by.setdefault((r["scenario"], r["controller"]), []).append(r)
    for (sc, ct), rs in sorted(by.items()):
        a = aggregate_runs(rs)
        print(f"{sc:12s} {ct:10s} {a['n']:3d} "
            f"{100*a['collision_rate']:6.1f}% {100*a['near_miss_rate']:9.1f}% "
            f"{a['t_lc_mean']:5.2f}s {a['a_y_peak_mean']:4.2f}")


if __name__ == "__main__":
    main()
