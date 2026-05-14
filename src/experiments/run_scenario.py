"""Renderable driver for the scripted lane-change scenarios.

Shares its setup with tests/test_near_miss_scenarios.py via
src.experiments.scenarios so the rendered behaviour matches CI.

Examples:
    python -m src.experiments.run_scenario --scenario hard_brake --render
    python -m src.experiments.run_scenario --scenario hard_brake --render \\
        --real-time --slowdown 3
    python -m src.experiments.run_scenario --scenario cutin --gif --seed 2

Display vs GIF are mutually exclusive (gym render_mode constraint).
Each run writes sim_output/<timestamp>_<scenario>/ with the README, six
plots, and -- if --gif was passed -- run.gif.
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import src.envs.cvar_lane_change_env  # noqa: F401  -- registers env id
from src.controllers.mppi import MPPI, MPPIConfig
from src.experiments.scenarios import (
    install_cutin,
    install_hard_brake_leader,
    min_traffic_distance,
)
from src.utils.observation import (
    ego_state,
    normalize_action,
    target_lane_y,
    traffic_state,
)
from src.utils.render_helpers import RenderHelper
from src.utils.run_logger import RunLogger

SCENARIOS = ("hard_brake", "cutin", "lane_change")


def run(
    scenario: str,
    seed: int,
    render: bool,
    gif: bool,
    slowdown: float,
    real_time: bool,
    max_steps: int,
    n_samples: int,
    horizon: int,
    target_lane: int | None,
) -> dict:
    rh = RenderHelper(
        render=render, gif=gif, slowdown=slowdown,
        real_time=real_time, policy_dt=0.05,
    )
    env = gym.make("CVaRLaneChange-v0", render_mode=rh.render_mode)
    env.reset(seed=seed)
    rh.configure_env(env)
    ego = env.unwrapped.vehicle

    extra: dict = {}
    if scenario == "hard_brake":
        leader = install_hard_brake_leader(env, dx_ahead=8.0, brake=-10.0)
        if leader is None:
            env.close()
            raise RuntimeError(
                "no forward same-lane vehicle to install hard-brake leader; "
                "try a different --seed"
            )
        extra = {"forced_dx_ahead": 8.0, "forced_brake_mps2": -10.0}
    elif scenario == "cutin":
        cutin = install_cutin(env, dx_ahead=6.0, dy=4.0, dv=4.0)
        if cutin is None:
            env.close()
            raise RuntimeError(
                "no adjacent-lane vehicle for cut-in; try a different --seed"
            )
        extra = {"forced_dx_ahead": 6.0, "forced_dy": 4.0, "forced_dv": 4.0}
    elif scenario == "lane_change":
        pass

    origin_lane = ego.lane_index[2]
    if target_lane is None:
        target_lane = origin_lane
    origin_y = target_lane_y(env, origin_lane)
    target_y = target_lane_y(env, target_lane)
    lanes_count = int(env.unwrapped.config["lanes_count"])
    lane_centers = [target_lane_y(env, i) for i in range(lanes_count)]
    dt = 1.0 / float(env.unwrapped.config["policy_frequency"])
    rh.policy_dt = dt

    ctrl = MPPI(
        MPPIConfig(
            horizon=horizon,
            n_samples=n_samples,
            dt=dt,
            target_y=target_y,
            seed=seed,
        )
    )
    logger = RunLogger(
        scenario=scenario,
        dt=dt,
        target_y=target_y,
        origin_y=origin_y,
        seed=seed,
        lane_centers=lane_centers,
        extra={
            "origin_lane_index": origin_lane,
            "target_lane_index": target_lane,
            "mppi_n_samples": n_samples,
            "mppi_horizon": horizon,
            **rh.summary(),
            **extra,
        },
    )

    min_d = float("inf")
    crashed = False
    k = 0
    for k in range(max_steps):
        x0 = ego_state(env)
        xt = traffic_state(env)
        u, info = ctrl.step(x0, xt)
        env.step(normalize_action(env, u))
        rh.after_step(env)
        crashed = bool(env.unwrapped.vehicle.crashed)
        logger.log(
            t=k * dt,
            ego_state=x0,
            action_cmd=u,
            traffic_state=xt,
            rollout_info=info,
            crashed=crashed,
        )
        min_d = min(min_d, min_traffic_distance(env))
        if crashed:
            break

    out_dir = logger.save()
    gif_path = rh.finalize(out_dir)
    env.close()
    return {
        "scenario": scenario,
        "crashed": crashed,
        "min_distance": min_d,
        "out_dir": str(out_dir),
        "gif": str(gif_path) if gif_path else None,
        "steps": k + 1,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--render", action="store_true",
                   help="open a pygame window (needs a display)")
    p.add_argument("--gif", action="store_true",
                   help="capture frames headlessly and save run.gif in out_dir")
    p.add_argument("--real-time", action="store_true",
                   help="with --render, pace window at wall-clock sim time")
    p.add_argument("--slowdown", type=float, default=1.0,
                   help="with --render, extra slowdown factor (>1 = slow-mo)")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--target-lane", type=int, default=None,
                   help="for scenario=lane_change; default = origin lane")
    args = p.parse_args()
    out = run(
        scenario=args.scenario,
        seed=args.seed,
        render=args.render,
        gif=args.gif,
        slowdown=args.slowdown,
        real_time=args.real_time,
        max_steps=args.max_steps,
        n_samples=args.n_samples,
        horizon=args.horizon,
        target_lane=args.target_lane,
    )
    for k, v in out.items():
        print(f"{k:>14s}: {v}")


if __name__ == "__main__":
    main()
