"""Run the MPPI lane-change baseline on CVaRLaneChange-v0.

Driver for the Week-4 'Initial Results' milestone: runs a single episode,
drives the ego with risk-neutral MPPI, and (by default) saves a
timestamped folder under sim_output/ containing plots, a performance
README.md, the raw per-step run.npz, and (optionally) run.gif.

    python -m src.experiments.run_baseline --target-lane 0 --seed 0
    python -m src.experiments.run_baseline --render --real-time --slowdown 2
    python -m src.experiments.run_baseline --gif --target-lane 0
    python -m src.experiments.run_baseline --no-log     # skip sim_output/
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import src.envs.cvar_lane_change_env  # noqa: F401  -- registers env id
from src.controllers.mppi import MPPI, MPPIConfig
from src.utils.observation import (
    ego_state,
    normalize_action,
    target_lane_y,
    traffic_state,
)
from src.utils.render_helpers import RenderHelper
from src.utils.run_logger import RunLogger


def run_episode(
    seed: int = 0,
    target_lane: int = 0,
    render: bool = False,
    gif: bool = False,
    slowdown: float = 1.0,
    real_time: bool = False,
    max_steps: int = 400,
    n_samples: int = 256,
    horizon: int = 30,
    log: bool = True,
    scenario: str = "baseline",
) -> dict:
    rh = RenderHelper(
        render=render, gif=gif, slowdown=slowdown,
        real_time=real_time, policy_dt=0.05,
    )
    env = gym.make("CVaRLaneChange-v0", render_mode=rh.render_mode)
    env.reset(seed=seed)
    rh.configure_env(env)

    target_y = target_lane_y(env, target_lane)
    origin_lane = env.unwrapped.vehicle.lane_index[2]
    origin_y = target_lane_y(env, origin_lane)
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

    logger = (
        RunLogger(
            scenario=scenario,
            dt=dt,
            target_y=target_y,
            origin_y=origin_y,
            seed=seed,
            extra={
                "target_lane_index": target_lane,
                "origin_lane_index": origin_lane,
                "mppi_n_samples": n_samples,
                "mppi_horizon": horizon,
                **rh.summary(),
            },
        )
        if log
        else None
    )

    total_reward = 0.0
    steps_taken = 0
    for steps_taken in range(1, max_steps + 1):
        x0 = ego_state(env)
        xt = traffic_state(env)
        u, info = ctrl.step(x0, xt)
        action = normalize_action(env, u)
        _, reward, terminated, truncated, _ = env.step(action)
        rh.after_step(env)
        total_reward += float(reward)
        if logger is not None:
            logger.log(
                t=(steps_taken - 1) * dt,
                ego_state=x0,
                action_cmd=u,
                traffic_state=xt,
                rollout_info=info,
                crashed=bool(env.unwrapped.vehicle.crashed),
            )
        if terminated or truncated:
            break

    crashed = bool(env.unwrapped.vehicle.crashed)
    final_y = float(env.unwrapped.vehicle.position[1])

    out_dir = logger.save() if logger is not None else None
    gif_path = rh.finalize(out_dir) if out_dir is not None else None
    env.close()
    return {
        "steps": steps_taken,
        "total_reward": total_reward,
        "crashed": crashed,
        "final_y": final_y,
        "target_y": float(target_y),
        "out_dir": str(out_dir) if out_dir is not None else None,
        "gif": str(gif_path) if gif_path else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-lane", type=int, default=0)
    p.add_argument("--render", action="store_true",
                   help="open a pygame window (needs a display)")
    p.add_argument("--gif", action="store_true",
                   help="capture frames headlessly and save run.gif in out_dir")
    p.add_argument("--real-time", action="store_true",
                   help="with --render, pace window at wall-clock sim time")
    p.add_argument("--slowdown", type=float, default=1.0,
                   help="with --render, extra slowdown factor (>1 = slow-mo)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--no-log", action="store_true",
                   help="skip sim_output/ folder generation")
    p.add_argument("--scenario", type=str, default="baseline",
                   help="scenario tag appended to the output folder name")
    args = p.parse_args()
    out = run_episode(
        seed=args.seed,
        target_lane=args.target_lane,
        render=args.render,
        gif=args.gif,
        slowdown=args.slowdown,
        real_time=args.real_time,
        max_steps=args.max_steps,
        n_samples=args.n_samples,
        horizon=args.horizon,
        log=not args.no_log,
        scenario=args.scenario,
    )
    for k, v in out.items():
        print(f"{k:>14s}: {v}")


if __name__ == "__main__":
    main()
