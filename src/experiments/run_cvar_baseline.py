"""Run the CVaR-MPPI lane-change baseline on CVaRLaneChange-v0.

Mirrors src/experiments/run_baseline.py so the two outputs are directly
comparable (same plots, same logger, same metrics).
"""

from __future__ import annotations
import argparse
import gymnasium as gym

import src.envs.cvar_lane_change_env  # noqa: F401
from src.controllers.cvar_mppi import CVaRMPPI, CVaRMPPIConfig
from src.utils.observation import (
    ego_state, normalize_action, target_lane_y, traffic_state,
)
from src.utils.render_helpers import RenderHelper
from src.utils.run_logger import RunLogger


def run_episode(
    seed=0, target_lane=0, render=False, gif=False, slowdown=1.0,
    real_time=False, max_steps=400, n_samples=256, horizon=30,
    n_disturb=24, alpha=0.9, risk_budget=4.0, penalty_weight=80.0,
    log=True, scenario="cvar_baseline",
):
    rh = RenderHelper(render=render, gif=gif, slowdown=slowdown,
                     real_time=real_time, policy_dt=0.05)
    env = gym.make("CVaRLaneChange-v0", render_mode=rh.render_mode)
    env.reset(seed=seed)
    rh.configure_env(env)

    target_y = target_lane_y(env, target_lane)
    origin_lane = env.unwrapped.vehicle.lane_index[2]
    origin_y = target_lane_y(env, origin_lane)
    lanes_count = int(env.unwrapped.config["lanes_count"])
    lane_centers = [target_lane_y(env, i) for i in range(lanes_count)]
    dt = 1.0 / float(env.unwrapped.config["policy_frequency"])
    rh.policy_dt = dt

    ctrl = CVaRMPPI(CVaRMPPIConfig(
        horizon=horizon, n_samples=n_samples, dt=dt,
        target_y=target_y, seed=seed,
        n_disturb=n_disturb, alpha=alpha,
        risk_budget=risk_budget, penalty_weight=penalty_weight,
    ))

    logger = RunLogger(
        scenario=scenario, dt=dt, target_y=target_y, origin_y=origin_y,
        seed=seed, lane_centers=lane_centers,
        extra={
            "controller": "cvar_mppi",
            "alpha": alpha, "risk_budget": risk_budget,
            "penalty_weight": penalty_weight, "n_disturb": n_disturb,
            "mppi_n_samples": n_samples, "mppi_horizon": horizon,
            "target_lane_index": target_lane,
            "origin_lane_index": origin_lane,
            **rh.summary(),
        },
    ) if log else None

    total_reward = 0.0; steps_taken = 0
    for steps_taken in range(1, max_steps + 1):
        x0 = ego_state(env); xt = traffic_state(env)
        u, info = ctrl.step(x0, xt)
        _, reward, terminated, truncated, _ = env.step(normalize_action(env, u))
        rh.after_step(env)
        total_reward += float(reward)
        if logger is not None:
            logger.log(t=(steps_taken-1)*dt, ego_state=x0, action_cmd=u,
                       traffic_state=xt, rollout_info=info,
                       crashed=bool(env.unwrapped.vehicle.crashed))
        if terminated or truncated:
            break

    out_dir = logger.save() if logger is not None else None
    gif_path = rh.finalize(out_dir) if out_dir is not None else None
    env.close()
    return {
        "steps": steps_taken, "total_reward": total_reward,
        "crashed": bool(env.unwrapped.vehicle.crashed),
        "final_y": float(env.unwrapped.vehicle.position[1]),
        "target_y": float(target_y),
        "out_dir": str(out_dir) if out_dir is not None else None,
        "gif": str(gif_path) if gif_path else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-lane", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--real-time", action="store_true")
    p.add_argument("--slowdown", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--n-disturb", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--risk-budget", type=float, default=4.0)
    p.add_argument("--penalty-weight", type=float, default=80.0)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--scenario", type=str, default="cvar_baseline")
    args = p.parse_args()
    out = run_episode(**{k: v for k, v in vars(args).items() if k != "no_log"},
                      log=not args.no_log)
    for k, v in out.items():
        print(f"{k:>14s}: {v}")


if __name__ == "__main__":
    main()
