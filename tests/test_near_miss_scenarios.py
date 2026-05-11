"""Scripted near-miss scenarios that stress the risk-neutral MPPI baseline.

These cases motivate the CVaR augmentation: a planner that minimises the
*expected* cost can still produce trajectories whose worst-case outcomes
are unsafe. We force two canonical setups:

1. Hard-brake leader: the closest forward same-lane vehicle is replaced
   by a scripted, non-IDM vehicle that decelerates at -10 m/s^2.
2. Cut-in: an adjacent-lane vehicle is replaced by a scripted vehicle
   placed close ahead with a heading aimed into the ego's lane.

Vehicles in HighwayEnv are IDM/MOBIL by default, and `IDMVehicle.act()`
overwrites any externally-set action -- so we replace the targeted
vehicle with a plain `Vehicle` instance whose `action` dict persists.

Each test produces a timestamped sim_output/<stamp>_<scenario>/ folder
with the README.md performance table and the six diagnostic plots.
We do NOT assert no-crash here -- the *point* is to expose tail-risk
failures of the risk-neutral baseline. Once CVaR-MPPI is wired in,
regression assertions can require strictly larger min-distance / fewer
crashes on these scenarios.
"""

import gymnasium as gym
import pytest

import src.envs.cvar_lane_change_env  # noqa: F401
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
from src.utils.run_logger import RunLogger


def _run_with_logger(env, scenario, seed, n_steps=40, n_samples=128, horizon=20,
                     target_lane_override=None, extra=None):
    ego = env.unwrapped.vehicle
    origin_lane = ego.lane_index[2]
    target_lane = origin_lane if target_lane_override is None else target_lane_override
    origin_y = target_lane_y(env, origin_lane)
    target_y = target_lane_y(env, target_lane)
    dt = 1.0 / float(env.unwrapped.config["policy_frequency"])

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
        extra={
            "origin_lane_index": origin_lane,
            "target_lane_index": target_lane,
            "mppi_n_samples": n_samples,
            "mppi_horizon": horizon,
            **(extra or {}),
        },
    )

    min_d = float("inf")
    for k in range(n_steps):
        x0 = ego_state(env)
        xt = traffic_state(env)
        u, info = ctrl.step(x0, xt)
        env.step(normalize_action(env, u))
        crashed_now = bool(env.unwrapped.vehicle.crashed)
        logger.log(
            t=k * dt,
            ego_state=x0,
            action_cmd=u,
            traffic_state=xt,
            rollout_info=info,
            crashed=crashed_now,
        )
        min_d = min(min_d, min_traffic_distance(env))
        if crashed_now:
            break
    return logger, min_d


def test_hard_brake_leader_runs_without_error():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=1)
    leader = install_hard_brake_leader(env, dx_ahead=8.0, brake=-10.0)
    if leader is None:
        env.close()
        pytest.skip("no forward same-lane vehicle available for setup")
    logger, min_d = _run_with_logger(
        env,
        scenario="hard_brake",
        seed=1,
        extra={"forced_dx_ahead": 8.0, "forced_brake_mps2": -10.0},
    )
    crashed = env.unwrapped.vehicle.crashed
    env.close()
    out_dir = logger.save()
    print({"scenario": "hard_brake", "crashed": crashed,
           "min_distance": min_d, "out_dir": str(out_dir)})


def test_cutin_vehicle_runs_without_error():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=2)
    cutin = install_cutin(env, dx_ahead=6.0, dy=4.0, dv=4.0)
    if cutin is None:
        env.close()
        pytest.skip("no adjacent-lane vehicle available for cut-in setup")
    logger, min_d = _run_with_logger(
        env,
        scenario="cutin",
        seed=2,
        n_steps=30,
        extra={"forced_dx_ahead": 6.0, "forced_dy": 4.0, "forced_dv": 4.0},
    )
    crashed = env.unwrapped.vehicle.crashed
    env.close()
    out_dir = logger.save()
    print({"scenario": "cutin", "crashed": crashed,
           "min_distance": min_d, "out_dir": str(out_dir)})
