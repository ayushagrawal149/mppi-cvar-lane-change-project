"""Reusable scripted near-miss scenarios.

Shared by tests/test_near_miss_scenarios.py and the visual driver
src/experiments/run_scenario.py, so a rendered window shows the *same*
behaviour that pytest exercises.

HighwayEnv's IDMVehicle.act() overwrites externally-set actions each
step, so to force a specific hazard we replace the targeted vehicle
with a plain `Vehicle` whose action dict persists across steps.
"""

from __future__ import annotations

import numpy as np
from highway_env.vehicle.kinematics import Vehicle


def min_traffic_distance(env) -> float:
    ego = env.unwrapped.vehicle
    others = [v for v in env.unwrapped.road.vehicles if v is not ego]
    if not others:
        return float("inf")
    return float(
        min(
            np.linalg.norm(np.array(v.position) - np.array(ego.position))
            for v in others
        )
    )


def replace_with_scripted(env, victim, position, heading, speed, action):
    road = env.unwrapped.road
    scripted = Vehicle(road, position, heading, speed)
    scripted.action = action
    road.vehicles[road.vehicles.index(victim)] = scripted
    return scripted


def install_hard_brake_leader(env, dx_ahead: float = 8.0, brake: float = -10.0):
    """Replace the closest forward same-lane vehicle with a scripted brake."""
    ego = env.unwrapped.vehicle
    candidates = [
        v
        for v in env.unwrapped.road.vehicles
        if v is not ego
        and v.position[0] > ego.position[0]
        and abs(v.position[1] - ego.position[1]) < 2.0
    ]
    if not candidates:
        return None
    leader = min(candidates, key=lambda v: v.position[0] - ego.position[0])
    return replace_with_scripted(
        env,
        leader,
        position=np.array([ego.position[0] + dx_ahead, ego.position[1]]),
        heading=0.0,
        speed=ego.speed,
        action={"acceleration": brake, "steering": 0.0},
    )


def install_cutin(
    env,
    dx_ahead: float = 6.0,
    dy: float = 4.0,
    dv: float = 4.0,
    into_lane_steer: float = -0.05,
):
    """Replace an adjacent-lane vehicle with a scripted cut-in."""
    ego = env.unwrapped.vehicle
    candidates = [
        v
        for v in env.unwrapped.road.vehicles
        if v is not ego and 3.0 < abs(v.position[1] - ego.position[1]) < 5.0
    ]
    if not candidates:
        return None
    cutin = candidates[0]
    return replace_with_scripted(
        env,
        cutin,
        position=np.array([ego.position[0] + dx_ahead, ego.position[1] + dy]),
        heading=np.sign(-dy) * abs(into_lane_steer),
        speed=ego.speed + dv,
        action={"acceleration": 0.0, "steering": into_lane_steer},
    )
