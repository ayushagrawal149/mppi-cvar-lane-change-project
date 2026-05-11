"""Extract continuous state vectors and lane geometry from a HighwayEnv.

The default Kinematics observation truncates / normalises in ways MPPI
doesn't want, so we read the underlying `road.vehicles` list directly.
"""

from __future__ import annotations

import numpy as np


def ego_state(env) -> np.ndarray:
    """Return the controlled vehicle's state (X, Y, psi, v)."""
    veh = env.unwrapped.vehicle
    return np.array(
        [veh.position[0], veh.position[1], veh.heading, veh.speed],
        dtype=np.float64,
    )


def traffic_state(env) -> np.ndarray:
    """Return surrounding-vehicle states as an (N, 4) array."""
    ego = env.unwrapped.vehicle
    others = [v for v in env.unwrapped.road.vehicles if v is not ego]
    if not others:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array(
        [
            [v.position[0], v.position[1], v.heading, v.speed]
            for v in others
        ],
        dtype=np.float64,
    )


def target_lane_y(env, lane_index: int) -> float:
    """Y-coordinate of the centerline of `lane_index` on the straight road."""
    road = env.unwrapped.road
    from_node = next(iter(road.network.graph.keys()))
    to_node = next(iter(road.network.graph[from_node].keys()))
    lane = road.network.graph[from_node][to_node][lane_index]
    return float(lane.position(0.0, 0.0)[1])


def normalize_action(env, u: np.ndarray) -> np.ndarray:
    """Map (a, delta) in physical units to the env's [-1, 1] action space."""
    at = env.unwrapped.action_type
    a_lo, a_hi = at.acceleration_range
    s_lo, s_hi = at.steering_range
    a_norm = 2.0 * (u[0] - a_lo) / (a_hi - a_lo) - 1.0
    d_norm = 2.0 * (u[1] - s_lo) / (s_hi - s_lo) - 1.0
    return np.clip(np.array([a_norm, d_norm], dtype=np.float32), -1.0, 1.0)
