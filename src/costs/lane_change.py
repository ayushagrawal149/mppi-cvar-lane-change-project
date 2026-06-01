"""Per-step cost for the highway lane-change task.

q(x, u) = w_lane         * nearest_lane_dist(Y)^2       (be on *any* lane)
        + w_target_pref  * (Y - Y_target)^2             (weak preference for target lane)
        + w_speed        * (v - v_desired)^2
        + w_heading      * psi^2                        (lateral damping)
        + w_collide      * collision_potential(x, traffic)
        + w_thw          * max(0, target_thw - THW)^2   (2-second rule)
        + w_road         * road_boundary_cost(Y)        (off-road barrier)
        + w_action       * effort

The lane and target_pref terms separate "stay on *a* lane" (active between
lane centers) from "prefer the target lane" (active anywhere except at
the target). This decoupling lets the controller switch lanes during
emergencies without fighting a strong target-tracking penalty.

The collision term is a smooth radial potential so MPPI's importance
weights have a useful gradient even when no rollout actually crashes; a
hard-collision indicator (hard_collision below) is used only for
post-hoc evaluation and tests.
"""


from __future__ import annotations

import numpy as np

EGO_LENGTH = 5.0
EGO_WIDTH = 2.0
LANE_CENTERS = np.array([0.0, 4.0, 8.0])              # 3-lane geometry
# TODO: The lane geometry is hardcoded (should be linked to env config)


def collision_potential(
    ego_xy: np.ndarray,
    traffic_xy: np.ndarray,
    sigma_long: float = 4.0,
    sigma_lat: float = 1.2,
) -> np.ndarray:
    """Sum of anisotropic Gaussian potentials around each traffic vehicle.

    ego_xy:     (..., 2)
    traffic_xy: (..., N, 2)  -- last axes broadcast against ego_xy[..., None, :]
    Returns:    (...)        scalar potential per ego sample.
    """
    diff = ego_xy[..., None, :] - traffic_xy
    dx = diff[..., 0]
    dy = diff[..., 1]
    return np.sum(
        np.exp(-0.5 * ((dx / sigma_long) ** 2 + (dy / sigma_lat) ** 2)),
        axis=-1,
    )


def time_headway_cost(
    ego_state: np.ndarray,
    traffic_xy: np.ndarray,
    target_thw: float = 2.0,
    lane_half_width: float = 2.0,
    v_min: float = 1.0,
) -> np.ndarray:
    """Squared shortfall of time-headway below `target_thw` (the 2-second rule).

    For each ego sample, find the closest same-lane vehicle *ahead*, compute
    THW = dx / v_ego, and return max(0, target_thw - THW)^2. No lead
    ahead -> 0 (no penalty). Distinct from CVaR's TTC risk cost (which
    uses closing rate, not ego speed) and lives in a separate file later.

    ego_state:  (..., 4)  [X, Y, psi, v]
    traffic_xy: (..., N, 2)
    Returns:    (...)     scalar per ego sample.
    """
    X = ego_state[..., 0]
    Y = ego_state[..., 1]
    v = np.maximum(ego_state[..., 3], v_min)         # guard div-by-zero

    dx = traffic_xy[..., 0] - X[..., None]            # (..., N)
    dy = traffic_xy[..., 1] - Y[..., None]
    same_lane_ahead = (np.abs(dy) < lane_half_width) & (dx > 0)

    dx_lead = np.where(same_lane_ahead, dx, np.inf).min(axis=-1)  # (...,)
    thw = dx_lead / v
    shortfall = np.maximum(0.0, target_thw - thw)
    return shortfall ** 2

def road_boundary_cost(
    Y: np.ndarray,
    Y_min: float = -2.0,     # top edge of Lane 0
    Y_max: float = 10.0,     # bottom edge of Lane 2
    margin: float = 0.5,     # start penalizing 0.5 m before the edge
) -> np.ndarray:
    """Quadratic barrier outside [Y_min, Y_max], with a soft inner margin.

    Zero inside [Y_min+margin, Y_max-margin]. Grows quadratically as the
    ego approaches the edge, then continues quadratically once past it.
    Keeps MPPI's importance weights gradient-informative even before the
    ego physically leaves the road.
    """
    over_top = np.maximum(0.0, (Y_min + margin) - Y)
    over_bot = np.maximum(0.0, Y - (Y_max - margin))
    return over_top ** 2 + over_bot ** 2


def step_cost(
    state: np.ndarray,
    action: np.ndarray,
    traffic_xy: np.ndarray,
    target_y: float,
    v_desired: float,
    w_lane: float = 0.3,
    w_target_pref: float = 0.3,
    w_speed: float = 0.1,
    w_collide: float = 200.0,
    w_action: float = 0.05,
    w_heading = 18.0,
    w_thw: float = 10.0,                
    target_thw: float = 2.0,   
    w_road=500.0,
    Y_min=-2.0, Y_max=10.0,
#    w_ydot: float = 4.0,     # gated dashpot — fires only near a lane center
) -> np.ndarray:
    """Sum of per-step costs. state and action broadcast against traffic_xy."""
    Y = state[..., 1]
    psi = state[..., 2]
    v = state[..., 3]
    a = action[..., 0]
    delta = action[..., 1]

    dists = np.abs(Y[..., None] - LANE_CENTERS)
    nearest_lane_dist = dists.min(axis=-1)
    lane_term  = w_lane * nearest_lane_dist ** 2          # "be on *any* lane"
    target_pref = w_target_pref * (Y - target_y) ** 2     # weak tie-break toward target

    # Gated lateral-velocity damping. A uniform Ẏ² penalty would fight every
    # lane change (the controller pays the dashpot in transit but only earns
    # target_pref at horizon-end → myopic optimum is "don't move"). Gating by
    # proximity to *any* lane center kills the on-lane limit cycle without
    # blocking the off-lane motion needed to reach one.
    #   gate ≈ 1 when within ~0.5 m of a lane center, ≈ 0 elsewhere.
    # Ẏ from kinematic bicycle: Ẏ = v·sin(ψ+β), β = atan(½ tan δ).
    # gate = np.exp(-(nearest_lane_dist / 0.5) ** 2)
    # beta = np.arctan(0.5 * np.tan(delta))
    # ydot = v * np.sin(psi + beta)
    # ydot_term = w_ydot * gate * ydot ** 2

    speed_term = w_speed * (v - v_desired) ** 2
    coll_term = w_collide * collision_potential(state[..., :2], traffic_xy)
    heading_term = w_heading * psi ** 2 # Damps lateral oscillation
    eff_term = w_action * (a ** 2 + 5.0 * delta ** 2)
    thw_term     = w_thw     * time_headway_cost(state, traffic_xy, target_thw) 
    road_term    = w_road    * road_boundary_cost(Y, Y_min, Y_max)
    return (lane_term + speed_term + heading_term + coll_term + eff_term
            + thw_term + road_term + target_pref)
#            + ydot_term)


def hard_collision(
    ego_state: np.ndarray, traffic_state: np.ndarray
) -> np.ndarray:
    """Boolean: any traffic vehicle within an L x W bounding box of ego.

    Used for post-hoc collision counting, not for MPPI optimisation.
    """
    dx = np.abs(ego_state[..., 0:1] - traffic_state[..., 0])
    dy = np.abs(ego_state[..., 1:2] - traffic_state[..., 1])
    return np.any((dx < EGO_LENGTH) & (dy < EGO_WIDTH), axis=-1)
