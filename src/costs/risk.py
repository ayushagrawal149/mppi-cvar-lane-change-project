"""Per-step risk cost ℓ(x̃) used by CVaR-MPPI.

Distinct from src/costs/lane_change.step_cost (which is the running task
cost q): this returns ONLY safety terms, so the sum L(τ) = Σ_k ℓ(x̃_k)
is the random variable whose tail we cap with CVaR.

References: theory.md §1.4, eq. (8)–(11); [P5] §III-B.
"""

from __future__ import annotations

import numpy as np


def ttc_risk(
    ego_xy: np.ndarray,        # (..., 2)
    ego_v: np.ndarray,         # (...,)
    traffic_xy: np.ndarray,    # (..., N, 2)
    traffic_v: np.ndarray,     # (..., N)
    ttc_safe: float = 2.0,
    lane_half_width: float = 2.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Squared shortfall of TTC below `ttc_safe` for the nearest closing lead.

    TTC uses CLOSING rate (v_ego - v_lead) -- distinct from
    time_headway_cost (lane_change.py) which uses v_ego only.
    No closing lead → 0.
    """
    dx = traffic_xy[..., 0] - ego_xy[..., None, 0]   # (..., N)
    dy = traffic_xy[..., 1] - ego_xy[..., None, 1]
    closing_v = ego_v[..., None] - traffic_v          # (..., N)
    same_lane = np.abs(dy) < lane_half_width
    ahead = dx > 0
    closing = closing_v > eps
    mask = same_lane & ahead & closing
    ttc = np.where(mask, dx / np.maximum(closing_v, eps), np.inf)
    ttc_min = ttc.min(axis=-1)
    return np.maximum(0.0, ttc_safe - ttc_min) ** 2


def lateral_encroachment(
    ego_xy: np.ndarray,
    traffic_xy: np.ndarray,
    long_box: float = 6.0,
    lat_box: float = 1.6,
) -> np.ndarray:
    """1.0 if any traffic vehicle is inside the ego's (long × lat) safety box."""
    dx = np.abs(traffic_xy[..., 0] - ego_xy[..., None, 0])
    dy = np.abs(traffic_xy[..., 1] - ego_xy[..., None, 1])
    inside = (dx < long_box) & (dy < lat_box)
    return inside.any(axis=-1).astype(np.float64)


def risk_cost(
    ego_state: np.ndarray,       # (..., 4)
    traffic_state: np.ndarray,   # (..., N, 4)
    w_ttc: float = 1.0,
    w_encroach: float = 5.0,
    ttc_safe: float = 2.0,
) -> np.ndarray:
    """ℓ(x̃) — the per-step risk cost (theory.md §1.4)."""
    ego_xy = ego_state[..., :2]
    ego_v = ego_state[..., 3]
    traffic_xy = traffic_state[..., :2]
    traffic_v = traffic_state[..., 3]
    return (
        w_ttc * ttc_risk(ego_xy, ego_v, traffic_xy, traffic_v, ttc_safe)
        + w_encroach * lateral_encroachment(ego_xy, traffic_xy)
    )
