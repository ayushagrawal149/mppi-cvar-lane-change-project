"""Lane-change performance metrics with ISO-derived comfort thresholds.

Thresholds match the table used in the Week-4 report:
- a_y,peak  -- ISO 11270 hard limit 2.5 m/s^2
- TTC_min   -- ISO 17387 warning at 2 s
- j_y,peak, e_y,RMS, T_LC -- driver-comfort literature consensus

The series outputs (a_y, j_y, e_y, TTC vs time) are returned alongside
the scalar peaks so the plotting layer can shade thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.dynamics.bicycle import LENGTH as EGO_LENGTH

THRESHOLDS = {
    "a_y_peak": {"comfortable": 1.5, "acceptable": 2.0, "hard_limit": 2.5},
    "j_y_peak": {"comfortable": 0.9, "acceptable": 2.0, "hard_limit": 5.0},
    "ttc_min": {"comfortable": 4.0, "acceptable": 3.0, "hard_limit": 2.0},
    "e_y_rms": {"comfortable": 0.05, "acceptable": 0.10, "hard_limit": 0.25},
    "t_lc": {
        "comfortable_range": (4.0, 5.0),
        "acceptable_range": (3.0, 6.0),
        "hard_range": (2.0, 8.0),
    },
}


def _ttc_series(t: np.ndarray, ego: np.ndarray, traffic_list) -> np.ndarray:
    """Time-to-collision to closest forward same-lane vehicle, per step.

    Returns +inf when no forward vehicle is closing.
    """
    out = np.full_like(t, np.inf, dtype=np.float64)
    for k in range(len(t)):
        traffic = traffic_list[k]
        if traffic.shape[0] == 0:
            continue
        dx = traffic[:, 0] - ego[k, 0]
        dy = traffic[:, 1] - ego[k, 1]
        forward = (dx > 0) & (np.abs(dy) < 2.0)
        if not np.any(forward):
            continue
        dv = ego[k, 3] - traffic[forward, 3]
        closing = dv > 1e-3
        if not np.any(closing):
            continue
        gap = np.maximum(dx[forward][closing] - EGO_LENGTH, 0.0)
        out[k] = float(np.min(gap / dv[closing]))
    return out


def _detect_lane_change(
    Y: np.ndarray, origin_y: float, target_y: float, dt: float
) -> tuple[float, int, int]:
    """Detect lane-change window. Returns (T_LC seconds, i_start, i_end).

    T_LC is NaN and i_start = i_end = -1 if no clear lane change is found.
    """
    if abs(target_y - origin_y) < 0.5:
        return float("nan"), -1, -1
    enter = np.where(np.abs(Y - origin_y) > 0.3)[0]
    if len(enter) == 0:
        return float("nan"), -1, -1
    i_start = int(enter[0])
    settle = np.where(np.abs(Y - target_y) < 0.1)[0]
    settle = settle[settle > i_start]
    if len(settle) == 0:
        return float("nan"), i_start, -1
    i_end = int(settle[0])
    return float((i_end - i_start) * dt), i_start, i_end


def compute_metrics(logger) -> dict[str, Any]:
    arrays = logger.to_arrays()
    t = arrays["t"]
    ego = arrays["ego"]
    dt = logger.dt

    Y = ego[:, 1]
    psi = ego[:, 2]
    v = ego[:, 3]
    # Body-frame lateral acceleration: a_y = v * psi_dot (centripetal).
    # Differentiating psi is much less noisy than differentiating Y twice,
    # because psi is the integral of yaw rate while Y is the integral of v*sin(psi+beta).
    if len(t) >= 3:
        psi_dot = np.gradient(psi, dt)
        a_y = v * psi_dot
        j_y = np.gradient(a_y, dt)
    else:
        a_y = np.zeros_like(Y)
        j_y = np.zeros_like(Y)
    e_y = Y - logger.target_y

    t_lc, i_start, i_end = _detect_lane_change(Y, logger.origin_y, logger.target_y, dt)
    # Per the Week-4 image, e_y is the *settled* tracking error vs the target
    # lane: measured *after* the lane change completes (i_end onward). When no
    # lane change happened, fall back to whole-trajectory tracking.
    if i_end >= 0 and i_end < len(e_y) - 1:
        settled = e_y[i_end:]
        e_y_rms = float(np.sqrt(np.mean(settled ** 2)))
    elif i_start < 0:
        e_y_rms = float(np.sqrt(np.mean(e_y ** 2))) if len(e_y) else 0.0
    else:
        e_y_rms = float("nan")

    ttc_series = _ttc_series(t, ego, logger.traffic) if len(t) else np.array([np.inf])
    finite_ttc = ttc_series[np.isfinite(ttc_series)]
    ttc_min = float(finite_ttc.min()) if finite_ttc.size > 0 else float("inf")

    return {
        "a_y_peak": float(np.max(np.abs(a_y))) if a_y.size else 0.0,
        "j_y_peak": float(np.max(np.abs(j_y))) if j_y.size else 0.0,
        "e_y_rms": e_y_rms,
        "t_lc": t_lc,
        "ttc_min": ttc_min,
        "a_y_series": a_y,
        "j_y_series": j_y,
        "e_y_series": e_y,
        "ttc_series": ttc_series,
        "lane_change_window": (i_start, i_end),
        "crashed": bool(logger.crashed),
        "steps": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
    }
