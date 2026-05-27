"""Traffic prediction model used inside MPPI rollouts.

For the Initial-Results milestone we use a constant-velocity predictor:
each surrounding vehicle keeps its current heading and speed over the
K-step horizon. This is the standard short-horizon predictor in highway
planning and is cheap enough to amortise across thousands of ego rollouts.

The CVaR augmentation in the next milestone will swap this for a
stochastic IDM predictor that draws acceleration noise per rollout, which
generates the tail-traffic outcomes that motivate CVaR. The interface is
kept identical so the controller is unchanged.
"""

from __future__ import annotations

import numpy as np


def constant_velocity_rollout(
    traffic_state0: np.ndarray, K: int, dt: float
) -> np.ndarray:
    """Propagate traffic at constant heading + speed over K steps.

    traffic_state0: (N, 4) [X, Y, psi, v]
    Returns:        (K+1, N, 4)
    """
    N = traffic_state0.shape[0]
    traj = np.empty((K + 1, N, 4), dtype=np.float64)
    traj[0] = traffic_state0
    if N == 0:
        return traj
    psi = traffic_state0[:, 2]
    v = traffic_state0[:, 3]
    vx = v * np.cos(psi)
    vy = v * np.sin(psi)
    for k in range(1, K + 1):
        traj[k, :, 0] = traffic_state0[:, 0] + vx * (k * dt)
        traj[k, :, 1] = traffic_state0[:, 1] + vy * (k * dt)
        traj[k, :, 2] = psi
        traj[k, :, 3] = v
    return traj
