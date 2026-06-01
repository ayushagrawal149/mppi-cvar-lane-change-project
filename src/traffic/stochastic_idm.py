"""Disturbed traffic predictor F̃ used to estimate per-rollout CVaR.

Each disturbed future is a constant-heading rollout with Gaussian
acceleration noise injected per step (and, optionally, a small per-step
discrete lateral 'lane-change attempt' modeled as a Y-jump).

Because the noise is zero-mean, E_w[F̃(x,u,w)] = F(x,u) holds at every
step -- the planner's nominal trajectory is the mean of the random one
(theory.md §1.2 eq. 2). Tail spread is what CVaR will quantify.
"""

from __future__ import annotations

import numpy as np


def stochastic_traffic_rollout(
    traffic_state0: np.ndarray,
    K: int,
    dt: float,
    n_disturb: int,
    sigma_a: float = 1.5,
    lane_change_prob: float = 0.0,
    lane_change_dy: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample N_disturb futures.

    traffic_state0: (N, 4)  [X, Y, psi, v]
    Returns:        (n_disturb, K+1, N, 4)
    """
    if rng is None:
        rng = np.random.default_rng()
    N = traffic_state0.shape[0]
    out = np.empty((n_disturb, K + 1, N, 4), dtype=np.float64)
    out[:, 0, :, :] = traffic_state0
    if N == 0:
        return out

    psi0 = traffic_state0[:, 2]
    cos0 = np.cos(psi0); sin0 = np.sin(psi0)

    # acceleration noise per (n, k, vehicle)
    a_noise = rng.normal(scale=sigma_a, size=(n_disturb, K, N))

    # state we evolve
    v = np.broadcast_to(traffic_state0[:, 3], (n_disturb, N)).copy()
    X = np.broadcast_to(traffic_state0[:, 0], (n_disturb, N)).copy()
    Y = np.broadcast_to(traffic_state0[:, 1], (n_disturb, N)).copy()

    for k in range(K):
        v = v + a_noise[:, k, :] * dt
        v = np.clip(v, 0.0, 40.0)
        X = X + v * cos0 * dt
        Y = Y + v * sin0 * dt
        if lane_change_prob > 0.0:
            jump = rng.random(size=(n_disturb, N)) < lane_change_prob
            direction = rng.choice([-1.0, 1.0], size=(n_disturb, N))
            Y = Y + jump * direction * lane_change_dy
        out[:, k + 1, :, 0] = X
        out[:, k + 1, :, 1] = Y
        out[:, k + 1, :, 2] = psi0
        out[:, k + 1, :, 3] = v
    return out
