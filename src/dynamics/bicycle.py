"""Kinematic bicycle model used for MPPI rollouts.

This intentionally mirrors HighwayEnv's `vehicle.kinematics.Vehicle.step`
(rear-axle-as-COG variant with slip angle beta = atan(0.5 tan delta) and
yaw-rate lever L/2) so that MPPI's predicted trajectories agree with the
simulator's actual integration. This keeps the Initial-Results comparison
free of model-mismatch artefacts; the standard Week-3 bicycle is recovered
by setting the lever to L instead of L/2.

State x = (X, Y, psi, v); control u = (a, delta).
"""

from __future__ import annotations

import numpy as np

LENGTH = 5.0  # m, HighwayEnv default vehicle length


def step(state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
    """One forward-Euler step.

    state:  (..., 4)  [X, Y, psi, v]
    action: (..., 2)  [a, delta]
    Shapes broadcast over the leading dims.
    """
    psi = state[..., 2]
    v = state[..., 3]
    a = action[..., 0]
    delta = action[..., 1]
    beta = np.arctan(0.5 * np.tan(delta))
    X_next = state[..., 0] + v * np.cos(psi + beta) * dt
    Y_next = state[..., 1] + v * np.sin(psi + beta) * dt
    psi_next = psi + (v * np.sin(beta) / (LENGTH / 2.0)) * dt
    v_next = v + a * dt
    return np.stack([X_next, Y_next, psi_next, v_next], axis=-1)


def rollout(state0: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    """Roll a control sequence forward in time.

    state0:   (M, 4) initial state, repeated across M sample sequences.
    controls: (M, K, 2)
    Returns:  (M, K+1, 4) trajectory including the initial state.
    """
    M = state0.shape[0]
    K = controls.shape[-2]
    traj = np.empty((M, K + 1, 4), dtype=np.float64)
    traj[:, 0, :] = state0
    state = state0
    for k in range(K):
        state = step(state, controls[:, k, :], dt)
        traj[:, k + 1, :] = state
    return traj
