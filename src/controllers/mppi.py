"""Risk-neutral MPPI controller (baseline for the CVaR comparison).

Implements the standard formulation from Williams et al. 2017:
- sample M control sequences u_m = v + eps_m, eps_m ~ N(0, Sigma_eps);
- propagate K-step rollouts under nominal bicycle dynamics;
- weight w_m = exp(-(S_m - min_m S_m) / lambda) and update
  v <- sum w_m u_m / sum w_m;
- apply v[0] and shift the nominal sequence forward (warm-start).

The CVaR augmentation will reuse this sampler and only modify the
per-rollout cost aggregator -- no other change to the controller.

Reference:
[1] G. Williams, A. Aldrich, E. A. Theodorou, "Model predictive path
    integral control: From theory to parallel computation,"
    J. Guid. Control Dyn., 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.dynamics.bicycle import rollout as bicycle_rollout
from src.traffic.prediction import constant_velocity_rollout
from src.costs.lane_change import step_cost


@dataclass
class MPPIConfig:
    horizon: int = 30
    n_samples: int = 256
    dt: float = 0.05
    temperature: float = 2.0  #1.0 -> 5.0
    sigma_a: float = 0.6      #1.5 -> 0.6
    sigma_delta: float = 0.03 #0.08 -> 0.03
    a_bounds: Tuple[float, float] = (-5.0, 5.0)
    delta_bounds: Tuple[float, float] = (-0.4, 0.4)
    target_y: float = 0.0
    v_desired: float = 25.0
    seed: int = 0
    w_jerk: float = 0.5
    w_steerrate: float = 100.0
    max_drate: float = 0.8   # rad/s — caps |a_y_ramp| ≈ ½·v·max_drate to ≤ 1 g at v=25 m/s


class MPPI:
    def __init__(self, config: MPPIConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.v = np.zeros((config.horizon, 2), dtype=np.float64)
        self.u_prev = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.v[:] = 0.0
        self.u_prev[:] = 0.0  

    def _sample_noise(self) -> np.ndarray:
        eps = self.rng.normal(
            size=(self.cfg.n_samples, self.cfg.horizon, 2)
        )
        eps[..., 0] *= self.cfg.sigma_a
        eps[..., 1] *= self.cfg.sigma_delta
        return eps

    def _clip_controls(self, U: np.ndarray) -> np.ndarray:
        a_lo, a_hi = self.cfg.a_bounds
        d_lo, d_hi = self.cfg.delta_bounds
        U[..., 0] = np.clip(U[..., 0], a_lo, a_hi)
        U[..., 1] = np.clip(U[..., 1], d_lo, d_hi)
        # Hard slew-rate cap on steering, anchored to the last applied control.
        # Models the front-wheel actuator's mechanical limit; unlike the soft
        # w_steerrate penalty (which competes against collision pressure), this
        # is a physical bound MPPI cannot trade off.
        if self.cfg.max_drate > 0.0:
            step = self.cfg.max_drate * self.cfg.dt   # max |Δδ| per dt
            prev = np.full(U.shape[0], self.u_prev[1])
            for k in range(U.shape[1]):
                U[:, k, 1] = np.clip(U[:, k, 1], prev - step, prev + step)
                prev = U[:, k, 1]
        return U

    def _rollout_cost(
        self, state0: np.ndarray, traffic_state0: np.ndarray, U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = self.cfg.horizon
        M = U.shape[0]
        state0_b = np.broadcast_to(state0, (M, 4)).copy()
        traj = bicycle_rollout(state0_b, U, self.cfg.dt)
        traffic_traj = constant_velocity_rollout(traffic_state0, K, self.cfg.dt)
        costs = np.zeros(M, dtype=np.float64)
        for k in range(K):
            costs += step_cost(
                state=traj[:, k + 1, :],
                action=U[:, k, :],
                traffic_xy=traffic_traj[k + 1, :, :2],
                target_y=self.cfg.target_y,
                v_desired=self.cfg.v_desired,
            )
        
        # Control-rate / jerk penalty: differences between consecutive controls in
        # the sampled sequence, with self.u_prev as the "step -1" baseline so the
        # very first commanded change is also penalized.
        u_prev_b = np.broadcast_to(self.u_prev, (M, 1, 2))
        U_aug = np.concatenate([u_prev_b, U], axis=1)           # (M, K+1, 2)
        dU = np.diff(U_aug, axis=1)                              # (M, K, 2)
        costs += self.cfg.w_jerk      * np.sum(dU[..., 0] ** 2, axis=-1)
        costs += self.cfg.w_steerrate * np.sum(dU[..., 1] ** 2, axis=-1)

        return costs, traj

    def step(
        self, state0: np.ndarray, traffic_state0: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        eps = self._sample_noise()
        U = self.v[None, :, :] + eps
        U = self._clip_controls(U)
        costs, traj = self._rollout_cost(state0, traffic_state0, U)
        S = costs - costs.min()
        w = np.exp(-S / self.cfg.temperature)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum < 1e-12:
            w = np.ones_like(w) / w.shape[0]
        else:
            w = w / w_sum
        self.v = np.einsum("m,mkj->kj", w, U)
        u0 = self.v[0].copy()
        self.u_prev = u0.copy()
        self.v[:-1] = self.v[1:]
        self.v[-1] = 0.0
        return u0, {"costs": costs, "weights": w, "traj": traj}
