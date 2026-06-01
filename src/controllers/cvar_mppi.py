"""Risk-aware MPPI (RA-MPPI) — CVaR-augmented controller.

Inherits the risk-neutral MPPI sampler/rollouter from src.controllers.mppi
and only modifies the per-rollout cost aggregator (theory.md §5.4–5.6):

    S̃_m  =  S_m  +  A · CVaR_α(L^{m,*}) · 1{CVaR_α(L^{m,*}) > C_u}
    ω_m  =  exp(−(S̃_m − min_m S̃_m) / λ)
    v⁺   =  Σ ω_m u_m  / Σ ω_m

where L^{m,n} is the trajectory-sum of risk_cost(ego^m, traffic^n) under
the disturbed traffic predictor F̃. The MPPI sampler is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from src.controllers.mppi import MPPI, MPPIConfig
from src.dynamics.bicycle import rollout as bicycle_rollout
from src.traffic.stochastic_idm import stochastic_traffic_rollout
from src.costs.risk import risk_cost
from src.risk.cvar import cvar_tail_average, variance_scale


@dataclass
class CVaRMPPIConfig(MPPIConfig):
    # CVaR-only knobs (sane defaults for first integration pass)
    n_disturb: int = 24
    alpha: float = 0.9
    risk_budget: float = 4.0
    penalty_weight: float = 80.0
    variance_scale_B: float = 1.0
    # disturbed-traffic predictor
    idm_sigma_a: float = 1.5
    idm_lane_change_prob: float = 0.0
    # risk cost weights
    risk_w_ttc: float = 1.0
    risk_w_encroach: float = 5.0
    risk_ttc_safe: float = 2.0


class CVaRMPPI(MPPI):
    cfg: CVaRMPPIConfig  # type: ignore[assignment]

    def __init__(self, config: CVaRMPPIConfig):
        super().__init__(config)
        self.cfg = config
        # Separate RNG for traffic disturbances so MPPI sampling stays reproducible
        self.disturb_rng = np.random.default_rng(config.seed + 1)

    def _disturbed_risk(
        self,
        ego_traj: np.ndarray,        # (M, K+1, 4)  — nominal ego rollouts
        traffic_state0: np.ndarray,  # (N_veh, 4)
    ) -> np.ndarray:                  # returns (M, n_disturb) — L^{m,n}
        """L^{m,n} = Σ_k ℓ(ego^m_k, traffic^n_k).

        The same N disturbed traffic futures are shared across all M ego
        rollouts (traffic disturbance independent of ego control) — much
        cheaper than per-rollout futures and unbiased for our F̃.
        """
        cfg = self.cfg
        K = cfg.horizon
        traffic_disturb = stochastic_traffic_rollout(
            traffic_state0,
            K=K,
            dt=cfg.dt,
            n_disturb=cfg.n_disturb,
            sigma_a=cfg.idm_sigma_a,
            lane_change_prob=cfg.idm_lane_change_prob,
            rng=self.disturb_rng,
        )                                  # (n_disturb, K+1, N_veh, 4)

        M = ego_traj.shape[0]
        N = cfg.n_disturb
        # accumulate L over k
        L = np.zeros((M, N), dtype=np.float64)
        for k in range(1, K + 1):
            ego_k = ego_traj[:, k, :]                       # (M, 4)
            traffic_k = traffic_disturb[:, k, :, :]         # (N, N_veh, 4)
            # broadcast to (M, N, ...) by inserting axes
            ego_b = ego_k[:, None, :]                       # (M, 1, 4)
            traffic_b = traffic_k[None, :, :, :]            # (1, N, N_veh, 4)
            L += risk_cost(
                ego_b,
                traffic_b,
                w_ttc=cfg.risk_w_ttc,
                w_encroach=cfg.risk_w_encroach,
                ttc_safe=cfg.risk_ttc_safe,
            )
        return L

    def step(
        self, state0: np.ndarray, traffic_state0: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        cfg = self.cfg
        # --- identical to base MPPI: sample, clip, base cost ---
        eps = self._sample_noise()
        U = self.v[None, :, :] + eps
        U = self._clip_controls(U)
        S, traj = self._rollout_cost(state0, traffic_state0, U)

        # --- NEW: CVaR penalty per rollout ---
        L = self._disturbed_risk(traj, traffic_state0)          # (M, N)
        L_scaled = variance_scale(L, cfg.variance_scale_B)
        cvar_m = cvar_tail_average(L_scaled, cfg.alpha)         # (M,)
        over = cvar_m > cfg.risk_budget
        J_C = cfg.penalty_weight * cvar_m * over                # (M,)
        S_tilde = S + J_C

        # --- identical to base MPPI: softmin & average ---
        S_shift = S_tilde - S_tilde.min()
        w = np.exp(-S_shift / cfg.temperature)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum < 1e-12:
            w = np.ones_like(w) / w.shape[0]
        else:
            w = w / w_sum
        self.v = np.einsum("m,mkj->kj", w, U)
        u0 = self.v[0].copy()
        self.u_prev = u0.copy()         # mirror base MPPI; slew cap + rate penalty depend on it
        self.v[:-1] = self.v[1:]
        self.v[-1] = 0.0
        return u0, {
            "costs": S, "weights": w, "traj": traj,
            "cvar": cvar_m, "J_C": J_C, "n_over_budget": int(over.sum()),
        }
