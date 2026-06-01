"""Per-step logger that produces a timestamped folder under sim_output/.

Usage:
    logger = RunLogger(scenario="hard_brake", dt=0.05,
                       target_y=0.0, origin_y=0.0, seed=1)
    for k in range(T):
        ...
        logger.log(t=k*dt, ego_state=..., action_cmd=...,
                   traffic_state=..., rollout_info=info)
    out_dir = logger.save()              # -> sim_output/<stamp>_<scenario>/

The save step writes run.npz, six plots, and a README.md (see
src.utils.report and src.utils.plots).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


class RunLogger:
    def __init__(
        self,
        scenario: str,
        dt: float = 0.05,
        target_y: float = 0.0,
        origin_y: float = 0.0,
        seed: int = 0,
        extra: dict | None = None,
        capture_rollout_at_step: int = 20,
        lane_centers: list[float] | None = None,
        lane_width: float = 4.0,
    ):
        self.scenario = scenario
        self.dt = float(dt)
        self.target_y = float(target_y)
        self.origin_y = float(origin_y)
        self.seed = int(seed)
        self.extra = dict(extra or {})
        self.capture_rollout_at_step = int(capture_rollout_at_step)
        # HighwayEnv defaults: 3 lanes at y = 0, 4, 8 with width 4 m. Drivers
        # should pass actual values read from the env so the plots match the
        # simulator faithfully.
        self.lane_centers = (
            [float(y) for y in lane_centers]
            if lane_centers is not None
            else [0.0, 4.0, 8.0]
        )
        self.lane_width = float(lane_width)

        self.t: list[float] = []
        self.ego: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.traffic: list[np.ndarray] = []
        self.crashed: bool = False
        self.rollout_snapshot: dict | None = None
        # Per-step CVaR diagnostics (empty dicts for baseline-MPPI runs).
        self.diag_steps: list[dict] = []
        self._step: int = 0

    def log(
        self,
        t: float,
        ego_state: np.ndarray,
        action_cmd: np.ndarray,
        traffic_state: np.ndarray,
        rollout_info: dict | None = None,
        crashed: bool = False,
    ) -> None:
        self.t.append(float(t))
        self.ego.append(np.asarray(ego_state, dtype=np.float64).copy())
        self.actions.append(np.asarray(action_cmd, dtype=np.float64).copy())
        self.traffic.append(np.asarray(traffic_state, dtype=np.float64).copy())
        if crashed:
            self.crashed = True
        if (
            rollout_info is not None
            and self.rollout_snapshot is None
            and self._step == self.capture_rollout_at_step
        ):
            self.rollout_snapshot = {
                "costs": np.asarray(rollout_info["costs"]).copy(),
                "weights": np.asarray(rollout_info["weights"]).copy(),
                "step": int(self._step),
                "t": float(t),
            }
        # Per-step CVaR diagnostics. Pulled from rollout_info if the
        # controller produced them; baseline MPPI has no such keys → empty
        # dict here, which `to_arrays` drops cleanly.
        diag: dict[str, float] = {}
        if rollout_info is not None:
            if "cvar" in rollout_info:
                cv = np.asarray(rollout_info["cvar"], dtype=np.float64)
                diag["cvar_median"] = float(np.median(cv))
                diag["cvar_p90"]    = float(np.percentile(cv, 90))
                diag["cvar_max"]    = float(np.max(cv))
            if "J_C" in rollout_info:
                jc = np.asarray(rollout_info["J_C"], dtype=np.float64)
                diag["jc_mean"] = float(jc.mean())
                diag["jc_max"]  = float(jc.max())
            if "n_over_budget" in rollout_info:
                diag["n_over_budget"] = float(rollout_info["n_over_budget"])
        self.diag_steps.append(diag)
        self._step += 1

    def to_arrays(self) -> dict[str, np.ndarray]:
        if not self.t:
            return {
                "t": np.zeros(0),
                "ego": np.zeros((0, 4)),
                "actions": np.zeros((0, 2)),
            }
        out = {
            "t": np.array(self.t, dtype=np.float64),
            "ego": np.stack(self.ego, axis=0),
            "actions": np.stack(self.actions, axis=0),
        }
        # CVaR diagnostics: only emitted if the controller logged any
        # (baseline MPPI runs ⇒ diag_steps is a list of empty dicts ⇒ no keys).
        diag_keys = set().union(*(d.keys() for d in self.diag_steps)) \
            if self.diag_steps else set()
        for k in sorted(diag_keys):
            out[k] = np.array(
                [d.get(k, np.nan) for d in self.diag_steps],
                dtype=np.float64,
            )
        return out

    def save(
        self,
        sim_output_root: str | Path = "sim_output",
        timestamp: str | None = None,
    ) -> Path:
        from src.utils.metrics import compute_metrics
        from src.utils.plots import save_plots
        from src.utils.report import write_readme

        stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
        slug = self.scenario.replace(" ", "_").replace("/", "-")
        out_dir = Path(sim_output_root) / f"{stamp}_{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)

        arrays = self.to_arrays()
        np.savez(out_dir / "run.npz", **arrays)

        metrics = compute_metrics(self)
        save_plots(self, metrics, out_dir)
        write_readme(self, metrics, out_dir)
        return out_dir

    def summary(self) -> dict[str, Any]:
        """Lightweight summary used by tests when full save is not desired."""
        from src.utils.metrics import compute_metrics

        m = compute_metrics(self)
        return {
            "steps": m["steps"],
            "crashed": m["crashed"],
            "a_y_peak": m["a_y_peak"],
            "j_y_peak": m["j_y_peak"],
            "ttc_min": m["ttc_min"],
            "e_y_rms": m["e_y_rms"],
            "t_lc": m["t_lc"],
        }
