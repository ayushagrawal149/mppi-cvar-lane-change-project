"""Plot generation for a single MPPI lane-change run.

Six figures, drawn from the literature and from the Week-4 metric image:
- trajectory_overview.png  -- X-Y top-down (Williams 2017)
- lateral_profile.png      -- Y(t) and e_y(t) vs target/origin lanes
- comfort.png              -- |a_y|(t) and |j_y|(t) with ISO thresholds
- safety.png               -- TTC(t) with ISO 17387 thresholds
- controls.png             -- a_cmd(t), delta_cmd(t)
- cost_distribution.png    -- rollout-cost histogram with VaR / CVaR (Yin 2022)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; safe for tests / CI
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import THRESHOLDS


def _save(fig, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)


def _plot_trajectory(logger, m, out_dir):
    arr = logger.to_arrays()
    t = arr["t"]
    ego = arr["ego"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ego[:, 0], ego[:, 1], color="steelblue", linewidth=2, label="ego")
    if logger.traffic[0].shape[0] > 0:
        tr0 = logger.traffic[0]
        ax.scatter(tr0[:, 0], tr0[:, 1], c="orange", s=30, alpha=0.6,
                   label="traffic (t=0)")
    if logger.traffic[-1].shape[0] > 0:
        trn = logger.traffic[-1]
        ax.scatter(trn[:, 0], trn[:, 1], c="firebrick", s=30, alpha=0.6,
                   label=f"traffic (t={t[-1]:.1f}s)")
    ax.axhline(logger.target_y, ls="--", color="green", alpha=0.6,
               label=f"target lane y={logger.target_y:.1f}")
    if abs(logger.origin_y - logger.target_y) > 0.5:
        ax.axhline(logger.origin_y, ls=":", color="navy", alpha=0.5,
                   label=f"origin lane y={logger.origin_y:.1f}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Trajectory: {logger.scenario}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    _save(fig, out_dir, "trajectory_overview.png")


def _plot_lateral(logger, m, out_dir):
    arr = logger.to_arrays()
    t = arr["t"]
    ego = arr["ego"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t, ego[:, 1], color="steelblue")
    axes[0].axhline(logger.target_y, ls="--", color="green", alpha=0.6, label="target y")
    axes[0].axhline(logger.origin_y, ls=":", color="navy", alpha=0.5, label="origin y")
    axes[0].set_ylabel("Y [m]")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, m["e_y_series"], color="steelblue")
    axes[1].axhline(THRESHOLDS["e_y_rms"]["comfortable"], ls=":", color="green", alpha=0.5)
    axes[1].axhline(-THRESHOLDS["e_y_rms"]["comfortable"], ls=":", color="green", alpha=0.5)
    axes[1].set_ylabel("e_y [m]")
    axes[1].set_xlabel("t [s]")
    suptitle = f"Lateral profile, e_y,RMS = {m['e_y_rms']:.3f} m"
    if not np.isnan(m["t_lc"]):
        suptitle += f", T_LC = {m['t_lc']:.2f} s"
    fig.suptitle(suptitle)
    _save(fig, out_dir, "lateral_profile.png")


def _plot_comfort(logger, m, out_dir):
    arr = logger.to_arrays()
    t = arr["t"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t, np.abs(m["a_y_series"]), color="steelblue")
    for label, val, col in (
        ("comfortable", THRESHOLDS["a_y_peak"]["comfortable"], "green"),
        ("acceptable", THRESHOLDS["a_y_peak"]["acceptable"], "orange"),
        ("hard limit", THRESHOLDS["a_y_peak"]["hard_limit"], "red"),
    ):
        axes[0].axhline(val, ls="--", color=col, alpha=0.6,
                        label=f"{label} ({val})")
    axes[0].set_ylabel("|a_y| [m/s^2]")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, np.abs(m["j_y_series"]), color="steelblue")
    for label, val, col in (
        ("comfortable", THRESHOLDS["j_y_peak"]["comfortable"], "green"),
        ("acceptable", THRESHOLDS["j_y_peak"]["acceptable"], "orange"),
        ("hard limit", THRESHOLDS["j_y_peak"]["hard_limit"], "red"),
    ):
        axes[1].axhline(val, ls="--", color=col, alpha=0.6,
                        label=f"{label} ({val})")
    axes[1].set_ylabel("|j_y| [m/s^3]")
    axes[1].set_xlabel("t [s]")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"Comfort: a_y,peak = {m['a_y_peak']:.2f} m/s^2, "
        f"j_y,peak = {m['j_y_peak']:.2f} m/s^3"
    )
    _save(fig, out_dir, "comfort.png")


def _plot_safety(logger, m, out_dir):
    arr = logger.to_arrays()
    t = arr["t"]
    ttc = m["ttc_series"].copy()
    cap = 10.0
    ttc_plot = np.where(np.isinf(ttc) | np.isnan(ttc), cap, np.minimum(ttc, cap))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, ttc_plot, color="steelblue")
    for label, val, col in (
        ("comfortable >= 4 s", THRESHOLDS["ttc_min"]["comfortable"], "green"),
        ("acceptable >= 3 s", THRESHOLDS["ttc_min"]["acceptable"], "orange"),
        ("hard limit 2 s", THRESHOLDS["ttc_min"]["hard_limit"], "red"),
    ):
        ax.axhline(val, ls="--", color=col, alpha=0.6, label=label)
    ax.set_ylabel(f"TTC [s] (capped at {cap:.0f})")
    ax.set_xlabel("t [s]")
    ax.set_ylim(0, cap + 0.5)
    ttc_min = m["ttc_min"]
    ttc_min_str = f"{ttc_min:.2f}" if np.isfinite(ttc_min) else "inf"
    ax.set_title(f"Safety: TTC_min = {ttc_min_str} s")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "safety.png")


def _plot_controls(logger, m, out_dir):
    arr = logger.to_arrays()
    t = arr["t"]
    u = arr["actions"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    axes[0].plot(t, u[:, 0], color="steelblue")
    axes[0].set_ylabel("a_cmd [m/s^2]")
    axes[1].plot(t, u[:, 1], color="steelblue")
    axes[1].set_ylabel("delta_cmd [rad]")
    axes[1].set_xlabel("t [s]")
    fig.suptitle(f"Commanded controls: {logger.scenario}")
    _save(fig, out_dir, "controls.png")


def _plot_cost_distribution(logger, m, out_dir):
    if logger.rollout_snapshot is None:
        return
    snap = logger.rollout_snapshot
    costs = np.asarray(snap["costs"], dtype=np.float64)
    beta = 0.9
    var_b = float(np.quantile(costs, beta))
    tail = costs[costs >= var_b]
    cvar_b = float(tail.mean()) if tail.size > 0 else var_b
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(costs, bins=40, color="steelblue", alpha=0.8)
    ax.axvline(var_b, color="orange", ls="--",
               label=f"VaR_{{{beta}}} = {var_b:.1f}")
    ax.axvline(cvar_b, color="red", ls="--",
               label=f"CVaR_{{{beta}}} = {cvar_b:.1f}")
    ax.set_xlabel("rollout cost")
    ax.set_ylabel("frequency")
    ax.set_title(
        f"Rollout cost distribution at t = {snap['t']:.2f} s "
        f"(M = {len(costs)}). The tail above VaR is what CVaR re-weights."
    )
    ax.legend()
    _save(fig, out_dir, "cost_distribution.png")


def save_plots(logger, m, out_dir):
    out_dir = Path(out_dir)
    _plot_trajectory(logger, m, out_dir)
    _plot_lateral(logger, m, out_dir)
    _plot_comfort(logger, m, out_dir)
    _plot_safety(logger, m, out_dir)
    _plot_controls(logger, m, out_dir)
    _plot_cost_distribution(logger, m, out_dir)
