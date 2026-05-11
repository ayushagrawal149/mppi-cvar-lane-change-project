"""Markdown summary written into each per-run output folder.

Each run produces a `sim_output/<timestamp>_<scenario>/README.md` containing
the run configuration, episode outcome, and the lane-change performance
table (a_y,peak / j_y,peak / TTC_min / e_y,RMS / T_LC) with status bands
derived from THRESHOLDS in src.utils.metrics.
"""

from __future__ import annotations

import math
from pathlib import Path

from src.utils.metrics import THRESHOLDS


def _band_le(value: float, t: dict) -> str:
    """Band for metrics that should be <= thresholds (a_y, j_y, e_y)."""
    if value <= t["comfortable"]:
        return "comfortable"
    if value <= t["acceptable"]:
        return "acceptable"
    if value <= t["hard_limit"]:
        return "exceeds acceptable, within hard limit"
    return "exceeds hard limit"


def _band_ge(value: float, t: dict) -> str:
    """Band for metrics that should be >= thresholds (TTC_min)."""
    if not math.isfinite(value):
        return "no forward closing vehicle (inf)"
    if value >= t["comfortable"]:
        return "comfortable"
    if value >= t["acceptable"]:
        return "acceptable"
    if value >= t["hard_limit"]:
        return "below acceptable, within hard limit"
    return "below hard limit"


def _band_range(value: float, t: dict) -> str:
    """Band for T_LC, which has comfortable / acceptable / hard ranges."""
    if value is None or math.isnan(value):
        return "no lane change detected"
    lo, hi = t["comfortable_range"]
    if lo <= value <= hi:
        return "comfortable"
    lo, hi = t["acceptable_range"]
    if lo <= value <= hi:
        return "acceptable"
    lo, hi = t["hard_range"]
    if lo <= value <= hi:
        return "outside acceptable range"
    return "outside [2, 8] s (flagged)"


def _fmt_t_lc(value: float) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2f}"


def _fmt_ttc(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return f"{value:.2f}"


def write_readme(logger, m, out_dir) -> Path:
    out_dir = Path(out_dir)
    lines: list[str] = []
    lines.append(f"# Run: {logger.scenario}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- folder: `{out_dir.name}`")
    lines.append(f"- scenario: `{logger.scenario}`")
    lines.append(f"- seed: {logger.seed}")
    lines.append(f"- control step dt: {logger.dt:.3f} s")
    lines.append(f"- target lane y: {logger.target_y:.2f} m")
    lines.append(f"- origin lane y: {logger.origin_y:.2f} m")
    for k, v in (logger.extra or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Outcome")
    lines.append(f"- steps simulated: {m['steps']}")
    lines.append(f"- simulated duration: {m['duration_s']:.2f} s")
    lines.append(f"- crashed: **{m['crashed']}**")
    lines.append("")
    lines.append("## Lane-change performance metrics")
    lines.append("")
    lines.append("| Metric | Value | Comfortable | Acceptable | Hard limit | Status |")
    lines.append("|---|---:|---:|---:|---:|:---|")
    lines.append(
        f"| a_y,peak [m/s^2] | {m['a_y_peak']:.2f} | <= 1.5 | <= 2.0 "
        f"| 2.5 (ISO 11270) | {_band_le(m['a_y_peak'], THRESHOLDS['a_y_peak'])} |"
    )
    lines.append(
        f"| j_y,peak [m/s^3] | {m['j_y_peak']:.2f} | <= 0.9 | <= 2.0 "
        f"| 5.0 | {_band_le(m['j_y_peak'], THRESHOLDS['j_y_peak'])} |"
    )
    lines.append(
        f"| TTC_min [s] | {_fmt_ttc(m['ttc_min'])} | >= 4 | >= 3 "
        f"| 2 (ISO 17387) | {_band_ge(m['ttc_min'], THRESHOLDS['ttc_min'])} |"
    )
    lines.append(
        f"| e_y,RMS [m] | {m['e_y_rms']:.3f} | <= 0.05 | <= 0.10 "
        f"| 0.25 | {_band_le(m['e_y_rms'], THRESHOLDS['e_y_rms'])} |"
    )
    lines.append(
        f"| T_LC [s] | {_fmt_t_lc(m['t_lc'])} | 4-5 | 3-6 "
        f"| outside [2, 8] flagged | {_band_range(m['t_lc'], THRESHOLDS['t_lc'])} |"
    )
    lines.append("")
    lines.append("## Plots")
    lines.append("- [trajectory_overview.png](trajectory_overview.png) -- X-Y top-down with traffic snapshots")
    lines.append("- [lateral_profile.png](lateral_profile.png) -- Y(t) and e_y(t) vs target lane")
    lines.append("- [comfort.png](comfort.png) -- |a_y|(t), |j_y|(t) with ISO 11270 bands")
    lines.append("- [safety.png](safety.png) -- TTC(t) with ISO 17387 bands")
    lines.append("- [controls.png](controls.png) -- commanded a, delta")
    if logger.rollout_snapshot is not None:
        lines.append("- [cost_distribution.png](cost_distribution.png) -- rollout-cost histogram with VaR / CVaR markers")
    lines.append("")
    lines.append("## Raw data")
    lines.append("- [run.npz](run.npz) -- numpy archive with `t`, `ego`, `actions`")
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path
