# Run: cvar_baseline

## Configuration
- folder: `20260602-024027_cvar_baseline`
- scenario: `cvar_baseline`
- seed: 1
- control step dt: 0.050 s
- target lane y: 0.00 m
- origin lane y: 4.00 m
- controller: cvar_mppi
- alpha: 0.9
- risk_budget: 5.0
- penalty_weight: 200.0
- n_disturb: 64
- idm_sigma_a: 5.0
- mppi_n_samples: 256
- mppi_horizon: 30
- target_lane_index: 0
- origin_lane_index: 1
- render: False
- gif: True
- slowdown: 1.0
- real_time: False
- frames_captured: 0

## Outcome
- steps simulated: 400
- simulated duration: 19.95 s
- crashed: **False**

## Lane-change performance metrics

| Metric | Value | Comfortable | Acceptable | Hard limit | Status |
|---|---:|---:|---:|---:|:---|
| a_y,peak [m/s^2] | 17.07 | <= 1.5 | <= 2.0 | 2.5 (ISO 11270) | exceeds hard limit |
| j_y,peak [m/s^3] | 68.36 | <= 0.9 | <= 2.0 | 5.0 | exceeds hard limit |
| TTC_min [s] | 0.75 | >= 4 | >= 3 | 2 (ISO 17387) | below hard limit |
| e_y,RMS [m] | 0.219 | <= 0.05 | <= 0.10 | 0.25 | exceeds acceptable, within hard limit |
| T_LC [s] | 0.65 | 4-5 | 3-6 | outside [2, 8] flagged | outside [2, 8] s (flagged) |

## Plots
- [trajectory_overview.png](trajectory_overview.png) -- X-Y top-down with traffic snapshots
- [lateral_profile.png](lateral_profile.png) -- Y(t) and e_y(t) vs target lane
- [comfort.png](comfort.png) -- |a_y|(t), |j_y|(t) with ISO 11270 bands
- [safety.png](safety.png) -- TTC(t) with ISO 17387 bands
- [controls.png](controls.png) -- commanded a, delta
- [cost_distribution.png](cost_distribution.png) -- rollout-cost histogram with VaR / CVaR markers

## Raw data
- [run.npz](run.npz) -- numpy archive with `t`, `ego`, `actions`
