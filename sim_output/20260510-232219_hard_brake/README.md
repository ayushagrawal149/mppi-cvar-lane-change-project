# Run: hard_brake

## Configuration
- folder: `20260510-232219_hard_brake`
- scenario: `hard_brake`
- seed: 1
- control step dt: 0.050 s
- target lane y: 4.00 m
- origin lane y: 4.00 m
- origin_lane_index: 1
- target_lane_index: 1
- mppi_n_samples: 128
- mppi_horizon: 20
- render: False
- gif: False
- slowdown: 1.0
- real_time: False
- frames_captured: 0
- forced_dx_ahead: 8.0
- forced_brake_mps2: -10.0

## Outcome
- steps simulated: 60
- simulated duration: 2.95 s
- crashed: **False**

## Lane-change performance metrics

| Metric | Value | Comfortable | Acceptable | Hard limit | Status |
|---|---:|---:|---:|---:|:---|
| a_y,peak [m/s^2] | 42.15 | <= 1.5 | <= 2.0 | 2.5 (ISO 11270) | exceeds hard limit |
| j_y,peak [m/s^3] | 508.90 | <= 0.9 | <= 2.0 | 5.0 | exceeds hard limit |
| TTC_min [s] | 1.44 | >= 4 | >= 3 | 2 (ISO 17387) | below hard limit |
| e_y,RMS [m] | 4.098 | <= 0.05 | <= 0.10 | 0.25 | exceeds hard limit |
| T_LC [s] | N/A | 4-5 | 3-6 | outside [2, 8] flagged | no lane change detected |

## Plots
- [trajectory_overview.png](trajectory_overview.png) -- X-Y top-down with traffic snapshots
- [lateral_profile.png](lateral_profile.png) -- Y(t) and e_y(t) vs target lane
- [comfort.png](comfort.png) -- |a_y|(t), |j_y|(t) with ISO 11270 bands
- [safety.png](safety.png) -- TTC(t) with ISO 17387 bands
- [controls.png](controls.png) -- commanded a, delta
- [cost_distribution.png](cost_distribution.png) -- rollout-cost histogram with VaR / CVaR markers

## Raw data
- [run.npz](run.npz) -- numpy archive with `t`, `ego`, `actions`
