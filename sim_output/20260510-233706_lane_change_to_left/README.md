# Run: lane_change_to_left

## Configuration
- folder: `20260510-233706_lane_change_to_left`
- scenario: `lane_change_to_left`
- seed: 0
- control step dt: 0.050 s
- target lane y: 0.00 m
- origin lane y: 8.00 m
- target_lane_index: 0
- origin_lane_index: 2
- mppi_n_samples: 256
- mppi_horizon: 30
- render: False
- gif: False
- slowdown: 1.0
- real_time: False
- frames_captured: 0

## Outcome
- steps simulated: 200
- simulated duration: 9.95 s
- crashed: **False**

## Lane-change performance metrics

| Metric | Value | Comfortable | Acceptable | Hard limit | Status |
|---|---:|---:|---:|---:|:---|
| a_y,peak [m/s^2] | 49.43 | <= 1.5 | <= 2.0 | 2.5 (ISO 11270) | exceeds hard limit |
| j_y,peak [m/s^3] | 612.44 | <= 0.9 | <= 2.0 | 5.0 | exceeds hard limit |
| TTC_min [s] | 2.49 | >= 4 | >= 3 | 2 (ISO 17387) | below acceptable, within hard limit |
| e_y,RMS [m] | 0.134 | <= 0.05 | <= 0.10 | 0.25 | exceeds acceptable, within hard limit |
| T_LC [s] | 1.20 | 4-5 | 3-6 | outside [2, 8] flagged | outside [2, 8] s (flagged) |

## Plots
- [trajectory_overview.png](trajectory_overview.png) -- X-Y top-down with traffic snapshots
- [lateral_profile.png](lateral_profile.png) -- Y(t) and e_y(t) vs target lane
- [comfort.png](comfort.png) -- |a_y|(t), |j_y|(t) with ISO 11270 bands
- [safety.png](safety.png) -- TTC(t) with ISO 17387 bands
- [controls.png](controls.png) -- commanded a, delta
- [cost_distribution.png](cost_distribution.png) -- rollout-cost histogram with VaR / CVaR markers

## Raw data
- [run.npz](run.npz) -- numpy archive with `t`, `ego`, `actions`
