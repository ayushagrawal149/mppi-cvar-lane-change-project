# Run: lane_change

## Configuration
- folder: `20260602-030608_lane_change`
- scenario: `lane_change`
- seed: 1
- control step dt: 0.050 s
- target lane y: 0.00 m
- origin lane y: 4.00 m
- origin_lane_index: 1
- target_lane_index: 0
- mppi_n_samples: 256
- mppi_horizon: 30
- render: False
- gif: True
- slowdown: 1.0
- real_time: False
- frames_captured: 0

## Outcome
- steps simulated: 131
- simulated duration: 6.50 s
- crashed: **True**

## Lane-change performance metrics

| Metric | Value | Comfortable | Acceptable | Hard limit | Status |
|---|---:|---:|---:|---:|:---|
| a_y,peak [m/s^2] | 15.21 | <= 1.5 | <= 2.0 | 2.5 (ISO 11270) | exceeds hard limit |
| j_y,peak [m/s^3] | 95.34 | <= 0.9 | <= 2.0 | 5.0 | exceeds hard limit |
| TTC_min [s] | 0.01 | >= 4 | >= 3 | 2 (ISO 17387) | below hard limit |
| e_y,RMS [m] | 0.762 | <= 0.05 | <= 0.10 | 0.25 | exceeds hard limit |
| T_LC [s] | 0.70 | 4-5 | 3-6 | outside [2, 8] flagged | outside [2, 8] s (flagged) |

## Plots
- [trajectory_overview.png](trajectory_overview.png) -- X-Y top-down with traffic snapshots
- [lateral_profile.png](lateral_profile.png) -- Y(t) and e_y(t) vs target lane
- [comfort.png](comfort.png) -- |a_y|(t), |j_y|(t) with ISO 11270 bands
- [safety.png](safety.png) -- TTC(t) with ISO 17387 bands
- [controls.png](controls.png) -- commanded a, delta
- [cost_distribution.png](cost_distribution.png) -- rollout-cost histogram with VaR / CVaR markers

## Raw data
- [run.npz](run.npz) -- numpy archive with `t`, `ego`, `actions`
