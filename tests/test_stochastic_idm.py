import numpy as np
from src.traffic.prediction import constant_velocity_rollout
from src.traffic.stochastic_idm import stochastic_traffic_rollout


def test_disturbed_mean_matches_nominal():
    rng = np.random.default_rng(0)
    s0 = np.array([[0., 0., 0., 20.], [10., 4., 0., 22.]])
    K, dt = 30, 0.05
    nominal = constant_velocity_rollout(s0, K, dt)              # (K+1,N,4)
    disturb = stochastic_traffic_rollout(s0, K, dt, n_disturb=2000,
                                         sigma_a=1.5, rng=rng)  # (.,K+1,N,4)
    mean = disturb.mean(axis=0)
    # X positions drift via integrated noise — tolerance grows with k
    np.testing.assert_allclose(mean[:, :, :3], nominal[:, :, :3], atol=0.5)
    np.testing.assert_allclose(mean[:, :, 3], nominal[:, :, 3], atol=0.1)
