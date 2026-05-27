"""End-to-end smoke tests: MPPI drives the ego in CVaRLaneChange-v0."""

import gymnasium as gym
import numpy as np

import src.envs.cvar_lane_change_env  # noqa: F401
from src.controllers.mppi import MPPI, MPPIConfig
from src.utils.observation import (
    ego_state,
    normalize_action,
    target_lane_y,
    traffic_state,
)


def test_mppi_returns_finite_action():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=0)
    target_y = target_lane_y(env, 0)
    ctrl = MPPI(MPPIConfig(target_y=target_y, n_samples=64, horizon=10))
    u, info = ctrl.step(ego_state(env), traffic_state(env))
    assert u.shape == (2,)
    assert np.all(np.isfinite(u))
    assert info["weights"].shape == (64,)
    np.testing.assert_allclose(info["weights"].sum(), 1.0, atol=1e-6)
    env.close()


def test_mppi_episode_runs_at_least_a_few_steps():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=0)
    target_y = target_lane_y(env, 0)
    ctrl = MPPI(MPPIConfig(target_y=target_y, n_samples=64, horizon=10))
    survived = 0
    for _ in range(20):
        u, _ = ctrl.step(ego_state(env), traffic_state(env))
        action = normalize_action(env, u)
        _, _, terminated, truncated, _ = env.step(action)
        survived += 1
        if terminated or truncated:
            break
    env.close()
    assert survived >= 1
