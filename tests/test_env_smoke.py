"""Smoke tests for the CVaRLaneChange-v0 environment wrapper."""

import gymnasium as gym
import numpy as np

import src.envs.cvar_lane_change_env  # noqa: F401  -- registers env id


def test_env_resets_with_three_lanes_and_ten_traffic():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=0)
    cfg = env.unwrapped.config
    assert cfg["lanes_count"] == 3
    assert cfg["vehicles_count"] == 10
    others = [
        v
        for v in env.unwrapped.road.vehicles
        if v is not env.unwrapped.vehicle
    ]
    assert len(others) == 10
    env.close()


def test_env_accepts_continuous_actions_for_a_few_steps():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=0)
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break
    env.close()


def test_action_space_is_2d_continuous_in_unit_box():
    env = gym.make("CVaRLaneChange-v0")
    env.reset(seed=0)
    space = env.action_space
    assert space.shape == (2,)
    assert np.all(space.low == -1.0) and np.all(space.high == 1.0)
    env.close()
