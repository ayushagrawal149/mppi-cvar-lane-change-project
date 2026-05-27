"""HighwayEnv subclass tailored to the CVaR-MPPI lane-change study.

Three lanes, 10 surrounding traffic vehicles, continuous (acceleration,
steering) control. The traffic uses HighwayEnv's IDM longitudinal +
MOBIL lane-change behaviour, so rare-but-significant tail events
(sudden hard-brakes, cut-ins) emerge naturally; scripted variants live
in tests/test_near_miss_scenarios.py.

We deliberately do not modify the upstream HighwayEnv submodule -- all
MPPI logic lives outside (src/dynamics, src/controllers, src/costs).
"""

from __future__ import annotations

import gymnasium as gym
from highway_env.envs.highway_env import HighwayEnv

ENV_ID = "CVaRLaneChange-v0"


class CVaRLaneChangeEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "lanes_count": 3,
                "vehicles_count": 10,
                "controlled_vehicles": 1,
                "duration": 40,
                "policy_frequency": 20,
                "simulation_frequency": 100,
                "ego_spacing": 2,
                "vehicles_density": 1,
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                },
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 11,
                    "features": ["presence", "x", "y", "vx", "vy", "heading"],
                    "absolute": True,
                    "normalize": False,
                    "see_behind": True,
                },
                "collision_reward": -100.0,
                "lane_change_reward": 0.0,
                "right_lane_reward": 0.0,
                "high_speed_reward": 0.0,
                "reward_speed_range": [20.0, 30.0],
            }
        )
        return cfg


if ENV_ID not in gym.envs.registry:
    gym.register(id=ENV_ID, entry_point=CVaRLaneChangeEnv)
