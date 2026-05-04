import gymnasium as gym
from highway_env.envs.highway_env import HighwayEnv

class CVaRLaneChangeEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "lane_count": 3,
            "vehicles_count": 8,
            "duration": 40,
            "collision_reward": -100.0,
            # ... add your shaping later

        })
        return cfg

gym.register(id="CVaRLaneChange-v0", entry_point=CVaRLaneChangeEnv)
