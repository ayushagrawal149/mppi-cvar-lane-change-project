import gymnasium as gym, numpy as np
import src.envs.cvar_lane_change_env  # noqa: F401
from src.controllers.cvar_mppi import CVaRMPPI, CVaRMPPIConfig
from src.utils.observation import ego_state, normalize_action, target_lane_y, traffic_state


def test_cvar_mppi_returns_finite_action():
    env = gym.make("CVaRLaneChange-v0"); env.reset(seed=0)
    target_y = target_lane_y(env, 0)
    ctrl = CVaRMPPI(CVaRMPPIConfig(
        target_y=target_y, n_samples=32, horizon=10, n_disturb=8,
    ))
    u, info = ctrl.step(ego_state(env), traffic_state(env))
    assert u.shape == (2,) and np.all(np.isfinite(u))
    assert info["cvar"].shape == (32,)
    np.testing.assert_allclose(info["weights"].sum(), 1.0, atol=1e-6)
    env.close()


def test_high_penalty_filters_unsafe_rollouts():
    """With A huge and budget 0, dangerous rollouts must lose weight."""
    env = gym.make("CVaRLaneChange-v0"); env.reset(seed=0)
    target_y = target_lane_y(env, 0)
    ctrl = CVaRMPPI(CVaRMPPIConfig(
        target_y=target_y, n_samples=64, horizon=10, n_disturb=12,
        alpha=0.9, risk_budget=0.0, penalty_weight=1e6,
    ))
    _, info = ctrl.step(ego_state(env), traffic_state(env))
    # weights on over-budget rollouts should be ~0
    over = info["cvar"] > 0
    if over.any():
        assert info["weights"][over].sum() < 1e-3
    env.close()
