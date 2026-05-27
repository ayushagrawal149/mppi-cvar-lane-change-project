"""Unit tests for the bicycle rollout used inside MPPI."""

import numpy as np

from src.dynamics import bicycle


def test_constant_velocity_straight_line():
    state0 = np.array([[0.0, 0.0, 0.0, 25.0]])
    K, dt = 10, 0.05
    U = np.zeros((1, K, 2))
    traj = bicycle.rollout(state0, U, dt)
    assert traj.shape == (1, K + 1, 4)
    np.testing.assert_allclose(traj[0, -1, 0], 25.0 * K * dt, atol=1e-9)
    np.testing.assert_allclose(traj[0, -1, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj[0, -1, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj[0, -1, 3], 25.0, atol=1e-9)


def test_pure_acceleration_no_steering():
    state0 = np.array([[0.0, 0.0, 0.0, 10.0]])
    K, dt = 20, 0.05
    U = np.tile(np.array([2.0, 0.0]), (1, K, 1))
    traj = bicycle.rollout(state0, U, dt)
    np.testing.assert_allclose(traj[0, -1, 3], 10.0 + 2.0 * K * dt, atol=1e-9)
    assert traj[0, -1, 0] > 0.0
    np.testing.assert_allclose(traj[0, -1, 1], 0.0, atol=1e-12)


def test_positive_steering_curves_left():
    """Positive steering should rotate heading toward +psi and shift +Y."""
    state0 = np.array([[0.0, 0.0, 0.0, 10.0]])
    K, dt = 30, 0.05
    U = np.tile(np.array([0.0, 0.05]), (1, K, 1))
    traj = bicycle.rollout(state0, U, dt)
    assert traj[0, -1, 2] > 0.0
    assert traj[0, -1, 1] > 0.0


def test_batched_rollout_shapes():
    M, K = 8, 12
    state0 = np.tile(np.array([0.0, 0.0, 0.0, 20.0]), (M, 1))
    U = np.zeros((M, K, 2))
    traj = bicycle.rollout(state0, U, 0.05)
    assert traj.shape == (M, K + 1, 4)
