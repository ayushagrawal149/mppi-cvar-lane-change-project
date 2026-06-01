"""CVaR estimators for RA-MPPI.

cvar_tail_average    — eq. (9) of theory.md / [P5] eq. 12
cvar_rockafellar     — eq. (8) variational form; useful for tests
variance_scale       — theory.md §5.5 / [P5] eq. 18–19
"""

from __future__ import annotations

import numpy as np


def cvar_tail_average(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Tail-average CVaR estimator.

    samples: (..., q) — last axis are i.i.d. samples of the loss L.
    alpha  : confidence level ∈ (0,1); the tail is the worst 1−α fraction.
    Returns: (...,) one CVaR estimate per leading index.

    Implementation: sort along last axis, average the top n_tail entries
    where n_tail = max(1, ceil(q * (1−α))).
    """
    samples = np.asarray(samples)
    q = samples.shape[-1]
    n_tail = max(1, int(np.ceil(q * (1.0 - alpha))))
    sorted_desc = np.sort(samples, axis=-1)[..., -n_tail:]  # top n_tail
    return sorted_desc.mean(axis=-1)


def cvar_rockafellar(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Sample CVaR via the Rockafellar–Uryasev form (eq. 8 of theory.md).

    Equivalent to cvar_tail_average for continuous loss distributions;
    kept for sanity tests against the closed-form Gaussian CVaR.
    """
    samples = np.asarray(samples)
    q = samples.shape[-1]
    # minimizer of w + 1/(q(1-α)) Σ (z_i − w)+ is the (αq)-th order statistic
    var_idx = int(np.ceil(alpha * q)) - 1
    var_idx = min(max(var_idx, 0), q - 1)
    w = np.partition(samples, var_idx, axis=-1)[..., var_idx]
    excess = np.maximum(samples - w[..., None], 0.0)
    return w + excess.sum(axis=-1) / (q * (1.0 - alpha))


def variance_scale(
    samples: np.ndarray, B: float
) -> np.ndarray:
    """L_a^{m,n} = B(L^{m,n} − \bar L^m) + \bar L^m, B ≥ 1 (§5.5)."""
    if B == 1.0:
        return samples
    mean = samples.mean(axis=-1, keepdims=True)
    return B * (samples - mean) + mean
