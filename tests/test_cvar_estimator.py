import numpy as np
from scipy import stats
from src.risk.cvar import cvar_tail_average, cvar_rockafellar, variance_scale


def test_cvar_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200_000)
    for alpha in (0.7, 0.9, 0.95):
        true = stats.norm.pdf(stats.norm.ppf(alpha)) / (1.0 - alpha)
        est_tail = cvar_tail_average(z, alpha)
        est_rock = cvar_rockafellar(z, alpha)
        assert abs(est_tail - true) < 0.03
        assert abs(est_rock - true) < 0.03


def test_cvar_monotone_in_alpha():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(10_000)
    prev = -np.inf
    for a in (0.5, 0.7, 0.9, 0.95, 0.99):
        c = cvar_tail_average(z, a)
        assert c > prev; prev = c


def test_variance_scale_preserves_mean():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((4, 100))
    out = variance_scale(z, B=3.0)
    np.testing.assert_allclose(out.mean(-1), z.mean(-1), atol=1e-10)
    assert out.var(-1).mean() > z.var(-1).mean() * 4  # B^2 ≈ 9
