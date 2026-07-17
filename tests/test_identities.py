"""Exact identities gating everything (PLAN.md B.4): Eq. (15), Lemma F.1."""
import numpy as np

from fors.diffusion import path_diffusion, C_LEMMA_F1


def test_eq15_identity():
    """a_r^2 + (1-a_r)^2/2 + b_r^2/2 == 1 for all r in [0,1], residual < 1e-14."""
    r = np.linspace(0.0, 1.0, 100_001)
    a, b, _, _ = path_diffusion(r)
    res = a**2 + 0.5 * (1 - a) ** 2 + 0.5 * b**2 - 1.0
    assert np.max(np.abs(res)) < 1e-14


def test_lemma_f1_constant():
    """3 a'^2/2 + b'^2/2 == c = 8 pi^2/27 for all r, residual < 1e-13."""
    r = np.linspace(0.0, 1.0, 100_001)
    _, _, da, db = path_diffusion(r)
    res = 1.5 * da**2 + 0.5 * db**2 - C_LEMMA_F1
    assert np.max(np.abs(res)) < 1e-13
    assert C_LEMMA_F1 <= 3.0


def test_boundary_values():
    """a_0 = b_1 = 0 and a_1 = b_0 = 1 (path endpoints, Sec. 4.2)."""
    a, b, _, _ = path_diffusion(np.array([0.0, 1.0]))
    assert abs(a[0]) < 1e-15 and abs(b[1]) < 1e-15
    assert abs(a[1] - 1) < 1e-15 and abs(b[0] - 1) < 1e-15


def test_lemma_f1_joint_law():
    """MC check (1e6): for x~N(g,eta), xhat~N(g,eta/2), z~N(0,eta/2), r~U[0,1]:
    gamma ~ N(g, eta), gamma_dot ~ N(0, c*eta), independent (4-sigma bands)."""
    rng = np.random.default_rng(7)
    n, g, eta = 1_000_000, 1.3, 0.37
    x = g + np.sqrt(eta) * rng.standard_normal(n)
    xh = g + np.sqrt(eta / 2) * rng.standard_normal(n)
    z = np.sqrt(eta / 2) * rng.standard_normal(n)
    r = rng.uniform(size=n)
    a, b, da, db = path_diffusion(r)
    gam = a * x + (1 - a) * xh + b * z
    gdot = da * (x - xh) + db * z

    se_mean = np.sqrt(eta / n)
    assert abs(gam.mean() - g) < 4 * se_mean
    assert abs(gdot.mean()) < 4 * np.sqrt(C_LEMMA_F1 * eta / n)
    # variances: SE(var) ~ var * sqrt(2/n)
    assert abs(gam.var() - eta) < 4 * eta * np.sqrt(2 / n)
    assert abs(gdot.var() - C_LEMMA_F1 * eta) < 4 * C_LEMMA_F1 * eta * np.sqrt(2 / n)
    # independence: Cov(gamma, gamma_dot) = 0; SE ~ sqrt(var1*var2/n)
    cov = np.mean((gam - gam.mean()) * (gdot - gdot.mean()))
    assert abs(cov) < 4 * np.sqrt(eta * C_LEMMA_F1 * eta / n)
    # and Cov(gamma^2, gdot^2) ~ 0 (true independence, not just uncorrelated)
    c2 = np.corrcoef(gam**2, gdot**2)[0, 1]
    assert abs(c2) < 4 / np.sqrt(n)
