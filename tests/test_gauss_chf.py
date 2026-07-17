"""Validation of the chf-based per-step chi^2 engine (EXP-2 machinery)
against the generic quadrature engine — two fully independent code paths."""
import numpy as np

from fors.targets import GaussianMixture
from fors.schedules import vp_schedule
from fors.diffusion import DiffusionSampler
from fors.quadrature import StepQuad, clip_gauss_mean
from fors.gauss_chf import (GaussStep, centered_density_from_chf,
                            _tail_h_integral)


def _mk(mix, sched, k, B):
    ds = DiffusionSampler(mix, sched, B=B)
    return StepQuad(sched.alpha[k], sched.eta[k], sched.sigma2[k],
                    sched.abar[k], B,
                    lambda x: ds.denoiser(k, x),
                    lambda x: ds.exact_score(k + 1, x),
                    lambda x: ds.denoiser(k + 1, x))


def test_density_pipeline_gaussian():
    """Known Gaussian chf -> E[h] matches the closed form to < 1e-10."""
    mu, sig, B = 0.3, 0.5, 0.8
    T_half = 40 * sig
    nt = 8192
    dt = np.pi / T_half
    t = (np.arange(nt) - nt // 2) * dt
    phi = np.exp(-0.5 * sig**2 * t**2)          # mean-shifted convention
    T, f = centered_density_from_chf(phi[None, :], dt)
    Eh = _tail_h_integral(T, f, -B - mu, B - mu)[0]
    truth = clip_gauss_mean(B, 1.0, np.array([mu]), np.array([sig]))[0] - mu
    assert abs(Eh - truth) < 1e-10  # measured 2.3e-12 at nt=8192


def test_chi2_matches_engine_d1():
    """chi^2 agreement with the generic engine at d=1 in the signal regime."""
    mix = GaussianMixture([1.0], [[0.0]], [[1.0]])
    B, G = 0.8, 8.0
    sched = vp_schedule(1e-3, G, 0.05)
    k = sched.K // 2
    step = _mk(mix, sched, k, B)
    gs = GaussStep(sched, k, B, d=1)
    pk = mix.noised(sched.abar[k], sched.sigma2[k])
    r = step.step_divergences(np.array([1.5]), pk.logpdf, n_r=32, n_u=96,
                              grid_n=2401, grid_w=16.0)
    c, _ = gs.chi2_given_xplus(1.5, n_r=32, nt=2048, n_x1=48)
    assert abs(c / r["chi2"] - 1) < 0.01, (c, r["chi2"])


def test_defect_matches_engine_d2():
    """Pointwise defect at d=2: chf triple-reduction vs 2D tensor-GH engine."""
    mix2 = GaussianMixture([1.0], [[0.0, 0.0]], [[1.0, 1.0]])
    B, G = 0.8, 8.0
    sched = vp_schedule(1e-3, G, 0.05)
    k = sched.K // 2
    step, big = _mk(mix2, sched, k, B), _mk(mix2, sched, k, 1e9)
    gs = GaussStep(sched, k, B, d=2)
    xp2 = np.array([1.5, 0.0])
    pts = np.array([[0.9, 0.3], [1.2, -0.5], [0.7, 0.0], [1.4, 0.8]])
    d_eng = (step.mean_w_reduced(pts, xp2, n_r=24, n_u=48)
             - big.mean_w_reduced(pts, xp2, n_r=24, n_u=48))
    d_chf = gs.defect_batch(pts[:, 0], pts[:, 1], pts[:, 1]**2, 1.5,
                            n_r=24, nt=2048)
    assert np.max(np.abs(d_eng - d_chf)) < 2e-4


def test_no_clip_zero_defect():
    """Huge B => defect identically zero (analytic branch)."""
    mix = GaussianMixture([1.0], [[0.0]], [[1.0]])
    sched = vp_schedule(1e-3, 50.0, 0.05)
    gs = GaussStep(sched, sched.K // 2, 1e6, d=8)
    d = gs.defect_batch(np.array([0.5]), np.array([1.0]), np.array([2.0]), 1.0)
    assert abs(d[0]) == 0.0
