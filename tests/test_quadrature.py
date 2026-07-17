"""Certification-engine tests.

The gating sanity (PLAN.md P3): for a pure-Gaussian target with exact scores
and huge B (clip never binds), the FORS step law rho_hat equals the true
backward kernel rho EXACTLY (the Sec-4.2 tilt identity is algebraically exact),
so the quadrature KL must vanish to numerical precision (< 1e-10).

Plus: the reduced rule (closed-form clip integral) agrees with PLAN.md's
brute-force GL x GH x GH rule, and is self-converged under node doubling.
"""
import numpy as np

from fors.targets import GaussianMixture, bimodal_1d
from fors.schedules import vp_schedule
from fors.quadrature import StepQuad, expected_step_divergence, clip_gauss_mean
from fors.diffusion import DiffusionSampler


def _make_step(mix, sched, k, B):
    ds = DiffusionSampler(mix, sched, B=B)
    return StepQuad(
        alpha_k=sched.alpha[k], eta_k=sched.eta[k], sigma2_k=sched.sigma2[k],
        abar_k=sched.abar[k], B=B,
        denoiser_k=lambda x: ds.denoiser(k, x),
        score_next=lambda x: ds.exact_score(k + 1, x),
        denoiser_next=lambda x: ds.denoiser(k + 1, x),
    )


def test_pure_gaussian_huge_B_kl_zero():
    """rho_hat == rho for Gaussian target, exact scores, no clipping."""
    mix = GaussianMixture([1.0], [[0.7]], [[1.3]])
    sched = vp_schedule(sigma0_sq=0.02, G=40.0, deltabar=0.3)
    k = sched.K // 2
    step = _make_step(mix, sched, k, B=1e8)
    pk = mix.noised(sched.abar[k], sched.sigma2[k])
    res = step.step_divergences(np.array([0.9]), pk.logpdf, n_r=32, n_u=64,
                                grid_n=1201)
    assert abs(res["kl"]) < 1e-10, f"KL = {res['kl']:.3e}"
    assert abs(res["chi2"]) < 1e-9


def test_mixture_huge_B_kl_zero():
    """Same exactness for the bimodal mixture (nonlinear score) — the tilt
    identity is exact for ANY target when B doesn't bind."""
    mix = bimodal_1d()
    sched = vp_schedule(sigma0_sq=0.05, G=60.0, deltabar=0.3)
    k = sched.K // 2
    step = _make_step(mix, sched, k, B=1e8)
    pk = mix.noised(sched.abar[k], sched.sigma2[k])
    res = step.step_divergences(np.array([1.4]), pk.logpdf, n_r=32, n_u=64,
                                grid_n=1201)
    assert abs(res["kl"]) < 1e-10, f"KL = {res['kl']:.3e}"


def test_reduced_vs_bruteforce():
    """The two independent quadrature rules agree on E[Clip_B W] with an
    ACTIVE clip (B = 1), on the bimodal mixture."""
    mix = bimodal_1d()
    sched = vp_schedule(sigma0_sq=0.05, G=60.0, deltabar=0.3)
    k = sched.K // 2
    step = _make_step(mix, sched, k, B=1.0)
    x_plus = np.array([1.4])
    x = step.local_grid(x_plus, half_width_sigmas=6.0, n=41)[:, None]
    m_red = step.mean_w_reduced(x, x_plus, n_r=32, n_u=64)
    m_bf = step.mean_w_bruteforce(x.ravel(), x_plus, n_r=32, n_z=48, n_xh=48)
    # brute-force GH must resolve the clip kink; agreement at its accuracy
    assert np.max(np.abs(m_red - m_bf)) < 2e-4, np.max(np.abs(m_red - m_bf))


def test_reduced_rule_self_convergence():
    """Node-doubling changes the reduced-rule result by < 1e-9 (smooth integrand)."""
    mix = bimodal_1d()
    sched = vp_schedule(sigma0_sq=0.05, G=60.0, deltabar=0.3)
    k = sched.K // 2
    step = _make_step(mix, sched, k, B=1.0)
    x_plus = np.array([1.4])
    x = step.local_grid(x_plus, half_width_sigmas=6.0, n=21)[:, None]
    m1 = step.mean_w_reduced(x, x_plus, n_r=24, n_u=48)
    m2 = step.mean_w_reduced(x, x_plus, n_r=48, n_u=96)
    assert np.max(np.abs(m1 - m2)) < 1e-9, np.max(np.abs(m1 - m2))


def test_clip_gauss_mean_closed_form():
    """Closed-form truncated-Gaussian moment vs dense numerical integral."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        B = rng.uniform(0.3, 3.0)
        lam = rng.uniform(0.2, 5.0)
        m = rng.uniform(-2, 2)
        s = rng.uniform(0.05, 2.0)
        from scipy.integrate import quad
        pdf = lambda t: np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        val_num, _ = quad(lambda t: np.clip(lam * t, -B, B) * pdf(t),
                          m - 14 * s, m + 14 * s,
                          points=[p for p in (-B / lam, B / lam)
                                  if abs(p - m) < 14 * s],
                          limit=200)
        val_cf = clip_gauss_mean(B, lam, np.array([m]), np.array([s]))[0]
        assert abs(val_num - val_cf) < 1e-11


def test_expected_step_divergence_runs():
    """Chain-rule building block: E_{x+ ~ p_{k+1}} KL is finite, small for a
    conservative schedule, and all node values are nonnegative."""
    mix = bimodal_1d()
    sched = vp_schedule(sigma0_sq=0.05, G=200.0, deltabar=0.3)
    k = sched.K // 2
    step = _make_step(mix, sched, k, B=1.0)
    p_next = mix.noised(sched.abar[k + 1], sched.sigma2[k + 1])
    pk = mix.noised(sched.abar[k], sched.sigma2[k])
    res = expected_step_divergence(step, p_next, pk.logpdf,
                                   n_xp=8, n_r=16, n_u=32, grid_n=401)
    assert np.isfinite(res["kl"]) and res["kl"] >= -1e-12
    assert np.all(res["kl_nodes"] > -1e-10)
    assert res["kl"] < 1e-2
