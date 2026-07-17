"""Mixture score vs torch autograd; posterior-covariance identity; schedules."""
import numpy as np
import torch

from fors.targets import GaussianMixture, bimodal_1d, subspace_mixture
from fors.schedules import vp_schedule, ve_schedule, G_of


def _torch_mixture_logpdf(x, w, mu, var):
    """Reference torch implementation (independent of the numpy code path)."""
    diff = x[:, None, :] - mu[None]
    q = (diff**2 / var[None]).sum(-1)
    logdet = torch.log(var).sum(-1)
    d = mu.shape[1]
    logc = -0.5 * (q + logdet[None] + d * np.log(2 * np.pi))
    return torch.logsumexp(logc + torch.log(w)[None], dim=1)


def test_score_vs_autograd():
    """Exact mixture score matches autograd of log p to < 1e-9 (2D, noised)."""
    mix = GaussianMixture([0.3, 0.7], [[-2.0, 0.5], [1.5, -1.0]],
                          [[0.3, 0.8], [0.5, 0.2]])
    pk = mix.noised(abar=0.8, sigma2=0.36)
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=2.0, size=(50, 2))

    x = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
    lp = _torch_mixture_logpdf(
        x, torch.tensor(pk.w), torch.tensor(pk.mu), torch.tensor(pk.var))
    (g,) = torch.autograd.grad(lp.sum(), x)
    assert np.max(np.abs(g.numpy() - pk.score(pts))) < 1e-9
    # logpdf itself agrees
    assert np.max(np.abs(lp.detach().numpy() - pk.logpdf(pts))) < 1e-12


def test_posterior_cov_is_gradm_times_tau():
    """grad m_tau = Cov(Y0|Y_tau)/tau; for a single Gaussian the closed form is
    diag(var/(var+tau)) — machine-exact, plus autograd cross-check on a mixture."""
    tau = 0.7
    g1 = GaussianMixture([1.0], [[0.5, -1.0]], [[0.9, 0.4]])
    y = np.array([[0.3, 0.2]])
    gm = g1.grad_m(y, tau)[0]
    expect = np.diag(g1.var[0] / (g1.var[0] + tau))
    assert np.max(np.abs(gm - expect)) < 1e-14

    mix = bimodal_1d()
    y = np.array([[0.4]])
    # autograd of m_tau(y) = y + tau * score_{q_tau}(y) (Tweedie)
    sm = mix.noised(1.0, tau)
    yt = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    lp = _torch_mixture_logpdf(
        yt, torch.tensor(sm.w), torch.tensor(sm.mu), torch.tensor(sm.var))
    (g,) = torch.autograd.grad(lp.sum(), yt, create_graph=True)
    m = yt + tau * g                     # Tweedie: m_tau(y) = y + tau grad log q_tau
    (dm,) = torch.autograd.grad(m.sum(), yt)
    assert abs(dm.item() - mix.grad_m(y, tau)[0, 0, 0]) < 1e-9


def test_subspace_target_trace():
    """Cor-E.4 mechanism: tr(grad m_tau) is flat in d for subspace targets but
    grows ~ linearly in d for full-rank targets (same tau, same y=0)."""
    tau = 0.5
    y16, y64 = np.zeros((1, 16)), np.zeros((1, 64))
    tr_sub16 = np.trace(subspace_mixture(16, thick=1e-8).grad_m(y16, tau)[0])
    tr_sub64 = np.trace(subspace_mixture(64, thick=1e-8).grad_m(y64, tau)[0])
    tr_full64 = np.trace(subspace_mixture(64, thick=0.25).grad_m(y64, tau)[0])
    assert abs(tr_sub64 - tr_sub16) < 1e-4          # intrinsic: d-independent
    assert tr_full64 > tr_sub64 + 10.0              # ambient: grows with d


def test_vp_schedule_identities():
    """eta_k = sigma_k^2/G; alpha_k^2 = (1-s_{k+1}^2)/(1-s_k^2);
    sigma_{k+1}^2 = alpha_k^2 (sigma_k^2 + eta_k); abar^2 + sigma^2 = 1."""
    s = vp_schedule(sigma0_sq=1e-4, G=100.0, deltabar=0.05)
    assert s.K > 0
    sig2, eta, al = s.sigma2, s.eta, s.alpha
    assert np.max(np.abs(eta - sig2[:-1] / 100.0)) < 1e-15
    a2 = (1 - sig2[1:]) / (1 - sig2[:-1])
    assert np.max(np.abs(al**2 - a2)) < 1e-12
    assert np.max(np.abs(sig2[1:] - a2 * (sig2[:-1] + eta))) < 1e-12
    assert np.max(np.abs(s.abar**2 + sig2 - 1.0)) < 1e-12
    assert 1 - sig2[-1] <= 0.05
    # etabar: 1/etabar = 1/eta + 1/sigma^2
    assert np.max(np.abs(1 / s.etabar - (1 / eta + 1 / sig2[:-1]))) < 1e-9


def test_ve_schedule_and_G():
    s = ve_schedule(sigma0_sq=1e-3, G=50.0, sigma_max_sq=10.0)
    assert np.max(np.abs(np.diff(s.sigma2) - s.eta)) < 1e-15
    assert np.max(np.abs(s.eta - s.sigma2[:-1] / 50.0)) < 1e-15
    assert G_of(2, 1e-3, 100) > 0
