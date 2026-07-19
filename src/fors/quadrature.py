"""Deterministic quadrature certification of per-step KL (arXiv:2602.01338).

Computes, with zero Monte-Carlo noise, the quantities the paper's own proof
(Sec. F.2 chain rule) runs on:

    m_k(x; x_plus) = E_{r,z,xhat}[ Clip_B W_hat ]          (pointwise tilt)
    rho_hat_k(. | x_plus) ∝ q_k(.) exp(m_k(.))             (FORS step law)
    KL(rho_k(.|x_plus) || rho_hat_k(.|x_plus))              (per-step error)
    sum_k E_{x_plus ~ p_{k+1}} KL(...)                      (chain bound)

Two independent rules for E[Clip_B W_hat]:

* `mean_w_bruteforce` — GL(r) x GH(z) x GH(xhat) tensor rule, exactly as
  PLAN.md specifies (1D).
* `mean_w_reduced` — GL(r) x GH(gamma) with the (z, xhat)-integral collapsed
  analytically: given (r, x), (gamma, gamma_dot) is jointly Gaussian with
  isotropic blocks, and W_hat depends on the latents only through the scalar
  t = <gamma_dot, D_k(gamma) - D_+>, whose law given gamma is 1D Gaussian, so
  E[Clip_B(lambda t) | gamma] is a closed-form truncated-Gaussian moment.
  This is exact (no extra approximation) and one full quadrature dimension
  cheaper; it also works in any ambient dimension d.

The two rules agree to quadrature precision — asserted in tests.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorized

from .diffusion import path_diffusion
from .metrics import grid_normalize, grid_kl, grid_chi2

SQRT2 = np.sqrt(2.0)


def _phi(x):
    # standard normal pdf; clip the argument so |x| beyond ~40 sigma (where the
    # density is < 1e-300 anyway) does not overflow x*x in float64 — the result
    # underflows to 0 correctly, this only silences a benign RuntimeWarning.
    x = np.clip(x, -1e150, 1e150)
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def clip_gauss_mean(B, lam, m, s):
    """E[Clip_B(lam * t)] for t ~ N(m, s^2), closed form. Vectorized.

    Clip_B(lam t) = -B on t < -B/lam, lam t in between, +B above (lam > 0).
    """
    s = np.maximum(s, 1e-300)
    lo, hi = -B / lam, B / lam
    al, be = (lo - m) / s, (hi - m) / s
    Fa, Fb = ndtr(al), ndtr(be)
    mid = lam * (m * (Fb - Fa) - s * (_phi(be) - _phi(al)))
    return -B * Fa + B * (1.0 - Fb) + mid


def gl_nodes(n):
    """Gauss-Legendre nodes/weights on [0, 1]."""
    r, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (r + 1.0), 0.5 * w


def gh_nodes(n):
    """Gauss-Hermite nodes/weights for N(0,1): x = sqrt(2) t, w/sqrt(pi)."""
    t, w = np.polynomial.hermite.hermgauss(n)
    return SQRT2 * t, w / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# step parameters bundle
# ---------------------------------------------------------------------------

class StepQuad:
    """Quadrature for backward step k of Algorithm 2.

    Parameters mirror DiffusionSampler.step_params:
      alpha_k, eta_k, etabar_k, sigma2_k (= sigma_k^2), abar_k, B,
      denoiser_k(x (n,d)) -> (n,d)  [uses the score ESTIMATE, so Arm-C
      perturbations flow through], score_next(x) for the proposal mean,
      denoiser_next(x) for D_{k+1}(x_plus).
    """

    def __init__(self, alpha_k, eta_k, sigma2_k, abar_k, B,
                 denoiser_k, score_next, denoiser_next):
        self.alpha, self.eta, self.sig2, self.abar, self.B = (
            alpha_k, eta_k, sigma2_k, abar_k, B)
        self.etabar = 1.0 / (1.0 / eta_k + 1.0 / sigma2_k)
        self.lam = abar_k / sigma2_k
        self.D_k, self.s_next, self.D_next = denoiser_k, score_next, denoiser_next

    def proposal(self, x_plus):
        """(Xbar (d,), etabar) for a single conditioning point x_plus (d,)."""
        xp = np.atleast_2d(x_plus)
        xbar = xp / self.alpha + self.alpha * self.eta * self.s_next(xp)
        return xbar[0], self.etabar

    # ---- reduced rule (any d) ---------------------------------------------
    def mean_w_reduced(self, x_grid, x_plus, n_r=24, n_u=48):
        """m(x) = E[Clip_B W] on x_grid (nx, d), conditioning on x_plus (d,).

        GL in r; GH in the 1D radial coordinate of gamma about its mean
        (gamma has isotropic covariance s_u^2 I; but D_k(gamma) is nonlinear in
        gamma, so gamma must be integrated per-dimension: for d == 1 a single
        GH axis; for d > 1 a tensor GH grid over gamma of size n_u^d — still
        one full dimension cheaper than brute force).
        """
        x_grid = np.atleast_2d(np.asarray(x_grid, dtype=np.float64))
        nx, d = x_grid.shape
        xbar, etab = self.proposal(x_plus)
        Dp = self.D_next(np.atleast_2d(x_plus))[0]      # (d,)
        half = etab / 2.0

        r, wr = gl_nodes(n_r)
        a, b, da, db = path_diffusion(r)                 # (nr,)
        s_u2 = ((1 - a) ** 2 + b ** 2) * half            # Var(gamma) scale
        s_v2 = (da ** 2 + db ** 2) * half                # Var(gamma_dot) scale
        c_uv = (-da * (1 - a) + b * db) * half           # Cov scale
        s_u = np.sqrt(s_u2)
        # conditional slope in the stable parametrization u = mu_u + s_u * t:
        # E[v|u] = mu_v + (c_uv/s_u) t ;  Var(v|u) = s_v2 - c_uv^2/s_u2
        slope = c_uv / np.maximum(s_u, 1e-300)
        s_cond2 = np.maximum(s_v2 - slope ** 2, 0.0)

        t, wt = gh_nodes(n_u)
        if d == 1:
            T = t[None, :, None]                          # (1, nu, 1)
            WT = wt
            tdim = 1
        else:
            grids = np.meshgrid(*([t] * d), indexing="ij")
            T = np.stack([g.ravel() for g in grids], axis=-1)[None]   # (1, nu^d, d)
            wg = np.meshgrid(*([wt] * d), indexing="ij")
            WT = np.prod(np.stack([g.ravel() for g in wg], axis=0), axis=0)
            tdim = T.shape[1]

        m_out = np.zeros(nx)
        for i in range(len(r)):                           # loop nr (small)
            mu_u = a[i] * x_grid + (1 - a[i]) * xbar      # (nx, d)
            mu_v = da[i] * (x_grid - xbar)                # (nx, d)
            gamma = mu_u[:, None, :] + s_u[i] * T         # (nx, nt, d)
            Dg = self.D_k(gamma.reshape(-1, d)).reshape(nx, -1, d)
            w_vec = Dg - Dp[None, None, :]                # (nx, nt, d)
            # t = <v, w>, v|gamma ~ N(mu_v + slope * t_node_vec, s_cond2 I)
            m_t = np.einsum("xd,xtd->xt", mu_v, w_vec) \
                + slope[i] * np.einsum("td,xtd->xt", np.atleast_2d(T[0]), w_vec)
            s_t = np.sqrt(s_cond2[i] * np.sum(w_vec ** 2, axis=-1))
            inner = clip_gauss_mean(self.B, self.lam, m_t, s_t)   # (nx, nt)
            m_out += wr[i] * inner @ WT
        return m_out

    # ---- brute-force rule (d == 1, PLAN.md spec) ---------------------------
    def mean_w_bruteforce(self, x_grid, x_plus, n_r=16, n_z=16, n_xh=16):
        """GL(r) x GH(z) x GH(xhat) tensor rule, 1D only (cross-check)."""
        x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
        xbar, etab = self.proposal(x_plus)
        xbar = float(xbar[0])
        Dp = float(self.D_next(np.atleast_2d(x_plus))[0, 0])
        sqh = np.sqrt(etab / 2.0)

        r, wr = gl_nodes(n_r)
        tz, wz = gh_nodes(n_z)
        th, wh = gh_nodes(n_xh)
        a, b, da, db = path_diffusion(r)

        X = x_grid[:, None, None, None]
        A, Bc, dA, dB = (v[None, :, None, None] for v in (a, b, da, db))
        Z = (sqh * tz)[None, None, :, None]
        XH = (xbar + sqh * th)[None, None, None, :]
        gamma = A * X + (1 - A) * XH + Bc * Z
        gdot = dA * (X - XH) + dB * Z
        shp = gamma.shape
        Dg = self.D_k(gamma.reshape(-1, 1)).reshape(shp)
        W = np.clip(self.lam * gdot * (Dg - Dp), -self.B, self.B)
        return np.einsum("xrzh,r,z,h->x", W, wr, wz, wh)

    def local_grid(self, x_plus, half_width_sigmas=12.0, n=801):
        """Grid adapted to the backward kernel: rho(.|x_plus) is p_k times a
        Gaussian factor of variance eta_k, so it lives within a few sqrt(eta)
        of x_plus/alpha. Centered on the proposal mean, width in sqrt(eta)."""
        xbar, _ = self.proposal(x_plus)
        c = float(np.atleast_1d(xbar)[0])
        w = half_width_sigmas * np.sqrt(self.eta)
        return np.linspace(c - w, c + w, n)

    # ---- per-step densities and divergences --------------------------------
    def step_divergences(self, x_plus, log_pk, x_grid=None, n_r=24, n_u=48,
                         grid_n=801, grid_w=12.0, mean_w=None):
        """Per-step true vs FORS backward kernels on a 1D grid, and KL/chi^2.

        log_pk : callable, log p_k on the grid (closed-form mixture).
        rho    ∝ p_k(x) exp(-||x - x_plus/alpha||^2 / (2 eta))
        rhohat ∝ q_k(x) exp(m(x))
        x_grid : optional; defaults to the kernel-adapted local grid.
        Returns dict(kl, chi2, rho, rhohat, x).
        """
        xp = float(np.asarray(x_plus).ravel()[0])
        x = (np.asarray(x_grid, dtype=np.float64).ravel() if x_grid is not None
             else self.local_grid(np.array([xp]), grid_w, grid_n))
        xbar, etab = self.proposal(np.array([xp]))
        xbar = float(xbar[0])

        log_rho = log_pk(x[:, None]) - (x - xp / self.alpha) ** 2 / (2 * self.eta)
        m = mean_w if mean_w is not None else self.mean_w_reduced(
            x[:, None], np.array([xp]), n_r=n_r, n_u=n_u)
        log_rhohat = -(x - xbar) ** 2 / (2 * etab) + m

        rho = grid_normalize(log_rho, x)
        rhohat = grid_normalize(log_rhohat, x)
        return dict(
            kl=grid_kl(rho, rhohat, x),
            chi2=grid_chi2(rho, rhohat, x),
            rho=rho, rhohat=rhohat, x=x,
        )


def expected_step_divergence(step: StepQuad, mix_next, log_pk,
                             n_xp=32, n_r=24, n_u=48, grid_n=801,
                             grid_w=12.0):
    """E_{x_plus ~ p_{k+1}}[KL and chi2]: GH nodes per component of the (1D)
    closed-form mixture p_{k+1} = mix_next; a kernel-adapted local grid is
    built per node.

    Returns dict(kl, chi2, kl_nodes, chi2_nodes, weights)."""
    t, wt = gh_nodes(n_xp)
    kls, chis, wts = [], [], []
    for h in range(mix_next.H):
        mu_h = float(mix_next.mu[h, 0])
        sd_h = float(np.sqrt(mix_next.var[h, 0]))
        for j in range(len(t)):
            xp = mu_h + sd_h * t[j]
            res = step.step_divergences(np.array([xp]), log_pk, n_r=n_r,
                                        n_u=n_u, grid_n=grid_n, grid_w=grid_w)
            kls.append(res["kl"])
            chis.append(res["chi2"])
            wts.append(float(mix_next.w[h]) * wt[j])
    kls, chis, wts = np.array(kls), np.array(chis), np.array(wts)
    return dict(kl=float(np.sum(kls * wts)), chi2=float(np.sum(chis * wts)),
                kl_nodes=kls, chi2_nodes=chis, weights=wts)
