"""Analytic target distributions with exact scores (arXiv:2602.01338 repro).

Everything here is closed-form so that score error is exactly zero and ground
truth is available to numerical precision. Notation follows the paper:

  forward VP process  X_k | X_0 ~ N(abar_k X_0, sigma_k^2 I),  abar^2 + sigma^2 = 1
  forward VE process  X_k | X_0 ~ N(X_0, sigma_k^2 I)          (abar == 1)

For a Gaussian mixture pdata = sum_h w_h N(mu_h, diag(s2_h)) the noised law is
  p_k = sum_h w_h N(abar mu_h, diag(abar^2 s2_h + sigma^2)),
which stays a mixture, so p_k, its score s*_k, the denoiser D*_k, and the
posterior-mean Jacobian grad m_tau = Cov(Y0|Y_tau)/tau are all exact.
"""
from __future__ import annotations

import numpy as np

_LOG2PI = np.log(2.0 * np.pi)


class GaussianMixture:
    """Mixture of axis-aligned Gaussians on R^d.

    Parameters
    ----------
    weights : (H,) positive, will be normalized.
    means : (H, d)
    var : (H, d) per-component per-coordinate variances (diagonal covariance).
    """

    def __init__(self, weights, means, var):
        self.w = np.asarray(weights, dtype=np.float64)
        self.w = self.w / self.w.sum()
        self.mu = np.atleast_2d(np.asarray(means, dtype=np.float64))
        self.var = np.atleast_2d(np.asarray(var, dtype=np.float64))
        if self.var.shape != self.mu.shape:
            # allow scalar/(H,) isotropic variance input
            self.var = np.broadcast_to(
                np.asarray(var, dtype=np.float64).reshape(-1, 1), self.mu.shape
            ).copy()
        self.H, self.d = self.mu.shape
        assert self.w.shape == (self.H,)

    # ---- helpers ------------------------------------------------------
    def noised(self, abar: float, sigma2: float) -> "GaussianMixture":
        """Law of abar*X0 + N(0, sigma2 I) — again a GaussianMixture."""
        return GaussianMixture(self.w, abar * self.mu, abar**2 * self.var + sigma2)

    def _log_comp(self, x):
        """(n, H) log N(x; mu_h, diag(var_h)) for x of shape (n, d)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        diff = x[:, None, :] - self.mu[None, :, :]            # (n, H, d)
        q = np.sum(diff**2 / self.var[None], axis=-1)          # (n, H)
        logdet = np.sum(np.log(self.var), axis=-1)             # (H,)
        return -0.5 * (q + logdet[None] + self.d * _LOG2PI)

    def resp(self, x):
        """(n, H) posterior responsibilities r_h(x) (softmax, stable)."""
        a = self._log_comp(x) + np.log(self.w)[None]
        a -= a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    # ---- densities / scores -------------------------------------------
    def logpdf(self, x):
        a = self._log_comp(x) + np.log(self.w)[None]
        m = a.max(axis=1)
        return m + np.log(np.exp(a - m[:, None]).sum(axis=1))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def score(self, x):
        """s(x) = grad log p(x), shape (n, d)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        r = self.resp(x)                                       # (n, H)
        grad_h = (self.mu[None] - x[:, None, :]) / self.var[None]  # (n, H, d)
        return np.sum(r[:, :, None] * grad_h, axis=1)

    def sample(self, n, rng):
        idx = rng.choice(self.H, size=n, p=self.w)
        return self.mu[idx] + rng.standard_normal((n, self.d)) * np.sqrt(self.var[idx])

    def second_moment(self):
        """M2^2 = E||X||^2."""
        return float(np.sum(self.w[:, None] * (self.mu**2 + self.var)))

    # ---- posterior quantities (VE smoothing q_tau = pdata * N(0, tau I)) ----
    def posterior_moments(self, y, tau):
        """Posterior mean m_tau(y)=E[Y0|Y_tau=y] and covariance Cov(Y0|Y_tau=y).

        Returns (mean (n,d), cov (n,d,d)). Mixture posterior: within-component
        Gaussian posteriors (diagonal) plus between-component spread.
        """
        y = np.atleast_2d(np.asarray(y, dtype=np.float64))
        n = y.shape[0]
        sm = self.noised(1.0, tau)          # marginal of Y_tau
        r = sm.resp(y)                       # (n, H)
        # per-component posterior: N(mh, diag(ch)) with
        # mh = (var*y + tau*mu)/(var+tau), ch = var*tau/(var+tau)
        denom = self.var + tau                                # (H, d)
        mh = (self.var[None] * y[:, None, :] + tau * self.mu[None]) / denom[None]
        ch = self.var * tau / denom                           # (H, d)
        mean = np.sum(r[:, :, None] * mh, axis=1)             # (n, d)
        cov = np.zeros((n, self.d, self.d))
        ii = np.arange(self.d)
        cov[:, ii, ii] = np.sum(r[:, :, None] * ch[None], axis=1)
        dm = mh - mean[:, None, :]                             # (n, H, d)
        cov += np.einsum("nh,nhi,nhj->nij", r, dm, dm)
        return mean, cov

    def grad_m(self, y, tau):
        """grad m_tau(y) = Cov(Y0|Y_tau=y)/tau, shape (n, d, d)."""
        _, cov = self.posterior_moments(y, tau)
        return cov / tau

    def denoiser(self, x, abar, sigma2):
        """D*(x) = E[X0 | abar X0 + sigma Z = x], computed directly from the
        mixture posterior (numerically stable even when abar -> 0, unlike the
        Tweedie form (x + sigma2 * s_k(x))/abar which cancels catastrophically
        near the terminal step)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        pk = self.noised(abar, sigma2)
        r = pk.resp(x)                                        # (n, H)
        denom = abar**2 * self.var + sigma2                   # (H, d)
        # E[X0 | x, h] = mu_h + abar*var_h/(abar^2 var_h + sigma2) * (x - abar mu_h)
        gain = abar * self.var / denom                        # (H, d)
        mh = self.mu[None] + gain[None] * (x[:, None, :] - abar * self.mu[None])
        return np.sum(r[:, :, None] * mh, axis=1)


def bimodal_1d():
    """The plan's primary EXP-1 target: 0.5 N(-2, 0.5^2) + 0.5 N(+2, 0.8^2)."""
    return GaussianMixture([0.5, 0.5], [[-2.0], [2.0]], [[0.25], [0.64]])


def subspace_mixture(d, sep=2.0, s2=0.25, thick=1e-6, rng=None):
    """2-component mixture supported on the first 2 coordinates of R^d,
    thickened by variance `thick` off-subspace: the intrinsic-dimension target
    (d* = O(1) while ambient dim = d)."""
    mu = np.zeros((2, d))
    mu[0, 0], mu[1, 0] = -sep, sep
    mu[0, 1], mu[1, 1] = -sep / 2, sep / 2
    var = np.full((2, d), thick)
    var[:, :2] = s2
    return GaussianMixture([0.5, 0.5], mu, var)


# ---- log-concave potentials (Section 5 targets) --------------------------

class Potential1D:
    """1D potential f with derivatives; target density mu ∝ exp(-f).

    beta1 = sup f'' (gradient-Lipschitz constant, s=1 Holder);
    beta0 = sup |f'(x)-f'(y)| (s=0 Holder / bounded-gradient-variation).
    """

    def __init__(self, f, df, d2f, beta1=None, beta0=None, name=""):
        self.f, self.df, self.d2f = f, df, d2f
        self.beta1, self.beta0 = beta1, beta0
        self.name = name

    def grid_density(self, lo=-12.0, hi=12.0, n=20001):
        """Normalized density on a uniform grid by trapezoid quadrature."""
        x = np.linspace(lo, hi, n)
        u = np.exp(-self.f(x) + np.min(self.f(x)))
        z = np.trapezoid(u, x)
        return x, u / z


def quadratic_potential(s2=1.0):
    return Potential1D(
        f=lambda x: x**2 / (2 * s2),
        df=lambda x: x / s2,
        d2f=lambda x: np.full_like(np.asarray(x, dtype=np.float64), 1.0 / s2),
        beta1=1.0 / s2,
        name=f"quadratic(s2={s2})",
    )


def logcosh_potential():
    """f(x) = x^2/2 + log cosh(2x): strongly log-concave, smooth.
    f'(x) = x + 2 tanh(2x); f''(x) = 1 + 4 sech^2(2x) in (1, 5] => beta1 = 5."""
    return Potential1D(
        f=lambda x: x**2 / 2 + np.logaddexp(2 * x, -2 * x) - np.log(2.0),
        df=lambda x: x + 2 * np.tanh(2 * x),
        d2f=lambda x: 1 + 4 / np.cosh(2 * x) ** 2,
        beta1=5.0,
        name="x^2/2+logcosh(2x)",
    )


def pseudo_huber_potential(s2=1.0):
    """f(x) = x^2/(2 s2) + sqrt(1+x^2): the extra term has |f'| variation <= 2,
    used for the s=0 (Lipschitz-f) flavor on top of a quadratic confinement."""
    return Potential1D(
        f=lambda x: x**2 / (2 * s2) + np.sqrt(1 + x**2),
        df=lambda x: x / s2 + x / np.sqrt(1 + x**2),
        d2f=lambda x: 1.0 / s2 + 1.0 / (1 + x**2) ** 1.5,
        beta1=1.0 / s2 + 1.0,
        beta0=2.0,
        name="quadratic+pseudoHuber",
    )
