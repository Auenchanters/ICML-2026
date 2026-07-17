"""Algorithm 2 of arXiv:2602.01338 with the Section-4.2 instantiation.

Backward step k (from X_{k+1} to X_k), all quantities per the paper v2:

  Xbar_k  = alpha_k^{-1} X_{k+1} + alpha_k eta_k s_{k+1}(X_{k+1})   (exp. integrator)
  q_k     = N(Xbar_k, etabar_k I),        1/etabar_k = 1/eta_k + 1/sigma_k^2
  lambda_k = abar_k / sigma_k^2
  draw:  r ~ U[0,1],  z ~ N(0, etabar_k/2 I),  xhat ~ N(Xbar_k, etabar_k/2 I)
  path (Eqs. 13-14):
     a_r = (1 + 2 cos((2 pi/3)(1-r))) / 3,   b_r = (2/sqrt 3) sin((2 pi/3)(1-r))
     gamma = a_r x + (1-a_r) xhat + b_r z,   gamma_dot = a'_r (x - xhat) + b'_r z
  W_hat = Clip_B( lambda_k < gamma_dot, D_k(gamma) - D_{k+1}(X_{k+1}) > )

Identities verified in tests:
  Eq. (15):   a_r^2 + (1-a_r)^2/2 + b_r^2/2 == 1  for all r
  Lemma F.1:  3 a'^2/2 + b'^2/2 == c = 8 pi^2/27, and (gamma, gamma_dot)
              independent with gamma ~ N(g, eta I), gamma_dot ~ N(0, c eta I).

The DDPM baseline shares the identical proposal q_k (mean AND variance) and
simply skips the FORS correction — so any accuracy difference is attributable
to the corrector alone.
"""
from __future__ import annotations

import numpy as np

from .fors import fors_batch, ForsStats

TWO_PI_3 = 2.0 * np.pi / 3.0
C_LEMMA_F1 = 8.0 * np.pi**2 / 27.0


def path_diffusion(r):
    """Eqs. (13)-(14) path coefficients and derivatives (w.r.t. r)."""
    u = TWO_PI_3 * (1.0 - r)
    a = (1.0 + 2.0 * np.cos(u)) / 3.0
    b = (2.0 / np.sqrt(3.0)) * np.sin(u)
    da = (2.0 * TWO_PI_3 / 3.0) * np.sin(u)          # = (4 pi / 9) sin u
    db = -(2.0 / np.sqrt(3.0)) * TWO_PI_3 * np.cos(u)
    return a, b, da, db


class DiffusionSampler:
    """Runs Algorithm 2 on an analytic target with exact (or perturbed) scores.

    target: targets.GaussianMixture (pdata); sched: schedules.Schedule.
    score_fn(k, x) -> s_k(x): defaults to the exact mixture score of p_k.
    B: clip level, Theta(1) per the paper.
    """

    def __init__(self, target, sched, B=1.0, score_fn=None):
        self.t, self.s, self.B = target, sched, B
        self._score = score_fn or self.exact_score

    # exact score of p_k = law of abar_k X0 + sigma_k xi
    def exact_score(self, k, x):
        pk = self.t.noised(self.s.abar[k], self.s.sigma2[k])
        return pk.score(x)

    def denoiser(self, k, x):
        """D_k(x) = (x + sigma_k^2 s_k(x)) / abar_k, using the (possibly
        perturbed) score estimate — matches the paper's D_k defined from s_k."""
        return (np.atleast_2d(x) + self.s.sigma2[k] * self._score(k, x)) / self.s.abar[k]

    # ---- one backward step ------------------------------------------------
    def step_params(self, k, x_next):
        """Proposal parameters for step k given X_{k+1} = x_next (n, d)."""
        alpha_k = self.s.alpha[k]
        eta_k = self.s.eta[k]
        etabar_k = self.s.etabar[k]
        xbar = x_next / alpha_k + alpha_k * eta_k * self._score(k + 1, x_next)
        lam = self.s.abar[k] / self.s.sigma2[k]
        return xbar, etabar_k, lam

    def fors_step(self, k, x_next, rng, stats=None):
        """FORS-corrected backward step: returns X_k (n, d)."""
        x_next = np.atleast_2d(x_next)
        n, d = x_next.shape
        xbar, etab, lam = self.step_params(k, x_next)
        d_next = self.denoiser(k + 1, x_next)          # (n, d), fixed per chain
        sq = np.sqrt(etab)
        sqh = np.sqrt(etab / 2.0)
        out = np.empty_like(x_next)
        st = stats if stats is not None else ForsStats()
        # vectorize across chains: each chain runs its own FORS accept loop.
        # We batch by running all chains' proposal rounds together.
        alive = np.arange(n)
        for _ in range(10**4):
            m = len(alive)
            x = xbar[alive] + sq * rng.standard_normal((m, d))
            st.proposals += m
            J = rng.poisson(2.0 * self.B, size=m)
            jmax = int(J.max())
            logp = np.zeros(m)
            if jmax > 0:
                r = rng.uniform(size=(m, jmax))
                z = sqh * rng.standard_normal((m, jmax, d))
                xh = xbar[alive][:, None, :] + sqh * rng.standard_normal((m, jmax, d))
                a, b, da, db = path_diffusion(r)
                gamma = (a[..., None] * x[:, None, :] + (1 - a)[..., None] * xh
                         + b[..., None] * z)
                gdot = da[..., None] * (x[:, None, :] - xh) + db[..., None] * z
                Dg = self.denoiser(k, gamma.reshape(-1, d)).reshape(m, jmax, d)
                W = lam * np.sum(gdot * (Dg - d_next[alive][:, None, :]), axis=-1)
                W = np.clip(W, -self.B, self.B)
                mask = np.arange(jmax)[None, :] < J[:, None]
                st.w_draws += int(J.sum())
                ratio = np.where(mask, (self.B + W) / (2 * self.B), 1.0)
                logp = np.log(np.maximum(ratio, 1e-300)).sum(axis=1)
            acc = np.log(rng.uniform(size=m)) < logp
            out[alive[acc]] = x[acc]
            st.accepts += int(acc.sum())
            alive = alive[~acc]
            if len(alive) == 0:
                return out, st
        raise RuntimeError(f"{len(alive)} chains failed to accept in 1e4 rounds")

    def ddpm_step(self, k, x_next, rng):
        """Baseline: X_k ~ q_k = N(Xbar_k, etabar_k I) — same proposal, no corrector."""
        x_next = np.atleast_2d(x_next)
        xbar, etab, _ = self.step_params(k, x_next)
        return xbar + np.sqrt(etab) * rng.standard_normal(x_next.shape)

    # ---- full backward chains ----------------------------------------------
    def init_xK(self, n, rng):
        """X_K ~ N(0, sigma_K^2 I) (VP init of Cor 4.4)."""
        return np.sqrt(self.s.sigma2[-1]) * rng.standard_normal((n, self.t.d))

    def sample(self, n, rng, method="fors", k_stop=1):
        """Run the backward chain from k = K-1 down to k_stop (law ≈ p_{k_stop}).
        Returns (X (n, d), ForsStats)."""
        x = self.init_xK(n, rng)
        st = ForsStats()
        for k in range(self.s.K - 1, k_stop - 1, -1):
            if method == "fors":
                x, _ = self.fors_step(k, x, rng, stats=st)
            elif method == "ddpm":
                x = self.ddpm_step(k, x, rng)
            else:
                raise ValueError(method)
        return x, st
