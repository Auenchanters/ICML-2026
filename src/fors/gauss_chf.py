"""Deterministic per-step chi^2 for pdata = N(0, I_d) at ANY ambient d.

Setting (EXP-2 / Claim 2): VP schedule, exact linear scores; p_k = N(0, I) for
every k and D_k(x) = abar_k x. Conditional on path time r and evaluation
point x, the pre-clip estimator is

    T = lam <gamma_dot, abar_k gamma - abar_{k+1} x_plus>
      = sum_i v_i (p u_i + q_i),   p = lam abar_k,  q_i = -lam abar_{k+1} x_plus_i,

a sum of d INDEPENDENT products of bivariate Gaussians whose means are affine
in x_i. Consequences, all exact:

  * log chf of T_i is exactly quadratic in x_i, so log chf of T depends on x
    only through (x1, S1, S2): the x_plus-aligned coordinate and the power
    sums of the remaining d-1 coordinates.
  * The clip defect  defect(x) = E[Clip_B W|x] - E[W|x]  (the ONLY deviation
    of the FORS step law from the true backward kernel — the unclipped tilt
    is exact) is a function of that triple: chf -> centered-DFT density ->
    integral, no Monte Carlo.
  * chi^2(rho||rho_hat) = E_rho[e^defect] E_rho[e^-defect] - 1, and under
    rho = N(m_rho, v_rho I): x1 Gaussian; S1 ~ N(0,(d-1)v_rho);
    S2 = S1^2/(d-1) + v_rho Q with Q ~ chi2(d-2) independent of S1.
  * x_plus ~ N(0, I_d) enters through ||x_plus||^2 ~ chi2(d) only.

Total: nested 1D Gauss rules x batched FFTs — cost independent of d.
Cross-checked against the generic quadrature engine at d = 1 (tests).
"""
from __future__ import annotations

import numpy as np
from scipy.special import roots_genlaguerre

from .diffusion import path_diffusion
from .quadrature import gh_nodes, gl_nodes


# ---------------------------------------------------------------------------
# bivariate-Gaussian product: chf (exact) and mean
# ---------------------------------------------------------------------------

def logchf_vw(t, mu_v, mu_w, s_vv, s_ww, s_vw):
    """log E[exp(i t v w)], (v,w) bivariate normal; vectorized over t.

    With v = mu_v + L0.Z, w = mu_w + L1.Z (Z ~ N(0,I2)) and Q = v w:
    E e^{sQ} = det(I-sA)^{-1/2} exp(s c + (s^2/2) b^T (I-sA)^{-1} b),
    A = L0 L1^T + L1 L0^T, b = mu_v L1 + mu_w L0, c = mu_v mu_w, s = it.
    """
    l00 = np.sqrt(max(s_vv, 0.0))
    l10 = s_vw / l00 if l00 > 0 else 0.0
    l11 = np.sqrt(max(s_ww - l10 * l10, 0.0))
    a00 = 2 * l00 * l10
    a01 = l00 * l11
    a11 = 0.0
    b0 = mu_v * l10 + mu_w * l00
    b1 = mu_v * l11
    c = mu_v * mu_w
    s = 1j * np.asarray(t, dtype=np.float64)
    m00 = 1.0 - s * a00
    m01 = -s * a01
    m11 = 1.0 - s * a11
    det = m00 * m11 - m01 * m01
    ib0 = (m11 * b0 - m01 * b1) / det
    ib1 = (-m01 * b0 + m00 * b1) / det
    quad = b0 * ib0 + b1 * ib1
    return -0.5 * np.log(det) + s * c + 0.5 * s * s * quad


def mean_vw(mu_v, mu_w, s_vw):
    return mu_v * mu_w + s_vw


def _tail_h_integral(T, f, cL, cR):
    """int h(T) f(T) dT for h(T) = (cR - T) on T > cR, (cL - T) on T < cL,
    0 between: Simpson on the strictly-outside grid points plus EXACT kink
    sub-cell pieces (f linear across the kink cell). f (m, n); returns (m,).

    Right piece over [cR, T_j] (L = T_j - cR, f linear fC -> f_j):
        int_0^L (-u)[fC + (f_j - fC) u/L] du = -L^2 (fC/6 + f_j/3);
    left piece symmetric with +L^2.
    """
    from scipy.integrate import simpson
    dT = T[1] - T[0]
    n = len(T)
    total = np.zeros(f.shape[0])
    j = int(np.searchsorted(T, cR, side="right"))      # first T_j > cR
    if j < n - 1:
        hr = cR - T[j:]
        total += simpson(f[:, j:] * hr[None, :], dx=dT, axis=-1)
        if j >= 1:
            L = T[j] - cR
            fC = f[:, j - 1] + (f[:, j] - f[:, j - 1]) * (cR - T[j - 1]) / dT
            total += -(L * L) * (fC / 6.0 + f[:, j] / 3.0)
    k = int(np.searchsorted(T, cL, side="right")) - 1  # last T_k <= cL
    if k >= 1:
        hl = cL - T[: k + 1]
        total += simpson(f[:, : k + 1] * hl[None, :], dx=dT, axis=-1)
        if k + 1 < n:
            L = cL - T[k]
            fC = f[:, k] + (f[:, k + 1] - f[:, k]) * (cL - T[k]) / dT
            total += (L * L) * (fC / 6.0 + f[:, k] / 3.0)
    return total


def centered_density_from_chf(phi, dt):
    """Given phi sampled on the symmetric grid t_j = (j - n/2) dt (n even),
    return (T_grid, f) with f the density on T_m = (m - n/2) dT,
    dT = 2 pi/(n dt). Uses the centered-DFT identity
    f = fftshift(fft(ifftshift(phi))) * dt/(2 pi). Batched over leading axes.
    """
    n = phi.shape[-1]
    dT = 2 * np.pi / (n * dt)
    T = (np.arange(n) - n // 2) * dT
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(phi, axes=-1), axis=-1),
                        axes=-1)
    f = np.real(F) * dt / (2 * np.pi)
    return T, np.maximum(f, 0.0)


# ---------------------------------------------------------------------------
# the per-step object
# ---------------------------------------------------------------------------

class GaussStep:
    """FORS step k for pdata = N(0, I_d), exact scores, clip level B."""

    def __init__(self, sched, k, B, d):
        self.B, self.d = float(B), int(d)
        self.abar_k = float(sched.abar[k])
        self.abar_n = float(sched.abar[k + 1])
        self.sig2 = float(sched.sigma2[k])
        self.eta = float(sched.eta[k])
        self.etab = 1.0 / (1.0 / self.eta + 1.0 / self.sig2)
        self.lam = self.abar_k / self.sig2
        self.alpha = float(sched.alpha[k])
        # proposal mean Xbar = kappa x_plus  (s_{k+1}(y) = -y exactly)
        self.kappa = 1.0 / self.alpha - self.alpha * self.eta
        # true kernel rho(.|x_plus) = N(m_coef x_plus, v_rho I)
        prec = 1.0 + 1.0 / self.eta
        self.v_rho = 1.0 / prec
        self.m_coef = (1.0 / (self.alpha * self.eta)) / prec

    def _vw_params(self, r, x, xbar_i, q_i):
        """Bivariate params of (v, w) for one coordinate at path time r."""
        a, b, da, db = (float(v[0]) for v in path_diffusion(np.asarray([r])))
        half = self.etab / 2.0
        mu_u = a * x + (1 - a) * xbar_i
        mu_v = da * (x - xbar_i)
        s_uu = ((1 - a) ** 2 + b ** 2) * half
        s_vv = (da ** 2 + db ** 2) * half
        s_uv = (-da * (1 - a) + b * db) * half
        p = self.lam * self.abar_k
        return mu_v, p * mu_u + q_i, s_vv, p * p * s_uu, p * s_uv

    def _class_coeffs(self, r, t, xbar_i, q_i):
        """log chf and mean of T_i as EXACT quadratics in the coordinate x:
        3-point fit (the dependence is exactly quadratic). Returns
        (c0,c1,c2) complex arrays over t and (e0,e1,e2) scalars."""
        ls, ms = [], []
        for xv in (-1.0, 0.0, 1.0):
            pars = self._vw_params(r, xv, xbar_i, q_i)
            ls.append(logchf_vw(t, *pars))
            ms.append(mean_vw(pars[0], pars[1], pars[4]))
        c0 = ls[1]
        c1 = 0.5 * (ls[2] - ls[0])
        c2 = 0.5 * (ls[2] + ls[0]) - ls[1]
        e0 = ms[1]
        e1 = 0.5 * (ms[2] - ms[0])
        e2 = 0.5 * (ms[2] + ms[0]) - ms[1]
        return (c0, c1, c2), (e0, e1, e2)

    def defect_batch(self, x1, S1, S2, x_plus_norm, n_r=16, nt=4096,
                     n_sigmas=40.0):
        """defect(x1, S1, S2) = E[Clip_B W] - E[W] for batched triples.

        Per r-node, each row is classified by (mean, sd) of T:
          * |mu| + n_sigmas*sd <= B  -> the clip never binds: contributes 0
            (tail bound e^{-n_sigmas^2/2} ~ e^{-338});
          * mu -+ n_sigmas*sd beyond +-B on one side -> whole density in one
            linear branch of h: E[h(T)] = +-B - mu exactly;
          * straddling rows (the only ones carrying signal): per-row FFT of
            the MEAN-SHIFTED chf e^{-i mu t} phi(t) on a local window
            +- n_sigmas*sd — the window always resolves the density, so no
            truncation ringing regardless of how narrow the row is.
        """
        x1, S1, S2 = (np.asarray(v, dtype=np.float64).ravel()
                      for v in (x1, S1, S2))
        m = len(x1)
        r_nodes, r_w = gl_nodes(n_r)
        q_axis = -self.lam * self.abar_n * x_plus_norm
        xbar_axis = self.kappa * x_plus_norm
        dm1 = self.d - 1
        B = self.B
        hfd = 1e-4

        out = np.zeros(m)
        for r, w in zip(r_nodes, r_w):
            # exact means and (FD) variances per row
            (c0, c1, c2), (e0, e1, e2) = self._class_coeffs(
                r, np.array([-hfd, 0.0, hfd]), xbar_axis, q_axis)
            (o0, o1, o2), (f0, f1, f2) = self._class_coeffs(
                r, np.array([-hfd, 0.0, hfd]), 0.0, 0.0)
            lp = (c0[None, :] + np.outer(x1, c1) + np.outer(x1**2, c2)
                  + dm1 * o0[None, :] + np.outer(S1, o1) + np.outer(S2, o2))
            mu = (e0 + e1 * x1 + e2 * x1**2 + dm1 * f0 + f1 * S1 + f2 * S2)
            # -d2 log phi/dt2 at 0 is the second CUMULANT = Var(T) directly
            var = np.real(-(lp[:, 0] - 2 * lp[:, 1] + lp[:, 2]) / hfd**2)
            sd = np.sqrt(np.maximum(var, 1e-16))
            wid = n_sigmas * sd

            below = mu + wid < -B          # fully in the left branch
            above = mu - wid > B           # fully in the right branch
            inside = (np.abs(mu) + wid <= B)
            out[below] += w * (-B - mu[below])
            out[above] += w * (B - mu[above])
            straddle = ~(below | above | inside)
            if not np.any(straddle):
                continue
            idx = np.nonzero(straddle)[0]
            # bucket rows by sd (factor-1.3 bands): after the mean shift the
            # window depends only on sd, so each bucket shares one t-grid and
            # the FFT is batched; only the O(nt) tail integral is per-row.
            keys = np.floor(np.log(sd[idx]) / np.log(1.3)).astype(int)
            for key in np.unique(keys):
                rows = idx[keys == key]
                T_half = n_sigmas * sd[rows].max()
                dt = np.pi / T_half
                t = (np.arange(nt) - nt // 2) * dt
                ax_c, _ = self._class_coeffs(r, t, xbar_axis, q_axis)
                lphi = (ax_c[0][None, :] + np.outer(x1[rows], ax_c[1])
                        + np.outer(x1[rows]**2, ax_c[2]))
                if dm1 > 0:
                    of_c, _ = self._class_coeffs(r, t, 0.0, 0.0)
                    lphi += (dm1 * of_c[0][None, :]
                             + np.outer(S1[rows], of_c[1])
                             + np.outer(S2[rows], of_c[2]))
                lphi -= 1j * np.outer(mu[rows], t)          # mean shift
                T, f = centered_density_from_chf(np.exp(lphi), dt)
                for jj, i in enumerate(rows):
                    out[i] += w * _tail_h_integral(
                        T, f[jj:jj + 1], -B - mu[i], B - mu[i])[0]
        return out

    # -------------------------------------------------------------------
    def chi2_given_xplus(self, x_plus_norm, n_r=16, nt=4096,
                         n_x1=16, n_s1=16, n_q=16):
        """chi^2(rho || rho_hat) for one conditioning norm ||x_plus||."""
        g1, w1 = gh_nodes(n_x1)
        m_rho = self.m_coef * x_plus_norm
        sv = np.sqrt(self.v_rho)
        x1 = m_rho + sv * g1

        if self.d == 1:
            X1 = x1; S1 = np.zeros_like(x1); S2 = np.zeros_like(x1); W = w1
        else:
            gs, ws = gh_nodes(n_s1)
            s1 = np.sqrt((self.d - 1) * self.v_rho) * gs
            if self.d >= 3:
                dfq = self.d - 2
                yq, wq = roots_genlaguerre(n_q, dfq / 2.0 - 1.0)
                q = 2.0 * yq * self.v_rho
                wq = wq / wq.sum()
            else:
                q = np.array([0.0]); wq = np.array([1.0])
            X1, S1g, Qg = np.meshgrid(x1, s1, q, indexing="ij")
            Wg = (w1[:, None, None] * ws[None, :, None] * wq[None, None, :])
            X1 = X1.ravel(); S1 = S1g.ravel()
            S2 = (S1g**2 / (self.d - 1) + Qg).ravel()
            W = Wg.ravel()
        dfc = self.defect_batch(X1, S1, S2, x_plus_norm, n_r=n_r, nt=nt)
        ep = np.sum(W * np.exp(dfc))
        em = np.sum(W * np.exp(-dfc))
        return float(ep * em - 1.0), dfc

    def chi2_expected(self, n_xp=10, **kw):
        """E over ||x_plus||: x_plus ~ N(0, I_d), ||x_plus||^2 ~ chi2(d).
        Returns (expected chi2, max over nodes)."""
        y, w = roots_genlaguerre(n_xp, self.d / 2.0 - 1.0)
        w = w / w.sum()
        norms = np.sqrt(2.0 * y)
        vals = []
        for nv in norms:
            c, _ = self.chi2_given_xplus(float(nv), **kw)
            vals.append(c)
        vals = np.array(vals)
        return float(np.sum(w * vals)), float(vals.max())
