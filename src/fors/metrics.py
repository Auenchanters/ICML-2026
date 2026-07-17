"""Error metrics: grid divergences (certified, 1-2D), exact Gaussian W2,
kernel MMD, sliced-W1, moment errors (sample-based, floor O(n^{-1/2})).

Grid divergences use Simpson's rule (O(h^4)) so that 1e3-point local grids
reach ~1e-12 quadrature error on the smooth integrands that arise here."""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

FLOOR = 1e-300


def grid_normalize(logu, x):
    logu = np.asarray(logu, dtype=np.float64)
    logu = logu - logu.max()
    u = np.exp(logu)
    return u / simpson(u, x=x)


def grid_kl(p, q, x):
    """KL(p || q) for densities on a common 1D grid (Simpson)."""
    mask = p > FLOOR
    integrand = np.where(
        mask, p * (np.log(np.maximum(p, FLOOR)) - np.log(np.maximum(q, FLOOR))), 0.0)
    return float(simpson(integrand, x=x))


def grid_chi2(p, q, x):
    """chi^2(p || q) = int p^2/q - 1."""
    return float(simpson(p * p / np.maximum(q, FLOOR), x=x) - 1.0)


def grid_tv(p, q, x):
    return float(0.5 * simpson(np.abs(p - q), x=x))


def gaussian_w2(m1, S1, m2, S2):
    """Exact W2^2 between N(m1, S1) and N(m2, S2) (full covariances)."""
    from scipy.linalg import sqrtm
    m1, m2 = np.asarray(m1), np.asarray(m2)
    S1, S2 = np.atleast_2d(S1), np.atleast_2d(S2)
    R = sqrtm(S2)
    cross = sqrtm(R @ S1 @ R)
    return float(np.sum((m1 - m2) ** 2) + np.trace(S1 + S2 - 2 * np.real(cross)))


def mmd2_unbiased(X, Y, bandwidth=None):
    """Unbiased MMD^2 U-statistic with Gaussian kernel; median-heuristic
    bandwidth (computed on the pooled sample) if not given."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    n, m = len(X), len(Y)

    def sqd(A, B):
        return np.maximum(
            (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)

    if bandwidth is None:
        Z = np.vstack([X[: min(n, 2000)], Y[: min(m, 2000)]])
        D = sqd(Z, Z)
        med = np.median(D[np.triu_indices_from(D, k=1)])
        bandwidth = np.sqrt(max(med, 1e-12) / 2.0)
    g = 1.0 / (2 * bandwidth**2)
    Kxx = np.exp(-g * sqd(X, X)); np.fill_diagonal(Kxx, 0.0)
    Kyy = np.exp(-g * sqd(Y, Y)); np.fill_diagonal(Kyy, 0.0)
    Kxy = np.exp(-g * sqd(X, Y))
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                 - 2 * Kxy.mean())


def sliced_w1(X, Y, n_dirs=200, rng=None):
    """Sliced Wasserstein-1: mean 1D W1 over random directions."""
    rng = rng or np.random.default_rng(0)
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    d = X.shape[1]
    dirs = rng.standard_normal((n_dirs, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n = min(len(X), len(Y))
    tot = 0.0
    for u in dirs:
        a = np.sort(X @ u)[:n]
        b = np.sort(Y @ u)[:n]
        tot += np.abs(a - b).mean()
    return float(tot / n_dirs)


def moment_errors(X, mean_true, cov_true):
    """(||mean err||, ||cov err||_F) of a sample vs exact moments."""
    X = np.atleast_2d(X)
    m = X.mean(0)
    C = np.cov(X.T) if X.shape[1] > 1 else np.array([[np.var(X, ddof=1)]])
    return (float(np.linalg.norm(m - np.asarray(mean_true))),
            float(np.linalg.norm(np.atleast_2d(C) - np.atleast_2d(cov_true))))


def hist_kl_vs_grid(samples, x, p_true, bins=200, lo=None, hi=None):
    """KDE-free KL estimate: histogram the samples on [lo, hi] and compare to
    the true density integrated per bin. Sample-noise floor O(bins/n)."""
    s = np.asarray(samples).ravel()
    lo = lo if lo is not None else x[0]
    hi = hi if hi is not None else x[-1]
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(s, bins=edges)
    ph = counts / counts.sum() / np.diff(edges)
    # true bin masses -> densities
    pt = np.interp(0.5 * (edges[1:] + edges[:-1]), x, p_true)
    pt = pt / np.trapezoid(pt, 0.5 * (edges[1:] + edges[:-1]))
    mask = (ph > 0) & (pt > 0)
    w = np.diff(edges)[mask]
    return float(np.sum(w * ph[mask] * np.log(ph[mask] / pt[mask])))
