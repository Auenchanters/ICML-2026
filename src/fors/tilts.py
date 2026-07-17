"""Section 3.2 of arXiv:2602.01338 — sampling Gaussian tilts by FORS.

Target:  nu(x) ∝ exp(-f(x) - ||x - x0||^2 / (2 eta)),  first-order access to f.

Recipe (Thm 3.3 instantiation):
  x_plus  ≈ prox_{eta f}(x0)          (Newton; Appendix G assumes exact prox)
  x_hat   = x0 - eta * grad_f(x_plus)
  proposal q = N(x_hat, eta I)
  path     gamma_{z,r}(x) = a_r x + (1-a_r) x_hat + b_r z,
           a_r = sin(pi r/2), b_r = cos(pi r/2),  z ~ N(0, eta I), r ~ U[0,1]
  estimator W = <gamma_dot, grad_f(x_plus) - grad_f(gamma)>, clipped to [-B, B].

Pre-clipping identity (verified in tests): E_{r,z} W = w(x) + const with
q(x) e^{w(x)} ∝ nu(x) exactly.
"""
from __future__ import annotations

import numpy as np

from .fors import fors_batch

HALF_PI = 0.5 * np.pi


def prox_newton(df, d2f, x0, eta, tol=1e-13, max_iter=60):
    """Solve x - x0 + eta * df(x) = 0 (the prox_{eta f}(x0) stationarity eq.)
    by 1D-per-coordinate Newton (targets here are products of 1D potentials).
    Returns x_plus with |residual| <= tol * max(1, |x0|); raises if not converged.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    x0 = np.asarray(x0, dtype=np.float64)
    scale = np.maximum(1.0, np.abs(x0))
    for _ in range(max_iter):
        g = x - x0 + eta * df(x)
        if np.all(np.abs(g) <= tol * scale):
            return x
        h = 1.0 + eta * d2f(x)
        x = x - g / h
    raise RuntimeError(f"prox Newton failed: max residual {np.max(np.abs(g)):.3e}")


def path_sin(r):
    """a_r = sin(pi r / 2), b_r = cos(pi r / 2) and derivatives."""
    a = np.sin(HALF_PI * r)
    b = np.cos(HALF_PI * r)
    da = HALF_PI * np.cos(HALF_PI * r)
    db = -HALF_PI * np.sin(HALF_PI * r)
    return a, b, da, db


def tilt_mean_w_quad(x_grid, x_hat, x_plus, df, eta, B, n_r=32, n_z=32):
    """E_{r,z}[Clip_B W] on a 1D grid of x by Gauss-Legendre (r) x
    Gauss-Hermite (z) quadrature. Returns (n_x,) array.

    1D only (d=1): z ~ N(0, eta)."""
    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    # GL nodes on [0,1]
    r, wr = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * (r + 1.0)
    wr = 0.5 * wr
    # GH nodes for N(0, eta):  z = sqrt(2 eta) t, weight w/sqrt(pi)
    t, wt = np.polynomial.hermite.hermgauss(n_z)
    z = np.sqrt(2.0 * eta) * t
    wz = wt / np.sqrt(np.pi)
    a, b, da, db = path_sin(r)
    # broadcast: x (nx,1,1), r (nr,1) via a[:,None]? build (nx, nr, nz)
    X = x_grid[:, None, None]
    A, Bc, dA, dB = (v[None, :, None] for v in (a, b, da, db))
    Z = z[None, None, :]
    gamma = A * X + (1 - A) * x_hat + Bc * Z
    gdot = dA * (X - x_hat) + dB * Z
    W = gdot * (df(np.asarray(x_plus)) - df(gamma))
    W = np.clip(W, -B, B)
    return np.einsum("xrz,r,z->x", W, wr, wz)


def rgo_fors(f_pot, x0, eta, B, n_out, rng, prox_tol=1e-13):
    """Sample n_out draws from nu ∝ exp(-f - ||.-x0||^2/(2 eta)) via FORS.

    f_pot: targets.Potential1D (1D; product structure extends coordinatewise).
    Returns (samples (n_out,), stats)."""
    x_plus = prox_newton(f_pot.df, f_pot.d2f, np.asarray([x0], dtype=np.float64),
                         eta, tol=prox_tol)[0]
    g_plus = f_pot.df(x_plus)
    x_hat = x0 - eta * g_plus
    sqrt_eta = np.sqrt(eta)

    def propose_n(n, rng):
        return x_hat + sqrt_eta * rng.standard_normal((n, 1))

    def draw_w_batch(x, J, rng):
        n, jmax = len(x), max(int(J.max()), 1)
        r = rng.uniform(size=(n, jmax))
        z = sqrt_eta * rng.standard_normal((n, jmax))
        a, b, da, db = path_sin(r)
        gamma = a * x + (1 - a) * x_hat + b * z
        gdot = da * (x - x_hat) + db * z
        return np.clip(gdot * (g_plus - f_pot.df(gamma)), -B, B)

    out, st = fors_batch(propose_n, draw_w_batch, B, n_out, rng)
    return out.ravel(), st


def true_tilt_grid(f_pot, x0, eta, lo, hi, n=8001):
    """Normalized nu ∝ exp(-f - ||.-x0||^2/(2 eta)) on a uniform grid."""
    x = np.linspace(lo, hi, n)
    logu = -f_pot.f(x) - (x - x0) ** 2 / (2 * eta)
    logu -= logu.max()
    u = np.exp(logu)
    return x, u / np.trapezoid(u, x)
