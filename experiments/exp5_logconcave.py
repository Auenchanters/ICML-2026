"""EXP-5 (Claim 5, Section 5 / Theorem G.1): high-accuracy log-concave
sampling from first-order queries via proximal sampler + FORS-RGO.

Target (primary): f(x) = x^2/2 + log cosh(2x), strongly log-concave, beta1 = 5.
s=0 flavor: f(x) = x^2/2 + sqrt(1+x^2) (pseudo-Huber, |f'|-variation 2).

All 1D chain quantities are DETERMINISTIC (density evolution on a grid):

1. RGO exactness (Thm 3.3): chi^2(nu || nu_hat) by quadrature across eta;
   nu_hat ∝ q e^{E[Clip_B W]} with the Section-3 sin/cos path; verify the
   superpolynomial decay in 1/eta and report the smallest working C in
   eta^{-1} >= C beta1 sqrt(d) log(1/delta).
2. Chain-level: mu_{n+1} = RGO-kernel . heat-kernel . mu_n on a grid, with the
   FORS-RGO law (clip included) computed pointwise ONCE. chi^2(mu_n || mu)
   decays exponentially to the certified RGO-bias floor (<= 1e-10 target).
   Query accounting is deterministic too: per-call draws = 2B/A(y),
   A(y) = int q_y e^{m_y - B} — integrated over the chain's y-law.
3. NC-4: ULA density evolution at h in {1e-1, 1e-2, 1e-3} plateaus at the
   O(h) bias floor; money plot #3 = queries vs achieved log(1/error).

Newton prox is used per Appendix G's proximal-oracle assumption ("we assume
that this error is zero") — residual < 1e-13 enforced.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.targets import logcosh_potential, pseudo_huber_potential  # noqa: E402
from fors.tilts import prox_newton, tilt_mean_w_quad                # noqa: E402
from fors.metrics import grid_chi2, grid_normalize                  # noqa: E402

OUT = ROOT / "results" / "exp5"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv

B = 1.0
LO, HI, NG = -9.0, 9.0, 901 if QUICK else 1401


def rgo_law_grid(pot, y, eta, x, n_r=32, n_z=32, clip=B):
    """FORS-RGO output density on grid x for tilt center y (normalized)."""
    xp = prox_newton(pot.df, pot.d2f, np.array([y]), eta)[0]
    xhat = y - eta * pot.df(xp)
    m = tilt_mean_w_quad(x, xhat, xp, pot.df, eta, clip, n_r=n_r, n_z=n_z)
    logu = -(x - xhat) ** 2 / (2 * eta) + m
    return grid_normalize(logu, x), xhat, m


def true_tilt(pot, y, eta, x):
    return grid_normalize(-pot.f(x) - (x - y) ** 2 / (2 * eta), x)


# ---------------- 1. RGO exactness sweep -----------------------------------

def rgo_sweep(pot, tag, y_probe=1.2):
    x = np.linspace(LO, HI, NG)
    rows = []
    for inv_eta in [2, 4, 8, 16, 32, 64, 128] if not QUICK else [4, 16, 64]:
        eta = 1.0 / inv_eta
        nu_hat, _, _ = rgo_law_grid(pot, y_probe, eta, x)
        nu = true_tilt(pot, y_probe, eta, x)
        c2 = grid_chi2(nu, nu_hat, x)
        rows.append(dict(pot=tag, inv_eta=inv_eta, eta=eta, chi2=max(c2, 0.0)))
        print(f"[RGO {tag}] 1/eta={inv_eta}: chi2 = {c2:.3e}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"rgo_sweep_{tag}.csv", index=False)
    return rows


# ---------------- 2 + 3. chain-level evolution ------------------------------

def build_kernels(pot, eta, x):
    """Heat kernel H[j,i] = N(y_j; x_i, eta) dx and RGO kernel R[i,j] =
    rho_hat(x_i | y_j) dx, both column-stochastic on the grid."""
    dx = x[1] - x[0]
    H = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * eta))
    H /= H.sum(axis=0, keepdims=True)
    R = np.empty((len(x), len(x)))
    A = np.empty(len(x))                      # acceptance per y-node
    for j, y in enumerate(x):
        rho, xhat, m = rgo_law_grid(pot, y, eta, x)
        R[:, j] = rho
        q = grid_normalize(-(x - xhat) ** 2 / (2 * eta), x)
        A[j] = np.trapezoid(q * np.exp(np.clip(m, -B, B) - B), x)
    R /= R.sum(axis=0, keepdims=True)
    return H * 1.0, R, A, dx


def chain_run(pot, tag, eta, n_iter=60):
    x = np.linspace(LO, HI, NG)
    mu_star = grid_normalize(-pot.f(x), x)
    t0 = time.time()
    H, R, A, dx = build_kernels(pot, eta, x)
    print(f"[CHAIN {tag}] kernels built in {time.time()-t0:.0f}s "
          f"(min acceptance {A.min():.3f})", flush=True)
    mu = grid_normalize(-(x**2) / 2.0, x)     # N(0,1) start
    rows = []
    for n in range(1, n_iter + 1):
        y_law = H @ mu * dx
        y_law /= np.trapezoid(y_law, x)
        mu = R @ y_law * dx
        mu /= np.trapezoid(mu, x)
        c2 = grid_chi2(mu, mu_star, x)        # chi2(mu_n || mu)
        # deterministic query accounting: E draws this iter = E_y[2B/A(y)]
        q_iter = np.trapezoid(y_law * 2 * B / A, x)
        rows.append(dict(pot=tag, n=n, chi2=max(c2, 1e-18), q_iter=q_iter))
        if n % 10 == 0:
            print(f"[CHAIN {tag}] n={n}: chi2 = {c2:.3e}", flush=True)
    df = pd.DataFrame(rows)
    df["queries_cum"] = df.q_iter.cumsum()
    df.to_csv(OUT / f"chain_{tag}.csv", index=False)
    return df


def ula_run(pot, tag, hs=(1e-1, 1e-2, 1e-3), n_iter=100000):
    x = np.linspace(LO, HI, NG)
    dx = x[1] - x[0]
    mu_star = grid_normalize(-pot.f(x), x)
    rows = []
    for h in hs:
        drift = x - h * pot.df(x)
        K = np.exp(-(x[:, None] - drift[None, :]) ** 2 / (4 * h))
        K /= K.sum(axis=0, keepdims=True)
        mu = grid_normalize(-(x**2) / 2.0, x)
        last = None
        for n in range(1, n_iter + 1):
            mu = K @ mu
            mu /= np.trapezoid(mu, x)
            if n % 200 == 0 or n <= 50:
                c2 = grid_chi2(mu, mu_star, x)
                rows.append(dict(pot=tag, h=h, n=n, chi2=max(c2, 1e-18)))
                if last is not None and abs(c2 - last) < 1e-4 * abs(last):
                    break                     # converged to the bias plateau
                last = c2
        print(f"[ULA {tag}] h={h}: plateau chi2 = {last:.3e} after n={n}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"ula_{tag}.csv", index=False)


def main():
    t0 = time.time()
    lc = logcosh_potential()
    ph = pseudo_huber_potential()
    rgo_sweep(lc, "logcosh")
    rgo_sweep(ph, "pseudohuber")
    eta = 1.0 / 32.0
    df = chain_run(lc, "logcosh", eta, n_iter=30 if QUICK else 120)
    ula_run(lc, "logcosh", n_iter=2000 if QUICK else 100000)

    # fits: exponential decay rate + queries vs log(1/eps)
    d = df[df.chi2 > 1e-14]
    if len(d) >= 5:
        fit = np.polyfit(d.n, np.log(d.chi2), 1)
        r2 = 1 - (np.sum((np.log(d.chi2) - np.polyval(fit, d.n))**2)
                  / np.sum((np.log(d.chi2) - np.log(d.chi2).mean())**2))
        print(f"[FIT] chain chi2 ~ exp({fit[0]:.3f} n), R^2 = {r2:.5f}")
        floor = df.chi2.iloc[-5:].mean()
        print(f"[FLOOR] chain chi2 floor = {floor:.3e} (target <= 1e-10)")
    json.dump(dict(B=B, eta=eta, grid=[LO, HI, NG],
                   runtime_s=time.time() - t0),
              open(OUT / "meta.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
