"""EXP-5 multi-d (Claim 5, Theorem G.1's kappa*sqrt(d) dimension dependence).

Addresses the d=1-only scope. Target: isotropic product
mu_d(x) ~ exp(-sum_i f1(x_i)), f1 = x^2/2 + logcosh 2x (strongly log-concave,
beta1 = 5). The proximal sampler with the joint d-dim FORS-RGO has, for a
product target, INDEPENDENT output coordinates each distributed as the exact
1D RGO law (the joint Bernoulli factory for e^{sum E[W_i]} = prod e^{E[W_i]}
factorizes), so:
    chi2_d(mu_N || mu) = (1 + chi2_1(N))^d - 1     (exact, no MC)
and per Theorem G.1 we set eta^{-1} = C beta1 sqrt(d) log(1/eps), i.e.
eta ~ 1/sqrt(d). The joint estimator W = sum_i W_i has Var ~ d * eta^2 = O(1),
so the joint acceptance A_d -- hence draws-per-call = 2B/A_d -- stays O(1) in d
(the whole point of the eta ~ 1/sqrt(d) scaling). Total first-order queries to
reach accuracy eps is therefore
    Q(d) = N(d) * (2B / A_d) ~ N(d) * O(1),
so the query scaling is the proximal-STEP scaling N(d). We measure N(d)
deterministically (1D grid chi2 evolution + product formula) and fit N(d) vs d.
Prediction: N(d) ~ sqrt(d) * polylog (the eta ~ 1/sqrt(d) mixing).

A_d is measured once per d by a small joint sampling run to confirm draws/call
is d-flat, closing the query-count argument.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.targets import logcosh_potential                 # noqa: E402
from fors.tilts import prox_newton, tilt_mean_w_quad, path_sin  # noqa: E402
from fors.fors import fors_batch                           # noqa: E402
from fors.metrics import grid_chi2, grid_normalize         # noqa: E402

OUT = ROOT / "results" / "exp5"
QUICK = "--quick" in sys.argv
B = 1.0
LO, HI, NG = -9.0, 9.0, 1201 if QUICK else 1601
EPS2_D = 1e-4          # target product chi^2 (above the d=1 RGO bias floor)


def kernels_1d(pot, eta, x):
    H = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * eta)); H /= H.sum(0, keepdims=True)
    R = np.empty((len(x), len(x)))
    for j, y in enumerate(x):
        xp = prox_newton(pot.df, pot.d2f, np.array([y]), eta)[0]
        xhat = y - eta * pot.df(xp)
        m = tilt_mean_w_quad(x, xhat, xp, pot.df, eta, B, n_r=24, n_z=24)
        R[:, j] = grid_normalize(-(x - xhat) ** 2 / (2 * eta) + m, x)
    R /= R.sum(0, keepdims=True)
    return H, R


def steps_to_accuracy(pot, d, eta, x, eps2_d=EPS2_D, max_n=6000):
    """Deterministic N(d): proximal steps until (1+chi2_1)^d - 1 <= eps2_d."""
    H, R = kernels_1d(pot, eta, x); dx = x[1] - x[0]
    mu_star = grid_normalize(-pot.f(x), x)
    mu = grid_normalize(-(x ** 2) / 2.0, x)
    for n in range(1, max_n + 1):
        y = H @ mu * dx; y /= np.trapezoid(y, x)
        mu = R @ y * dx; mu /= np.trapezoid(mu, x)
        c1 = max(grid_chi2(mu, mu_star, x), 0.0)
        if (1.0 + c1) ** d - 1.0 <= eps2_d:
            return n, c1
    return max_n, c1


def measure_draws_per_call(pot, d, eta, rng, n=4000):
    """Joint d-dim FORS-RGO: one proximal step, measure draws/output (= 2B/A_d)."""
    Y = np.zeros((n, d))                       # tilt centered at 0 (typical)
    xp = prox_newton(pot.df, pot.d2f, Y.ravel(), eta).reshape(n, d)
    gxp = pot.df(xp); xhat = Y - eta * gxp; sq = np.sqrt(eta)

    def propose_n(m, rng):
        idx = rng.integers(0, n, size=m); propose_n.idx = idx
        return xhat[idx] + sq * rng.standard_normal((m, d))

    def draw_w(xx, J, rng):
        m, jm = len(xx), max(int(J.max()), 1); idx = propose_n.idx
        r = rng.uniform(size=(m, jm)); z = sq * rng.standard_normal((m, jm, d))
        a, b, da, db = path_sin(r)
        gamma = a[..., None] * xx[:, None, :] + (1 - a)[..., None] * xhat[idx][:, None, :] + b[..., None] * z
        gdot = da[..., None] * (xx[:, None, :] - xhat[idx][:, None, :]) + db[..., None] * z
        gd = gxp[idx][:, None, :] - pot.df(gamma.reshape(-1, d)).reshape(m, jm, d)
        return np.clip(np.sum(gdot * gd, axis=-1), -B, B)

    _, st = fors_batch(propose_n, draw_w, B, n, rng, batch=2 * n)
    return st.w_draws / n, st.accept_rate


def main():
    t0 = time.time()
    pot = logcosh_potential()
    x = np.linspace(LO, HI, NG)
    ds = [1, 2, 4, 8] if QUICK else [1, 2, 4, 8, 16, 32]
    rng = np.random.default_rng(0)
    rows = []
    for d in ds:
        eta = 1.0 / (16.0 * np.sqrt(d))         # eta ~ 1/sqrt(d), Thm G.1
        N, c1 = steps_to_accuracy(pot, d, eta, x)
        dpc, acc = measure_draws_per_call(pot, d, eta, rng)
        rows.append(dict(d=d, eta=eta, N_steps=N, chi2_1_final=c1,
                         draws_per_call=dpc, accept_rate=acc,
                         total_queries=N * dpc))
        print(f"[d={d:3d}] eta={eta:.4f} N={N} steps, draws/call={dpc:.2f} "
              f"(A={acc:.3f}), Q=N*dpc={N*dpc:.0f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multid.csv", index=False)
    fN = np.polyfit(np.log(df.d), np.log(df.N_steps), 1)
    r2N = 1 - np.sum((np.log(df.N_steps) - np.polyval(fN, np.log(df.d)))**2) / \
        np.sum((np.log(df.N_steps) - np.log(df.N_steps).mean())**2)
    fQ = np.polyfit(np.log(df.d), np.log(df.total_queries), 1)
    print(f"[FIT] proximal steps N(d) ~ d^{fN[0]:.3f} (R^2={r2N:.4f}) "
          f"-- Thm G.1 eta~1/sqrt(d) predicts ~0.5")
    print(f"[FIT] total queries Q(d) ~ d^{fQ[0]:.3f}; draws/call d-flat: "
          f"{df.draws_per_call.min():.2f}-{df.draws_per_call.max():.2f}")
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
