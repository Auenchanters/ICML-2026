"""EXP-3 (Claim 3, Corollary 4.4): complexity tracks intrinsic dimension d*.

Targets: 2-component Gaussian mixture supported on the first 2 coordinates of
R^d with sigma* = 1e-3 thickening (var 1e-6) off-subspace — d* = O(1) by
Example 4.2 — vs the full-rank analogue (same means, var 0.25 everywhere).

1. STRUCTURAL (deterministic): E[tr grad m_tau] = E[tr cov_tau]/tau vs d:
   flat (~ intrinsic) for subspace targets, linear in d for full-rank —
   Corollary E.4's mechanism. Computed by the law of total variance:
   E[tr cov] = tr Cov(Y0) - tr Cov(m(Y)); the posterior factorizes across the
   on/off blocks (components share the off-block law), so E||m||^2 needs only
   a 2-D tensor quadrature on the block plus closed forms off-block.

2. MONEY PLOT #2: critical G*(d, delta=1e-3) for subspace targets (flat)
   overlaid on the full-rank isotropic-Gaussian curve from EXP-2 (linear).
   Subspace chi^2 estimator (the plan's importance-weighted MC, upgraded):
   x+ ~ p_{k+1} and x ~ rho(.|x+) sampled EXACTLY (the tilted mixture is a
   closed-form mixture); per sample the (r, gamma) latent uses the
   closed-form conditional clip integral; clipped and unclipped share common
   random numbers so the defect estimator variance collapses; inner-MC bias
   is estimated and subtracted; bootstrap CIs over x-samples reported.

3. End-to-end: Algorithm 2 at d = 128 on the subspace target with the
   d*-schedule (G from d* = 2, NOT d = 128): healthy acceptance and correct
   2-D projection — operationally, the d*-schedule suffices.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.targets import GaussianMixture, subspace_mixture  # noqa: E402
from fors.schedules import vp_schedule                      # noqa: E402
from fors.diffusion import DiffusionSampler, path_diffusion  # noqa: E402
from fors.quadrature import clip_gauss_mean, gl_nodes        # noqa: E402
from fors.metrics import hist_kl_vs_grid                     # noqa: E402

OUT = ROOT / "results" / "exp3"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv

B = 1.0
SIGMA0 = 1e-4
DELTABAR = 1e-2
THICK = 1e-6          # sigma* = 1e-3 thickening


# ---------------- 1. structural: E[tr grad m_tau] ---------------------------

def e_tr_gradm(mix, tau, n_grid=241, span=8.0):
    """E_{q_tau}[tr grad m_tau] = (tr Cov(Y0) - tr Cov(m(Y_tau)))/tau, exact.

    Blockwise: coordinates where all components share mean/var are linear
    (closed form); the first 2 coordinates are handled by tensor quadrature."""
    w, mu, var = mix.w, mix.mu, mix.var
    m0 = np.sum(w[:, None] * mu, axis=0)
    tr_cov_y0 = float(np.sum(w[:, None] * (var + (mu - m0) ** 2)))
    # off-block: per-coordinate linear posterior mean m_i = g_i y_i + c_i,
    # g = v/(v+tau); E[m_i^2] - (E m_i)^2 = g_i^2 Var(Y_tau,i) = g_i^2 (v_i+tau)
    off = slice(2, mix.d)
    v_off = var[0, off]                     # identical across components
    tr_cov_m_off = float(np.sum((v_off / (v_off + tau)) ** 2 * (v_off + tau)))
    # on-block (2D mixture): quadrature for E||m_on||^2
    sm = GaussianMixture(w, mu[:, :2], var[:, :2] + tau)
    lim = float(np.abs(mu[:, :2]).max() + span * np.sqrt(var[:, :2].max() + tau))
    g = np.linspace(-lim, lim, n_grid)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    dens = sm.pdf(pts)
    on_mix = GaussianMixture(w, mu[:, :2], var[:, :2])
    m_on, _ = on_mix.posterior_moments(pts, tau)
    e_m2 = np.trapezoid(np.trapezoid(
        (dens * np.sum(m_on**2, axis=1)).reshape(n_grid, n_grid), g, axis=1), g)
    e_m = np.trapezoid(np.trapezoid(
        (dens[:, None] * m_on).reshape(n_grid, n_grid, 2), g, axis=1), g, axis=0)
    tr_cov_m_on = float(e_m2 - np.sum(e_m**2))
    return (tr_cov_y0 - tr_cov_m_on - tr_cov_m_off) / tau


def structural():
    rows = []
    for d in [8, 32, 128, 512]:
        for tau in [0.1, 0.5, 2.0]:
            sub = subspace_mixture(d, thick=THICK)
            full = subspace_mixture(d, thick=0.25)
            rows.append(dict(d=d, tau=tau,
                             tr_sub=e_tr_gradm(sub, tau),
                             tr_full=e_tr_gradm(full, tau)))
            print(f"[S] d={d} tau={tau}: subspace {rows[-1]['tr_sub']:.4f}  "
                  f"full-rank {rows[-1]['tr_full']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "structural_trace.csv", index=False)


# ---------------- 2. subspace chi^2 by exact-sampling MC --------------------

def rho_mixture(mix_k, x_plus, alpha, eta):
    """rho(.|x+) ∝ p_k(x) N(x; x+/alpha, eta I) as an exact mixture."""
    c = x_plus / alpha
    v = mix_k.var                               # (H, d)
    vp = v * eta / (v + eta)
    mp = vp * (mix_k.mu / v + c[None, :] / eta)
    # log component weights pick up N(c; mu_h, v_h + eta)
    d2 = np.sum((c[None, :] - mix_k.mu) ** 2 / (v + eta), axis=1)
    logw = np.log(mix_k.w) - 0.5 * d2 - 0.5 * np.sum(np.log(v + eta), axis=1)
    logw -= logw.max()
    return GaussianMixture(np.exp(logw), mp, vp)


def defect_mc(ds, k, x, x_plus, n_lat, rng):
    """defect(x) = E[Clip_B W|x] - E[W|x] by MC over (r, gamma) with the
    closed-form conditional clip integral and common random numbers.
    Returns (defect (n,), inner_var (n,))."""
    sched = ds.s
    n, d = x.shape
    eta_b = sched.etabar[k]
    lam = sched.abar[k] / sched.sigma2[k]
    xbar = (x_plus / sched.alpha[k]
            + sched.alpha[k] * sched.eta[k] * ds.exact_score(k + 1, x_plus[None])[0])
    half = eta_b / 2.0
    r = rng.uniform(size=(n, n_lat))
    a, b, da, db = path_diffusion(r)
    s_u2 = ((1 - a)**2 + b**2) * half
    s_v2 = (da**2 + db**2) * half
    c_uv = (-da * (1 - a) + b * db) * half
    s_u = np.sqrt(s_u2)
    slope = c_uv / np.maximum(s_u, 1e-300)
    s_c2 = np.maximum(s_v2 - slope**2, 0.0)
    xi = rng.standard_normal((n, n_lat, d))
    gamma = (a[..., None] * x[:, None, :] + (1 - a)[..., None] * xbar[None, None, :]
             + s_u[..., None] * xi)
    Dg = ds.denoiser(k, gamma.reshape(-1, d)).reshape(n, n_lat, d)
    Dp = ds.denoiser(k + 1, x_plus[None])[0]
    wv = Dg - Dp[None, None, :]
    mu_v = da[..., None] * (x[:, None, :] - xbar[None, None, :])
    m_t = np.einsum("nld,nld->nl", mu_v + slope[..., None] * xi, wv)
    s_t = np.sqrt(s_c2 * np.einsum("nld,nld->nl", wv, wv))
    clipped = clip_gauss_mean(B, lam, m_t, s_t)
    samples = clipped - lam * m_t               # common random numbers
    defect = samples.mean(axis=1)
    ivar = samples.var(axis=1, ddof=1) / n_lat
    return defect, ivar


def subspace_chi2(mix, sched, k, n_xp, n_x, n_lat, rng):
    """E_{x+} chi^2(rho||rho_hat) with inner-bias correction and 95% CI."""
    ds = DiffusionSampler(mix, sched, B=B)
    p_next = mix.noised(sched.abar[k + 1], sched.sigma2[k + 1])
    pk = mix.noised(sched.abar[k], sched.sigma2[k])
    chis = []
    for xp in p_next.sample(n_xp, rng):
        rho = rho_mixture(pk, xp, sched.alpha[k], sched.eta[k])
        x = rho.sample(n_x, rng)
        dfc, ivar = defect_mc(ds, k, x, xp, n_lat, rng)
        ep, em = np.exp(dfc).mean(), np.exp(-dfc).mean()
        chi = ep * em - 1.0 - ivar.mean()       # subtract inner-MC inflation
        chis.append(chi)
    chis = np.array(chis)
    return float(chis.mean()), float(1.96 * chis.std(ddof=1) / np.sqrt(n_xp))


def critical_G_subspace(d, delta, seed, n_xp=48, n_x=48, n_lat=64,
                        tol_ratio=1.15):
    mix = subspace_mixture(d, thick=THICK)

    def worst(G):
        rng = np.random.default_rng(seed)       # common seeds across G
        sched = vp_schedule(SIGMA0, G, DELTABAR)
        vals = [subspace_chi2(mix, sched, max(1, int(f * sched.K)),
                              n_xp, n_x, n_lat, rng)[0]
                for f in (0.1, 0.5, 0.9)]
        return max(vals)

    lo = 2.0
    if worst(lo) <= delta:
        return lo, 0.0, lo
    hi = lo
    while True:
        hi *= 4.0
        if worst(hi) <= delta:
            break
        lo = hi
        if hi > 1e6:
            raise RuntimeError("no bracket")
    while hi / lo > tol_ratio:
        mid = np.sqrt(lo * hi)
        if worst(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi)), lo, hi


def money2():
    rows = []
    ds_list = [8, 32] if QUICK else [8, 32, 128]
    for d in ds_list:
        t1 = time.time()
        for seed in ([0] if QUICK else [0, 1, 2]):
            g, lo, hi = critical_G_subspace(d, 1e-3, seed)
            rows.append(dict(d=d, seed=seed, G_star=g, lo=lo, hi=hi,
                             secs=time.time() - t1))
            print(f"[G*sub] d={d} seed={seed}: G* = {g:.1f} [{lo:.1f}, {hi:.1f}]",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "critical_G_subspace.csv", index=False)
    gm = df.groupby("d").G_star.mean()
    fit = np.polyfit(np.log(gm.index), np.log(gm.values), 1)
    print(f"[FIT] subspace G* ~ d^{fit[0]:.3f} (accept |slope| <= 0.15)")


# ---------------- 3. end-to-end at d = 128 ---------------------------------

def end_to_end():
    d = 32 if QUICK else 128
    mix = subspace_mixture(d, thick=THICK)
    # d*-schedule: G from d* = 2, NOT d
    L = np.log(4000 / 1e-2**2)
    G = 0.55 * (2.0 + L) * L
    sched = vp_schedule(1e-4, G, 1e-2)
    ds = DiffusionSampler(mix, sched, B=B)
    rng = np.random.default_rng(7)
    n = 5000 if QUICK else 20000
    t0 = time.time()
    xs, st = ds.sample(n, rng, method="fors")
    p1 = mix.noised(sched.abar[1], sched.sigma2[1])
    # 2-D projection metrics (marginal of coord 0 vs exact 1D mixture)
    g = np.linspace(-6, 6, 3001)
    m0 = GaussianMixture(p1.w, p1.mu[:, :1], p1.var[:, :1])
    kl0 = hist_kl_vs_grid(xs[:, 0], g, m0.pdf(g[:, None]), bins=50)
    amb = float(np.abs(xs[:, 2:].std(axis=0)**2 - p1.var[0, 2:]).max())
    res = dict(d=d, K=int(sched.K), G=float(G), n=n,
               proj_hist_kl=float(kl0), floor=50 / (2 * n),
               ambient_var_err_max=amb,
               accept_rate=float(st.accept_rate),
               q_per_step=float(st.w_draws / (n * (sched.K - 1))),
               secs=time.time() - t0)
    json.dump(res, open(OUT / "end_to_end.json", "w"), indent=1)
    print(f"[E2E] d={d}: K={sched.K} (d*-schedule G={G:.0f}), proj-KL={kl0:.2e} "
          f"(floor {res['floor']:.1e}), acc={res['accept_rate']:.3f}, "
          f"q/step={res['q_per_step']:.2f}, {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    structural()
    money2()
    end_to_end()
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")
