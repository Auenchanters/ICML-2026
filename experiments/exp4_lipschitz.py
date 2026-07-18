"""EXP-4 (Claim 4, Theorem 4.9): the non-uniform Lipschitz refinement.

Empirical structure (see the claim page for the full provenance):

1. ASSUMPTION CHECKS (exact / closed form):
   (a) log-concave => ||grad m_tau||_op <= 1 pointwise: exact v/(v+tau) for
       Gaussians; 1D quadrature posterior for the non-Gaussian log-concave
       logcosh potential, sup over (y, tau) grids.
   (b) H-mixture => L_op tails ~ O(log H log(d/delta)): quantiles of
       ||grad m_tau||_op over p_tau samples for H in {2, 8, 32} (closed form
       per sample), d = 16.
   (c) Prop 4.7: L_F <= C sqrt(L_op (d* + log(1/delta))): smallest working C
       from the same samples.

2. THE REFINEMENT ITSELF: for the log-concave Gaussian target (L_op = 1),
   the measured critical G*(d) from EXP-2's deterministic engine scales as
   ~ sqrt(d) — Theorem 4.9's sqrt(d L_op) — while condition (16)'s generic
   d-schedule is verified SUFFICIENT with a margin that grows accordingly.
   This script adds the VE-setting version (Thm 4.9 is stated for VE) at a
   few d to confirm the VP-measured exponent transfers.

3. NC-3: at fixed d, a far-separated 2-component mixture (L_op >> 1 at the
   critical tau) run at the Gaussian-calibrated sqrt(d)-budget G blows up;
   restoring the generic d-schedule G brings chi^2 back under delta. The
   Lipschitz condition, not luck, buys sqrt(d).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.targets import GaussianMixture, logcosh_potential   # noqa: E402
from fors.schedules import vp_schedule, ve_schedule           # noqa: E402
from fors.gauss_chf import GaussStep                          # noqa: E402
from fors.diffusion import DiffusionSampler                   # noqa: E402
sys.path.insert(0, str(ROOT / "experiments"))
from exp3_intrinsic import subspace_chi2                      # noqa: E402

OUT = ROOT / "results" / "exp4"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
B = 1.0


# ---------------- 1. assumption checks -------------------------------------

def check_logconcave_opnorm():
    """(a) sup ||grad m_tau||_op <= 1 for log-concave targets."""
    rows = []
    # Gaussian: exact
    for v in [0.3, 1.0, 4.0]:
        for tau in [0.05, 0.5, 5.0]:
            rows.append(dict(target=f"gauss(v={v})", tau=tau,
                             sup_opnorm=v / (v + tau), exact=True))
    # non-Gaussian log-concave (logcosh) by 1D quadrature posterior
    pot = logcosh_potential()
    xg = np.linspace(-10, 10, 4001)
    w0 = np.exp(-pot.f(xg))
    for tau in [0.05, 0.2, 1.0, 5.0]:
        sup = 0.0
        for y in np.linspace(-8, 8, 161):
            wy = w0 * np.exp(-(y - xg) ** 2 / (2 * tau))
            wy /= np.trapezoid(wy, xg)
            m1 = np.trapezoid(wy * xg, xg)
            var = np.trapezoid(wy * (xg - m1) ** 2, xg)
            sup = max(sup, var / tau)
        rows.append(dict(target="logcosh", tau=tau, sup_opnorm=sup, exact=False))
        print(f"[A(a)] logcosh tau={tau}: sup ||grad m||_op = {sup:.6f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "opnorm_logconcave.csv", index=False)
    assert (df.sup_opnorm <= 1.0 + 1e-9).all()


def check_mixture_tails(d=16, n=20000 if True else 100000, seed=0):
    """(b) L_op quantiles vs H; (c) Prop 4.7 constant."""
    rng = np.random.default_rng(seed)
    rows = []
    delta = 1e-3
    for H in [2, 8, 32]:
        mu = rng.standard_normal((H, d)) * 2.0
        mix = GaussianMixture(np.ones(H), mu, np.full((H, d), 0.25))
        for tau in [0.1, 0.5]:
            y = mix.noised(1.0, tau).sample(n, rng)
            _, cov = mix.posterior_moments(y, tau)
            gm = cov / tau
            eig = np.linalg.eigvalsh(gm)
            op = eig[:, -1]
            fro = np.sqrt((gm ** 2).sum(axis=(1, 2)))
            q_op = np.quantile(op, 1 - delta)
            q_f = np.quantile(fro, 1 - delta)
            # Prop 4.7: L_F <= C sqrt(L_op (d* + log(1/delta))); d* ~ intrinsic
            dstar = 2.0 + np.log(H)          # point-cloud cover heuristic
            C = q_f / np.sqrt(q_op * (dstar + np.log(1 / delta)))
            rows.append(dict(H=H, d=d, tau=tau, q_op=q_op, q_fro=q_f,
                             prop47_C=C, n=n))
            print(f"[A(b,c)] H={H} tau={tau}: L_op(1-d)={q_op:.2f} "
                  f"L_F(1-d)={q_f:.2f} C={C:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "mixture_tails.csv", index=False)


# ---------------- 2. VE-setting sqrt(d) confirmation ------------------------

class GaussStepVE(GaussStep):
    """VE flavor: abar == 1, p_k = N(0, (1 + sigma_k^2) I) for unit-variance
    data; denoiser gains g_k = 1/(1 + sigma_k^2)."""

    def __init__(self, sched, k, B, d):
        self.B, self.d = float(B), int(d)
        self.sig2 = float(sched.sigma2[k])
        sig2n = float(sched.sigma2[k + 1])
        self.eta = float(sched.eta[k])
        self.etab = 1.0 / (1.0 / self.eta + 1.0 / self.sig2)
        self.lam = 1.0 / self.sig2            # lam = abar/sigma^2, abar = 1
        self.alpha = 1.0
        g_next = 1.0 / (1.0 + sig2n)
        self.abar_k = 1.0 / (1.0 + self.sig2)   # denoiser gain D_k = g_k x
        self.abar_n = g_next                     # D_{k+1} = g_next x
        # proposal: Xbar = x_plus + eta s_{k+1}(x_plus), s(y) = -y g_next
        self.kappa = 1.0 - self.eta * g_next
        # rho ∝ N(0, 1+sig2) e^{-(x - x_plus)^2/(2 eta)}
        prec = 1.0 / (1.0 + self.sig2) + 1.0 / self.eta
        self.v_rho = 1.0 / prec
        self.m_coef = (1.0 / self.eta) / prec


def worst_chi2_ve(d, G, n_xp=6):
    sched = ve_schedule(1e-4, G, sigma_max_sq=25.0)
    ks = [max(1, int(f * sched.K)) for f in (0.1, 0.5, 0.9)]
    worst = 0.0
    for k in ks:
        gs = GaussStepVE(sched, k, B, d=d)
        c, _ = gs.chi2_expected(n_xp=n_xp, n_r=12, nt=2048,
                                n_x1=12, n_s1=12, n_q=12)
        worst = max(worst, c)
    return worst


def ve_sweep():
    rows = []
    ds = [4, 16] if QUICK else [4, 16, 64]
    for d in ds:
        lo = 2.0
        hi = lo
        if worst_chi2_ve(d, lo) <= 1e-3:
            rows.append(dict(d=d, G_star=lo)); continue
        while worst_chi2_ve(d, hi) > 1e-3:
            lo = hi; hi *= 4
        while hi / lo > 1.1:
            mid = np.sqrt(lo * hi)
            if worst_chi2_ve(d, mid) <= 1e-3:
                hi = mid
            else:
                lo = mid
        rows.append(dict(d=d, G_star=float(np.sqrt(lo * hi)), lo=lo, hi=hi))
        print(f"[VE] d={d}: G* = {rows[-1]['G_star']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ve_critical_G.csv", index=False)
    if len(df) >= 3:
        fit = np.polyfit(np.log(df.d), np.log(df.G_star), 1)
        print(f"[FIT] VE Gaussian G* ~ d^{fit[0]:.3f} (Thm 4.9: 0.5 +- 0.1)")


# ---------------- 3. NC-3 ---------------------------------------------------

def nc3(d=32, delta=1e-3):
    """Separated mixture at the Gaussian sqrt(d)-budget G -> blowup;
    at the generic d-schedule G -> back under delta."""
    rows = []
    G_gauss = None
    p = ROOT / "results" / "exp2" / "critical_G.csv"
    if p.exists():
        dfg = pd.read_csv(p)
        m = dfg[(dfg.d == d) & (dfg.delta == delta)]
        if len(m):
            G_gauss = float(m.G_star.iloc[0])
    if G_gauss is None:
        G_gauss = 5.5 * np.sqrt(d / 2.0)      # sqrt-extrapolation fallback
    G_generic = 0.55 * (d + np.log(1 / delta)) * np.log(1 / delta) * 4
    for S in [4.0, 8.0]:
        mu = np.zeros((2, d)); mu[0, 0], mu[1, 0] = -S / 2, S / 2
        mix = GaussianMixture([0.5, 0.5], mu, np.full((2, d), 0.25))
        for tag, G in [("sqrt_d_budget", G_gauss * 1.2),
                       ("generic_d_schedule", G_generic)]:
            rng = np.random.default_rng(11)
            sched = vp_schedule(1e-4, G, 1e-2)
            # the mode-splitting ridge for separation S sits where
            # sigma^2/(1-sigma^2) ~ (S/2)^2; probe it plus +- half/full
            # e-folds (one e-fold of rho = G steps) and a mid-chain control
            rho = sched.sigma2[:-1] / np.maximum(sched.tbar[:-1], 1e-300)
            k_r = int(np.argmin(np.abs(np.log(rho) - 2 * np.log(S / 2.0))))
            step = max(int(G) // 2, 1)
            ks = sorted({min(max(1, k), sched.K - 1) for k in
                         [k_r - 2 * step, k_r - step, k_r, k_r + step,
                          k_r + 2 * step, sched.K // 2]})
            vals = [subspace_chi2(mix, sched, k,
                                  24 if QUICK else 48, 32 if QUICK else 48,
                                  64, rng)[0]
                    for k in ks]
            rows.append(dict(S=S, G=G, tag=tag, worst_chi2=max(vals),
                             k_ridge=k_r, K=sched.K, target_delta=delta))
            print(f"[NC-3] sep={S} {tag} (G={G:.0f}, ridge k={k_r}/{sched.K}): "
                  f"worst chi2 = {max(vals):.3e} vs delta = {delta:.0e}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nc3.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    if "--nc3-only" not in sys.argv:
        check_logconcave_opnorm()
        check_mixture_tails(n=20000 if QUICK else 100000)
        ve_sweep()
    nc3()
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")
