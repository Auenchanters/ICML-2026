"""EXP-2 Arm A (Claim 2): critical G*(d, delta) scaling for pdata = N(0, I_d).

Condition (16) of Theorem 4.3 prescribes sigma_k^2/eta_k = G >> d log(1/delta)
+ log^2(1/delta) (d* = d for the full-rank Gaussian). We binary-search the
smallest G such that   max over probe steps of E_{x+~p_{k+1}} chi^2(rho_k ||
rho_hat_k) <= delta,   with the chi^2 computed DETERMINISTICALLY at any d by
the chf reduction (src/fors/gauss_chf.py; validated against the generic
engine at d = 1, 2 — see tests/test_gauss_chf.py).

Accept (PLAN.md C.EXP-2): log-log slope of G* vs d = 1.00 +- 0.1, R^2 >= 0.99
at each delta in {1e-3, 1e-5}.

Schedule shape held fixed across (d, G): sigma0^2 = 1e-4, terminal
1 - sigma_K^2 <= 1e-2; probe steps at 10% / 50% / 90% of the chain.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.schedules import vp_schedule           # noqa: E402
from fors.gauss_chf import GaussStep             # noqa: E402

OUT = ROOT / "results" / "exp2"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv

B = 1.0
SIGMA0 = 1e-4
DELTABAR = 1e-2
QUAD = dict(n_r=12, nt=2048, n_x1=12, n_s1=12, n_q=12)
QUAD_DOUBLE = dict(n_r=24, nt=4096, n_x1=24, n_s1=24, n_q=24)


def worst_chi2(d, G, n_xp=6, quad=QUAD):
    sched = vp_schedule(SIGMA0, G, DELTABAR)
    ks = [max(1, int(f * sched.K)) for f in (0.1, 0.5, 0.9)]
    worst = 0.0
    for k in ks:
        gs = GaussStep(sched, k, B, d=d)
        c, mx = gs.chi2_expected(n_xp=n_xp, **quad)
        worst = max(worst, c)
    return worst


def critical_G(d, delta, tol_ratio=1.08, verbose=True):
    """Smallest G with worst chi^2 <= delta, by bisection in log G
    (chi^2 is decreasing in G)."""
    lo = 2.0
    if worst_chi2(d, lo) <= delta:
        return lo, 0.0, lo          # already sub-threshold at the floor value
    hi = lo
    while True:
        hi *= 4.0
        if hi > 1e7:
            raise RuntimeError("no upper bracket below 1e7")
        if worst_chi2(d, hi) <= delta:
            break
        lo = hi
    while hi / lo > tol_ratio:
        mid = np.sqrt(lo * hi)
        c = worst_chi2(d, mid)
        if c <= delta:
            hi = mid
        else:
            lo = mid
        if verbose:
            print(f"    d={d} delta={delta:.0e}: G in [{lo:.1f}, {hi:.1f}] "
                  f"(chi2({mid:.1f}) = {c:.2e})", flush=True)
    return np.sqrt(lo * hi), lo, hi


def main():
    t0 = time.time()
    ds = [2, 4, 8] if QUICK else [2, 4, 8, 16, 32, 64]
    deltas = [1e-3] if QUICK else [1e-3, 1e-5]
    rows = []
    for delta in deltas:
        for d in ds:
            t1 = time.time()
            gstar, lo, hi = critical_G(d, delta)
            rows.append(dict(d=d, delta=delta, G_star=gstar, lo=lo, hi=hi,
                             secs=time.time() - t1))
            print(f"[G*] d={d} delta={delta:.0e}: G* = {gstar:.1f} "
                  f"[{lo:.1f}, {hi:.1f}] ({rows[-1]['secs']:.0f}s)", flush=True)
        sub = pd.DataFrame([r for r in rows if r["delta"] == delta])
        fit = np.polyfit(np.log(sub.d), np.log(sub.G_star), 1)
        r2 = 1 - (np.sum((np.log(sub.G_star) - np.polyval(fit, np.log(sub.d)))**2)
                  / np.sum((np.log(sub.G_star) - np.log(sub.G_star).mean())**2))
        print(f"[FIT] delta={delta:.0e}: G* ~ d^{fit[0]:.3f}, R^2 = {r2:.5f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "critical_G.csv", index=False)

    # quadrature convergence check on one cell (doubled nodes)
    d0, G0 = ds[-1], rows[len(ds) - 1]["G_star"]
    c1 = worst_chi2(d0, G0)
    c2 = worst_chi2(d0, G0, n_xp=12, quad=QUAD_DOUBLE)
    json.dump(dict(B=B, sigma0=SIGMA0, deltabar=DELTABAR, quad=QUAD,
                   conv_check=dict(d=d0, G=G0, chi2=c1, chi2_doubled=c2,
                                   rel=abs(c2 - c1) / max(c2, 1e-300)),
                   runtime_s=time.time() - t0),
              open(OUT / "meta.json", "w"), indent=1)
    print(f"[CONV] d={d0}, G={G0:.0f}: chi2 = {c1:.4e} vs doubled {c2:.4e}")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
