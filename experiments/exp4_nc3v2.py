"""NC-3 v2 (Claim 3): a target that is hard for BOTH complexity routes.

The v1 attempt (far-separated 2-component mixture) stayed easy at the
sqrt(d)-budget — correctly so: a 2-component mixture has d* = O(1) no matter
the separation, so the paper's OWN intrinsic-dimension route (Thm 4.6)
predicts a small critical G. That non-blowup is reported as a finding.

The correct stress target must have d* ~ d AND L_op >> 1: the i.i.d. product
of d separated bimodal 1D mixtures (2^d modes, so log-covering-number ~ d;
per-coordinate grad-m spikes ~ S^2/4tau at the ridge). Prediction: the
Gaussian-calibrated sqrt(d) budget G ~ 20 fails at the mode-splitting ridge,
while the generic condition-(16) d-schedule restores chi^2 <= delta.

chi^2 estimated exactly as in EXP-3 (exact rho sampling per coordinate,
closed-form conditional clip integral, common random numbers, inner-bias
subtracted), seeds and CIs reported.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from fors.targets import GaussianMixture      # noqa: E402
from fors.schedules import vp_schedule        # noqa: E402
from fors.diffusion import DiffusionSampler   # noqa: E402
from exp3_intrinsic import defect_mc, rho_mixture  # noqa: E402

OUT = ROOT / "results" / "exp4"
QUICK = "--quick" in sys.argv
B = 1.0


class IIDProduct:
    """Product over d coordinates of the SAME 1D mixture (2^d modes)."""

    def __init__(self, mix1d, d):
        self.m, self.d = mix1d, d

    def noised(self, abar, sig2):
        return IIDProduct(self.m.noised(abar, sig2), self.d)

    def score(self, x):
        x = np.atleast_2d(x)
        return self.m.score(x.reshape(-1, 1)).reshape(x.shape)

    def denoiser(self, x, abar, sig2):
        x = np.atleast_2d(x)
        return self.m.denoiser(x.reshape(-1, 1), abar, sig2).reshape(x.shape)

    def sample(self, n, rng):
        return self.m.sample(n * self.d, rng).reshape(n, self.d)


def product_chi2(target, sched, k, n_xp, n_x, n_lat, rng):
    """E_{x+} chi^2 for the product target: rho(.|x+) factorizes, so exact
    sampling runs the 1D tilted-mixture construction per coordinate."""
    ds = DiffusionSampler(target, sched, B=B)
    p_next = target.noised(sched.abar[k + 1], sched.sigma2[k + 1])
    pk1d = target.m.noised(sched.abar[k], sched.sigma2[k])
    chis = []
    for xp in p_next.sample(n_xp, rng):
        x = np.empty((n_x, target.d))
        for i in range(target.d):
            rho_i = rho_mixture(pk1d, xp[i:i + 1], sched.alpha[k], sched.eta[k])
            x[:, i] = rho_i.sample(n_x, rng)[:, 0]
        dfc, ivar = defect_mc(ds, k, x, xp, n_lat, rng)
        chis.append(np.exp(dfc).mean() * np.exp(-dfc).mean() - 1.0 - ivar.mean())
    chis = np.array(chis)
    return float(chis.mean()), float(1.96 * chis.std(ddof=1) / np.sqrt(len(chis)))


def main(d=32, S=4.0, delta=1e-3):
    mix1d = GaussianMixture([0.5, 0.5], [[-S / 2], [S / 2]], [[0.25], [0.25]])
    target = IIDProduct(mix1d, d)
    G_sqrt = 5.5 * np.sqrt(d / 2.0) * 1.2       # Gaussian-calibrated budget
    L = np.log(1 / delta)
    G_gen = 0.55 * (d + L) * L * 4              # generic condition-(16) level
    rows = []
    for tag, G in [("sqrt_d_budget", G_sqrt), ("generic_d_schedule", G_gen)]:
        sched = vp_schedule(1e-4, G, 1e-2)
        rho = sched.sigma2[:-1] / np.maximum(sched.tbar[:-1], 1e-300)
        k_r = int(np.argmin(np.abs(np.log(rho) - 2 * np.log(S / 2.0))))
        step = max(int(G) // 2, 1)
        ks = sorted({min(max(1, k), sched.K - 1) for k in
                     [k_r - step, k_r, k_r + step, sched.K // 2]})
        n_xp, n_x, n_lat = (12, 24, 32) if QUICK else (32, 48, 64)
        rng = np.random.default_rng(23)
        worst, worst_ci, worst_k = -1.0, 0.0, -1
        for k in ks:
            c, ci = product_chi2(target, sched, k, n_xp, n_x, n_lat, rng)
            print(f"  {tag} k={k}/{sched.K}: chi2 = {c:.3e} +- {ci:.1e}",
                  flush=True)
            if c > worst:
                worst, worst_ci, worst_k = c, ci, k
        rows.append(dict(d=d, S=S, tag=tag, G=G, K=sched.K, k_ridge=k_r,
                         worst_chi2=worst, ci=worst_ci, worst_k=worst_k,
                         target_delta=delta))
        print(f"[NC-3v2] {tag} (G={G:.0f}): worst chi2 = {worst:.3e} "
              f"+- {worst_ci:.1e} vs delta = {delta:.0e}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nc3_v2.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
