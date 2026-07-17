"""EXP-1 Arms B + C and NC-2 (PLAN.md C.EXP-1, prompt P6).

Arm B  — end-to-end sampling corroboration: run Algorithm 2 at delta in
         {1e-1, 1e-2}, multiple seeds; histogram-KL and moment errors vs the
         closed-form p_1; verify query counts are O(K) with the Thm-3.1(c)
         per-call structure. (Sample metrics floor at O(bins/n) — the
         verified-grade evidence for Claim 1 is the certified Arm A.)

Arm C  — the claim's conditional clause ("given sufficiently accurate score
         estimates"): inject s_k = s*_k + eps * g, g(x) = sin(3x). Then
         eps_{k,score}^2 = eps^2 E_{p_k}[sin^2(3X)] has a CLOSED FORM for
         mixtures: E[sin^2 3X] = (1 - sum_h w_h cos(6 abar mu_h) e^{-18 v_h})/2.
         Certified chain KL vs eps must scale as eps^2 (log-log slope 2) and
         track C * sum_k eta_k eps_{k,score}^2 with a moderate constant C.

NC-2   — violate condition (16): G at 0.1x the calibrated threshold. The
         certified per-step KL must blow up by orders of magnitude vs the
         per-step target — the schedule condition is load-bearing.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.targets import bimodal_1d                     # noqa: E402
from fors.schedules import vp_schedule                  # noqa: E402
from fors.diffusion import DiffusionSampler             # noqa: E402
from fors.quadrature import StepQuad, expected_step_divergence  # noqa: E402
from fors.metrics import hist_kl_vs_grid                # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from exp1_certified import build_schedule, strata_indices, FLOOR, B  # noqa: E402

OUT = ROOT / "results" / "exp1"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv


# ---------------- Arm B ----------------------------------------------------

def arm_b():
    mix = bimodal_1d()
    rows = []
    # CPU-sized: bigger n and delta=1e-3 go to HF GPU Job #1 (PLAN.md P10)
    plans = [(1e-1, 100_000, 3), (1e-2, 30_000, 2)]
    if QUICK:
        plans = [(1e-1, 20_000, 2)]
    for delta, n, n_seeds in plans:
        sched, meta = build_schedule(delta, mix)
        ds = DiffusionSampler(mix, sched, B=B)
        p1 = mix.noised(sched.abar[1], sched.sigma2[1])
        g = np.linspace(-6, 6, 4001)
        p1d = p1.pdf(g[:, None])
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            t0 = time.time()
            xs, st = ds.sample(n, rng, method="fors")
            kl = hist_kl_vs_grid(xs, g, p1d, bins=60)
            m_err = abs(xs.mean() - 0.0)     # symmetric target: mean 0... use exact
            mu1 = float(np.sum(p1.w[:, None] * p1.mu))
            v1 = float(np.sum(p1.w[:, None] * (p1.mu**2 + p1.var)) - mu1**2)
            rows.append(dict(
                delta=delta, K=sched.K, n=n, seed=seed,
                hist_kl=kl, noise_floor=60 / (2 * n),
                mean_err=float(abs(xs.mean() - mu1)),
                var_err=float(abs(xs.var() - v1)),
                queries=st.w_draws, queries_per_step_chain=st.w_draws / (n * (sched.K - 1)),
                accept_rate=st.accept_rate, secs=time.time() - t0))
            print(f"[B] delta={delta:.0e} seed={seed}: hist-KL={kl:.2e} "
                  f"(floor {60/(2*n):.1e}), q/step/chain="
                  f"{rows[-1]['queries_per_step_chain']:.2f}, "
                  f"acc={st.accept_rate:.3f}, {rows[-1]['secs']:.0f}s")
    pd.DataFrame(rows).to_csv(OUT / "arm_b.csv", index=False)


# ---------------- Arm C ----------------------------------------------------

def eps_score_sq(mix, abar, sigma2):
    """E_{p_k}[sin^2(3X)] closed form; p_k component means abar*mu, var v."""
    mu = abar * mix.mu[:, 0]
    v = abar**2 * mix.var[:, 0] + sigma2
    return float(0.5 * (1.0 - np.sum(mix.w * np.cos(6 * mu) * np.exp(-18 * v))))


def make_perturbed_step(mix, sched, k, eps):
    ds = DiffusionSampler(mix, sched, B=B)

    def score(kk, x):
        return ds.exact_score(kk, x) + eps * np.sin(3 * np.atleast_2d(x))

    def denoiser(kk, x):
        return (np.atleast_2d(x) + sched.sigma2[kk] * score(kk, x)) / sched.abar[kk]

    return StepQuad(
        alpha_k=sched.alpha[k], eta_k=sched.eta[k], sigma2_k=sched.sigma2[k],
        abar_k=sched.abar[k], B=B,
        denoiser_k=lambda x: denoiser(k, x),
        score_next=lambda x: score(k + 1, x),
        denoiser_next=lambda x: denoiser(k + 1, x),
    )


def certify_perturbed(mix, sched, eps, n_strata=20):
    K = sched.K
    edges = strata_indices(K, n_strata)
    cache = {}
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        for k in {int(lo), int(hi)}:
            kk = min(k, K - 2)
            if kk in cache:
                continue
            step = make_perturbed_step(mix, sched, kk, eps)
            p_next = mix.noised(sched.abar[kk + 1], sched.sigma2[kk + 1])
            pk = mix.noised(sched.abar[kk], sched.sigma2[kk])
            res = expected_step_divergence(step, p_next, pk.logpdf,
                                           n_xp=16, n_r=16, n_u=32, grid_n=401)
            cache[kk] = res["kl"]
        seg = max(cache[min(int(lo), K - 2)], cache[min(int(hi), K - 2)], FLOOR)
        total += seg * (hi - lo)
    return float(total)


def arm_c():
    mix = bimodal_1d()
    sched, meta = build_schedule(1e-2, mix)
    # exact robustness term: sum_k eta_k * eps^2 * E_{p_k}[g^2]
    base = sum(sched.eta[k] * eps_score_sq(mix, sched.abar[k], sched.sigma2[k])
               for k in range(sched.K - 1))
    rows = []
    eps_list = [1e-3, 1e-2] if QUICK else [1e-4, 1e-3, 1e-2, 1e-1]
    for eps in eps_list:
        tot = certify_perturbed(mix, sched, eps,
                                n_strata=10 if QUICK else 20)
        pred = base * eps**2
        rows.append(dict(eps=eps, chain_kl=tot, prediction=pred,
                         ratio=tot / pred, sum_eta_g2=base, K=sched.K))
        print(f"[C] eps={eps:.0e}: certified chain KL={tot:.3e}, "
              f"paper term sum(eta_k eps_k^2)={pred:.3e}, ratio={tot/pred:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "arm_c.csv", index=False)
    if len(df) >= 3:
        fit = np.polyfit(np.log(df.eps), np.log(df.chain_kl), 1)
        r2 = 1 - np.sum((np.log(df.chain_kl) - np.polyval(fit, np.log(df.eps)))**2) \
            / np.sum((np.log(df.chain_kl) - np.log(df.chain_kl).mean())**2)
        print(f"[C-FIT] chain KL ~ eps^{fit[0]:.3f} (theory 2), R²={r2:.5f}")


# ---------------- NC-2 -----------------------------------------------------

def nc2():
    mix = bimodal_1d()
    sched_ok, meta = build_schedule(1e-2, mix)
    G_ok = sched_ok.G
    delta_step = 1e-2**2 / sched_ok.K
    sig0, dbar = meta["sigma0_sq"], meta["deltabar"]
    rows = []
    for fac, tag in [(1.0, "calibrated"), (0.1, "violated (0.1x)"),
                     (0.03, "violated (0.03x)")]:
        sched = vp_schedule(sig0, G_ok * fac, dbar)
        ks = [1, sched.K // 4, sched.K // 2, 3 * sched.K // 4, sched.K - 2]
        for k in ks:
            step_kl = None
            ds = DiffusionSampler(mix, sched, B=B)
            step = StepQuad(
                alpha_k=sched.alpha[k], eta_k=sched.eta[k],
                sigma2_k=sched.sigma2[k], abar_k=sched.abar[k], B=B,
                denoiser_k=lambda x: ds.denoiser(k, x),
                score_next=lambda x: ds.exact_score(k + 1, x),
                denoiser_next=lambda x: ds.denoiser(k + 1, x))
            p_next = mix.noised(sched.abar[k + 1], sched.sigma2[k + 1])
            pk = mix.noised(sched.abar[k], sched.sigma2[k])
            res = expected_step_divergence(step, p_next, pk.logpdf,
                                           n_xp=16, n_r=16, n_u=32, grid_n=401)
            step_kl = res["kl"]
            rows.append(dict(G_factor=fac, tag=tag, G=sched.G, K=sched.K,
                             k=k, per_step_kl=step_kl,
                             per_step_target=delta_step,
                             excess=step_kl / delta_step))
        worst = max(r["excess"] for r in rows if r["G_factor"] == fac)
        print(f"[NC-2] {tag}: G={sched.G:.1f}, worst per-step KL / target = "
              f"{worst:.2e}")
    pd.DataFrame(rows).to_csv(OUT / "nc2.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    arm_c()
    nc2()
    arm_b()
    print(f"all done in {time.time() - t0:.0f}s")
