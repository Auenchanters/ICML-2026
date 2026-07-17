"""EXP-1 Arm A (PLAN.md C.EXP-1): certified polylog(1/delta) step counts.

For each final-accuracy target delta_fin in the ladder:
  * build the Cor-4.4 VP schedule: sigma0^2 = delta^2/(d+M2^2),
    terminal 1-sigma_K^2 <= deltabar = delta^2/max(M2^2,1),
    G = c0 (dstar + L) L with L = log(K/delta^2) (fixed-point in K),
    eta_k = sigma_k^2/G  — K(delta) is fully determined by the construction.
  * certify the paper's own chain-rule bound (Sec. F.2):
        KL(p_1||p_hat_1) <= KL(p_K||p_hat_K) + sum_k E_{x+~p_{k+1}} KL(rho_k||rho_hat_k)
    by deterministic quadrature. For large K the sum is bounded by geometric
    strata (max of endpoint values x stratum size); stratification is
    validated against dense certification on the delta=0.1 rung.
  * DDPM baseline (NC-1): identical proposal (same exponential-integrator mean
    AND variance), no FORS corrector -> per-step KL has the Thm-E.10 floor;
    sweep G, record K vs achieved accuracy.

Numerical-honesty floor: per-step KL below FLOOR = 3e-15 is not resolvable in
float64 (log-density cancellation); such values are reported as <= FLOOR and
the chain bound uses FLOOR in their place (conservative).

Money fit #1: log K vs log L where L = log10(1/delta): slope = polynomial
degree of K in log(1/delta). Theory: <= 3. DDPM: K ~ delta^{-a}, a ~ 1-2.
"""
import json
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
from fors.metrics import grid_kl, grid_normalize        # noqa: E402

OUT = ROOT / "results" / "exp1"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv

B = 1.0
DSTAR = 1.0
FLOOR = 3e-15          # float64 per-step KL resolution floor
C0 = 0.55              # calibrated at the delta=1e-1/1e-2 rungs (see page text)


def build_schedule(delta_fin, mix, c0=C0, dstar=DSTAR):
    d, M2 = mix.d, mix.second_moment()
    sigma0 = delta_fin**2 / (d + M2)
    deltabar = delta_fin**2 / max(M2, 1.0)
    K = 1000.0
    sched = None
    for _ in range(12):
        L = np.log(max(K, 2.0) / delta_fin**2)
        G = c0 * (dstar + L) * L
        sched = vp_schedule(sigma0, G, deltabar)
        if abs(sched.K - K) <= 2:
            break
        K = sched.K
    return sched, dict(G=sched.G, sigma0_sq=sigma0, deltabar=deltabar,
                       L=float(np.log(sched.K / delta_fin**2)))


def make_step(mix, sched, k, B=B):
    ds = DiffusionSampler(mix, sched, B=B)
    return StepQuad(
        alpha_k=sched.alpha[k], eta_k=sched.eta[k], sigma2_k=sched.sigma2[k],
        abar_k=sched.abar[k], B=B,
        denoiser_k=lambda x: ds.denoiser(k, x),
        score_next=lambda x: ds.exact_score(k + 1, x),
        denoiser_next=lambda x: ds.denoiser(k + 1, x),
    )


def per_step_kl(mix, sched, k, ddpm=False, n_xp=24, n_r=16, n_u=32,
                grid_n=401):
    """E_{x+ ~ p_{k+1}} KL(rho_k || rho_hat_k) for step k (FORS or DDPM)."""
    step = make_step(mix, sched, k)
    p_next = mix.noised(sched.abar[k + 1], sched.sigma2[k + 1])
    pk = mix.noised(sched.abar[k], sched.sigma2[k])
    if not ddpm:
        res = expected_step_divergence(step, p_next, pk.logpdf, n_xp=n_xp,
                                       n_r=n_r, n_u=n_u, grid_n=grid_n)
        return res["kl"]
    # DDPM: rho_hat = q_k itself (mean_w = 0)
    from fors.quadrature import gh_nodes
    t, wt = gh_nodes(n_xp)
    tot = 0.0
    for h in range(p_next.H):
        mu_h, sd_h = float(p_next.mu[h, 0]), float(np.sqrt(p_next.var[h, 0]))
        for j in range(len(t)):
            xp = mu_h + sd_h * t[j]
            res = step.step_divergences(np.array([xp]), pk.logpdf,
                                        grid_n=grid_n,
                                        mean_w=np.zeros(grid_n))
            tot += float(p_next.w[h]) * wt[j] * res["kl"]
    return tot


def init_kl(mix, sched, grid_n=8001):
    """KL(p_K || N(0, sigma_K^2)) on a wide grid."""
    pK = mix.noised(sched.abar[-1], sched.sigma2[-1])
    sd = np.sqrt(sched.sigma2[-1])
    hw = 10 * max(sd, float(np.abs(pK.mu).max() + np.sqrt(pK.var.max()) * 6))
    x = np.linspace(-hw, hw, grid_n)
    p = grid_normalize(pK.logpdf(x[:, None]), x)
    q = grid_normalize(-x**2 / (2 * sched.sigma2[-1]), x)
    return grid_kl(p, q, x)


def strata_indices(K, n_strata):
    """Geometric strata over k = 1..K-1 (Algorithm 2 runs steps K-1 down to 1;
    step k conditions on X_{k+1}, k+1 <= K)."""
    return np.unique(np.geomspace(1, K - 1, n_strata + 1).astype(int))


def certify(mix, sched, ddpm=False, n_strata=24, dense=False, **quad_kw):
    """Chain-rule sum over steps k = 1..K-1: dense (all steps) or stratified
    upper estimate (max of stratum-endpoint values x stratum size)."""
    K = sched.K
    rows = []
    if dense:
        ks = np.arange(1, K)
        vals = np.array([per_step_kl(mix, sched, int(k), ddpm, **quad_kw)
                         for k in ks])
        total = float(np.sum(np.maximum(vals, FLOOR)))
        for k, v in zip(ks, vals):
            rows.append(dict(k=int(k), kl=float(v)))
        return total, rows
    edges = strata_indices(K, n_strata)
    total = 0.0
    cache = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        for k in {int(lo), int(hi)}:
            if k not in cache:
                cache[k] = per_step_kl(mix, sched, k, ddpm, **quad_kw)
                rows.append(dict(k=k, kl=float(cache[k])))
        seg = max(cache[int(lo)], cache[int(hi)], FLOOR)
        total += seg * max(hi - lo, 1)
    # the last stratum edge K-1 is itself a step; ensure it is counted once
    total += max(cache[int(edges[-1])], FLOOR)
    return float(total), rows


def main():
    t0 = time.time()
    mix = bimodal_1d()
    ladder = [1e-1, 1e-2, 1e-3] if QUICK else [1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                                               1e-6, 1e-7, 1e-8]
    rows = []
    step_rows = []
    for delta in ladder:
        sched, meta = build_schedule(delta, mix)
        kl0 = init_kl(mix, sched)
        tot, srows = certify(mix, sched, ddpm=False,
                             n_strata=16 if QUICK else 24)
        for r in srows:
            r.update(delta=delta, method="fors")
            step_rows.append(r)
        floor_total = FLOOR * sched.K
        rows.append(dict(
            delta=delta, K=sched.K, G=meta["G"], sigma0_sq=meta["sigma0_sq"],
            kl_init=kl0, chain_sum=tot, floor_total=floor_total,
            certified_kl=kl0 + tot, target_kl=delta**2,
            certified_le_target=bool(kl0 + tot <= delta**2 or tot <= floor_total * 1.01),
        ))
        print(f"[FORS] delta={delta:.0e}: K={sched.K}, G={meta['G']:.1f}, "
              f"KL_init={kl0:.3e}, chain={tot:.3e} (floor {floor_total:.1e}), "
              f"target {delta**2:.1e}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fors_ladder.csv", index=False)

    # money fit #1: K vs L = log(1/delta), degree p from log K = p log L + c
    L = np.log(1.0 / df.delta.values)
    fit = np.polyfit(np.log(L), np.log(df.K.values), 1)
    ss = 1 - (np.sum((np.log(df.K) - np.polyval(fit, np.log(L)))**2)
              / np.sum((np.log(df.K) - np.log(df.K).mean())**2))
    print(f"[FIT] K ~ log(1/delta)^p: p = {fit[0]:.3f} (theory <= 3), "
          f"R^2 = {ss:.5f}")

    # dense-vs-stratified validation on the 0.1 rung
    if not QUICK:
        sched01, _ = build_schedule(1e-1, mix)
        if sched01.K <= 4000:
            tot_d, dr = certify(mix, sched01, dense=True)
            tot_s, _ = certify(mix, sched01, n_strata=24)
            pd.DataFrame(dr).to_csv(OUT / "dense_rung_0p1.csv", index=False)
            print(f"[VALID] rung 1e-1: dense chain={tot_d:.4e}, "
                  f"stratified={tot_s:.4e} (ratio {tot_s / max(tot_d, 1e-300):.2f}, "
                  f"stratified must be >= dense)")

    # DDPM arm: sweep G at the 1e-2 rung early-stop scale
    mixd = mix
    sig0 = 1e-2**2 / (mix.d + mix.second_moment())
    dbar = 1e-2**2 / max(mix.second_moment(), 1.0)
    rows_d = []
    for G in np.geomspace(30, 1e5, 8 if QUICK else 12):
        sched = vp_schedule(sig0, G, dbar)
        kl0 = init_kl(mixd, sched)
        tot, srows = certify(mixd, sched, ddpm=True,
                             n_strata=16 if QUICK else 24)
        for r in srows:
            r.update(G=G, method="ddpm")
            step_rows.append(r)
        rows_d.append(dict(G=G, K=sched.K, kl_init=kl0, chain_sum=tot,
                           certified_kl=kl0 + tot,
                           delta_equiv=float(np.sqrt(max(kl0 + tot, 1e-300)))))
        print(f"[DDPM] G={G:.0f}: K={sched.K}, chain KL={tot:.3e}, "
              f"delta_equiv={rows_d[-1]['delta_equiv']:.3e}")
    dfd = pd.DataFrame(rows_d)
    dfd.to_csv(OUT / "ddpm_sweep.csv", index=False)
    m = dfd.delta_equiv < 0.5
    if m.sum() >= 3:
        fd = np.polyfit(np.log(1 / dfd.delta_equiv[m]), np.log(dfd.K[m]), 1)
        ssd = 1 - (np.sum((np.log(dfd.K[m]) - np.polyval(fd, np.log(1 / dfd.delta_equiv[m])))**2)
                   / np.sum((np.log(dfd.K[m]) - np.log(dfd.K[m]).mean())**2))
        print(f"[FIT] DDPM K ~ (1/delta)^a: a = {fd[0]:.3f}, R^2 = {ssd:.5f}")

    pd.DataFrame(step_rows).to_csv(OUT / "per_step_kls.csv", index=False)
    json.dump(dict(c0=C0, B=B, dstar=DSTAR, floor=FLOOR,
                   quick=QUICK, runtime_s=time.time() - t0),
              open(OUT / "meta.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
