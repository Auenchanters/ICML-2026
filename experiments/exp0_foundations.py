"""EXP-0 (PLAN.md C.EXP-0): FORS exactness — Theorem 3.1 verified empirically.

Setup: q = N(0,1), tilt w(x) = 0.6 sin(2x), estimator W_x = w(x) + U[-c, c]
(mean-zero noise, c = 0.5), B = 1.2 (support check: |w| + c = 1.1 < B, no
clipping binds, so E[W|x] = w(x) exactly).

Tests
  1. exact law        : chi-square GOF on 200 equal-mass bins vs the
                        quadrature-normalized q e^w; W1 scaling ~ n^{-1/2}
  2. acceptance       : per-x acceptance == e^{w(x) - B} (Thm 3.1(b) identity)
  3. query complexity : P(N_draws > 3 B e^{2B} log(2/delta)) <= delta, and
                        mean draws per call == 2B / A, A = E_q e^{w - B}
  4. NC-0             : x-DEPENDENT estimator bias (w + 0.3 cos x — a constant
                        bias would cancel in normalization) => GOF rejects
Seeds 0..9 for every stochastic quantity. All raw numbers -> results/exp0/.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import chi2 as chi2_dist, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fors.fors import fors_batch  # noqa: E402

OUT = ROOT / "results" / "exp0"
OUT.mkdir(parents=True, exist_ok=True)

B = 1.2
C_NOISE = 0.5
QUICK = "--quick" in sys.argv


def w_tilt(x):
    return 0.6 * np.sin(2 * x)


def bias(x):
    return 0.3 * np.cos(x)


# ---- ground truth by quadrature -------------------------------------------
G = np.linspace(-8, 8, 64001)
DENS = np.exp(-G**2 / 2 + w_tilt(G))
DENS /= np.trapezoid(DENS, G)
CDF = cumulative_trapezoid(DENS, G, initial=0.0)
CDF /= CDF[-1]

Q_DENS = np.exp(-G**2 / 2) / np.sqrt(2 * np.pi)
A_TRUE = float(np.trapezoid(Q_DENS * np.exp(w_tilt(G) - B), G))   # E_q e^{w-B}


def sample_fors(n, rng, biased=False):
    def propose_n(m, rng):
        return rng.standard_normal((m, 1))

    def draw_w(x, J, rng):
        jm = max(int(J.max()), 1)
        base = w_tilt(x) + (bias(x) if biased else 0.0)
        # biased run keeps support inside [-B, B]: 0.6 + 0.3 + 0.25 < 1.2
        c = 0.25 if biased else C_NOISE
        return base + rng.uniform(-c, c, size=(len(x), jm))

    got, chunks = 0, []
    while got < n:
        m = min(n - got, 400_000)
        out, _ = fors_batch(propose_n, draw_w, B, m, rng, batch=800_000)
        chunks.append(out.ravel())
        got += m
    return np.concatenate(chunks)


def gof_equal_mass(x, n_bins=200):
    """Chi-square GOF against the quadrature truth on equal-mass bins."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.interp(qs, CDF, G)
    edges[0], edges[-1] = -np.inf, np.inf
    cnt, _ = np.histogram(x, bins=edges)
    exp = np.full(n_bins, len(x) / n_bins)
    stat = float(np.sum((cnt - exp) ** 2 / exp))
    p = float(chi2_dist.sf(stat, n_bins - 1))
    return stat, p, cnt


def w1_to_truth(x):
    xs = np.sort(x)
    Fn = np.searchsorted(xs, G, side="right") / len(xs)
    return float(np.trapezoid(np.abs(Fn - CDF), G))


def main():
    t0 = time.time()
    rows_gof, rows_w1 = [], []

    # ---- Test 1: exact law --------------------------------------------------
    n_big = 10**6 if QUICK else 10**7
    rng = np.random.default_rng(0)
    xs_big = sample_fors(n_big, rng)
    stat, p, cnt = gof_equal_mass(xs_big)
    rows_gof.append(dict(seed=0, n=n_big, stat=stat, p=p, biased=False))
    print(f"[T1] n={n_big:.0e}: GOF stat={stat:.1f} (df=199), p={p:.4f}")

    for i, n in enumerate([10**4, 10**5, 10**6] + ([] if QUICK else [10**7])):
        r = np.random.default_rng(50 + i)          # independent sample per n
        rows_w1.append(dict(n=n, w1=w1_to_truth(sample_fors(n, r)),
                            independent=True))
    for seed in range(1, 4 if QUICK else 10):
        r = np.random.default_rng(seed)
        x = sample_fors(10**6 if QUICK else 10**6, r)
        s, pv, _ = gof_equal_mass(x)
        rows_gof.append(dict(seed=seed, n=len(x), stat=s, p=pv, biased=False))
    pd.DataFrame(rows_gof).to_csv(OUT / "gof.csv", index=False)
    pd.DataFrame(rows_w1).to_csv(OUT / "w1_scaling.csv", index=False)
    lw = np.polyfit(np.log10([r["n"] for r in rows_w1]),
                    np.log10([r["w1"] for r in rows_w1]), 1)
    print(f"[T1] W1 scaling slope = {lw[0]:.3f} (theory -1/2)")

    # histogram CSV for the overlay figure (100 uniform bins on [-4,4])
    edges = np.linspace(-4, 4, 101)
    cnt_u, _ = np.histogram(xs_big, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dens_emp = cnt_u / len(xs_big) / np.diff(edges)
    dens_true = np.interp(centers, G, DENS)
    pd.DataFrame(dict(x=centers, dens_emp=dens_emp, dens_true=dens_true)
                 ).to_csv(OUT / "law_histogram.csv", index=False)

    # ---- Test 2: per-x acceptance identity ---------------------------------
    rows_acc = []
    n_prop = 4 * 10**5 if QUICK else 2 * 10**6
    for seed in range(3 if QUICK else 10):
        r = np.random.default_rng(100 + seed)
        x = r.standard_normal(n_prop)
        J = r.poisson(2 * B, size=n_prop)
        jm = J.max()
        W = w_tilt(x)[:, None] + r.uniform(-C_NOISE, C_NOISE, size=(n_prop, jm))
        mask = np.arange(jm)[None, :] < J[:, None]
        logp = np.where(mask, np.log((B + W) / (2 * B)), 0.0).sum(axis=1)
        acc = np.log(r.uniform(size=n_prop)) < logp
        edges_a = np.linspace(-3, 3, 61)
        idx = np.digitize(x, edges_a)
        for b in range(1, 61):
            sel = idx == b
            nb = int(sel.sum())
            if nb < 100:
                continue
            emp = float(acc[sel].mean())
            true = float(np.exp(w_tilt(x[sel]) - B).mean())
            se = np.sqrt(true * (1 - true) / nb)
            rows_acc.append(dict(seed=seed, bin_lo=edges_a[b - 1],
                                 bin_hi=edges_a[b], n=nb, emp=emp,
                                 true=true, z=(emp - true) / se))
    acc_df = pd.DataFrame(rows_acc)
    acc_df.to_csv(OUT / "acceptance_identity.csv", index=False)
    n_cells = len(acc_df)
    bonf_z = float(norm.isf(0.001 / n_cells / 2))
    print(f"[T2] max|z| over {n_cells} (bin, seed) cells = "
          f"{acc_df.z.abs().max():.2f} (Bonferroni-0.001 threshold {bonf_z:.2f})")

    # ---- Test 3: query complexity -------------------------------------------
    rows_q = []
    n_calls = 4 * 10**4 if QUICK else 2 * 10**5
    for Bq in [0.5, 1.0, 2.0]:
        # tilt scaled to B so the support requirement |W| <= B holds:
        # w_B = 0.4 B sin(2x), noise half-width 0.4 B  =>  |W| <= 0.8 B
        amp = 0.4 * Bq
        A_B = float(np.trapezoid(Q_DENS * np.exp(amp * np.sin(2 * G) - Bq), G))
        cw = 0.4 * Bq
        for seed in range(3 if QUICK else 10):
            r = np.random.default_rng(200 + seed)
            draws = np.zeros(n_calls, dtype=np.int64)
            active = np.arange(n_calls)
            for _ in range(10**4):
                m = len(active)
                if m == 0:
                    break
                x = r.standard_normal(m)
                J = r.poisson(2 * Bq, size=m)
                jm = max(int(J.max()), 1)
                W = amp * np.sin(2 * x)[:, None] + r.uniform(-cw, cw, size=(m, jm))
                mask = np.arange(jm)[None, :] < J[:, None]
                logp = np.where(mask, np.log((Bq + W) / (2 * Bq)), 0.0).sum(axis=1)
                draws[active] += J
                acc = np.log(r.uniform(size=m)) < logp
                active = active[~acc]
            mean_draws = float(draws.mean())
            row = dict(B=Bq, seed=seed, mean_draws=mean_draws,
                       pred_mean=2 * Bq / A_B, A=A_B)
            for delta in [0.1, 0.01, 0.001]:
                bound = 3 * Bq * np.exp(2 * Bq) * np.log(2 / delta)
                row[f"tail_{delta}"] = float((draws > bound).mean())
                row[f"bound_{delta}"] = bound
            rows_q.append(row)
    qdf = pd.DataFrame(rows_q)
    qdf.to_csv(OUT / "query_complexity.csv", index=False)
    for delta in [0.1, 0.01, 0.001]:
        worst = qdf[f"tail_{delta}"].max()
        print(f"[T3] delta={delta}: worst empirical tail = {worst:.5f} "
              f"(bound requires <= {delta})")
    print("[T3] mean draws vs 2B/A prediction:")
    print(qdf.groupby("B")[["mean_draws", "pred_mean"]].mean().to_string())

    # ---- Test 4: NC-0 --------------------------------------------------------
    rng = np.random.default_rng(999)
    x_nc = sample_fors(10**5 if QUICK else 10**6, rng, biased=True)
    s_nc, p_nc, _ = gof_equal_mass(x_nc)
    pd.DataFrame([dict(seed=999, n=len(x_nc), stat=s_nc, p=p_nc, biased=True)]
                 ).to_csv(OUT / "nc0.csv", index=False)
    print(f"[NC-0] biased estimator: GOF stat={s_nc:.1f}, p={p_nc:.3e} "
          f"(rejects: {p_nc < 1e-6})")

    print(f"done in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
