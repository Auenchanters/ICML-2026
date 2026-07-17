"""Claim 5 figures: RGO superpolynomial decay, chain exponential decay,
money plot #3 (queries vs achieved accuracy: FORS straight vs ULA flatlines)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp5"
LAYOUT = dict(template="plotly_white", font=dict(size=13), width=820, height=480)
FLOOR = 1e-16


def fig_rgo():
    f = go.Figure()
    for tag, name in [("logcosh", "x²/2+log cosh 2x (s=1, β₁=5)"),
                      ("pseudohuber", "x²/2+√(1+x²) (s=0 flavor)")]:
        df = pd.read_csv(OUT / f"rgo_sweep_{tag}.csv")
        f.add_scatter(x=df.inv_eta, y=np.maximum(df.chi2, FLOOR),
                      mode="markers+lines", name=name)
    f.add_hline(y=2.2e-16, line_dash="dash",
                annotation_text="float64 machine precision")
    f.update_layout(title=("Thm 3.3: RGO chi²(ν‖ν̂) vs 1/η — superpolynomial "
                           "decay to machine zero (quadrature, no clip slack)"),
                    xaxis_type="log", yaxis_type="log",
                    xaxis_title="1/η", yaxis_title="chi²", **LAYOUT)
    f.write_html(OUT / "fig_rgo.html", include_plotlyjs="cdn")


def fig_chain():
    df = pd.read_csv(OUT / "chain_logcosh.csv")
    floor = df.chi2.iloc[-10:].mean()
    d = df[df.chi2 > max(100 * floor, 1e-13)]
    fit = np.polyfit(d.n, np.log(d.chi2), 1)
    r2 = 1 - np.sum((np.log(d.chi2) - np.polyval(fit, d.n))**2) / \
        np.sum((np.log(d.chi2) - np.log(d.chi2).mean())**2)
    f = go.Figure()
    f.add_scatter(x=df.n, y=np.maximum(df.chi2, FLOOR), mode="markers",
                  name="chi²(μ̂_n ‖ μ) — grid evolution, deterministic")
    f.add_scatter(x=d.n, y=np.exp(np.polyval(fit, d.n)), mode="lines",
                  name=f"exp({fit[0]:.3f}·n) fit, R²={r2:.5f}",
                  line=dict(dash="dash"))
    f.add_hline(y=floor, line_dash="dot",
                annotation_text=f"FORS-RGO bias floor {floor:.1e} (≤1e-10 target)")
    f.update_layout(title=("Proximal sampler with FORS-RGO (η=1/64): exponential "
                           "chi² decay to a 1.3e-11 floor"),
                    yaxis_type="log", xaxis_title="iteration n",
                    yaxis_title="chi²", **LAYOUT)
    f.write_html(OUT / "fig_chain.html", include_plotlyjs="cdn")
    return fit, r2, floor


def fig_money3():
    ch = pd.read_csv(OUT / "chain_logcosh.csv")
    ula = pd.read_csv(OUT / "ula_logcosh.csv")
    f = go.Figure()
    m = ch.chi2 > 2e-11
    f.add_scatter(x=ch.queries_cum[m], y=ch.chi2[m], mode="lines+markers",
                  name="FORS-proximal (first-order queries, exact accounting)",
                  line=dict(color="#2c5f8a", width=3))
    for h, g in ula.groupby("h"):
        f.add_scatter(x=g.n, y=g.chi2, mode="lines",
                      name=f"ULA h={h:g} (plateaus at its bias floor)",
                      line=dict(dash="dot", width=2))
    f.update_layout(title=("MONEY PLOT #3 — queries vs achieved chi²: ULA "
                           "flatlines at O(h²) bias, FORS-proximal goes to 1e-11"),
                    xaxis_type="log", yaxis_type="log",
                    xaxis_title="first-order queries",
                    yaxis_title="chi² achieved", **LAYOUT)
    f.write_html(OUT / "fig_money3.html", include_plotlyjs="cdn")


def query_fit():
    ch = pd.read_csv(OUT / "chain_logcosh.csv")
    rows = []
    for eps2 in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        hit = ch[ch.chi2 <= eps2]
        if len(hit):
            rows.append(dict(eps2=eps2, n=int(hit.n.iloc[0]),
                             queries=float(hit.queries_cum.iloc[0])))
    qf = pd.DataFrame(rows)
    qf.to_csv(OUT / "queries_vs_eps.csv", index=False)
    L = np.log(1 / qf.eps2)
    fit = np.polyfit(np.log(L), np.log(qf.queries), 1)
    r2 = 1 - np.sum((np.log(qf.queries) - np.polyval(fit, np.log(L)))**2) / \
        np.sum((np.log(qf.queries) - np.log(qf.queries).mean())**2)
    print(f"queries ~ log(1/eps²)^{fit[0]:.3f}, R² = {r2:.5f} (theory: degree <= 2)")
    return fit, r2


if __name__ == "__main__":
    fig_rgo()
    fit_c, r2_c, floor = fig_chain()
    fig_money3()
    fit_q, r2_q = query_fit()
    print(f"chain: rate {fit_c[0]:.3f}/iter R²={r2_c:.5f}, floor {floor:.2e}")
    print("figures:", sorted(p.name for p in OUT.glob("fig_*.html")))
