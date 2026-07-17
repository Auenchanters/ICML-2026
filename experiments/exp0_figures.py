"""Build the EXP-0 plotly figures from results/exp0/*.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp0"

LAYOUT = dict(template="plotly_white", font=dict(size=13), width=760, height=440)


def fig_law():
    df = pd.read_csv(OUT / "law_histogram.csv")
    gof = pd.read_csv(OUT / "gof.csv")
    r0 = gof[(gof.seed == 0) & (~gof.biased)].iloc[0]
    f = go.Figure()
    f.add_bar(x=df.x, y=df.dens_emp, name=f"FORS empirical (n={int(r0.n):.0e})",
              marker_color="#88a8d8", opacity=0.75)
    f.add_scatter(x=df.x, y=df.dens_true, name="q·e^w / Z (quadrature truth)",
                  line=dict(color="#c0392b", width=2))
    f.update_layout(title=(f"Exact law (Thm 3.1a): chi2 GOF stat={r0.stat:.1f} "
                           f"(df=199), p={r0.p:.3f}"),
                    xaxis_title="x", yaxis_title="density", **LAYOUT)
    f.write_html(OUT / "fig_law.html", include_plotlyjs="cdn")


def fig_w1():
    df = pd.read_csv(OUT / "w1_scaling.csv")
    sl = np.polyfit(np.log10(df.n), np.log10(df.w1), 1)
    f = go.Figure()
    f.add_scatter(x=df.n, y=df.w1, mode="markers+lines", name="W1(empirical, truth)")
    ref = df.w1.iloc[0] * (df.n / df.n.iloc[0]) ** -0.5
    f.add_scatter(x=df.n, y=ref, mode="lines", name="n^(-1/2) reference",
                  line=dict(dash="dash", color="gray"))
    f.update_layout(title=f"W1 vs n: fitted slope {sl[0]:.3f} (theory -1/2)",
                    xaxis_type="log", yaxis_type="log",
                    xaxis_title="n samples", yaxis_title="W1", **LAYOUT)
    f.write_html(OUT / "fig_w1.html", include_plotlyjs="cdn")


def fig_acceptance():
    df = pd.read_csv(OUT / "acceptance_identity.csv")
    d0 = df[df.seed == 0].copy()
    xc = 0.5 * (d0.bin_lo + d0.bin_hi)
    se = np.sqrt(d0.true * (1 - d0.true) / d0.n)
    f = go.Figure()
    f.add_scatter(x=xc, y=d0.true, name="e^(w(x)-B)  (Thm 3.1b identity)",
                  line=dict(color="#c0392b", width=2))
    f.add_scatter(x=xc, y=d0.emp, mode="markers", name="empirical acceptance",
                  error_y=dict(type="data", array=2 * se, visible=True),
                  marker=dict(color="#2c5f8a", size=5))
    mz = df.z.abs().max()
    f.update_layout(title=(f"Per-x acceptance identity: max|z| = {mz:.2f} over "
                           f"{len(df)} (bin,seed) cells (seed 0 shown, 2SE bars)"),
                    xaxis_title="x", yaxis_title="P(accept | x)", **LAYOUT)
    f.write_html(OUT / "fig_acceptance.html", include_plotlyjs="cdn")


def fig_queries():
    df = pd.read_csv(OUT / "query_complexity.csv")
    f = go.Figure()
    deltas = [0.1, 0.01, 0.001]
    for Bq, g in df.groupby("B"):
        f.add_scatter(x=deltas, y=[g[f"tail_{d}"].mean() for d in deltas],
                      mode="markers+lines", name=f"empirical tail, B={Bq}")
    f.add_scatter(x=deltas, y=deltas, mode="lines", name="Thm 3.1(c) bound (delta)",
                  line=dict(dash="dash", color="black"))
    f.update_layout(title="P(N_draws > 3Be^{2B}log(2/delta)) vs the guaranteed delta",
                    xaxis_type="log", yaxis_type="log", xaxis_title="delta",
                    yaxis_title="tail probability",
                    yaxis_range=[-5.2, 0], **LAYOUT)
    f.write_html(OUT / "fig_queries.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    fig_law(); fig_w1(); fig_acceptance(); fig_queries()
    print("figures written:", sorted(p.name for p in OUT.glob("fig_*.html")))
