"""Money plot #1 + supporting figures from results/exp1/*.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp1"
LAYOUT = dict(template="plotly_white", font=dict(size=13), width=820, height=480)


def money_plot():
    df = pd.read_csv(OUT / "fors_ladder.csv")
    dd = pd.read_csv(OUT / "ddpm_sweep.csv")
    L = np.log(1 / df.delta)
    p_fit = np.polyfit(np.log(L), np.log(df.K), 1)
    r2f = 1 - np.sum((np.log(df.K) - np.polyval(p_fit, np.log(L)))**2) / \
        np.sum((np.log(df.K) - np.log(df.K).mean())**2)
    m = dd.delta_equiv < 0.5
    a_fit = np.polyfit(np.log(1 / dd.delta_equiv[m]), np.log(dd.K[m]), 1)
    r2d = 1 - np.sum((np.log(dd.K[m]) - np.polyval(a_fit, np.log(1 / dd.delta_equiv[m])))**2) / \
        np.sum((np.log(dd.K[m]) - np.log(dd.K[m]).mean())**2)

    f = go.Figure()
    f.add_scatter(x=1 / df.delta, y=df.K, mode="markers+lines",
                  name=f"FORS (Alg. 2): K ~ log(1/δ)^{p_fit[0]:.2f}, R²={r2f:.4f}",
                  line=dict(color="#2c5f8a", width=3), marker=dict(size=9))
    f.add_scatter(x=1 / dd.delta_equiv[m], y=dd.K[m], mode="markers+lines",
                  name=f"DDPM baseline: K ~ (1/δ)^{a_fit[0]:.2f}, R²={r2d:.4f}",
                  line=dict(color="#c0392b", width=3, dash="dot"),
                  marker=dict(size=9, symbol="square"))
    f.update_layout(
        title=("Money plot #1 — steps to reach KL ≤ δ²: polylog (FORS) vs "
               "poly(1/δ) (DDPM), certified by quadrature"),
        xaxis_type="log", yaxis_type="log",
        xaxis_title="1/δ (final accuracy)", yaxis_title="K (backward steps)",
        legend=dict(x=0.02, y=0.98), **LAYOUT)
    f.write_html(OUT / "fig_money1.html", include_plotlyjs="cdn")
    print(f"FORS degree p={p_fit[0]:.3f} R2={r2f:.5f} | "
          f"DDPM a={a_fit[0]:.3f} R2={r2d:.5f}")


def per_step_plot():
    df = pd.read_csv(OUT / "per_step_kls.csv")
    meta = pd.read_json(OUT / "meta.json", typ="series")
    f = go.Figure()
    for delta, g in df[df.method == "fors"].groupby("delta"):
        g = g.sort_values("k")
        f.add_scatter(x=g.k + 1, y=np.maximum(g.kl, 1e-18),
                      mode="markers+lines", name=f"FORS δ={delta:.0e}")
    f.add_hline(y=meta["floor"], line_dash="dash",
                annotation_text="float64 resolution floor 3e-15")
    f.update_layout(title=("Certified per-step E KL(ρ_k‖ρ̂_k) at sampled steps "
                           "(FORS, exact scores) — at/below the numerical floor"),
                    xaxis_type="log", yaxis_type="log",
                    xaxis_title="step k", yaxis_title="per-step KL", **LAYOUT)
    f.write_html(OUT / "fig_per_step.html", include_plotlyjs="cdn")


def dense_plot():
    p = OUT / "dense_rung_0p1.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    f = go.Figure()
    f.add_scatter(x=df.k + 1, y=np.maximum(df.kl, 1e-18), mode="lines",
                  name="dense per-step KL (every k)")
    f.add_hline(y=3e-15, line_dash="dash", annotation_text="float64 floor")
    f.update_layout(title=("Dense certification of every step, δ=0.1 rung — "
                           "validates the stratified bound"),
                    xaxis_type="log", yaxis_type="log",
                    xaxis_title="step k", yaxis_title="per-step KL", **LAYOUT)
    f.write_html(OUT / "fig_dense.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    money_plot(); per_step_plot(); dense_plot()
    print("figures:", sorted(p.name for p in OUT.glob("fig_*.html")))
