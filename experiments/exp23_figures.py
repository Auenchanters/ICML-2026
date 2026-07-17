"""Figures for Claims 2 and 3: money plot #2 (G* flat vs growing), the
structural trace plot, and the EXP-2 G*(d) scaling with sufficiency margins."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
E2, E3 = ROOT / "results" / "exp2", ROOT / "results" / "exp3"
LAYOUT = dict(template="plotly_white", font=dict(size=13), width=840, height=500)


def load_exp2():
    """critical_G.csv if the sweep finished, else parse run.log lines."""
    p = E2 / "critical_G.csv"
    if p.exists():
        return pd.read_csv(p)
    rows = []
    for line in (E2 / "run.log").read_text().splitlines():
        m = re.match(r"\[G\*\] d=(\d+) delta=(\S+): G\* = ([\d.]+) "
                     r"\[([\d.]+), ([\d.]+)\]", line)
        if m:
            rows.append(dict(d=int(m.group(1)), delta=float(m.group(2)),
                             G_star=float(m.group(3)), lo=float(m.group(4)),
                             hi=float(m.group(5))))
    return pd.DataFrame(rows)


def money2():
    g2 = load_exp2()
    g3 = pd.read_csv(E3 / "critical_G_subspace.csv")
    sub = g3.groupby("d").G_star.agg(["mean", "std"]).reset_index()
    f = go.Figure()
    for delta, g in g2.groupby("delta"):
        g = g.sort_values("d")
        fit = np.polyfit(np.log(g.d), np.log(g.G_star), 1)
        f.add_scatter(x=g.d, y=g.G_star, mode="markers+lines",
                      name=(f"full-rank N(0,I_d), δ={delta:.0e}: "
                            f"G* ~ d^{fit[0]:.2f}"),
                      line=dict(width=3))
    fit_s = np.polyfit(np.log(sub["d"]), np.log(sub["mean"]), 1)
    f.add_scatter(x=sub["d"], y=sub["mean"], mode="markers+lines",
                  error_y=dict(type="data", array=sub["std"]),
                  name=(f"subspace mixture (d* ≈ 2), δ=1e-3: "
                        f"G* ~ d^{fit_s[0]:.2f} (FLAT)"),
                  line=dict(width=3, dash="dash", color="#1a7a4a"))
    f.update_layout(
        title=("MONEY PLOT #2 — critical schedule stiffness G*(d): intrinsic "
               "dimension governs, not ambient"),
        xaxis_type="log", yaxis_type="log",
        xaxis_title="ambient dimension d", yaxis_title="critical G*",
        legend=dict(x=0.02, y=0.98), **LAYOUT)
    f.write_html(E3 / "fig_money2.html", include_plotlyjs="cdn")
    print(f"subspace slope {fit_s[0]:.3f}")


def structural():
    df = pd.read_csv(E3 / "structural_trace.csv")
    f = go.Figure()
    for tau, g in df.groupby("tau"):
        f.add_scatter(x=g.d, y=g.tr_full, mode="markers+lines",
                      name=f"full-rank, τ={tau:g} (∝ d)")
        f.add_scatter(x=g.d, y=g.tr_sub, mode="markers+lines",
                      line=dict(dash="dash"),
                      name=f"subspace, τ={tau:g} (flat ≈ d*)")
    f.update_layout(
        title=("Cor E.4's mechanism, exact: E[tr ∇m_τ] linear in d for "
               "full-rank targets, d-independent for subspace targets"),
        xaxis_type="log", yaxis_type="log", xaxis_title="d",
        yaxis_title="E[tr grad m_tau]  (deterministic quadrature)", **LAYOUT)
    f.write_html(E3 / "fig_structural.html", include_plotlyjs="cdn")


def exp2_margins():
    """Claim-2 sufficiency: the condition-(16) schedule G_d = c0(d+L)L vs the
    measured critical G*: margin G_d/G* grows with d (the schedule is
    sufficient with room, exactly as an upper-bound theorem requires)."""
    g2 = load_exp2()
    if not len(g2):
        return
    f = go.Figure()
    for delta, g in g2.groupby("delta"):
        g = g.sort_values("d")
        L = np.log(1 / delta)
        G_suff = 0.55 * (g.d + L) * L
        f.add_scatter(x=g.d, y=G_suff / g.G_star, mode="markers+lines",
                      name=f"δ={delta:.0e}: margin (G_cond16 / G*)")
    f.add_hline(y=1.0, line_dash="dash", annotation_text="margin = 1")
    f.update_layout(
        title=("Claim-2 sufficiency margin: condition (16)'s d-schedule over "
               "the measured critical G* — ≥ 1 everywhere, growing with d"),
        xaxis_type="log", yaxis_type="log", xaxis_title="d",
        yaxis_title="G_cond16 / G*", **LAYOUT)
    f.write_html(E2 / "fig_margins.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    money2()
    structural()
    exp2_margins()
    print("figures:", [p.name for p in E3.glob("fig_*.html")]
          + [p.name for p in E2.glob("fig_*.html")])
