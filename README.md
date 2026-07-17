# Reproduction: High-accuracy sampling for diffusion models and log-concave distributions

Independent numerical reproduction of **arXiv:2602.01338** (Chen, Chewi, Daskalakis,
Rakhlin — ICML 2026 Outstanding Paper Award) for the **ICML 2026 agent
reproducibility challenge** (Hugging Face × alphaXiv × Trackio), paper orid 71132.

The paper is pure theory — *"we are working toward implementation and experimental
evaluation, which will be left for future work"* — so a faithful independent
implementation verified against closed-form ground truth **is** the reproduction.
Every theorem is tested at its own level of generality on analytic targets with
exact scores, and per-step error is **certified by deterministic quadrature**
through the paper's own chain-rule decomposition (Sec. F.2): zero Monte-Carlo
noise in the headline results.

Logbook: `Auenchanters/repro-2602-01338-fors` (Trackio) · Headline: FORS reaches
KL ≤ δ² in K ~ log(1/δ)^2.24 steps (R²=0.997) where the DDPM baseline needs
K ~ (1/δ)^2.06 (R²=0.9995) — the paper's exponential separation, certified.

## Layout

```text
PLAN.md              master plan (single source of truth, prompts P0-P14)
src/fors/            library: targets, schedules, FORS (Alg 1), tilts,
                     diffusion (Alg 2 + DDPM baseline), quadrature
                     certification, chf engine (any-d Gaussian chi2), metrics
experiments/         one script per experiment arm (exp0..exp5, figures, GPU job)
tests/               28 pytest gates (exact identities, cross-validations)
sources/             arXiv v1/v2 TeX + SHA256 + version-audit provenance
results/exp*/        raw CSVs + plotly figures (attached to logbook cells)
```

## Reproduce

```bash
py -3.13 -m venv .venv && .venv/Scripts/pip install -U trackio numpy scipy torch matplotlib plotly pandas pytest
.venv/Scripts/python -m pytest tests/ -q          # 28 green, ~1 min
.venv/Scripts/python experiments/exp0_foundations.py       # Thm 3.1 exactness (~1 min)
.venv/Scripts/python experiments/exp1_certified.py         # certified ladder (~7 h CPU; --quick ~2 min)
.venv/Scripts/python experiments/exp1_arms_bc.py           # arms B/C + NC-2
.venv/Scripts/python experiments/exp2_criticalG.py         # critical G*(d) sweep
.venv/Scripts/python experiments/exp3_intrinsic.py         # intrinsic-dimension arm
.venv/Scripts/python experiments/exp4_lipschitz.py         # Lipschitz refinement arm
.venv/Scripts/python experiments/exp5_logconcave.py        # log-concave sampler (~15 min)
```

Every experiment supports `--quick` for a minutes-scale smoke run. All float64;
stochastic arms log seeds; deterministic arms need none.

GPU job (scaled corroboration, `hf jobs uv run --flavor a10g-small`):
`experiments/gpu_job1.py` — Arm B at n=10⁶ and the d=512 subspace end-to-end run.

## Numerical-honesty conventions

- Per-step KL below 3e-15 is under the float64 log-density cancellation floor and
  is reported as "≤ floor", never as a smaller number.
- Chain sums over large K use geometric strata (endpoint-max × size, conservative);
  validated against dense certification of every step on the δ=0.1 rung.
- Quadrature results carry node-doubling convergence checks; MC results carry
  seeds and CIs; sample-based metrics state their O(n^{-1/2}) floors.
