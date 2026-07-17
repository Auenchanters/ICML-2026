# Negative Controls & Robustness


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_553f21d35552", "created_at": "2026-07-17T16:25:30+00:00", "title": "NC-0, NC-1, NC-2: the tests have power"}
-->
Every 'verified' cell in this logbook is paired with a negative control that demonstrates the measuring instrument rejects wrong things at the reported scales.

**NC-0 — biased FORS estimator (Claim 1 page):** estimator with x-dependent bias E[W] = w(x) + 0.3·cos(x) (a constant bias would cancel in normalization). Chi-square GOF: **stat = 14,706 (df=199), p < 1e-300** at n = 10^6 — decisive rejection, while the unbiased estimator passes with p = 0.196 at n = 10^7. Raw: `results/exp0/nc0.csv`.

**NC-1 — DDPM baseline (Claim 2 page):** identical proposal (same exponential-integrator mean AND variance η̄_k), corrector removed. Certified K to reach KL ≤ δ²: **K ~ (1/δ)^2.06, R² = 0.9995** across K = 659 → 2.16M — the poly(1/δ) wall the paper's Theorem E.10 predicts for one-step Gaussian kernels, against FORS's certified log^2.24(1/δ) on the same targets and schedule family. Raw: `results/exp1/ddpm_sweep.csv`.

**NC-2 — condition (16) violated (Claim 2):** with the calibrated G, worst certified per-step KL sits **3.1e-9× below** the per-step target; at G × 0.1 it exceeds the target **19.8×**; at G × 0.03, **16,209×**. The schedule condition of Theorem 4.3 is load-bearing, and the certification pipeline resolves the transition over 13 orders of magnitude. Raw: `results/exp1/nc2.csv`. (Mechanism note: this is also PLAN risk H.1 — small G collapses the FORS acceptance rate; our sampler raises `ForsBudgetExceeded` rather than silently truncating.)

**Robustness (Arm C, Claim 2):** controlled score-noise injection verifies the KL floor scales exactly as ε² (ratio to the paper's Σ η_k ε²_{k,score} term = 0.593, constant to 4 decimals over four decades of ε). Raw: `results/exp1/arm_c.csv`.

NC-3 (heavy-tailed ∇m target vs the √d schedule, Claim 3) and NC-4 (ULA bias plateau vs FORS-proximal, Claim 5) land with their claim pages.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_af0ff6cbe5fc", "created_at": "2026-07-17T22:32:38+00:00", "title": "NC-4: ULA bias plateaus (Claim 5)"}
-->
ULA at step sizes h ∈ {1e-1, 1e-2, 1e-3} on the logcosh target plateaus at χ² = 2.5e-2 / 2.1e-4 / 2.1e-6 — the O(h) discretization-bias wall (χ² ~ h²) — while the FORS-proximal chain on the identical target and query model decays exponentially to 1.3e-11. Money plot #3 on the Claim 5 page; raw `results/exp5/ula_logcosh.csv`. This is the entire point of the paper made visible: prior first-order high-accuracy samplers pay poly(1/ε) because of discretization bias; FORS removes the bias.
