# Foundations: FORS Exactness


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_96bded3c0849", "created_at": "2026-07-17T08:16:37+00:00", "title": "Core library unit tests (P3 gate)"}
-->
The library implements Algorithms 1–3 with the Section-4.2 instantiation (paths Eqs. 13–14, exponential-integrator proposal, Clip_B estimator), the Cor-4.4 VP schedule, Newton prox, and a deterministic quadrature engine for per-step KL certification. Unit-test gates (all at spec precision): Eq. (15) identity residual < 1e-14 over 1e5 r-values; Lemma F.1 joint law (γ, γ̇ independent, γ ~ N(g, ηI), γ̇ ~ N(0, cηI), c = 8π²/27) at 4σ on 1e6 MC draws; exact mixture score vs torch autograd < 1e-9; FORS exact-law chi-square GOF; Thm 3.1(b) acceptance identity; Thm 3.1(c) draw-count bound; prox Newton residual < 1e-13; and the gating sanity — quadrature KL(ρ‖ρ̂) < 1e-10 for pure-Gaussian AND bimodal-mixture targets with exact scores and non-binding B, certifying the entire Sec-4.2 path algebra. The reduced quadrature rule (closed-form clipped-Gaussian inner integral) is cross-checked against the brute-force GL×GH×GH rule and node-doubling self-convergence < 1e-9.

```
$ pytest -q
........................                                                 [100%]
24 passed in 11.21s
```
