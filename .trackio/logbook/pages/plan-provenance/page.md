# Plan & Provenance


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bddb4acfe7bb", "created_at": "2026-07-17T08:15:58+00:00", "title": "Paper, claims, and reproduction strategy"}
-->
**Paper:** [High-accuracy sampling for diffusion models and log-concave distributions](https://arxiv.org/abs/2602.01338) (arXiv:2602.01338v2) — Fan Chen (MIT), Sinho Chewi (Yale), Constantinos Daskalakis (MIT), Alexander Rakhlin (MIT). ICML 2026 Outstanding Paper Award — [ICML oral 71132](https://icml.cc/virtual/2026/oral/71132).

**Summary.** The paper introduces FORS (first-order rejection sampling), a Bernoulli-factory-based rejection sampler that produces samples from a tilted proposal q·e^w using only *unbiased estimates* of w — no density or zeroth-order access (Algorithm 1, Theorem 3.1, an exact identity). Applying it to the backward kernels of a diffusion process with a path-integral estimator built from score/denoiser differences (Algorithm 2, Eqs. 13–14), it achieves δ-accurate sampling in **polylog(1/δ)** steps — an exponential improvement in δ over all prior diffusion samplers, which pay poly(1/δ) for their per-step Gaussian-approximation bias (Theorem 4.3, Corollary 4.4). Complexity depends on the data distribution only through its intrinsic dimension d⋆ (≤ ambient d), and refines to Frobenius-norm-Lipschitz rates under a non-uniform smoothness condition (Theorem 4.9, Props 4.7/4.10). The same machinery implements the restricted Gaussian oracle of the proximal sampler, yielding the first polylog(1/ε) high-accuracy log-concave/isoperimetric samplers from **first-order queries only** (Section 5, Theorem G.1). The paper is purely theoretical: *"Although this is a primarily theoretical work, we are working toward implementation and experimental evaluation, which will be left for future work"* (Conclusion, v2) — **no code and no experiments exist**, so a faithful independent numerical implementation verified against closed-form ground truth constitutes the reproduction.

**The five registered claims (verbatim):**

> **C1.** The diffusion sampler attains delta-error in polylog(1/delta) steps given sufficiently accurate score estimates (Theorem 4.3)
> **C2.** Under minimal data assumptions, the diffusion sampling complexity is stated as Õ(d polylog(1/delta)), where d is the data dimension (Theorem 4.3)
> **C3.** When the data distribution has intrinsic dimension d*, the complexity reduces to Õ(d* polylog(1/delta)) (Corollary 4.4)
> **C4.** Under a non-uniform Lipschitz condition, the diffusion sampling complexity is refined to Õ(sqrt(dL) polylog(1/delta)) (Theorem 4.9)
> **C5.** The same framework yields a polylog(1/delta)-accuracy sampler for log-concave and more general isoperimetric distributions using first-order gradient queries (Section 5)

**Reproduction strategy.** Every theorem is tested at its own level of generality on analytic targets (Gaussian mixtures, log-concave 1D/2D potentials) with **exact scores** (ε_score = 0), so the theorems' preconditions hold by construction. Per-step error is **certified by deterministic quadrature** through the paper's own chain-rule decomposition (Sec. F.2): KL(p₁‖p̂₁) ≤ KL(p_K‖p̂_K) + Σₖ E KL(ρₖ‖ρ̂ₖ), each term computable to numerical precision with zero Monte-Carlo noise — the strongest available notion of "verified" for a theory paper. End-to-end sampling runs corroborate the certified bounds and measure query counts; negative controls (DDPM baseline, condition-(16) violation, biased estimators, ULA bias floor) demonstrate the tests have power. The conditional clause of C1 ("sufficiently accurate scores") is tested directly by controlled score-noise injection against the paper's Σ η̄ₖ ε²ₖ robustness term.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ef650f692cb3", "created_at": "2026-07-17T08:16:08+00:00", "title": "Version audit: v1 vs v2 provenance (TeX-source diff)"}
-->
# Version audit: arXiv:2602.01338 v1 vs v2 (P2 deliverable)

**Paper:** "High-accuracy sampling for diffusion models and log-concave distributions"
— Fan Chen, Sinho Chewi, Constantinos Daskalakis, Alexander Rakhlin.

**Sources fetched:** `https://arxiv.org/e-print/2602.01338v{1,2}` on 2026-07-17.

```
SHA256(v1.tar.gz) = 01eb277ee0469fce54ba8c241a4a16281b23b4ea9d5ef23b67351e03ec2b0027
SHA256(v2.tar.gz) = 4461cfd4938c5024e1c41f0096a020d60654c049eabe0dc4c9ece81ad2be3aa9
```

v1 dated 2026-02-03, v2 dated 2026-04-28 (tarball mtimes). Full keyword sweep in
`grep-audit.txt` (68 hits). Structural change: v1's `appdx_DM_simple.tex` (analysis of the
simple straight-line estimator) is dropped in v2; v2 adds `appdx_structural.tex`
(structural/intrinsic-dimension machinery).

## Theorem-level diff

| Item | v1 statement (TeX source) | v2 statement (TeX source) | Relationship |
|---|---|---|---|
| Main diffusion theorem | `thm:DM-clip-simple`: condition `σ_k²/η_k ≫ d·log(1/δ) + log²(1/δ)`; straight-line path estimator `Ŵ_{r,x} = Clip⟨x − x̄_t, s_t(rx + (1−r)x̄_t)⟩`; KL ≲ KL(p_T‖p̂_T) + Tδ + Σ η_t ε²_t | `thm:DM-intrinsic` (= **Thm 4.3**): condition (16) `σ_k²/η_k ≫ d⋆·log(1/δ) + log²(1/δ)` with **intrinsic** d⋆; new 3-point path (Eqs. 13–14) with z, x̂ halves and denoiser-difference estimator; same KL conclusion | **Sharpened** (d → d⋆; d⋆ ≤ d always) |
| Complexity corollary | `cor:poly-log`: `T ≤ O((d + log(1/δ))·log²(κ/δ))`, κ = (d+M₂²)/σ₀²; boxed total `max{d, log(1/δ)}·log²((d+M₂²)/δ²)` | `cor:poly-log` (= **Cor 4.4**): `K ≤ O((d⋆ + log(κ/δ))·log²(d⋆κ/δ))`, κ = M₂²/σ₀² + 1; boxed total `d⋆·log³((d+M₂²)/δ²)` | **Sharpened** (same schedule mechanism, d → d⋆) |
| Intrinsic-dimension result | separate adaptive theorem `thm:DM-clip-path-adapt`, boxed `max{d⋆, log(d/δ)}·log²((d+M₂²)/δ²)` | folded into the main theorem (Thm 4.3 / Cor 4.4) | **Mainlined** |
| Refined (Lipschitz) theorem | `thm:DM-clip-path`: condition `σ_t²/η_t ≫ √(dL_δ log(d/δ)) + L_δ log(d/δ)`; guarantee in TV²≤2H² with **degraded score-error term** `√(d/L_δ)·Σ η_t ε²_t` ("we do not know if this is fundamental"); boxed `max{√(dL_δ log(d/δ)), L_δ log(d/δ)}·log((d+M₂²)/δ²)` | `thm:DM-Lip` (= **Thm 4.9**): Frobenius-norm Assumptions 4.6+4.8, condition (20) `σ_k²/η_k ≫ L_F log(d⋆/δ) + log²(1/δ)`; **clean KL** guarantee with undegraded score term; boxed `L_F·log³((d+M₂²)/δ²)`; `prop:Lip-op-to-Frob` (= Prop 4.7): `L_F ≤ C√(L_op(d⋆+log(1/δ)))`; `prop:DDPM-Lip` (= Prop 4.10) boxed `min{√(d·L_op), d⋆^{2/3}L_op^{1/3}}·polylog` | **Sharpened & unified** (√(dL) verbatim in v1; v2 recovers it via Prop 4.7/4.10 and fixes the score-error degradation) |
| Log-concave (Sec. 5) | analogous summary via proximal sampler | LSI: `Õ(κ(d^{1/2}log^{3/2}(R/ε²) + log²(R/ε²)))`; PI; log-concave `Õ(β₁d^{1/2}W₂²/ε²)`; s=0 variants; v2 adds KLS remark (`C_PI ≤ O(log d)·‖E XXᵀ‖_op`) | **Essentially unchanged**, remark added |
| FORS core (Thm 3.1) | `thm:fors` | `thm:fors` — identical statement: output density ∝ q·e^{E[W₁|x]}; ≤ 3Be^{2B}log(2/δ) draws w.p. 1−δ; O(Be^{2B}(T+log(1/δ))) over T calls | **Identical** |

## Claim-by-claim provenance (challenge claims, orid 71132, verbatim)

| # | Claim text | Verdict of audit |
|---|---|---|
| 1 | "delta-error in polylog(1/delta) steps given sufficiently accurate score estimates (Theorem 4.3)" | Matches v2 Thm 4.3 + Cor 4.4 directly. Note the conditional clause ("given sufficiently accurate scores") maps to the `Σ η_k ε²_{k,score}` robustness term — tested in EXP-1 Arm C. |
| 2 | "complexity … Õ(d polylog(1/delta)), where d is the data dimension (Theorem 4.3)" | Claim wording matches **v1** `thm:DM-clip-simple` / boxed `max{d, log(1/δ)}log²(...)`. v2's Thm 4.3 states the sharper d⋆ bound; since d⋆ ≤ d always, the claim as written is **implied a fortiori** by v2. No contradiction. |
| 3 | "intrinsic dimension d*, complexity reduces to Õ(d* polylog(1/delta)) (Corollary 4.4)" | Matches v2 Cor 4.4 (boxed d⋆·log³) exactly; in v1 this was the separate adaptive theorem. |
| 4 | "Õ(sqrt(dL) polylog(1/delta)) (Theorem 4.9)" | "√(dL)" appears **verbatim** in v1 `thm:DM-clip-path`'s boxed bound. v2's Thm 4.9 is stated as `L_F·log³` with Prop 4.7 giving `L_F ≲ √(L_op·d⋆)` and Prop 4.10 giving `min{√(d·L_op), d⋆^{2/3}L_op^{1/3}}` — so √(dL) is a true (weaker) consequence of the v2 statements. **No falsification available; do not chase one.** |
| 5 | "polylog(1/delta)-accuracy sampler for log-concave and more general isoperimetric distributions using first-order gradient queries (Section 5)" | Matches v2 Section 5 / Theorem G.1 (appendix `appdx_log_concave.tex`). Unchanged between versions. |

**Conclusion.** Claims 2 and 4 were phrased against v1's statements; v2 sharpened both
(d → d⋆, √(dL·log) → L_F with a clean KL bound). Every claim as written is a true
consequence of the v2 theorems. The reproduction targets the paper's actual (sharper) v2
statements and reports both readings on the relevant claim pages.

One notable substantive v1→v2 fix worth reporting: v1's refined theorem carried a
`√(d/L_δ)` degradation on the score-error term, flagged by the authors as possibly
non-fundamental; v2 removes it via the Frobenius-norm route (Assumption 4.8, Lemma-level
analysis in `appdx_structural.tex`). This strengthens the case that the Frobenius condition
is the "right" assumption (cf. v2 §ssec:why-Frob, used to frame Claim 4).
