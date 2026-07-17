"""Independent numerical implementation of arXiv:2602.01338
"High-accuracy sampling for diffusion models and log-concave distributions"
(Chen, Chewi, Daskalakis, Rakhlin) for the ICML 2026 reproducibility challenge.

Modules
-------
targets    : analytic targets with exact scores (Gaussian mixtures, 1D potentials)
schedules  : Cor-4.4 VP schedule, VE schedule
fors       : Algorithm 1 (FORS), scalar + batched, query-instrumented
tilts      : Section-3.2 Gaussian-tilt sampler (RGO building block), Newton prox
diffusion  : Algorithm 2 with the Section-4.2 instantiation + DDPM baseline
quadrature : deterministic certification of per-step KL (Sec. F.2 chain rule)
metrics    : grid KL/chi2/TV, Gaussian W2, MMD, sliced-W1, moment errors
"""
__version__ = "0.1.0"
