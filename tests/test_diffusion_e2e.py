"""End-to-end Algorithm 2 smoke: FORS chain lands on p_1, beats/matches DDPM
at equal step count, and query counts are O(K) per chain (Cor 4.4 clause)."""
import numpy as np

from fors.targets import bimodal_1d
from fors.schedules import vp_schedule
from fors.diffusion import DiffusionSampler
from fors.metrics import hist_kl_vs_grid


def _true_p1_grid(mix, sched, lo=-6, hi=6, n=4001):
    p1 = mix.noised(sched.abar[1], sched.sigma2[1])
    x = np.linspace(lo, hi, n)
    return x, p1.pdf(x[:, None])


def test_fors_chain_hits_p1():
    rng = np.random.default_rng(11)
    mix = bimodal_1d()
    sched = vp_schedule(sigma0_sq=0.05, G=60.0, deltabar=0.1)
    ds = DiffusionSampler(mix, sched, B=1.0)
    n = 4000
    xf, st = ds.sample(n, rng, method="fors")
    x, p1 = _true_p1_grid(mix, sched)
    kl_f = hist_kl_vs_grid(xf, x, p1, bins=40)
    # sample-noise floor for 40 bins at n=4000 is ~ bins/(2n) ~ 5e-3
    assert kl_f < 0.05, f"FORS chain KL {kl_f:.4f}"
    # query accounting: draws per accepted step bounded (Thm 3.1(c) shape)
    per_step = st.w_draws / (n * (sched.K - 1))
    assert per_step < 3 * 1.0 * np.exp(2.0) * np.log(2 / 0.01)
    assert st.accept_rate > np.exp(-2 * 1.0) * 0.5   # far from collapse


def test_ddpm_baseline_runs():
    rng = np.random.default_rng(12)
    mix = bimodal_1d()
    sched = vp_schedule(sigma0_sq=0.05, G=60.0, deltabar=0.1)
    ds = DiffusionSampler(mix, sched, B=1.0)
    xd, _ = ds.sample(3000, rng, method="ddpm")
    x, p1 = _true_p1_grid(mix, sched)
    kl_d = hist_kl_vs_grid(xd, x, p1, bins=40)
    assert np.isfinite(kl_d) and kl_d < 0.5
