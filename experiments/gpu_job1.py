# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "pandas"]
# ///
"""HF GPU Job #1 (PLAN.md P10) — scaled corroboration for the FORS repro
(arXiv:2602.01338, logbook Auenchanters/repro-2602-01338-fors).

Self-contained torch-CUDA float64 port of Algorithm 2 (FORS diffusion
sampler, Sec-4.2 instantiation, stable tbar VP schedule):

  JOB-A  Claim-1 Arm B at scale: bimodal 1D mixture, exact scores,
         delta in {1e-2, 1e-3} (K ~ 3.8k / 9.3k), n = 1e6 chains each.
         Metrics: histogram-KL vs closed-form p_1 (floor bins/2n = 3e-5),
         queries/step/chain (predict 2B/A = 5.44), acceptance.
  JOB-B  Claim-3 end-to-end at ambient d = 512: subspace mixture (d* ~ 2),
         d*-schedule (G from d* = 2, NOT d = 512), n = 1e5 chains.
         Metrics: on-subspace projection histogram-KL, ambient moment errors,
         acceptance (healthy acceptance at d = 512 under the d*-budget is the
         operational claim-3 signature).

All results printed as CSV blocks between BEGIN/END markers for logbook
capture. Deterministic seeds. Est. wall-clock < 30 min on a10g-small.
"""
import sys
import time

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
B = 1.0
C0, DSTAR1 = 0.55, 1.0
print(f"# device={DEV}, torch={torch.__version__}, "
      f"gpu={torch.cuda.get_device_name(0) if DEV == 'cuda' else 'none'}",
      flush=True)


# ---------------- schedule (tbar parametrization, exact) --------------------

def vp_schedule(sigma0_sq, G, deltabar):
    rho = sigma0_sq / (1.0 - sigma0_sq)
    sig2, tbar, eta, alpha = [sigma0_sq], [1.0 / (1.0 + rho)], [], []
    r = 1.0 + 1.0 / G
    while tbar[-1] > deltabar:
        s2 = sig2[-1]
        rho *= r
        tb = 1.0 / (1.0 + rho)
        eta.append(s2 / G)
        alpha.append(np.sqrt(tb / tbar[-1]))
        sig2.append(rho / (1.0 + rho))
        tbar.append(tb)
    return (np.array(sig2), np.array(tbar), np.array(eta), np.array(alpha))


def build_exp1_schedule(delta, d, M2sq, dstar=DSTAR1):
    sigma0 = delta**2 / (d + M2sq)
    deltabar = delta**2 / max(M2sq, 1.0)
    K = 1000.0
    for _ in range(12):
        L = np.log(max(K, 2.0) / delta**2)
        G = C0 * (dstar + L) * L
        sch = vp_schedule(sigma0, G, deltabar)
        if abs(len(sch[2]) - K) <= 2:
            break
        K = len(sch[2])
    return sch, G


# ---------------- diagonal mixture in torch --------------------------------

class Mix:
    def __init__(self, w, mu, var):
        self.w = torch.tensor(w, dtype=DT, device=DEV)
        self.w /= self.w.sum()
        self.mu = torch.tensor(mu, dtype=DT, device=DEV)
        self.var = torch.tensor(var, dtype=DT, device=DEV)
        self.H, self.d = self.mu.shape

    def noised_params(self, abar, sig2):
        return abar * self.mu, abar**2 * self.var + sig2

    def resp(self, x, mu, var):
        diff = x[:, None, :] - mu[None]
        logc = -0.5 * ((diff**2 / var[None]).sum(-1)
                       + torch.log(var).sum(-1)[None])
        a = logc + torch.log(self.w)[None]
        return torch.softmax(a, dim=1)

    def score(self, x, abar, sig2):
        mu, var = self.noised_params(abar, sig2)
        r = self.resp(x, mu, var)
        return torch.einsum("nh,nhd->nd", r, (mu[None] - x[:, None, :]) / var[None])

    def denoiser(self, x, abar, sig2):
        """Posterior-mean form (stable as abar -> 0)."""
        mu, var = self.noised_params(abar, sig2)
        r = self.resp(x, mu, var)
        gain = abar * self.var / var          # (H, d)
        mh = self.mu[None] + gain[None] * (x[:, None, :] - abar * self.mu[None])
        return torch.einsum("nh,nhd->nd", r, mh)


TWO_PI_3 = 2 * np.pi / 3


def path(r):
    u = TWO_PI_3 * (1.0 - r)
    return ((1 + 2 * torch.cos(u)) / 3, (2 / np.sqrt(3)) * torch.sin(u),
            (2 * TWO_PI_3 / 3) * torch.sin(u),
            -(2 / np.sqrt(3)) * TWO_PI_3 * torch.cos(u))


def sample_chain(mix, sch, n, seed, chunk=200_000):
    """Algorithm 2, FORS steps, k = K-1 .. 1. Returns samples + stats."""
    sig2, tbar, eta, alpha = sch
    K = len(eta)
    abar = np.sqrt(tbar)
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    tot_draws = tot_props = tot_acc = 0
    for c0 in range(0, n, chunk):
        m0 = min(chunk, n - c0)
        x = (torch.randn(m0, mix.d, generator=g, device=DEV, dtype=DT)
             * np.sqrt(sig2[K]))
        for k in range(K - 1, 0, -1):
            lam = abar[k] / sig2[k]
            etab = 1.0 / (1.0 / eta[k] + 1.0 / sig2[k])
            xbar = (x / alpha[k]
                    + alpha[k] * eta[k] * mix.score(x, abar[k + 1], sig2[k + 1]))
            d_next = mix.denoiser(x, abar[k + 1], sig2[k + 1])
            newx = torch.empty_like(x)
            alive = torch.arange(m0, device=DEV)
            for _ in range(10_000):
                mm = len(alive)
                xb = xbar[alive]
                prop = xb + np.sqrt(etab) * torch.randn(
                    mm, mix.d, generator=g, device=DEV, dtype=DT)
                tot_props += mm
                J = torch.poisson(torch.full((mm,), 2 * B, device=DEV),
                                  generator=g).long()
                jmax = int(J.max().item()) if mm else 0
                logp = torch.zeros(mm, device=DEV, dtype=DT)
                if jmax > 0:
                    r = torch.rand(mm, jmax, generator=g, device=DEV, dtype=DT)
                    z = np.sqrt(etab / 2) * torch.randn(
                        mm, jmax, mix.d, generator=g, device=DEV, dtype=DT)
                    xh = xb[:, None, :] + np.sqrt(etab / 2) * torch.randn(
                        mm, jmax, mix.d, generator=g, device=DEV, dtype=DT)
                    a, b, da, db = path(r)
                    gam = (a[..., None] * prop[:, None, :]
                           + (1 - a)[..., None] * xh + b[..., None] * z)
                    gdot = da[..., None] * (prop[:, None, :] - xh) + db[..., None] * z
                    Dg = mix.denoiser(gam.reshape(-1, mix.d), abar[k],
                                      sig2[k]).reshape(mm, jmax, mix.d)
                    W = lam * ((gdot * (Dg - d_next[alive][:, None, :])).sum(-1))
                    W = W.clamp(-B, B)
                    mask = (torch.arange(jmax, device=DEV)[None, :] < J[:, None])
                    tot_draws += int(J.sum().item())
                    ratio = torch.where(mask, (B + W) / (2 * B),
                                        torch.ones_like(W))
                    logp = torch.log(ratio.clamp_min(1e-300)).sum(1)
                acc = torch.log(torch.rand(mm, generator=g, device=DEV,
                                           dtype=DT)) < logp
                newx[alive[acc]] = prop[acc]
                tot_acc += int(acc.sum().item())
                alive = alive[~acc]
                if len(alive) == 0:
                    break
            else:
                raise RuntimeError("acceptance collapse")
            x = newx
            if k % 1000 == 0:
                print(f"#   chunk@{c0}: k={k}", flush=True)
        out.append(x)
    xs = torch.cat(out)
    return xs, dict(draws=tot_draws, props=tot_props, accs=tot_acc,
                    q_per_step_chain=tot_draws / (n * (K - 1)),
                    accept_rate=tot_acc / max(tot_props, 1))


def hist_kl(xs, w, mu, var, lo=-6, hi=6, bins=60):
    """Histogram KL of 1D samples vs the closed-form mixture density."""
    xs = xs.cpu().numpy().ravel()
    edges = np.linspace(lo, hi, bins + 1)
    cnt, _ = np.histogram(xs, bins=edges)
    ph = cnt / cnt.sum() / np.diff(edges)
    cc = 0.5 * (edges[1:] + edges[:-1])
    pt = sum(wi * np.exp(-(cc - m)**2 / (2 * v)) / np.sqrt(2 * np.pi * v)
             for wi, m, v in zip(w, mu, var))
    pt /= np.trapezoid(pt, cc)
    msk = (ph > 0) & (pt > 0)
    return float(np.sum(np.diff(edges)[msk] * ph[msk] * np.log(ph[msk] / pt[msk])))


# ---------------- JOB-A -----------------------------------------------------

def job_a():
    mix = Mix([0.5, 0.5], [[-2.0], [2.0]], [[0.25], [0.64]])
    M2 = 0.5 * (4 + 0.25) + 0.5 * (4 + 0.64)
    print("BEGIN CSV job_a", flush=True)
    print("delta,K,n,seed,hist_kl,noise_floor,q_per_step_chain,accept_rate,secs")
    for delta in [1e-2, 1e-3]:
        sch, G = build_exp1_schedule(delta, 1, M2)
        K = len(sch[2])
        for seed in [0, 1]:
            t0 = time.time()
            n = 1_000_000
            xs, st = sample_chain(mix, sch, n, seed=1000 + seed)
            abar1 = np.sqrt(sch[1][1])
            w = [0.5, 0.5]
            mu1 = [abar1 * -2.0, abar1 * 2.0]
            v1 = [abar1**2 * 0.25 + sch[0][1], abar1**2 * 0.64 + sch[0][1]]
            kl = hist_kl(xs, w, mu1, v1)
            print(f"{delta},{K},{n},{seed},{kl:.6e},{60/(2*n):.1e},"
                  f"{st['q_per_step_chain']:.4f},{st['accept_rate']:.4f},"
                  f"{time.time()-t0:.0f}", flush=True)
    print("END CSV job_a", flush=True)


# ---------------- JOB-B -----------------------------------------------------

def job_b():
    d = 512
    mu = np.zeros((2, d)); mu[0, 0], mu[1, 0] = -2, 2
    mu[0, 1], mu[1, 1] = -1, 1
    var = np.full((2, d), 1e-6); var[:, :2] = 0.25
    mix = Mix([0.5, 0.5], mu, var)
    L = np.log(4000 / 1e-2**2)
    G = C0 * (2.0 + L) * L                      # d* = 2 budget, NOT d = 512
    sch = vp_schedule(1e-4, G, 1e-2)
    K = len(sch[2])
    t0 = time.time()
    n = 100_000
    xs, st = sample_chain(mix, sch, n, seed=7, chunk=20_000)
    abar1 = np.sqrt(sch[1][1])
    kl0 = hist_kl(xs[:, 0], [0.5, 0.5], [abar1 * -2, abar1 * 2],
                  [abar1**2 * 0.25 + sch[0][1]] * 2)
    amb_err = float((xs[:, 2:].var(dim=0).cpu().numpy()
                     - (abar1**2 * 1e-6 + sch[0][1])).max())
    print("BEGIN CSV job_b", flush=True)
    print("d,K,G,n,proj_hist_kl,noise_floor,ambient_var_err_max,"
          "q_per_step_chain,accept_rate,secs")
    print(f"{d},{K},{G:.1f},{n},{kl0:.6e},{60/(2*n):.1e},{amb_err:.3e},"
          f"{st['q_per_step_chain']:.4f},{st['accept_rate']:.4f},"
          f"{time.time()-t0:.0f}", flush=True)
    print("END CSV job_b", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    job_a()
    job_b()
    print(f"# total wall-clock {time.time()-t0:.0f}s", flush=True)
