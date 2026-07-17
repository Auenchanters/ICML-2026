"""Noise schedules per arXiv:2602.01338 Corollary 4.4 (VP) and the VE analogue.

VP recursion (paper Sec. 4.2 bullet list, v2):
    sigma_{k+1}^2 / (1 - sigma_{k+1}^2) = sigma_k^2/(1 - sigma_k^2) * (1 + 1/G)
which is exactly eta_k = sigma_k^2 / G, with
    alpha_k^2 = (1 - sigma_{k+1}^2)/(1 - sigma_k^2),   abar_k^2 = 1 - sigma_k^2,
    sigma_{k+1}^2 = alpha_k^2 (sigma_k^2 + eta_k),
    1/etabar_k = 1/eta_k + 1/sigma_k^2.

G := C (dstar + log(K/delta)) log(K/delta)  — helper `G_of` below.
Terminal condition: run until 1 - sigma_K^2 <= deltabar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Schedule:
    """Arrays indexed k = 0..K (sigma2 has K+1 entries; per-step arrays have K).

    VP schedules also carry tbar_k := 1 - sigma_k^2 computed WITHOUT the
    catastrophic subtraction 1 - sigma2 (via 1/(1+rho)): near the terminal
    sigma^2 -> 1 the subtraction loses ~4 digits and breaks the exact tilt
    identity at the 1e-5 log-density level."""

    kind: str                      # "VP" or "VE"
    sigma2: np.ndarray             # (K+1,) noise variances, increasing
    eta: np.ndarray                # (K,)   eta_k
    alpha: np.ndarray = field(default=None)   # (K,) per-step alpha_k (VP); 1 for VE
    G: float = None
    tbar: np.ndarray = field(default=None)    # (K+1,) 1 - sigma_k^2, VP only

    @property
    def K(self):
        return len(self.eta)

    @property
    def abar(self):
        """(K+1,) cumulative signal coefficient abar_k."""
        if self.kind == "VE":
            return np.ones_like(self.sigma2)
        if self.tbar is not None:
            return np.sqrt(self.tbar)
        return np.sqrt(1.0 - self.sigma2)

    @property
    def etabar(self):
        """(K,) 1/etabar_k = 1/eta_k + 1/sigma_k^2 (uses sigma_k, the *lower* index)."""
        return 1.0 / (1.0 / self.eta + 1.0 / self.sigma2[:-1])


def G_of(dstar: float, delta: float, K: float, C: float = 1.0) -> float:
    """G = C (dstar + log(K/delta)) log(K/delta), per Cor 4.4."""
    L = np.log(K / delta)
    return float(C * (dstar + L) * L)


def sigma0_of(delta: float, d: int, M2sq: float, c: float = 1.0) -> float:
    """sigma_0^2 ≍ delta^2/(d + M2^2) — the early-stopping choice of Cor 4.4."""
    return float(c * delta**2 / (d + M2sq))


def vp_schedule(sigma0_sq: float, G: float, deltabar: float, max_K: int = 10**7) -> Schedule:
    """Build the Cor-4.4 VP schedule until 1 - sigma_K^2 <= deltabar.

    All terminal-sensitive quantities go through tbar = 1/(1+rho) (relative
    precision preserved as sigma^2 -> 1); alpha_k^2 = tbar_{k+1}/tbar_k."""
    rho = sigma0_sq / (1.0 - sigma0_sq)
    sig2, tbar = [sigma0_sq], [1.0 / (1.0 + rho)]
    eta, alpha = [], []
    r = 1.0 + 1.0 / G
    while tbar[-1] > deltabar:
        if len(eta) >= max_K:
            raise RuntimeError(f"VP schedule exceeded max_K={max_K}")
        s2 = sig2[-1]
        rho *= r
        tb = 1.0 / (1.0 + rho)
        eta.append(s2 / G)             # exact: eta_k = sigma_k^2/G by the recursion
        alpha.append(np.sqrt(tb / tbar[-1]))
        sig2.append(rho / (1.0 + rho))
        tbar.append(tb)
    return Schedule("VP", np.array(sig2), np.array(eta), np.array(alpha), G=G,
                    tbar=np.array(tbar))


def ve_schedule(sigma0_sq: float, G: float, sigma_max_sq: float, max_K: int = 10**7) -> Schedule:
    """VE: sigma_{k+1}^2 = sigma_k^2 (1 + 1/G) up to sigma_max_sq; eta_k = sigma_k^2/G."""
    sig2 = [sigma0_sq]
    eta = []
    while sig2[-1] < sigma_max_sq:
        if len(eta) >= max_K:
            raise RuntimeError(f"VE schedule exceeded max_K={max_K}")
        eta.append(sig2[-1] / G)
        sig2.append(sig2[-1] + eta[-1])
    K = len(eta)
    return Schedule("VE", np.array(sig2), np.array(eta), np.ones(K), G=G)
