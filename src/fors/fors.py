"""Algorithm 1 of arXiv:2602.01338 — First-Order Rejection Sampling (FORS).

    loop:
        x ~ q;  J ~ Poisson(2B);  W_1..W_J ~ iid W_x  (supported on [-B, B])
        accept x with probability  prod_j (B + W_j) / (2B)

Theorem 3.1: the output density is exactly p_hat(x) ∝ q(x) exp(E[W|x]); the
per-call acceptance probability given x is exp(E[W|x] - B); the number of W
draws per output is <= 3 B e^{2B} log(2/delta) w.p. >= 1 - delta.

Two implementations: a scalar reference (`fors_scalar`, transparently close to
the paper's pseudocode) and a batched vectorized version (`fors_batch`). Both
count proposals and W-draws (= first-order queries) and RAISE on a hard cap —
never silently truncate (PLAN.md risk H.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ForsBudgetExceeded(RuntimeError):
    """Raised when the proposal loop exceeds its hard cap: a loud signal that
    the acceptance rate collapsed (e.g. schedule condition (16) violated)."""


@dataclass
class ForsStats:
    proposals: int = 0
    accepts: int = 0
    w_draws: int = 0
    draws_per_accept: list = field(default_factory=list)  # W-draws consumed per output

    @property
    def accept_rate(self):
        return self.accepts / max(self.proposals, 1)


def fors_scalar(propose, draw_w, B, rng, stats=None, max_iters=10**6):
    """One FORS output, scalar reference implementation.

    propose(rng) -> x (d,);  draw_w(x, n, rng) -> (n,) iid samples of W_x.
    """
    st = stats if stats is not None else ForsStats()
    draws_this_call = 0
    for _ in range(max_iters):
        x = propose(rng)
        st.proposals += 1
        J = rng.poisson(2.0 * B)
        p_acc = 1.0
        if J > 0:
            W = draw_w(x, J, rng)
            if np.any(np.abs(W) > B + 1e-12):
                raise ValueError("estimator W outside [-B, B]; clip before FORS")
            st.w_draws += J
            draws_this_call += J
            p_acc = float(np.prod((B + W) / (2.0 * B)))
        if rng.uniform() < p_acc:
            st.accepts += 1
            st.draws_per_accept.append(draws_this_call)
            return x, st
    raise ForsBudgetExceeded(f"no accept in {max_iters} proposals (rate collapse?)")


def fors_batch(propose_n, draw_w_batch, B, n_out, rng, max_rounds=10**4,
               batch=None, stats=None):
    """Vectorized FORS: produce `n_out` independent outputs.

    propose_n(n, rng) -> (n, d) proposals;
    draw_w_batch(x (n,d), J (n,), rng) -> (n, Jmax) W-matrix (entries beyond
        each row's J_i are ignored; ONLY the first J_i entries may cost queries
        — implementations should generate exactly sum(J) evaluations).

    Returns (out (n_out, d), stats). Query accounting: w_draws += sum(J_i) over
    all proposals, matching the paper's query model (each W draw = one
    first-order/score evaluation).
    """
    st = stats if stats is not None else ForsStats()
    d = np.atleast_2d(propose_n(1, rng)).shape[1]
    out = np.empty((n_out, d))
    filled = 0
    for _ in range(max_rounds):
        n = batch or max(2 * (n_out - filled), 64)
        x = propose_n(n, rng)
        st.proposals += n
        J = rng.poisson(2.0 * B, size=n)
        logp = np.zeros(n)
        if J.max() > 0:
            W = draw_w_batch(x, J, rng)          # (n, Jmax)
            if np.any(np.abs(W) > B + 1e-12):
                raise ValueError("estimator W outside [-B, B]; clip before FORS")
            mask = np.arange(W.shape[1])[None, :] < J[:, None]
            # log-space product for numerical safety; (B+W)/(2B) in [0,1]
            ratio = np.where(mask, (B + W) / (2.0 * B), 1.0)
            if np.any(ratio == 0.0):
                logp = np.where((ratio == 0.0).any(axis=1), -np.inf,
                                np.log(np.maximum(ratio, 1e-300)).sum(axis=1))
            else:
                logp = np.log(ratio).sum(axis=1)
            st.w_draws += int(J.sum())
        acc = np.log(rng.uniform(size=n)) < logp
        take = min(int(acc.sum()), n_out - filled)
        if take > 0:
            out[filled:filled + take] = x[acc][:take]
            filled += take
            st.accepts += take
        if filled == n_out:
            return out, st
    raise ForsBudgetExceeded(
        f"only {filled}/{n_out} accepts in {max_rounds} rounds (rate collapse?)")
