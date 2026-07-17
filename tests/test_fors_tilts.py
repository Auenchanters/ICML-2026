"""FORS exact-law GOF (Thm 3.1); Gaussian-tilt instantiation (Thm 3.3); prox."""
import numpy as np
from scipy.stats import chisquare

from fors.fors import fors_batch, fors_scalar, ForsBudgetExceeded
from fors.tilts import prox_newton, rgo_fors, tilt_mean_w_quad, true_tilt_grid
from fors.targets import logcosh_potential, quadratic_potential


def _fors_known_tilt(n, B, c_noise, rng, w=lambda x: 0.6 * np.sin(2 * x)):
    """FORS with q=N(0,1) and estimator W_x = w(x) + U[-c, c] (mean-zero noise)."""
    def propose_n(m, rng):
        return rng.standard_normal((m, 1))

    def draw_w(x, J, rng):
        jm = max(int(J.max()), 1)
        return w(x) + rng.uniform(-c_noise, c_noise, size=(len(x), jm))

    return fors_batch(propose_n, draw_w, B, n, rng)


def test_fors_exact_law_gof():
    """Thm 3.1(a): output density ∝ q e^w. Chi-square GOF on 60 bins, n=2e5."""
    rng = np.random.default_rng(1)
    n = 200_000
    out, st = _fors_known_tilt(n, B=1.2, c_noise=0.5, rng=rng)
    x = out.ravel()
    # ground truth by quadrature
    g = np.linspace(-5, 5, 4001)
    dens = np.exp(-g**2 / 2 + 0.6 * np.sin(2 * g))
    dens /= np.trapezoid(dens, g)
    edges = np.linspace(-3.5, 3.5, 61)
    cnt, _ = np.histogram(x, bins=edges)
    from scipy.integrate import cumulative_trapezoid
    F = cumulative_trapezoid(dens, g, initial=0.0)
    probs = np.diff(np.interp(edges, g, F))
    inside = (x > -3.5) & (x < 3.5)
    exp_cnt = probs / probs.sum() * inside.sum()
    stat, p = chisquare(cnt, exp_cnt)
    assert p > 1e-4, f"GOF rejected: stat={stat:.1f}, p={p:.2e}"


def test_fors_acceptance_identity():
    """Thm 3.1(b): overall acceptance rate = E_q[e^{w(x)-B}] (4-sigma band)."""
    rng = np.random.default_rng(2)
    B = 1.2
    out, st = _fors_known_tilt(100_000, B=B, c_noise=0.5, rng=rng)
    g = np.linspace(-8, 8, 8001)
    q = np.exp(-g**2 / 2) / np.sqrt(2 * np.pi)
    a_true = np.trapezoid(q * np.exp(0.6 * np.sin(2 * g) - B), g)
    se = np.sqrt(a_true * (1 - a_true) / st.proposals)
    assert abs(st.accept_rate - a_true) < 4 * se


def test_fors_scalar_matches_and_counts():
    """Scalar reference runs, counts draws, and draw counts respect Thm 3.1(c)."""
    rng = np.random.default_rng(3)
    B = 1.0

    def propose(rng):
        return rng.standard_normal(1)

    def draw_w(x, n, rng):
        return 0.6 * np.sin(2 * x[0]) + rng.uniform(-0.4, 0.4, size=n)

    draws = []
    for _ in range(2000):
        _, st = fors_scalar(propose, draw_w, B, rng)
        draws.append(st.draws_per_accept[-1])
    draws = np.array(draws)
    delta = 0.05
    bound = 3 * B * np.exp(2 * B) * np.log(2 / delta)
    assert np.mean(draws > bound) <= delta


def test_fors_raises_on_collapse():
    """Hard cap raises (never silently truncates) when acceptance ~ 0."""
    rng = np.random.default_rng(4)
    B = 1.0

    def propose_n(m, rng):
        return rng.standard_normal((m, 1))

    def draw_w(x, J, rng):
        jm = max(int(J.max()), 1)
        return np.full((len(x), jm), -B)     # e^{E W - B} = e^{-2B}, tiny-ish

    # max_rounds=1 with tiny batch forces failure
    try:
        fors_batch(propose_n, draw_w, B, 5000, rng, max_rounds=1, batch=8)
    except ForsBudgetExceeded:
        pass
    else:
        raise AssertionError("expected ForsBudgetExceeded")


def test_prox_newton_residual():
    """|x - x0 + eta f'(x)| <= 1e-13 * scale on a grid of x0 (logcosh)."""
    pot = logcosh_potential()
    x0 = np.linspace(-8, 8, 41)
    eta = 0.3
    xp = prox_newton(pot.df, pot.d2f, x0, eta)
    res = xp - x0 + eta * pot.df(xp)
    assert np.max(np.abs(res) / np.maximum(1, np.abs(x0))) < 1e-13


def test_tilt_quadrature_identity_quadratic():
    """Section-3 pre-clip identity: nu ∝ q e^{E W} exactly. For quadratic f the
    check is closed-form: E[W](x) - (log nu - log q)(x) must be x-constant."""
    pot = quadratic_potential(s2=1.5)
    x0, eta, B = 2.0, 0.4, 1e9         # huge B: clip never binds
    xp = prox_newton(pot.df, pot.d2f, np.array([x0]), eta)[0]
    xhat = x0 - eta * pot.df(xp)
    xs = np.linspace(-2, 4, 31)
    m = tilt_mean_w_quad(xs, xhat, xp, pot.df, eta, B, n_r=48, n_z=48)
    log_nu_minus_log_q = (-pot.f(xs) - (xs - x0) ** 2 / (2 * eta)
                          + (xs - xhat) ** 2 / (2 * eta))
    diff = m - log_nu_minus_log_q
    assert np.max(np.abs(diff - diff.mean())) < 1e-9


def test_rgo_fors_gaussian_moments():
    """RGO via FORS on quadratic f = exact Gaussian: moments within 5 sigma."""
    rng = np.random.default_rng(5)
    s2, eta, x0 = 1.0, 0.1, 1.5
    pot = quadratic_potential(s2=s2)
    mean_true = x0 * s2 / (s2 + eta)
    var_true = eta * s2 / (s2 + eta)
    n = 50_000
    out, st = rgo_fors(pot, x0, eta, B=2.0, n_out=n, rng=rng)
    se_m = np.sqrt(var_true / n)
    assert abs(out.mean() - mean_true) < 5 * se_m
    assert abs(out.var() - var_true) < 5 * var_true * np.sqrt(2 / n)
