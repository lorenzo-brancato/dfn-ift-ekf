"""Spherical particle, exact modal integrator (ZOH on the flux) + Crank-Nicolson reference.
c(r,t) = c_avg + sum_n a~_n(t) * [sin(lam_n r/R) R/(r sin lam_n)] + coda;  tan(lam)=lam.
For each mode: a~_n' = -kap_n a~_n - (2/R) j/F, kap_n = D lam_n^2/R^2; c_surf = c_avg + sum a~_n + s_tail.
Tail (n > N_m): single effective mode s^+ = rho s - (1-rho) W j/F, rho = sum_n w_n e^{-kap_n dt} / W (exact DC gain
via sum 1/lam_n^2 = 1/10) -> corrects the truncated surface response (equivalent to the semi-infinite correction at short times)."""
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import brentq
jax.config.update("jax_enable_x64", True)

NLAM = 600                                          # number of modes in the exact sum behind the tail fit
_LAM = np.array([brentq(lambda x: np.tan(x) - x, n * np.pi + 1e-9, n * np.pi + np.pi / 2 - 1e-9) for n in range(1, NLAM + 1)])
LAM = jnp.asarray(_LAM)
F = 96485.33212


def consts(R, D, dt, Nm, K=2):
    """Step constants. Tail (n > Nm) in K geometric bands [Nm,2Nm),[2Nm,4Nm),...,[.., inf): one effective mode per band."""
    lam = LAM[:Nm]
    kap = D * lam ** 2 / R ** 2
    e = jnp.exp(-kap * dt)
    gain = (1 - e) * 2 / (R * kap)                       # a~ <- e a~ - gain * j/F
    edges = [Nm * 2 ** i for i in range(K)] + [NLAM]
    rho, W = [], []
    Wtot = (2 * R / D) * (0.1 - jnp.sum(1 / lam ** 2))   # total DC gain of the tail (exact)
    Wexp = 0.0
    for i in range(K):
        lo, hi = edges[i], edges[i + 1]
        lt = LAM[lo:hi]
        kt = D * lt ** 2 / R ** 2
        w = 2 / (R * kt)
        Wi = jnp.sum(w)
        if i == K - 1:                                   # open band: adds the remainder beyond NLAM
            Wi = Wtot - Wexp
        Wexp = Wexp + Wi
        rho.append(jnp.sum(w * jnp.exp(-kt * dt)) / Wi)
        W.append(Wi)
    return dict(e=e, gain=gain, rho=jnp.stack(rho), W=jnp.stack(W), kap=kap, dt=dt, R=R, K=K)


def step(state, j, C):
    """state = (c_avg [nodes], a [Nm,nodes], s [K,nodes]); j [A/m2] at the nodes (ZOH over dt)."""
    c_avg, a, s = state
    jF = j / F
    c_avg = c_avg - 3.0 / C["R"] * jF * C["dt"]
    a = C["e"][:, None] * a - C["gain"][:, None] * jF[None, :]
    s = C["rho"][:, None] * s - ((1 - C["rho"]) * C["W"])[:, None] * jF[None, :]
    return c_avg, a, s


def surf(state):
    c_avg, a, s = state
    return c_avg + a.sum(0) + s.sum(0)


# ---------------- Crank-Nicolson reference (spherical finite volumes, ZOH flux)
def crank_surface(c0, R, D, jseq, dt, nsub=4, Nr=400):
    """Spherical finite-volume reference, ZOH flux. CN with Rannacher start (2 implicit-Euler sub-steps at every
    flux jump) to remove the CN ringing on steps. Returns c_surf (extrapolated) and c_avg at the end of the step."""
    from scipy.linalg import solve_banded
    h = R / Nr
    re = np.arange(Nr + 1) * h
    vol = (re[1:] ** 3 - re[:-1] ** 3) / 3.0
    up = D * re[1:-1] ** 2 / h
    main = np.zeros(Nr); main[:-1] -= up; main[1:] -= up
    ds = dt / nsub

    def band(theta, ds_):
        ab = np.zeros((3, Nr))
        ab[1] = vol - theta * ds_ * main
        ab[0, 1:] = -theta * ds_ * up
        ab[2, :-1] = -theta * ds_ * up
        return ab

    ab_cn, ab_be = band(0.5, ds), band(1.0, ds / 2)
    c = np.full(Nr, float(c0))
    out_s, out_a = [], []
    jprev = None
    for j in jseq:
        flux_in = -(j / F) * R ** 2
        rann = (jprev is None) or (j != jprev)
        for m in range(nsub):
            if rann and m < 4:                             # 4 implicit half steps (= 2 sub-steps)
                dsl = ds / 2
                for _ in range(2 if m < 2 else 0):
                    rhs = vol * c; rhs[-1] += dsl * flux_in
                    c = solve_banded((1, 1), ab_be, rhs)
                if m < 2:
                    continue
            rhs = vol * c + 0.5 * ds * (main * c)
            rhs[:-1] += 0.5 * ds * up * c[1:]
            rhs[1:] += 0.5 * ds * up * c[:-1]
            rhs[-1] += ds * flux_in
            c = solve_banded((1, 1), ab_cn, rhs)
        jprev = j
        out_s.append(c[-1] - (h / 2) * j / (F * D))
        out_a.append(np.sum(c * vol) * 3 / R ** 3)
    return np.array(out_s), np.array(out_a)
