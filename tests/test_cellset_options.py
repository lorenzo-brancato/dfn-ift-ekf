"""The options used by the cell sets other than Chen2020 (dfnekf.cellset): per-node particle constants, the cathode
D_s(y) law frozen per step, and the implicit (end-of-step) closure. No external data: they run on Chen2020."""
import numpy as np
import jax.numpy as jnp

from dfnekf import params as prm, particles as pt
from dfnekf.rom import simulate
from dfnekf.step import DFN

DIMS = (10, 5, 10, 8)


def _run(P, dt=0.5, n=240, I=5.0):
    m = DFN(P, *DIMS, 2, dt, "uniform")
    z0 = m.init_state(prm.C0_N_100, prm.C0_P_100)
    out = simulate(m, z0, jnp.full(n, I), jnp.array([1.0, 1.0, 1.0]), k=None)
    return np.asarray(out["V"]), np.asarray(out["resF"])


def test_vector_D_consts_match_scalar():
    D = np.array([1e-14, 4e-15, 2.5e-14])
    Cv = pt.consts(5e-6, jnp.asarray(D), 0.5, 8, 2)
    for i, d in enumerate(D):
        Cs = pt.consts(5e-6, d, 0.5, 8, 2)
        for k in ("e", "gain", "rho", "W"):
            np.testing.assert_allclose(np.asarray(Cv[k])[..., i], np.asarray(Cs[k]), rtol=1e-12)


def test_dp_law_without_drop_is_constant_D():
    P = prm.base()
    V0, _ = _run(P)
    Q = dict(P, dp_law=dict(drop=0.0, shape=1.0, y0=0.4, y1=0.95))
    V1, _ = _run(Q)
    np.testing.assert_allclose(V1, V0, atol=1e-10)


def test_ift_jacobian_with_all_options():
    """The structured Jacobian (structjac.jac_full: implicit closure, D_p(y) columns, k_p sensitivity, entropic V) agrees
    with automatic differentiation through the converged implicit Newton closure."""
    import jax
    from dfnekf.structjac import Struct
    P = dict(prm.base(), implicit_closure=True, ds_on="p", R_contact=1e-3, entropic=(3.0, 4e-4, -1e-3, 0.0),
             dp_law=dict(drop=1.5, shape=1.4, y0=0.3, y1=0.95))
    dt = 5.0
    m = DFN(P, *DIMS, 2, dt, "4:1"); S = Struct(m)
    th = jnp.array([1.5, 0.95, 0.8])
    z0 = m.init_state(prm.C0_N_100, prm.C0_P_100)
    z = jax.jit(lambda: simulate(m, z0, jnp.full(300, 5.0), th, k=None)["zT"])()
    I = 5.0
    csn, csp, ce = m.surf(z)
    g = m.A.solve_robust(m.A.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=1e-12, imp=m.imp_coeffs(z, th))[0]
    Jz, Jt = [np.asarray(a) for a in jax.jit(S.jac_full)(z, I, th, g)]

    def phi_v(z_, th_):
        g_ = m.closure_newton(z_, I, th_, None, tol=1e-13)
        c1, c2, _ = m.surf(z_)
        return jnp.concatenate([m.advance(z_, g_, th_), m.A.V_of(g_, I, c1, c2)[None]])
    Jz_ref = np.asarray(jax.jacfwd(phi_v, 0)(z, th)); Jt_ref = np.asarray(jax.jacfwd(phi_v, 1)(z, th))
    assert np.linalg.norm(Jz - Jz_ref) / np.linalg.norm(Jz_ref) < 1e-9
    assert np.linalg.norm(Jt - Jt_ref) / np.linalg.norm(Jt_ref) < 1e-9


def test_implicit_closure_converges_to_explicit():
    """Same model, explicit vs implicit coupling: the difference is O(dt) and vanishes as dt shrinks."""
    P = prm.base()
    Q = dict(P, implicit_closure=True)
    d = []
    for dt, n in ((1.0, 120), (0.25, 480)):
        Ve, re = _run(P, dt, n)
        Vi, ri = _run(Q, dt, n)
        assert re.max() < 1e-8 and ri.max() < 1e-8
        d.append(np.abs(Vi[-1] - Ve[-1]))
    assert d[1] < 0.5 * d[0] + 1e-6
    assert d[1] < 2e-3
