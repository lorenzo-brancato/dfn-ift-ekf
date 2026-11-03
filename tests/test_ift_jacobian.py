"""The implicit-function-theorem Jacobian of the step agrees with automatic differentiation through the converged Newton solver (Sec. II-C)."""
import numpy as np
import jax
import jax.numpy as jnp
from dfnekf import params as prm
from dfnekf.step import DFN
from dfnekf.rom import simulate
from dfnekf.structjac import Struct
from dfnekf.protocols import current_series


def test_ift_jacobian_matches_autodiff():
    dt = 0.5
    m = DFN(prm.base(), 10, 5, 10, 8, 2, dt, "4:1"); S = Struct(m)
    th = jnp.array([4.0, 0.9, 0.55]); I_all = current_series("P6L", dt)
    z0 = m.init_state(prm.C0_N_50, prm.C0_P_50)
    z = jax.jit(lambda: simulate(m, z0, jnp.asarray(I_all[:240]), th, k=None)["zT"])(); I = float(I_all[240])
    csn, csp, ce = m.surf(z); g = m.A.solve_robust(m.A.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=1e-12)[0]
    Jz, Jt = [np.asarray(a) for a in jax.jit(S.jac_full)(z, I, th, g)]

    def phi_v(z_, th_):
        g_ = m.closure_newton(z_, I, th_, None, tol=1e-13)
        return jnp.concatenate([m.advance(z_, g_, th_), m.A.V_of(g_, I)[None]])
    Jz_ref = np.asarray(jax.jacfwd(phi_v, 0)(z, th)); Jt_ref = np.asarray(jax.jacfwd(phi_v, 1)(z, th))
    assert np.linalg.norm(Jz - Jz_ref) / np.linalg.norm(Jz_ref) < 1e-9
    assert np.linalg.norm(Jt - Jt_ref) / np.linalg.norm(Jt_ref) < 1e-9
