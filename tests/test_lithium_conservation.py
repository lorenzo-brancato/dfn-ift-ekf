"""Lithium is conserved to machine precision whatever the convergence of the nonlinear rows of the closure (App. A)."""
import numpy as np
import jax
import jax.numpy as jnp
from dfnekf import params as prm
from dfnekf.step import DFN
from dfnekf.protocols import current_series


def test_lithium_drift():
    dt = 0.5
    m = DFN(prm.base(), 10, 5, 10, 8, 2, dt, "4:1"); th = jnp.array([4.0, 0.9, 0.55])
    I = jnp.asarray(current_series("P6L", dt)[:1000]); z0 = m.init_state(prm.C0_N_50, prm.C0_P_50)

    def body(z, i):
        csn, csp, ce = m.surf(z); g, _ = m.A.newton_fixed(m.A.guess(csn, csp, i, th), csn, csp, ce, i, th, k=1)       # ONE Newton iteration
        z2 = m.advance(z, g, th); return z2, m.lithium(z2, th)
    _, li = jax.jit(lambda: jax.lax.scan(body, z0, I))()
    li = np.asarray(li); li0 = float(m.lithium(z0, th))
    assert np.max(np.abs(li - li0)) / abs(li0) < 1e-11
