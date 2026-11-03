"""A short EKF run (twin experiment, 4:1 grid, 300 steps) against values stored from the code that produced the manuscript (tests/data/expected_short_run.npz)."""
import os
import numpy as np
import jax
import jax.numpy as jnp
from dfnekf import params as prm
from dfnekf.step import DFN
from dfnekf.rom import simulate
from dfnekf.ekf import EKF, GuardedEKF
from dfnekf.protocols import current_series

EXP = np.load(os.path.join(os.path.dirname(__file__), "data", "expected_short_run.npz"))


def _setup():
    dt = 0.5; NS = 300
    m = DFN(prm.base(), 20, 10, 20, 12, 2, dt, "4:1"); I = jnp.asarray(current_series("P6L", dt)[:NS])
    true = np.array([4.0, 0.1, 0.55]); th = jnp.array([true[0], 1 - true[1], true[2]]); z0 = m.init_state(prm.C0_N_50, prm.C0_P_50)
    V = np.asarray(jax.jit(lambda: simulate(m, z0, I, th, k=None)["V"])()); y = V + 2e-3 * np.random.default_rng(3).normal(size=NS)
    rng = np.array([9.0, 0.3, 0.7]); th0 = np.clip(true + np.random.default_rng(200).uniform(-0.4, 0.4, 3) * rng, [1, 0, 0.3], [10, 0.3, 1.0])
    kw = dict(closure="newton", coords=("Rf", "LAM", "Ds"), fixed=dict(Rf=1.0, LAM=0.0, Ds=1.0)); ek = EKF(m, **kw); gk = GuardedEKF(m, **kw)
    c0 = ek.coord_of(th0); dc = np.asarray(ek.dval_dc(jnp.asarray(c0))); nel = m.nel; x0 = np.concatenate([np.asarray(z0), c0])
    P0d = np.concatenate([np.full(nel, 10.0), np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0), 0.3 * rng / dc]); Qd = np.concatenate([np.zeros(m.nz), 1e-4 * rng * np.sqrt(dt) / dc])
    xt = np.tile(np.concatenate([np.zeros(m.nz), ek.coord_of(true)]), (NS, 1))
    return m, ek, gk, I, V, y, x0, P0d, Qd, xt


def test_short_run_matches_stored():
    # tolerance 1e-8: the same code gives identical numbers within one process, but across processes XLA CPU summation order can differ in the last bits (up to ~4e-10 here after the filter amplifies it)
    m, ek, gk, I, V, y, x0, P0d, Qd, xt = _setup()
    assert np.max(np.abs(V - EXP["V"])) < 1e-10
    H = jax.jit(lambda *a: ek.run(a[0], a[1], a[2], 4e-6, a[3], a[4], a[5]))(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), I, jnp.asarray(y), jnp.asarray(xt))
    for k in ["vals", "ppl", "eta", "nees"]: assert np.max(np.abs(np.asarray(H[k]) - EXP[k])) < 1e-8, k
    G = jax.jit(lambda *a: gk.run_g(a[0], a[1], a[2], 4e-6, a[3], a[4], guard=True))(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), I, jnp.asarray(y))
    assert np.max(np.abs(np.asarray(G["vals"]) - EXP["g_vals"])) < 1e-8 and np.max(np.abs(np.asarray(G["sd_vals"]) - EXP["g_sd"])) < 1e-8
    assert not np.asarray(G["rejected"]).any() and not EXP["g_rej"].any()
