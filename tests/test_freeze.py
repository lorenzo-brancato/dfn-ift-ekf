"""The CV-freeze option of the guarded EKF (``make_step_g(freeze=True)``, used by ``ageing_track.py --freeze``): with an all-zero code it is the paper's filter bit for bit;
code 1 (parameter rows of the gain zero) and code 2 (no measurement update) keep the parameter estimates constant, and the frozen steps are reported."""
import numpy as np
import jax
import jax.numpy as jnp
from dfnekf import params as prm
from dfnekf.step import DFN
from dfnekf.ekf import GuardedEKF
from dfnekf.rom import simulate


def _run(code):
    m = DFN(prm.base(), 6, 3, 6, 4, 2, 1.0, "uniform"); ek = GuardedEKF(m, closure="newton", coords=("Rf", "Ds"))
    N = 120; I = jnp.asarray(np.where(np.arange(N) < 60, 5.0, -2.5)); th = jnp.array([2.0, 1.0, 0.8])
    z0 = m.init_state(prm.C0_N_50, prm.C0_P_50); V = np.asarray(simulate(m, z0, I, th, k=None)["V"]); y = V + 2e-3 * np.random.default_rng(0).normal(size=N)
    c0 = ek.coord_of(np.array([1.5, 0.9])); dc = np.asarray(ek.dval_dc(jnp.asarray(c0))); rng = np.array([9.0, 0.7])
    x0 = np.concatenate([np.asarray(z0), c0]); nz = m.nz
    P0d = np.concatenate([np.full(m.nel, 10.0), np.full(m.Nm * m.nel, 1.0), np.full(m.K * m.nel, 1.0), np.full(m.N, 3.0), 0.3 * rng / dc]); Qd = np.concatenate([np.zeros(nz), 1e-4 * rng / dc])
    I_next = jnp.concatenate([I[1:], I[-1:]]); frz = code is not None
    step, scale = ek.make_step_g(x0, P0d, Qd, (2e-3) ** 2, guard=True, freeze=frz)
    g0 = ek.init_g(z0, I[0], ek.th_of(jnp.asarray(c0)))
    inp = (I, jnp.asarray(y), I_next) + ((jnp.asarray(code),) if frz else ())
    (_, P, _), H = jax.lax.scan(step, (jnp.zeros(ek.n), jnp.diag((jnp.asarray(P0d) / scale) ** 2), g0), inp)
    return {k: np.asarray(v) for k, v in H.items()}


def test_freeze_codes():
    base = _run(None); zero = _run(np.zeros(120, np.int32))
    assert np.array_equal(base["vals"], zero["vals"]), "an all-zero freeze code must reproduce the unfrozen filter exactly"
    code = np.zeros(120, np.int32); code[30:50] = 1; code[80:100] = 2
    f = _run(code)
    assert np.array_equal(f["frozen"], code)
    for a, b in ((30, 50), (80, 100)):                                      # the estimate output at a frozen step equals the one before it
        assert np.allclose(f["vals"][a:b], f["vals"][a - 1], rtol=0, atol=1e-12), (a, b)
    assert not np.allclose(f["vals"][50:60], f["vals"][49], rtol=0, atol=1e-6)      # the update resumes after the hold
    assert np.all(np.isfinite(f["vals"]))
