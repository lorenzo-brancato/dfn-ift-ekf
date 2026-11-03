"""Shared machinery of the charge and discharge scripts: one grey-box EKF run with the state guard plus the SPMe unscented-filter baseline on a PyBaMM reference.

``GreyBoxRunner`` owns the jitted guarded filter of one ageing state (compiled once, reused over seeds); ``run_pair`` returns the filter histories and the SPMe outputs
of one seed with identical initial offsets, noise, P0 and Q (``Case.offsets``, ``Case.x0_P0_Q``, ``np.random.default_rng(seed)`` noise of 2 mV).

Plating potential and reaction overpotential of the reference (``reference_interface``) are evaluated at the anode/separator face, extrapolated linearly from the
last two anode cells, exactly as ``step.DFN.interface`` does for the filter (this is what ``step.interface_ref`` implements).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from .experiment import Case, run_spme
from .ekf import GuardedEKF
from .step import interface_ref

# fields of the guarded filter history stored in the per-run trajectory files
TRAJ_KEYS = ("vals", "eta", "ppl", "ce_min", "F_warm", "F_newton", "F_next", "csn_max", "csn_min", "rejected", "innov")
FIXED = dict(Rf=1.0, LAM=0.0, Ds=1.0)          # LLI is fixed (taken from the reference initial state); R_f, LAM_n, D_s are estimated


def truncate(case, n):
    """Keep only the first ``n`` steps of a case (smoke tests). The reference fields keep their full length; callers index them with ``[:case.N]``."""
    case.N = int(n); case.I = case.I[:n]; case.Vtrue = case.Vtrue[:n]
    return case


def reference_interface(case):
    """(N, 2) array: [eta_lit, phi_pl] of the reference at the anode/separator face (extrapolated from the last two anode cells)."""
    r = case.r
    return np.array([interface_ref(r, i, case.P0, r["Rf"]) for i in range(case.N)])


class GreyBoxRunner:
    """Guarded grey-box EKF (closure: converged Newton, state guard on) for one ageing state / protocol."""

    def __init__(self, case):
        self.case = case
        self.ek = GuardedEKF(case.m, closure="newton", coords=case.coords, fixed=dict(FIXED))
        ek = self.ek
        self.fn = jax.jit(lambda x0, P0d, Qd, y, I: ek.run_g(x0, P0d, Qd, (2e-3) ** 2, I, y, guard=True))

    def run(self, seed):
        """One seed: returns (H, cpu_s) with H the dict of numpy histories."""
        case = self.case
        th0 = case.offsets(seed); x0, P0d, Qd = case.x0_P0_Q(th0)
        y = case.Vtrue + 2e-3 * np.random.default_rng(seed).normal(size=case.N)
        c0 = time.process_time()
        H = self.fn(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), jnp.asarray(y), case.I)
        jax.block_until_ready(H["vals"]); H = {k: np.asarray(v) for k, v in H.items()}
        return H, time.process_time() - c0


def run_pair(runner, seed):
    """Grey-box EKF with guard and SPMe UKF on the same case and seed. Returns (H, out_spme, cpu_ekf_s, cpu_spme_s)."""
    H, cpu_e = runner.run(seed)
    c1 = time.process_time(); _, out_s = run_spme(runner.case, seed); cpu_s = time.process_time() - c1
    return H, out_s, cpu_e, cpu_s


def theta_end_errors(case, H):
    """Final parameter errors in % of the ageing range for (R_f, LAM_n, D_s)."""
    v = H["vals"]; return 100 * np.abs(v[-1] - case.true_vals) / case.range
