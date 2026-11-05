"""Shared helpers of the Monte Carlo scripts (``static_mc.py``, ``twin_nees.py``): one filter run with the conventions of the paper, and the statistics used by the tables.

Conventions (identical for every run): measurement noise ``default_rng(seed)`` with sigma_V = 2 mV, initial offsets ``default_rng(200 + seed)`` uniform in +-40 % of
the parameter range (see ``experiment.Case.offsets``), initial covariance 30 % of the range, random-walk process noise 1e-4 x range x sqrt(dt).
Import this module only after ``runtime.setup()`` (it imports jax).
"""
import time
import numpy as np
import jax, jax.numpy as jnp
from .experiment import summarize
from .ukf import UKF

RANGE_W = {"Rf": 9.0, "LAM": 0.3, "Ds": 0.7}                  # width of the ageing range of each parameter (the unit of the '% of range' errors)
SIGMA_V = 2e-3
QUICK_STEPS = 1200                                            # --quick: 600 s of the protocol


def grid_label(grid):
    return "20-10-20_" + ("4to1" if grid == "4:1" else "uniform") + "_Nm12"


def truncate(case, n):
    """Restrict a Case to its first ``n`` samples (smoke tests)."""
    case.N = int(n); case.I = case.I[:n]; case.Vtrue = np.asarray(case.Vtrue)[:n]
    return case


def iters_label(closure_iters):
    return "conv" if closure_iters is None else str(int(closure_iters))


def parse_iters(s):
    """'--ukf-iters' value: an integer number of Newton iterations per sigma point, or 'conv' (Newton to convergence at every sigma point)."""
    return None if str(s).lower() in ("conv", "none", "converged") else int(s)


_JIT = {}


def run_one(case, filt, seed, closure_iters=1, p0_scale_rf=1.0, init_error="absolute", chunk=0):
    """One filter run on ``case`` (PyBaMM reference or twin). Returns (summary dict, trajectories dict).

    filt: 'ekf' or 'ukf' (``closure_iters`` Newton iterations per sigma point; None = converged).
    p0_scale_rf: factor on the initial standard deviation of R_f only (2.0 = initial variance x4).
    chunk: UKF only: steps per chunk of the (non-jitted) outer loop, which prints the progress after every chunk; 0 (default) = one jitted scan without progress output; a chunked loop changes the NEES at the 1e-3 relative level (different compilation), so it is not used where results are compared.
    init_error: 'absolute' (offset uniform in +-40 % of the range) or 'relative' (R_f_0 = R_f,true x exp(u), u in +-ln 2 from the same random number, clipped to [0.6, 12];
                the other parameters keep the absolute offsets).
    """
    ek = case.ek; key = (filt, id(case), closure_iters, case.N); chunked = filt == "ukf" and bool(chunk)
    if not chunked and key not in _JIT:
        R = SIGMA_V ** 2
        if filt == "ekf": f = lambda x0, P0d, Qd, y, xt, I: ek.run(x0, P0d, Qd, R, I, y, xt)
        else:
            uk = UKF(ek, closure_iters=closure_iters); f = lambda x0, P0d, Qd, y, xt, I: uk.run(x0, P0d, Qd, R, I, y, xt)
        _JIT[key] = jax.jit(f)
    th0 = case.offsets(seed)
    if init_error == "relative":
        draw = np.random.default_rng(200 + seed).uniform(-0.4, 0.4, case.nth) * case.range
        u = np.log(2.0) * draw[0] / (0.4 * case.range[0]); th0 = th0.copy(); th0[0] = float(np.clip(case.true_vals[0] * np.exp(u), 0.6, 12.0))
    elif init_error != "absolute": raise ValueError(init_error)
    x0, P0d, Qd = case.x0_P0_Q(th0); P0d = np.array(P0d)
    if p0_scale_rf != 1.0: P0d[case.nz + case.coords.index("Rf")] *= p0_scale_rf
    y = case.Vtrue + SIGMA_V * np.random.default_rng(seed).normal(size=case.N)
    c0, w0 = time.process_time(), time.time()
    if chunked:
        def progress(k, N, Hc, wall_s):
            print(f"seed {seed} step {k}/{N} wall {wall_s:.0f} s innov rms {1e3 * float(np.sqrt(np.mean(np.asarray(Hc['innov']) ** 2))):.2f} mV nees2 {float(np.mean(np.asarray(Hc['nees2']))):.2f} "
                  f"max sigma-point residual {float(np.max(np.asarray(Hc['resMax']))):.1e}", flush=True)
        H = UKF(ek, closure_iters=closure_iters).run(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), SIGMA_V ** 2, case.I, jnp.asarray(y), jnp.asarray(case.x_true()), chunk=chunk, progress=progress)
    else: H = _JIT[key](jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), jnp.asarray(y), jnp.asarray(case.x_true()), case.I)
    jax.block_until_ready(H["vals"])
    wall = time.time() - w0; cpu = time.process_time() - c0
    res, out = summarize(case, filt, seed, th0, H, wall, "")
    res.update(cpu_s=cpu, grid=grid_label(case.grid_name))
    if filt == "ukf": res.update(closure_iters=iters_label(closure_iters), alpha=1.0)
    return res, out


def ci95(x):
    """Half-width of the 95 % confidence interval of the mean (1.96 s / sqrt(n))."""
    x = np.asarray(x, float); return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def p90(x):
    return float(np.quantile(np.asarray(x, float), 0.9))
