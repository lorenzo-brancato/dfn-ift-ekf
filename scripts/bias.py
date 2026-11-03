"""Batch-fit bias of the grey-box model (orange bars of Fig. 5, Appendix D).

Fits (R_f, LAM, D_s) of the open-loop model (LLI fixed through the initial state) to the NOISE-FREE PyBaMM voltage of the static protocol P6Lx2 by
Levenberg-Marquardt / Gauss-Newton with exact sensitivities, starting from the true values. What is left is the structural bias of the reduced model for that
ageing state: the floor below which no filter can go on that reference. ``static_mc.py --aggregate`` reads these files to subtract the bias from the errors.

Output: <results>/bias_<state>.csv (one row), columns bias_Rf_pct, bias_LAM_pct, bias_Ds_pct (signed, % of the parameter range, fit minus true) and the fit itself.

  python scripts/bias.py --grid 4:1 --data data --results results --states th1,th2,th3,th4,th5      # ~85 s per state, one thread
  python scripts/bias.py --quick
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

args = cli.parser("Fig. 5 (orange bars), Appendix D", "batch-fit bias of the model against the noise-free PyBaMM reference").parse_args()
runtime.setup(args.core)
import time
import numpy as np, pandas as pd, jax, jax.numpy as jnp
from dfnekf import params as prm, refio, particles
from dfnekf.step import DFN
from dfnekf.sens import traj_sens
from dfnekf.montecarlo import QUICK_STEPS

PROTO, DT, NM = "P6Lx2", 0.5, 12
RNG = np.array([9.0, 0.3, 0.7]); LO = np.array([0.5, -0.05, 0.15]); HI = np.array([12.0, 0.4, 1.3])      # ageing range widths; bounds of the logit box


def fit(tag):
    m = DFN(prm.base(), 20, 10, 20, NM, 2, DT, args.grid)
    r = refio.load(PROTO, tag, period=DT, data_dir=args.data)
    true = np.array([r["Rf"], r["LAM"], r["Ds"]]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]]); I = jnp.asarray(r["I"]); Vref = r["V"]
    if args.quick: I, Vref = I[:QUICK_STEPS], Vref[:QUICK_STEPS]
    f = jax.jit(lambda th, z0, I: traj_sens(m, th, z0, I, par=("Rf", "LAM", "Ds"), return_V=True))

    def model(p):
        dV, V = f(jnp.array([p[0], 1 - p[1], p[2]]), z0, I); return np.asarray(dV), np.asarray(V)

    c0, t0 = time.process_time(), time.time()
    p = true.copy(); lam = 1e-3; hist = []
    dV, V = model(p); cost = np.sum((V - Vref) ** 2)
    for _ in range(12):
        res = V - Vref; Jn = dV * RNG[None, :]; H = Jn.T @ Jn; gv = Jn.T @ res; ok = False
        for _ in range(8):
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H)), gv) * RNG; pn = np.clip(p + step, LO, HI); dVn, Vn = model(pn); cn = np.sum((Vn - Vref) ** 2)
            if cn < cost: ok = True; break
            lam *= 10
        if not ok: break
        conv = np.max(np.abs(pn - p) / RNG) < 1e-6; p, dV, V, cost = pn, dVn, Vn, cn; lam = max(lam / 10, 1e-9); hist.append(cost)
        if conv: break
    bias = p - true; res = V - Vref; V0 = model(true)[1]
    return dict(grid=args.grid, Nm=NM, dt=DT, NLAM=particles.NLAM, theta_tag=tag, Rf_fit=p[0], LAM_fit=p[1], Ds_fit=p[2],
                bias_Rf_pct=100 * bias[0] / RNG[0], bias_LAM_pct=100 * bias[1] / RNG[1], bias_Ds_pct=100 * bias[2] / RNG[2],
                resid_rms_mV=1e3 * np.sqrt(np.mean(res ** 2)), resid_rms_at_true_mV=1e3 * np.sqrt(np.mean((V0 - Vref) ** 2)), iters=len(hist), cpu_s=time.process_time() - c0, wall_s=time.time() - t0)


if __name__ == "__main__":
    clock = cli.Clock(args, "bias.py")
    for tag in (["th2"] if args.quick else cli.states(args)):
        row = fit(tag)
        print(tag, {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in row.items() if k.startswith("bias") or k in ("iters", "wall_s")}, flush=True)
        cli.write_csv(pd.DataFrame([row]), args, f"bias_{tag}{'_quick' if args.quick else ''}.csv", note="batch-fit bias against the noise-free reference")
    clock.log()
