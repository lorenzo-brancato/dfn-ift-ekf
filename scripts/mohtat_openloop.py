"""Open-loop check of the grey-box DFN on a measured Mohtat cell with an exported parameter set (no filter).

    python scripts/mohtat_openloop.py --cell C01 --tests C20,VAL,HPPC --results results/mohtat

Data: the tests prepared by ParamIdentification, in data/exp_Mohtat/tests (DB_REAL_<cell>_<test>_BOL/cycles.csv: t_s, I_A > 0 on
discharge = mean over [t, t + dt), V_V and T_C at mid-interval, eis = 1 on samples excluded from the residual).
Initial state at uniform concentrations, placed as ParamIdentification does (identification.initial_charges): a test
that starts under current starts at the top of the cell's step-1 window (x100, y100); one that starts at rest (C20, at
the bottom) at the charge q0 that inverts the window's OCV at its first voltage. The run is at the test's mean temperature (constant; see dfnekf.cellset)
with the converged Newton closure, model step = the sampling period / n_sub <= --dt-max. Reported: RMSE over the samples with
eis = 0, and over those with V >= V_min (3.0 V) too; a figure and an .npz per test.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import runtime  # noqa: E402,F401  (thread settings before JAX)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dfnekf import cellset  # noqa: E402
from dfnekf.rom import simulate  # noqa: E402
from dfnekf.step import DFN  # noqa: E402
from dfnekf import datasets  # noqa: E402

V_MIN = 3.0


def load_test(data_dir, cell, test):
    d = os.path.join(data_dir, "DB_REAL_%s_%s_BOL" % (cell, test))
    df = pd.read_csv(os.path.join(d, "cycles.csv"))
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    return df, cfg


def discharge_leg(df, i_rest=0.02):
    """The C/20 test's discharge leg as the identification scores it ("from_discharge"): from the last rest sample
    before the first discharge that follows the top of charge, to the end."""
    I, V = df.I_A.values, df.V_V.values
    k_top = int(np.argmax(V))
    k_dis = k_top + int(np.argmax(I[k_top:] > i_rest))
    k0 = k_dis - 1 if k_dis > 0 and abs(I[k_dis - 1]) <= i_rest else k_dis
    out = df.iloc[k0:].reset_index(drop=True)
    out["t_s"] = out.t_s - out.t_s.iloc[0]
    return out


def run(cs, cell, df, dt, dims, grid, dt_max, q0):
    """Model step dt / n_sub <= dt_max (the explicit surface/flux coupling is unstable at 10 s with this set's fast
    anode kinetics); the current is held over the sample interval and V is read at its midpoint (data convention)."""
    T_K = float(df.T_C.mean()) + 273.15
    P = cellset.params_for(cs, cell, T_K)
    n_sub = max(1, int(np.ceil(dt / dt_max - 1e-9)))
    m = DFN(P, *dims, 2, dt / n_sub, grid)
    z0 = m.init_state(*cellset.c0_at(cs, cell, q0))
    th = jnp.array([1.0, 1.0, 1.0])
    t0 = time.time()
    I = jnp.repeat(jnp.asarray(df.I_A.values), n_sub)
    out = jax.jit(lambda: simulate(m, z0, I, th, k=None))()
    pick = lambda v: np.asarray(v).reshape(len(df), n_sub, *np.shape(v)[1:])[:, n_sub // 2]
    return dict(V=pick(out["V"]), x_s=pick(out["cs_n"]).mean(1) / P["cmax_n"], y_s=pick(out["cs_p"]).mean(1) / P["cmax_p"],
                resF=np.asarray(out["resF"]).reshape(len(df), n_sub).max(1), T_K=T_K, n_sub=n_sub, wall_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=datasets.cellset_path("exp_Mohtat"))
    ap.add_argument("--data", default=datasets.MOHTAT_TESTS, help="folder of the DB_REAL_<cell>_<test>_BOL tests")
    ap.add_argument("--cell", default="C01")
    ap.add_argument("--tests", default="C20,C20:dis,VAL,HPPC", help="C20:dis = the C/20 discharge leg alone")
    ap.add_argument("--q-shift", type=float, default=0.0,
                    help="[A.h] moves the initial state up the window (the fit's C/20 axis_shift, e.g. 0.058 for C01)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "mohtat"))
    ap.add_argument("--grid", default="4:1")
    ap.add_argument("--dims", default="20,10,20,12", help="Nn,Ns,Np,Nm")
    ap.add_argument("--dt-max", type=float, default=2.0, help="largest model step [s]")
    a = ap.parse_args()
    cs = cellset.load(a.set)
    dims = tuple(int(v) for v in a.dims.split(","))
    os.makedirs(a.results, exist_ok=True)
    ref = cs["source"].get("rmse_per_test_mV") or {}
    rows = []
    for test in a.tests.split(","):
        df, cfg = load_test(a.data, a.cell, test.split(":")[0])
        if test == "C20:dis":                           # the discharge leg alone, rested at the top (as in the fit)
            df = discharge_leg(df)
        q0 = cellset.q_start(cs, a.cell, float(df.I_A.values[0]), float(df.V_V.values[0])) - a.q_shift
        r = run(cs, a.cell, df, float(cfg["dt_s"]), dims, a.grid, a.dt_max, q0)
        Vm = df.V_V.values
        ok = (df.eis.values == 0) & np.isfinite(r["V"])
        err = (r["V"] - Vm) * 1e3
        rm = float(np.sqrt(np.mean(err[ok] ** 2)))
        okv = ok & (Vm >= V_MIN)
        rmv = float(np.sqrt(np.mean(err[okv] ** 2)))
        nbad = int(np.sum(~np.isfinite(r["V"]))) + int(np.sum(r["resF"] > 1e-8))
        row = dict(cell=a.cell, test=test, q0_Ah=q0, n=len(df), dt_s=cfg["dt_s"], n_sub=r["n_sub"], T_C=r["T_K"] - 273.15, rmse_mV=rm,
                   rmse_above_vmin_mV=rmv, max_abs_mV=float(np.max(np.abs(err[ok]))), mean_mV=float(np.mean(err[ok])),
                   lm3c_rmse_mV=ref.get("%s %s" % (a.cell, test)), bad_steps=nbad, wall_s=r["wall_s"])
        if test == "C20:dis":                           # the fit's C/20 score starts at 3.9 A.h discharged (its knee)
            qd = np.cumsum(df.I_A.values) * float(cfg["dt_s"]) / 3600.0
            okq = ok & (qd >= 3.9)
            row["rmse_from_3p9Ah_mV"] = float(np.sqrt(np.mean(err[okq] ** 2)))
            row["lm3c_rmse_mV"] = ref.get("%s C20" % a.cell)
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
        np.savez(os.path.join(a.results, "openloop_%s_%s.npz" % (a.cell, test.replace(":", "_"))), t=df.t_s.values, I=df.I_A.values,
                 V_meas=Vm, V_dfn=r["V"], T_C=df.T_C.values, x_s=r["x_s"], y_s=r["y_s"], resF=r["resF"])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
            th = df.t_s.values / 3600
            ax[0].plot(th, Vm, "k", lw=1, label="measured")
            ax[0].plot(th, r["V"], "C3", lw=1, ls="--", label="DFN, %s" % cs["id"])
            ax[0].set_ylabel("V [V]"); ax[0].legend(frameon=False)
            ax[1].plot(th, np.where(ok, err, np.nan), "C0", lw=0.8)
            ax[1].set_ylabel("DFN - meas. [mV]"); ax[1].axhline(0, color="0.6", lw=0.5)
            ax[2].plot(th, df.I_A.values, "0.3", lw=0.8); ax[2].set_ylabel("I [A] (+ dis.)"); ax[2].set_xlabel("t [h]")
            ax[0].set_title("%s %s  |  RMSE %.1f mV (V >= %.1f V: %.1f mV)  |  T = %.1f C"
                            % (a.cell, test, rm, V_MIN, rmv, r["T_K"] - 273.15), fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(a.results, "openloop_%s_%s.png" % (a.cell, test.replace(":", "_"))), dpi=130)
            plt.close(fig)
        except ImportError:
            pass
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "openloop_%s.csv" % a.cell), index=False)


if __name__ == "__main__":
    main()
