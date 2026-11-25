"""Open loop of the grey-box DFN on a measured Mohtat test with the MEASURED temperature, sample by sample (no filter).

    .venv/Scripts/python.exe scripts/mohtat_phm/openloop_T.py --cells C08,C11 --tests VAL

scripts/mohtat_openloop.py runs a test at its mean temperature; the cold 1.5-2C tests self-heat by ~6 K, and at -5 degC
the model is so sensitive to T (fit B, C08 VAL: 27 mV at +3 degC, a collapse at -3 degC) that a constant T is not a fair
check. Here every sample is simulated at its own temperature, rounded to a bin of --dT K (one DFN, i.e. one set of
parameters from cellset.params_for, per bin, built and compiled once): kinetics, diffusion, electrolyte, chi and the
entropic term all follow the sample's T. The state is carried across bins. MPM-LE (mpmle srcs/ms_model) does the same for
the kinetics and the electrolyte, and holds the particles' diffusion at the mean T of windows of constant current.
Same initial state and RMSE convention as mohtat_openloop.py; results/mohtat_phm/openloop_T/.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dfnekf import runtime  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dfnekf import cellset, datasets  # noqa: E402
from dfnekf.rom import simulate  # noqa: E402
from dfnekf.step import DFN  # noqa: E402
import mohtat_openloop as mo  # noqa: E402

MPMLE_HELDOUT = os.path.join(os.environ.get("MPMLE_ROOT", r"C:\Users\alexa\OneDrive - Politecnico di Milano\Desktop"
                                                          r"\Batteries\Python codes\MIONet\mpmle_identification"),
                             "runs", "studies", "mohtat_B_fin3", "heldout_val_rmse.csv")


def run_T(cs, cell, df, dt, dims, grid, dt_max, q0, dT):
    n_sub = max(1, int(np.ceil(dt / dt_max - 1e-9)))
    th = jnp.array([1.0, 1.0, 1.0])
    bins = np.round(df.T_C.values / dT) * dT
    fns = {}

    def fn_of(b):
        if b not in fns:
            P = cellset.params_for(cs, cell, b + 273.15)
            m = DFN(P, *dims, 2, dt / n_sub, grid)
            f = jax.jit(lambda z, I: (lambda o: (o["zT"], o["V"], o["resF"]))(simulate(m, z, I, th, k=None)))
            fns[b] = (m, f)
        return fns[b]

    m0, _ = fn_of(bins[0])
    z = m0.init_state(*cellset.c0_at(cs, cell, q0))
    V = np.full(len(df), np.nan)
    res = np.zeros(len(df))
    t0 = time.time()
    for k, (I, b) in enumerate(zip(df.I_A.values, bins)):
        _, f = fn_of(b)
        z, v, r = f(z, jnp.full(n_sub, float(I)))
        V[k] = float(v[n_sub // 2])
        res[k] = float(jnp.max(r))
    return dict(V=V, resF=res, n_sub=n_sub, n_bins=len(fns), wall_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(ROOT, "data", "exp_Mohtat", "params", "mohtat_B_fin3.json"))
    ap.add_argument("--data", default=datasets.MOHTAT_TESTS)
    ap.add_argument("--cells", default="C08")
    ap.add_argument("--tests", default="VAL")
    ap.add_argument("--dT", type=float, default=0.5, help="temperature bin [K]")
    ap.add_argument("--dims", default="20,10,20,12")
    ap.add_argument("--grid", default="4:1")
    ap.add_argument("--dt-max", type=float, default=2.0)
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "mohtat_phm", "openloop_T"))
    a = ap.parse_args()
    cs = cellset.load(a.set)
    dims = tuple(int(v) for v in a.dims.split(","))
    os.makedirs(a.results, exist_ok=True)
    ref = {}
    if os.path.exists(MPMLE_HELDOUT):
        ref = {r["cell"]: float(r["rmse_mV"]) for r in csv.DictReader(open(MPMLE_HELDOUT))}
    ref.update({c: v for k, v in (cs["source"].get("rmse_per_test_mV") or {}).items()
                for c in [k.split()[0]] if k.endswith(" VAL")})
    for cell in a.cells.split(","):
        for test in a.tests.split(","):
            df, cfg = mo.load_test(a.data, cell, test)
            q0 = cellset.q_start(cs, cell, float(df.I_A.values[0]), float(df.V_V.values[0]))
            r = run_T(cs, cell, df, float(cfg["dt_s"]), dims, a.grid, a.dt_max, q0, a.dT)
            Vm = df.V_V.values
            ok = (df.eis.values == 0) & np.isfinite(r["V"])
            err = (r["V"] - Vm) * 1e3
            row = dict(cell=cell, test=test, T_min_C=float(df.T_C.min()), T_max_C=float(df.T_C.max()),
                       rmse_mV=float(np.sqrt(np.mean(err[ok] ** 2))), max_abs_mV=float(np.max(np.abs(err[ok]))),
                       mean_mV=float(np.mean(err[ok])), mpmle_rmse_mV=ref.get(cell) if test == "VAL" else None,
                       bad_steps=int(np.sum(~np.isfinite(r["V"]))) + int(np.sum(r["resF"] > 1e-8)),
                       n_bins=r["n_bins"], wall_s=r["wall_s"])
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            np.savez(os.path.join(a.results, "openloopT_%s_%s.npz" % (cell, test)), t=df.t_s.values, I=df.I_A.values,
                     V_meas=Vm, V_dfn=r["V"], T_C=df.T_C.values, resF=r["resF"])


if __name__ == "__main__":
    main()
