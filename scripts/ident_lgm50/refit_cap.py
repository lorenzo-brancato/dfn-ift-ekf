"""Capacity of each held-out cell calibrated on one discharge, then the cell's other tests with it (dynamics fixed).

    python refit_cap.py [--fit results/exp_LGM50/fit_m50_v4/best.json] [--cells B,C,E,G,H,5B,5C,5E,5G,5H] [--workers 4]

One factor s per cell scales both electrode capacities (Q_n, Q_p -> s Q_n, s Q_p; x100, y100 kept), so the window
width (the cell capacity) changes and the stoichiometry range does not. The start of every test follows from the new
window (OCV inversion of its rested voltage, as in validate.py). s minimises the time-weighted RMSE of ONE discharge:
  Expt 4 cells: the C/2 discharge of CC05 and the rest after it (up to the charge); checked on C10, P05, P10, CC05
    (whole), GITT, G20;
  Expt 5 cells: the cycle 0 1C discharge (CYC); checked on CYC + the cycle 1 CC-CV charge (diag_cv.cycle_profile).
Output: results/exp_LGM50/refit_cap/<fit tag>/{summary.json, per_test.csv}
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
os.environ.setdefault("LP_CELL", "lgm50")
sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
sys.path.insert(0, HERE)
KINDS4 = ("C10", "P05", "P10", "CC05", "GITT", "G20")


def run_cell(args):
    cell, fit = args
    import common as C
    import m50data as M
    from diag_cv import cycle_profile
    from lpmodel import Model, prepare
    from scipy.optimize import minimize_scalar
    m = Model()
    p0 = json.load(open(fit))["params"]
    if cell.startswith("5"):
        T = M.AGE_T[cell[-1]]
        d = cycle_profile(cell)
        pr = prepare(d); pr["V_rest"] = d["V_rest"]
        calib = (cell + "_CYC", C.profiles([cell + "_CYC"])[cell + "_CYC"], T)
        tests = [(cell + "_CYCCV", pr, T)]
    else:
        d = M.cut(cell + "_CC05")
        n = int(np.flatnonzero(d["I"] < -0.1)[0])
        d = {k: (v[:n] if isinstance(v, np.ndarray) and len(v) == len(d["t"]) else v) for k, v in d.items()}
        pc = prepare(d); pc["V_rest"] = d["V_rest"]
        calib = (cell + "_CC05dis", pc, 25.0)
        prof = C.profiles(["%s_%s" % (cell, k) for k in KINDS4])
        tests = [(k, pr, 25.0) for k, pr in prof.items()]
    base = C.params_for(p0, dict(win=cell))

    def errs(s, tests=tests):
        p = dict(base, Q_n=base["Q_n"] * s, Q_p=base["Q_p"] * s)
        out = {}
        for k, pr, T in tests:
            dq = m.dq_from_ocv(C.model_params(p), T, pr["V_rest"])
            V, F, _, _ = m.simulate(C.model_params(p), pr, T, dq=dq, full=True)
            e = (V - pr["V"]) * 1e3
            w = pr["dt"]; ok = np.isfinite(e)
            rm = float(np.sqrt(np.sum(w[ok] * e[ok] ** 2) / np.sum(w[ok]))) if ok.mean() > 0.98 else 1e3
            r = dict(rmse_mV=rm, bad=int(np.sum(~ok) + np.sum(F > 1e-6)))
            if k.endswith("CYCCV"):
                t = np.cumsum(pr["dt"]); I = pr["I"]
                ch = t > t[np.flatnonzero(I > 1)[-1]]
                cv = ch & (pr["V"] > 4.195) & (I < -0.02)
                r.update(cv_err_mV=float(np.nanmean(e[cv])), cv_end_err_mV=float(np.nanmean(e[cv][-20:])),
                         dis_rmse_mV=float(np.sqrt(np.nanmean(e[~ch] ** 2))))
            out[k] = r
        return out

    cost = lambda s: errs(s, [calib])[calib[0]]["rmse_mV"]
    e0 = errs(1.0, [calib] + tests)
    sol = minimize_scalar(cost, bounds=(0.85, 1.15), method="bounded", options=dict(xatol=1e-3))
    e1 = errs(sol.x, [calib] + tests)
    res = dict(cell=cell, calib=calib[0], s=float(sol.x), Q_window_before_Ah=None, nfev=int(sol.nfev),
               Q_n=base["Q_n"] * sol.x, Q_p=base["Q_p"] * sol.x, before=e0, after=e1)
    print(json.dumps(dict(cell=cell, s=round(sol.x, 4), before={k: round(v["rmse_mV"], 1) for k, v in e0.items()},
                          after={k: round(v["rmse_mV"], 1) for k, v in e1.items()})), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(ROOT, "results", "exp_LGM50", "fit_m50_v4", "best.json"))
    ap.add_argument("--cells", default="B,C,E,G,H,5B,5C,5E,5G,5H")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    fit = os.path.abspath(a.fit)
    import m50data as M
    out = os.path.join(M.RES, "refit_cap", os.path.basename(os.path.dirname(fit))); os.makedirs(out, exist_ok=True)
    W = json.load(open(os.path.join(M.RES, "windows.json")))["cells"]
    with Pool(a.workers) as pool:
        res = pool.map(run_cell, [(c, fit) for c in a.cells.split(",")], chunksize=1)
    rows = []
    for r in res:
        r["Q_window_before_Ah"] = W[r["cell"]]["Q_window_Ah"]
        r["Q_window_after_Ah"] = W[r["cell"]]["Q_window_Ah"] * r["s"]
        for k in r["before"]:
            rows.append(dict(cell=r["cell"], test=k + (" (calib)" if k == r["calib"] else ""), s=r["s"], rmse_before=r["before"][k]["rmse_mV"],
                             rmse_after=r["after"][k]["rmse_mV"], bad_before=r["before"][k]["bad"],
                             bad_after=r["after"][k]["bad"],
                             cv_end_before=r["before"][k].get("cv_end_err_mV"), cv_end_after=r["after"][k].get("cv_end_err_mV"),
                             dis_before=r["before"][k].get("dis_rmse_mV"), dis_after=r["after"][k].get("dis_rmse_mV")))
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "per_test.csv"), index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
