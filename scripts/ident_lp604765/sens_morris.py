"""Step 0 for the LP604765: Morris elementary effects of every parameter on the simulated voltage of the three HPPCs.

    python sens_morris.py --r 12 --workers 16

Unit cube over common.BOUNDS (log axis where marked), p = 4 levels, Delta = 2/3. Effect of parameter i on test k:
EE = RMS_t(V(u + Delta e_i) - V(u)) [mV, time-weighted] / Delta. Reported: mu* (mean EE), sigma (std), per test and the
mean over the tests; failed or non-finite runs are counted and skipped. Output: results/sens/morris.{json,csv,png}.
"""
import argparse
import json
import os
import time

import numpy as np

import common as C

OUT = os.path.join(C.RES, "sens")
# prior box of the screening (plausible values; common.BOUNDS is the wider search box of the fit)
SCREEN = dict(
    Ds_n=(6e-14, 6e-12, "log"), Ds_p=(1e-14, 1e-12, "log"), drop=(0.0, 3.0, "lin"), shape=(0.7, 3.0, "lin"),
    k0_n=(0.1, 10.0, "log"), k0_p=(0.1, 10.0, "log"), R_contact=(1e-3, 2e-2, "log"), tau=(0.5, 5.0, "log"),
    Ea_Dn=(10e3, 80e3, "lin"), Ea_Dp=(10e3, 80e3, "lin"), Ea_r=(10e3, 80e3, "lin"), Ea_s=(0.0, 40e3, "lin"),
    dUdT0=(-1e-3, 1e-3, "lin"), dUdT_n=(-2e-3, 2e-3, "lin"),
    x100=(0.60, 0.72, "lin"), y100=(0.38, 0.43, "lin"), Q_n=(3.8, 5.0, "lin"), Q_p=(3.5, 4.2, "lin"),
    dq_40=(-0.12, 0.02, "lin"), dq_00=(-0.05, 0.15, "lin"),
)
V_CLIP = (2.0, 4.5)


def from_unit(n, u):
    lo, hi, kind = SCREEN[n]
    return lo * (hi / lo) ** u if kind == "log" else lo + u * (hi - lo)


def clip(v):
    return None if v is None else np.clip(np.nan_to_num(v, nan=V_CLIP[0], posinf=V_CLIP[1], neginf=V_CLIP[0]), *V_CLIP)


def trajectories(k, r, p=4, seed=0):
    rng = np.random.default_rng(seed)
    delta = p / (2.0 * (p - 1))
    grid = np.arange(p // 2) / (p - 1)           # start levels that allow +Delta
    out = []
    for _ in range(r):
        x = rng.choice(grid, size=k)
        sgn = np.ones(k)
        order = rng.permutation(k)
        pts = [x.copy()]
        for i in order:
            x = x.copy(); x[i] += delta * sgn[i]
            pts.append(x)
        out.append((np.array(pts), order))
    return out, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tests", default="HPPC20,HPPC40,HPPC00")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = C.ALL_NAMES
    keys = a.tests.split(",")
    prof = C.profiles(keys)
    w = {k: C.weights(prof[k]) for k in keys}
    trajs, delta = trajectories(len(names), a.r)
    jobs = []
    for ti, (pts, order) in enumerate(trajs):
        for pi, u in enumerate(pts):
            p = {n: from_unit(n, u[j]) for j, n in enumerate(names)}
            for key in keys:
                jobs.append(((ti, pi, key), (key, p, False)))
    t0 = time.time()
    with C.pool(keys, a.workers) as pl:
        res = pl.map(C._sim, [j[1] for j in jobs], chunksize=1)
    print("%d runs in %.0f s" % (len(jobs), time.time() - t0), flush=True)
    import pickle
    pickle.dump(dict(jobs=[j[0] for j in jobs], params=[j[1][1] for j in jobs], V=[r[0] for r in res], trajs=trajs,
                     delta=delta, names=names, keys=keys), open(os.path.join(OUT, "morris_runs.pkl"), "wb"))
    V = {j[0]: clip(r[0]) for j, r in zip(jobs, res)}
    nfail = {k: sum(1 for j, r in zip(jobs, res) if j[0][2] == k and (r[0] is None or not np.all(np.isfinite(r[0]))
                                                                       or np.min(r[0]) < V_CLIP[0] + 0.5)) for k in keys}
    errs = [r[1] for r in res if r[1]]
    EE = {k: [[] for _ in names] for k in keys}
    EEn = {k: [[] for _ in names] for k in keys}           # normal operation: samples where both runs stay >= 2.7 V
    nbad = {k: 0 for k in keys}
    for ti, (pts, order) in enumerate(trajs):
        for step, i in enumerate(order):
            for key in keys:
                v0, v1 = V[(ti, step, key)], V[(ti, step + 1, key)]
                if v0 is None or v1 is None:
                    nbad[key] += 1
                    continue
                EE[key][i].append(1e3 * np.sqrt(np.sum((w[key] * (v1 - v0)) ** 2)) / delta)
                ok = (v0 >= 2.7) & (v1 >= 2.7)
                EEn[key][i].append(1e3 * np.sqrt(np.sum((w[key] * (v1 - v0))[ok] ** 2) / np.sum(w[key][ok] ** 2)) / delta)
    rows = []
    for i, n in enumerate(names):
        row = dict(param=n)
        for key in keys:
            e = np.array(EE[key][i])
            row["mu_" + key] = float(e.mean()) if len(e) else np.nan
            row["sd_" + key] = float(e.std()) if len(e) else np.nan
            en = np.array(EEn[key][i])
            row["muN_" + key] = float(en.mean()) if len(en) else np.nan
        row["mu_mean"] = float(np.nanmean([row["mu_" + k] for k in keys]))
        row["muN_mean"] = float(np.nanmean([row["muN_" + k] for k in keys]))
        rows.append(row)
    rows.sort(key=lambda r: -r["mu_mean"])
    print("runs with V < 2.5 V or non-finite (of %d per test):" % (len(jobs) // len(keys)), nfail)
    json.dump(dict(rows=rows, n_bad=nbad, n_collapse=nfail, box=SCREEN, errors=errs[:5], r=a.r, delta=delta), open(os.path.join(OUT, "morris.json"), "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "morris.csv"), index=False)
    print(df.round(2).to_string(index=False))
    print("non-finite pairs:", nbad, "errors:", len(errs))


if __name__ == "__main__":
    main()
