"""Step 0 for the LP604765 in parameter groups: global (Morris) screening of the 13 dynamic candidates of step 2.

    python sens_groups.py --r 30 --levels 6 --workers 7 [--tests B]

As in the ApEn draft (Section 2.4.1): elementary effects over the literature boxes of common.BOUNDS (LP_GROUPS=1, set
here), unit cube with log axes where marked, p levels, Delta = p / (2 (p - 1)); the window, entropic terms and HPPC
offsets of step 1 and the stack are fixed (common.NOMINAL). No measured data and no fitted value enter.
Effect of group i on test k: EE = RMS_t(V(u + Delta e_i) - V(u)) [mV, time-weighted] / Delta, on the whole test and
separately on discharge (I > 0) and charge (I < 0) samples; "N" variants keep only samples where both runs are >= 2.7 V
(normal operation; outside it a collapsed run dominates the RMS). Reported per test: mean (mu*), median and sd.
Discard rule of the draft: a group whose median EE stays below 1 mV on every test (and on every phase).
Output: results/exp_LP604765/sens_groups/{morris.json, morris.csv, morris_runs.pkl}.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

os.environ["LP_GROUPS"] = "1"           # before common: inherited by the spawned workers
os.environ.pop("LP_THERMAL", None)
import common as C  # noqa: E402

OUT = os.path.join(C.RES, "sens_groups")
V_CLIP = (2.0, 4.5)


def from_unit(n, u):
    return C.from_unit(n, u)


def trajectories(k, r, p, seed=0):
    rng = np.random.default_rng(seed)
    delta = p / (2.0 * (p - 1))
    grid = np.arange(p // 2) / (p - 1)            # start levels that allow +Delta
    out = []
    for _ in range(r):
        x = rng.choice(grid, size=k)
        order = rng.permutation(k)
        pts = [x.copy()]
        for i in order:
            x = x.copy(); x[i] += delta
            pts.append(x)
        out.append((np.array(pts), order))
    return out, delta


def clip(v):
    return None if v is None else np.clip(np.nan_to_num(v, nan=V_CLIP[0], posinf=V_CLIP[1], neginf=V_CLIP[0]), *V_CLIP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tests", default="B")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = list(C.GROUPS)
    keys = C.TEST_SETS.get(a.tests, a.tests.split(","))
    prof = C.profiles(keys)
    trajs, delta = trajectories(len(names), a.r, a.levels)
    jobs = []
    for ti, (pts, order) in enumerate(trajs):
        for pi, u in enumerate(pts):
            p = {**C.NOMINAL, **{n: from_unit(n, u[j]) for j, n in enumerate(names)}}
            for key in keys:
                jobs.append(((ti, pi, key), (key, p, False)))
    print("%d groups, %d trajectories, %d levels: %d runs on %s" % (len(names), a.r, a.levels, len(jobs), keys), flush=True)
    t0 = time.time()
    res = []
    with C.pool(keys, a.workers) as pl:
        for i, r_ in enumerate(pl.imap(C._sim, [j[1] for j in jobs], chunksize=1)):
            res.append(r_)
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print("  %d / %d runs, %.0f s, ETA %.0f min" % (i + 1, len(jobs), el, el / (i + 1) * (len(jobs) - i - 1) / 60), flush=True)
    print("%d runs in %.0f s" % (len(jobs), time.time() - t0), flush=True)
    pickle.dump(dict(jobs=[j[0] for j in jobs], params=[j[1][1] for j in jobs], V=[r[0] for r in res], trajs=trajs,
                     delta=delta, names=names, keys=keys), open(os.path.join(OUT, "morris_runs.pkl"), "wb"))
    V = {j[0]: clip(r[0]) for j, r in zip(jobs, res)}
    ncol = {k: sum(1 for j, r in zip(jobs, res) if j[0][2] == k and (r[0] is None or not np.all(np.isfinite(r[0]))
                                                                     or np.min(r[0]) < 2.5)) for k in keys}
    phases = {}
    for k in keys:
        I = prof[k]["I"]; w2 = C.weights(prof[k]) ** 2
        phases[k] = {"all": np.ones(len(I), bool), "dis": I > 0.01, "ch": I < -0.01}
        phases[k] = {ph: m for ph, m in phases[k].items() if m.sum() > 10}
    EE = {}                                       # (key, phase, normal) -> list per group
    for ti, (pts, order) in enumerate(trajs):
        for step, i in enumerate(order):
            for k in keys:
                v0, v1 = V[(ti, step, k)], V[(ti, step + 1, k)]
                if v0 is None or v1 is None:
                    continue
                w2 = C.weights(prof[k]) ** 2
                d2 = (v1 - v0) ** 2
                ok = (v0 >= 2.7) & (v1 >= 2.7)
                for ph, m in phases[k].items():
                    for nrm, mm in ((False, m), (True, m & ok)):
                        s = w2[mm].sum()
                        e = 1e3 * np.sqrt(np.sum(w2[mm] * d2[mm]) / s) / delta if s > 0 else np.nan
                        EE.setdefault((k, ph, nrm), [[] for _ in names])[i].append(e)
    rows = []
    for i, n in enumerate(names):
        row = dict(param=n)
        for (k, ph, nrm), L in EE.items():
            e = np.array(L[i], float); e = e[np.isfinite(e)]
            tag = "%s_%s%s" % (k, ph, "_N" if nrm else "")
            row["mu_" + tag] = float(e.mean()) if len(e) else np.nan
            row["med_" + tag] = float(np.median(e)) if len(e) else np.nan
            row["sd_" + tag] = float(e.std()) if len(e) else np.nan
        row["mu_mean"] = float(np.nanmean([row["mu_%s_all" % k] for k in keys]))
        row["muN_mean"] = float(np.nanmean([row["mu_%s_all_N" % k] for k in keys]))
        row["medN_max"] = float(np.nanmax([row[c] for c in row if c.startswith("med_") and c.endswith("_N")]))
        row["discard"] = bool(row["medN_max"] < 1.0)
        rows.append(row)
    rows.sort(key=lambda r: -r["muN_mean"])
    json.dump(dict(rows=rows, n_collapse=ncol, runs_per_test=len(jobs) // len(keys), box={n: C.BOUNDS[n] for n in names},
                   fixed={k: v for k, v in C.NOMINAL.items() if k not in names}, r=a.r, levels=a.levels, delta=delta,
                   keys=keys), open(os.path.join(OUT, "morris.json"), "w"), indent=1, default=float)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "morris.csv"), index=False)
    cols = ["param", "muN_mean", "mu_mean", "medN_max", "discard"] + ["mu_%s_all_N" % k for k in keys]
    print(df[cols].round(2).to_string(index=False))
    print("runs below 2.5 V or non-finite (of %d per test):" % (len(jobs) // len(keys)), ncol)


if __name__ == "__main__":
    main()
