"""Step 2 for the LP604765: joint fit of the dynamics, the temperature laws and the window on the three HPPCs.

    python fit.py --starts 6 --workers 26 --tag v1

Free parameters (FREE): chosen from the Morris screening (results/sens/morris.csv); the anode ones (Ds_n, Ea_Dn, k0_n)
have no visible effect on these data and stay at NOMINAL / FIXED (Ea_s is free: R_contact carries most of the resistance). Cost: mean over the tests of the
time-weighted MSE (sqrt(dt / T_test) weights), V clipped to [2.0, 4.5] V. Optimiser: scipy least_squares (TRF) in unit
coordinates of common.BOUNDS, forward-difference Jacobian evaluated in parallel (one process per simulation). Starts:
the nominal set and the Morris points with the lowest cost against the data (fixed parameters reset to FIXED).
Output: results/fit_<tag>/{log.txt, start_<i>.json, best.json}.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.optimize import least_squares

import common as C

FREE = ["y100", "Q_p", "x100", "Q_n", "tau", "k0_p", "Ds_p", "drop", "shape", "R_contact", "Ea_Dp", "Ea_r", "Ea_s",
        "dUdT0", "dUdT_n", "dq_40", "dq_00"]
FIXED = dict(C.NOMINAL)
TESTS = list(C.FIT)
V_CLIP = (2.0, 4.5)
H = 1e-3


class Problem:
    def __init__(self, pool, tests, prof, log):
        self.pool, self.tests, self.prof, self.log = pool, tests, prof, log
        self.w = {k: C.weights(prof[k]) / np.sqrt(len(tests)) for k in tests}
        self.cache = {}
        self.nev = 0
        self.best = (np.inf, None)

    def params(self, u):
        p = dict(FIXED)
        for j, n in enumerate(FREE):
            p[n] = float(C.from_unit(n, float(np.clip(u[j], 0.0, 1.0))))
        return p

    def _res(self, V, k):
        V = np.clip(np.nan_to_num(V, nan=V_CLIP[0]), *V_CLIP) if V is not None else np.full(len(self.prof[k]["V"]), V_CLIP[0])
        return 1e3 * self.w[k] * (V - self.prof[k]["V"])

    def eval_many(self, U):
        """Residual vectors at the rows of U (one parallel batch)."""
        jobs = [(k, self.params(u), False) for u in U for k in self.tests]
        out = self.pool.map(C._sim, jobs, chunksize=1)
        self.nev += len(U)
        R = []
        for i in range(len(U)):
            R.append(np.concatenate([self._res(out[i * len(self.tests) + j][0], k) for j, k in enumerate(self.tests)]))
        return R

    def fun(self, u):
        key = tuple(np.round(u, 14))
        if key not in self.cache:
            self.cache = {key: self.eval_many([u])[0]}
            r = self.cache[key]
            per = self.per_test(r)
            c = float(np.sum(r ** 2))
            if c < self.best[0]:
                self.best = (c, np.array(u))
            self.log("  eval %4d  cost %.3f mV rms  %s" % (self.nev, np.sqrt(c), "  ".join("%s %.2f" % kv for kv in per.items())))
        return self.cache[key]

    def per_test(self, r):
        out, i = {}, 0
        for k in self.tests:
            n = len(self.prof[k]["V"])
            out[k] = float(np.sqrt(len(self.tests) * np.sum(r[i:i + n] ** 2)))
            i += n
        return out

    def jac(self, u):
        r0 = self.fun(u)
        U = []
        hs = []
        for j in range(len(u)):
            h = H if u[j] + H <= 1.0 else -H
            v = np.array(u, dtype=float); v[j] += h
            U.append(v); hs.append(h)
        R = self.eval_many(U)
        return np.stack([(R[j] - r0) / hs[j] for j in range(len(u))], axis=1)


def morris_starts(prof, n):
    f = os.path.join(C.RES, "sens", "morris_runs.pkl")
    if not os.path.exists(f) or n <= 0:
        return []
    d = pickle.load(open(f, "rb"))
    cost = {}
    params = {}
    for (ti, pi, k), p, V in zip(d["jobs"], d["params"], d["V"]):
        if k not in prof:
            continue
        Vc = np.clip(np.nan_to_num(V, nan=V_CLIP[0]), *V_CLIP) if V is not None else None
        e = np.inf if Vc is None else np.sum((C.weights(prof[k]) * (Vc - prof[k]["V"])) ** 2)
        cost[(ti, pi)] = cost.get((ti, pi), 0.0) + e
        params[(ti, pi)] = p
    order = sorted(cost, key=cost.get)
    out = []
    for key in order[:n]:
        p = params[key]
        u = np.array([float(np.clip(C.to_unit(nm, p[nm]), 0.0, 1.0)) for nm in FREE])
        out.append((u, float(np.sqrt(cost[key] / len(prof)) * 1e3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=6, help="Morris starts in addition to the nominal one")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--from-json", default=None, help="start from a previous best.json only")
    a = ap.parse_args()
    out = os.path.join(C.RES, "fit_" + a.tag)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    prof = C.profiles(TESTS)
    starts = []
    if a.from_json:                     # warm start first (its fixed parameters keep FIXED), then the screening starts
        p = dict(FIXED, **{k: v for k, v in json.load(open(a.from_json))["params"].items() if k in FREE})
        starts.append((np.array([float(np.clip(C.to_unit(n, p[n]), 0, 1)) for n in FREE]), np.nan))
    else:
        starts.append((np.array([C.to_unit(n, FIXED[n]) for n in FREE]), np.nan))
    starts += morris_starts(prof, a.starts)
    log("free: %s\nfixed: %s\n%d starts" % (FREE, {k: v for k, v in FIXED.items() if k not in FREE}, len(starts)))
    results = []
    with C.pool(TESTS, a.workers) as pl:
        for i, (u0, c0) in enumerate(starts):
            log("=== start %d (screening cost %.1f mV)" % (i, c0))
            pb = Problem(pl, TESTS, prof, log)
            t0 = time.time()
            s = least_squares(pb.fun, np.clip(u0, 1e-3, 1 - 1e-3), jac=pb.jac, bounds=(0.0, 1.0), method="trf",
                              x_scale="jac", max_nfev=a.max_nfev, xtol=1e-6, ftol=1e-6, gtol=1e-8)
            u = pb.best[1] if pb.best[0] < 2 * s.cost else s.x
            r = pb.fun(u)
            res = dict(start=i, cost_mV=float(np.sqrt(np.sum(r ** 2))), per_test_mV=pb.per_test(r),
                       params=pb.params(u), u=list(map(float, u)), nfev=int(s.nfev), status=int(s.status),
                       message=s.message, wall_s=time.time() - t0,
                       at_bound=[n for n, x in zip(FREE, u) if x < 1e-3 or x > 1 - 1e-3])
            json.dump(res, open(os.path.join(out, "start_%d.json" % i), "w"), indent=1)
            log("start %d: cost %.2f mV  %s  (%.0f s, nfev %d, at bound %s)"
                % (i, res["cost_mV"], res["per_test_mV"], res["wall_s"], s.nfev, res["at_bound"]))
            results.append(res)
            best = min(results, key=lambda r_: r_["cost_mV"])
            json.dump(dict(best, free=FREE, fixed={k: v for k, v in FIXED.items() if k not in FREE}),
                      open(os.path.join(out, "best.json"), "w"), indent=1)
    log("best start %d: %.2f mV" % (best["start"], best["cost_mV"]))


if __name__ == "__main__":
    main()
