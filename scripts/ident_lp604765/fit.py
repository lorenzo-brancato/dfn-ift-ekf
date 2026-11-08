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
                if getattr(self, "ckpt", None):         # resumable: --from-json <out>/progress.json
                    json.dump(dict(params=self.params(u), cost_mV=float(np.sqrt(c)), per_test_mV=self.per_test(r), nev=self.nev),
                              open(self.ckpt, "w"), indent=1)
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
        t0 = time.time()
        J = self.jac_ift(u) if getattr(self, "mode", "fd") == "ift" else self.jac_fd(u)
        self.t_jac = getattr(self, "t_jac", 0.0) + time.time() - t0; self.n_jac = getattr(self, "n_jac", 0) + 1
        return J

    def jac_ift(self, u):
        """Exact Jacobian: per test, V and dV/dp from forward-mode AD through the model with the closure (and the
        OCV start inversion) differentiated by the implicit function theorem (common._vjac, lpmodel ift=True);
        then the chain rule to the unit coordinates, the voltage clip and the weights."""
        p = self.params(u)
        out = self.pool.map(C._vjac, [(k, list(FREE), p, False) for k in self.tests], chunksize=1)
        dpdu = []
        for j, n in enumerate(FREE):
            lo, hi, kind = C.BOUNDS[n]
            inside = 0.0 <= u[j] <= 1.0
            dpdu.append((p[n] * np.log(hi / lo) if kind == "log" else hi - lo) if inside else 0.0)
        dpdu = np.array(dpdu)
        R, Jr = [], []
        for (res, err), k in zip(out, self.tests):
            if res is None:
                raise RuntimeError("exact Jacobian failed on %s: %s" % (k, err))
            V, Jp = res
            Jp = np.where(np.isfinite(Jp), Jp, 0.0)        # a collapsed model (start far from the optimum) gives NaN rows
            act = np.isfinite(V) & (V >= V_CLIP[0]) & (V <= V_CLIP[1])
            R.append(self._res(V, k))
            Jr.append((1e3 * self.w[k] * act)[:, None] * Jp * dpdu[None, :])
        r = np.concatenate(R)
        key = tuple(np.round(u, 14))
        if key not in self.cache:                        # the primal comes with the Jacobian: no extra simulation
            self.cache = {key: r}
        self.nev += 1
        return np.concatenate(Jr, axis=0)

    def jac_fd(self, u):
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
        if Vc is not None and len(Vc) != len(prof[k]["V"]):
            continue                    # run made on another profile (older step classes): not comparable
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
    ap.add_argument("--free", default=None, help="free groups (comma list), default FREE")
    ap.add_argument("--fix", default="", help="fixed values overriding NOMINAL, e.g. Ea_r=39570,dUdT_n=0 (literature)")
    ap.add_argument("--bounds", default="", help="search boxes overriding common.BOUNDS, e.g. Ea_r=5e3:150e3")
    ap.add_argument("--note", default="", help="free text stored in best.json (the justification of the choices)")
    ap.add_argument("--jac", default="fd", choices=("fd", "ift"), help="Jacobian: forward differences (fd) or exact, by AD with the IFT (ift)")
    ap.add_argument("--tests", default="hppc", help="fit set: hppc (the three HPPCs) or B (+ full 1C cycles at 0/20/40 C), or a list")
    ap.add_argument("--ocp-p", default=None, help="cathode OCP table (ocp_sub.py json) instead of lm3c")
    a = ap.parse_args()
    if a.ocp_p:
        os.environ["LP_OCP_P"] = os.path.abspath(a.ocp_p)       # inherited by the worker processes
    if a.free:
        FREE[:] = a.free.split(",")
    TESTS[:] = C.TEST_SETS.get(a.tests, a.tests.split(","))
    for k, v in (it.split("=") for it in a.fix.split(",") if it):
        FIXED[k] = float(v)
    for k, v in (it.split("=") for it in a.bounds.split(",") if it):
        lo, hi = (float(x) for x in v.split(":"))
        C.BOUNDS[k] = (lo, hi, C.BOUNDS[k][2])
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
    if len(starts) - 1 < a.starts:      # no comparable screening runs: seeded perturbations of the first start
        rg = np.random.default_rng(0)
        for _ in range(a.starts - (len(starts) - 1)):
            starts.append((np.clip(starts[0][0] + rg.uniform(-0.05, 0.05, len(FREE)), 0.0, 1.0), np.nan))   # small: the window groups collapse the model beyond a few %
    log("free: %s\nfixed: %s\n%d starts" % (FREE, {k: v for k, v in FIXED.items() if k not in FREE}, len(starts)))
    results = []
    with C.pool(TESTS, a.workers) as pl:
        for i, (u0, c0) in enumerate(starts):
            log("=== start %d (screening cost %.1f mV)" % (i, c0))
            pb = Problem(pl, TESTS, prof, log)
            pb.ckpt = os.path.join(out, "progress.json")
            pb.mode = a.jac
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
            res.update(jac=a.jac, n_jac=getattr(pb, "n_jac", 0), t_jac_s=getattr(pb, "t_jac", 0.0))
            log("   Jacobians (%s): %d, %.0f s in total, %.1f s each" % (a.jac, res["n_jac"], res["t_jac_s"], res["t_jac_s"] / max(res["n_jac"], 1)))
            results.append(res)
            best = min(results, key=lambda r_: r_["cost_mV"])
            json.dump(dict(best, free=FREE, fixed={k: v for k, v in FIXED.items() if k not in FREE},
                           bounds={k: list(C.BOUNDS[k]) for k in FREE}, note=a.note, argv=vars(a)),
                      open(os.path.join(out, "best.json"), "w"), indent=1)
    log("best start %d: %.2f mV" % (best["start"], best["cost_mV"]))


if __name__ == "__main__":
    main()
