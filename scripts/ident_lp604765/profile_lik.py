"""Profile likelihood of single parameter groups (Raue et al., Bioinformatics 25 (2009) 1923): practical identifiability.

    python profile_lik.py --fit <results>/fit_v2/best.json --ident <results>/fit_v2/ident_v2.json \
        --profiles "Ds_p=0.25,0.5,2,4x;shape=1.0,1.5,2.5;Ea_r=20e3,40e3,60e3" --max-nfev 30 --workers 26 --tag v2

For each group theta_j and each grid value: theta_j fixed there, every other free group of the fit re-optimised (TRF,
warm start at the fit, --max-nfev), cost recorded. A grid value given as 'Nx' is N times the fitted value.
The group is practically non-identifiable when the profile stays below the threshold
    RMSE(theta_j) / RMSE_opt - 1 < sqrt(1 + chi2_{1,0.95} / N_eff) - 1
(N_eff from ident.py: the residual is model error, correlated in time; the white-noise threshold would be far too
strict). The same profile gives the cost of replacing a group by a literature value.
Output: <fit folder>/profile_<tag>.json (appended per point, so a run can be resumed; rerun the same command to resume), profile_<tag>.png/.pdf.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import least_squares

import common as C
import fit as Fm

HERE = os.path.dirname(os.path.abspath(__file__))


def grid_of(spec, v0):
    out = []
    for s in spec.split(","):
        out.append(float(s[:-1]) * v0 if s.endswith("x") else float(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--ident", required=True, help="ident.py json of the same fit (N_eff, threshold)")
    ap.add_argument("--profiles", required=True, help="name=v1,v2,...;name2=... (Nx = N times the fitted value)")
    ap.add_argument("--max-nfev", type=int, default=30)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    out = os.path.dirname(os.path.abspath(a.fit))
    best = json.load(open(a.fit))
    idn = json.load(open(a.ident))
    thr = idn["profile_threshold_rel"]
    free0 = list(best.get("free", Fm.FREE))
    p_opt = dict(C.NOMINAL, **best["params"])
    fo = os.path.join(out, "profile_%s.json" % a.tag)
    done = json.load(open(fo)) if os.path.exists(fo) else dict(threshold_rel=thr, N_eff=idn["N_eff"], rmse_opt=best["cost_mV"], points=[])
    have = {(q["param"], round(q["value"], 12)) for q in done["points"]}
    prof = C.profiles(Fm.TESTS)
    logf = open(os.path.join(out, "profile_%s.log" % a.tag), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    with C.pool(Fm.TESTS, a.workers) as pl:
        for spec in (s for s in a.profiles.split(";") if s):
            nm, g = spec.split("=")
            for v in grid_of(g, p_opt[nm]):
                lo, hi, _ = C.BOUNDS[nm]
                if not lo <= v <= hi:
                    log("%s = %.4g outside the box [%.3g, %.3g]: skipped" % (nm, v, lo, hi)); continue
                if (nm, round(v, 12)) in have:
                    continue
                free = [f for f in free0 if f != nm]
                Fm.FREE[:] = free
                Fm.FIXED.clear(); Fm.FIXED.update({k: p_opt[k] for k in C.BOUNDS}); Fm.FIXED[nm] = v
                pb = Fm.Problem(pl, Fm.TESTS, prof, lambda s: None)
                u0 = np.clip([C.to_unit(f, p_opt[f]) for f in free], 1e-3, 1 - 1e-3)
                t0 = time.time()
                s = least_squares(pb.fun, u0, jac=pb.jac, bounds=(0.0, 1.0), method="trf", x_scale="jac",
                                  max_nfev=a.max_nfev, xtol=1e-6, ftol=1e-6, gtol=1e-8)
                u = pb.best[1] if pb.best[0] < 2 * s.cost else s.x
                r = pb.fun(u)
                cost = float(np.sqrt(np.sum(r ** 2)))
                pt = dict(param=nm, value=v, value_rel=v / p_opt[nm] if p_opt[nm] else None, rmse_mV=cost,
                          rel_increase=cost / best["cost_mV"] - 1.0, per_test_mV=pb.per_test(r), nfev=int(s.nfev),
                          params=pb.params(u), wall_s=time.time() - t0,
                          at_bound=[f for f, x in zip(free, u) if x < 1e-3 or x > 1 - 1e-3])
                done["points"].append(pt)
                json.dump(done, open(fo, "w"), indent=1)
                log("%-9s = %-10.4g RMSE %.3f mV (%+.2f %%, threshold %.2f %%)  nfev %d  %.0f s  at bound %s"
                    % (nm, v, cost, 100 * pt["rel_increase"], 100 * thr, s.nfev, pt["wall_s"], pt["at_bound"]))
    plot(done, p_opt, out, a.tag)


def plot(done, p_opt, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    names = list(dict.fromkeys(q["param"] for q in done["points"]))
    nc = min(3, len(names)); nr = int(np.ceil(len(names) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(170 * mm, 50 * mm * nr), layout="constrained", squeeze=False)
    for i, nm in enumerate(names):
        q = sorted([p for p in done["points"] if p["param"] == nm], key=lambda p: p["value"])
        x = [p["value"] for p in q] + [p_opt[nm]]; y = [100 * p["rel_increase"] for p in q] + [0.0]
        o = np.argsort(x)
        A = ax[i // nc][i % nc]
        A.plot(np.array(x)[o], np.array(y)[o], "o-", ms=3)
        A.plot([p_opt[nm]], [0.0], "k*", ms=7)
        A.axhline(100 * done["threshold_rel"], color="C3", lw=0.8, ls="--")
        if C.BOUNDS[nm][2] == "log":
            A.set_xscale("log")
        A.set_xlabel(nm); A.set_ylabel(r"$\Delta$RMSE [%]")
    for j in range(len(names), nr * nc):
        ax[j // nc][j % nc].axis("off")
    fig.savefig(os.path.join(out, "profile_%s.png" % tag), dpi=200); fig.savefig(os.path.join(out, "profile_%s.pdf" % tag))


if __name__ == "__main__":
    main()
