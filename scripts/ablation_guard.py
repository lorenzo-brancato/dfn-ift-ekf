"""Ablation of the state guard in the final configuration (Sec. IV-D and Appendix B).

Two separate claims:
 (1) "without the state guard the filter breaks down" - tested here on the ageing reference with the final tracker (4 parameters, grid given by --grid, converged Newton closure),
     unchanged except guard=False. The unguarded run is stopped at the first breakdown: non-finite estimate / state, or closure residual above 1e-2 at the current or at the next
     step (Newton failure). The breakdown time is reported together with the first time each of the four guard conditions would have fired, and the first time Rf or LAM is more than 20 % of its range from the truth after the first 2 h (a collapse without NaN) (electrolyte depletion, surface
     concentration out of [0, c_max], closure residual, non-finite). The guarded tracker (same noise, same everything) is run to the same time plus a margin and must show no breakdown.
 (2) number of rejected updates of the guarded tracker over the whole ageing run (needs the full runs of ageing_track.py in --tracking), counted per guard condition. Appendix B states
     that the guard did NOT trigger on the protocols of the static tests: this is a different claim, taken from the per-run CSVs of the static-test scripts (column n_rejected_total);
     pass their folder with --static to have it summed here.

  python scripts/ablation_guard.py --grid 4:1 --data data --results results/ablation_guard --seeds 0 --core 0     # the unguarded run stops at ~6.4 h of simulated time: about 20-30 min
  python scripts/ablation_guard.py --aggregate --results results/ablation_guard --tracking results/ageing --static results
  python scripts/ablation_guard.py --quick --data data --results results/quick                                    # 0.05 h, no breakdown expected

Outputs: ablation_guard_breakdown.csv (one row per seed x guard: breakdown time, first time of every condition, hours simulated), ablation_guard_rejections.csv (claim 2),
traj/ablation_guard_seed<s>_{guard,noguard}.npz.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

REF_NAME = "aging_dyn_rptslow_full.npz"


def first_error(d, P, at, limit=20.0, after_h=2.0):
    """First time [h] after ``after_h`` (end of the initial transient: Ds is not yet identified before) at which Rf or LAM is more than ``limit`` % of its ageing range from the truth, or any
    estimate is non-finite: the descriptor of a collapse that does not end in NaN. NaN if it never happens in the run."""
    import numpy as np
    v = d["vals"]; k = np.minimum(np.arange(len(v)) * 20, len(P["Rf"]) - 1); coords = list(d["coords"]); bad = ~np.all(np.isfinite(v), axis=1)
    for i, c in enumerate(coords):
        if c not in ("Rf", "LAM"): continue
        rg = at.RANGE[c][1] - at.RANGE[c][0]; bad |= ~(100 * np.abs(v[:, i] - at.true_value(P, c, k)) / rg <= limit)
    bad &= (k * at.DT / 3600 >= after_h)
    return float(k[np.argmax(bad)] * at.DT / 3600) if bad.any() else float("nan")


def exit_times(d, P, at, after_h=2.0):
    """When the estimate violates the acceptance criteria, per parameter (R_f, LAM, D_s, LLI), after ``after_h`` hours (the initial transient is excluded):
    first time the error exceeds 10 % / 20 % of the ageing range, or the estimate leaves the physical (ageing) range, and the 'sustained' time (the criterion then stays violated to the end of the run;
    the definition of the first ageing study). NaN = never. Columns ``first_<crit>_<param>_h`` and ``sustained_<crit>_<param>_h``, crit in err10, err20, range."""
    import numpy as np
    v = d["vals"]; k = np.minimum(np.arange(len(v)) * 20, len(P["Rf"]) - 1); t = k * at.DT / 3600; late = t >= after_h; out = {}
    for i, c in enumerate(list(d["coords"])):
        lo, hi = at.RANGE[c]; err = 100 * np.abs(v[:, i] - at.true_value(P, c, k)) / (hi - lo)
        for crit, bad in (("err10", err > 10.0), ("err20", err > 20.0), ("range", (v[:, i] < lo) | (v[:, i] > hi))):
            m = bad & late; out[f"first_{crit}_{c}_h"] = float(t[np.argmax(m)]) if m.any() else float("nan")
            stay = (np.flip(np.cumprod(np.flip(bad | ~late).astype(int))) > 0) & late
            out[f"sustained_{crit}_{c}_h"] = float(t[np.argmax(stay)]) if stay.any() else float("nan")
    return out


def guarded_row_from_tracker(a, seed, P, hours):
    """The guarded half of the pair taken from the full tracker run of ``ageing_track.py`` (same noise seed, same everything, guard on): the first ``hours`` of its trajectory.
    Used instead of running the guarded tracker a second time over a long window. max_resF is not available at this resolution (NaN)."""
    import numpy as np
    from dfnekf import ageing_tracker as at
    d = dict(np.load(os.path.join(a.guarded_from, "traj", f"ageing_track_seed{seed}.npz"))); n = int(round(hours * 3600 / at.DT)); nv = int(np.ceil(n / 20))
    d["vals"] = d["vals"][:nv]
    r = dict(seed=seed, guard=True, simulated_h=hours, breakdown=int(d["breakdown"]) >= 0, breakdown_h=float("nan"), n_rejected=int(d["ps_rejected"][:n].sum()), max_resF=float("nan"),
             n_nonfinite=int((~np.all(np.isfinite(d["vals"]), axis=1)).sum()), source="ageing_track.py trajectory")
    r.update(first_error_gt_20pct_h=first_error(d, P, at)); r.update(exit_times(d, P, at)); return r


def run_pair(a, seed, P):
    import numpy as np
    from dfnekf import ageing_tracker as at
    chunk = 120 if a.quick else a.chunk; rows = []; cap = 360 if a.quick else int(a.max_hours * 3600 / at.DT) + 1
    tr0 = at.Tracker(P, a.grid, seed=seed, guard=False, nsteps=cap)
    d0 = at.track(tr0, stop_on_breakdown=True, chunk=chunk)
    at.save_run(os.path.join(a.results, "traj", f"ablation_guard_seed{seed}_noguard.npz"), d0)
    bd = int(d0["breakdown"])
    n_g = min(len(tr0.t), bd + int(a.margin_hours * 3600 / at.DT)) if bd >= 0 else cap
    runs = [(False, d0, tr0)]
    if not a.guarded_from:
        tr1 = at.Tracker(P, a.grid, seed=seed, guard=True, nsteps=n_g)
        d1 = at.track(tr1, stop_on_breakdown=False, chunk=chunk)
        at.save_run(os.path.join(a.results, "traj", f"ablation_guard_seed{seed}_guard.npz"), d1); runs.append((True, d1, tr1))
    for guard, d, tr in runs:
        bd_ = int(d["breakdown"])
        r = dict(seed=seed, guard=guard, simulated_h=float(len(d["ps_innov"])) * at.DT / 3600, breakdown=bd_ >= 0, breakdown_h=tr.t[bd_] / 3600 if bd_ >= 0 else float("nan"),
                 n_rejected=int(d["n_rejected"].sum()), max_resF=float(np.max(d["resF_max"])), n_nonfinite=int(d["n_nonfinite"].sum()))
        r.update(first_error_gt_20pct_h=first_error(d, P, at)); r.update(exit_times(d, P, at))
        for k in ("non_finite", "resF", "F_next") + at.VIOL: r["first_" + k + "_h"] = tr.t[int(d["first_" + k])] / 3600 if "first_" + k in d else float("nan")
        rows.append(r)
    if a.guarded_from: rows.append(guarded_row_from_tracker(a, seed, P, rows[0]["simulated_h"]))
    return rows


def aggregate(a):
    import numpy as np, pandas as pd, glob
    from dfnekf import ageing_tracker as at
    rows = []
    for s in cli.parse_ids(a.seeds):
        f = os.path.join(a.tracking, "traj", f"ageing_track_seed{s}.npz")
        if not os.path.exists(f): continue
        d = np.load(f)
        r = dict(seed=s, scope="ageing run, guarded tracker, whole run", n_rejected=int(d["n_rejected"].sum()), n_would_reject=int(d["n_would_reject"].sum()), n_nonfinite=int(d["n_nonfinite"].sum()) if "n_nonfinite" in d else -1)
        for k in at.VIOL: r["n_" + k] = int(d["n_" + k].sum()) if "n_" + k in d else -1
        r["max_resF"] = float(np.max(d["resF_max"])); rows.append(r)
    if a.static:
        st = 0; nf = 0
        for f in glob.glob(os.path.join(a.static, "**", "*.csv"), recursive=True):
            try: D = pd.read_csv(f, comment="#")
            except Exception: continue
            if "n_rejected_total" in D.columns: st += int(D["n_rejected_total"].sum()); nf += len(D)
        rows.append(dict(seed=-1, scope=f"static tests: sum of n_rejected_total over {nf} runs in {a.static}", n_rejected=st if nf else float("nan")))
    if not rows: print("no full tracker run found in --tracking: claim 2 not produced"); return
    sa = type("A", (), dict(results=a.results, item=a.item, grid=a.grid))
    cli.write_csv(pd.DataFrame(rows), sa, "ablation_guard_rejections.csv", note="claim 2: rejected updates of the guarded filter (ageing run) and, separately, in the static tests")
    print(pd.DataFrame(rows).to_string())


def main():
    p = cli.parser("Sec. IV-D and App. B (state guard)", "guard ablation on the ageing reference: breakdown time without the guard, rejected updates with it", seeds="0")
    p.add_argument("--aggregate", action="store_true"); p.add_argument("--max-hours", type=float, default=12.0, help="give up if the unguarded run has not broken down by then")
    p.add_argument("--margin-hours", type=float, default=1.0, help="the guarded run goes this far beyond the breakdown time of the unguarded one")
    p.add_argument("--chunk", type=int, default=5000); p.add_argument("--tracking", default="results/ageing"); p.add_argument("--static", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--guarded-from", default=None, help="results folder of ageing_track.py: take the guarded half of the pair from its trajectory (first N hours) instead of running it again")
    a = p.parse_args()
    runtime.setup(a.core)
    if a.aggregate: return aggregate(a)
    import pandas as pd
    from dfnekf import ageing
    P = ageing.load(a.ref or os.path.join(a.data, REF_NAME)); clock = cli.Clock(a, "ablation_guard"); rows = []
    for s in cli.parse_ids(a.seeds):
        rows += run_pair(a, s, P)
        cli.write_csv(pd.DataFrame(rows), a, "ablation_guard_breakdown.csv", note="claim 1: breakdown time of the ageing tracker without / with the guard"); print(pd.DataFrame(rows).T.to_string(), flush=True)
    clock.log()
    if not a.quick: aggregate(a)


if __name__ == "__main__":
    main()
