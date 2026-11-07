"""CV freeze of the ageing tracker: evaluation of the runs of ``ageing_track.py --freeze ...`` (Sec. III-D, "Two remedies on the filter side ... did not help"; see FREEZE_CV.md).

Reads <results>/traj/ageing_track_seed<s><suffix>.npz for the variants  none ('' = the paper's run), phase (``--freeze phase``: parameter rows of the gain zero in the CV of every RPT, cell
state still updated) and phase-noupdate (``--freeze phase --cv-update none``: no measurement update in that CV), voltage (``--freeze voltage``), and writes in <results>:
  tables/freeze_rpt_<variant>.csv          one row per seed x RPT x read-out point (precv = before the CV, cvend = end of the CV hold, end = end of the RPT): errors in % of the range,
                                           standard deviations, z, criteria A, B, z and innovation C (chunk RMS <= 3 mV), pass flag
  source_data/trk_rf_<variant>.csv         the format of figures/data/trk_rf.csv (rpt, seed0_precv, seed1_precv, seed0_end) plus seed*_cvend, seed1_end: R_f error of RPTs 1-10
  tables/freeze_summary.csv                per variant and seed: RPTs passing at each read-out point, hour up to which all estimates pass, change of every estimate during the CV and in the
                                           30 min of rest after it (RPTs 3-6 mean), mean CV innovation, guard rejections in the CV holds and in the whole run
  freeze_cv_tables.md                      the same as markdown tables
Usage:  python scripts/freeze_cv.py --data data --results results/freeze [--variants none,phase,phase-noupdate] [--seeds 0,1]
        python scripts/freeze_cv.py --quick --data data --results results/quick     (evaluates the variants found in the folder; with a short trajectory only the RPTs it covers)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

SUFFIX = {"none": "", "voltage": "_freeze-voltage", "phase": "_freeze-phase", "phase-noupdate": "_freeze-phase-noupdate"}
RNG = {"Rf": 9.0, "LAM": 0.3, "Ds": 0.7, "LLI": 0.2}
POINT_LABEL = {"precv": "before CV", "cvend": "end of CV", "end": "end of RPT (30 min after the CV)"}


def main():
    p = cli.parser("Sec. III-D (CV freeze)", "evaluation of the ageing-tracker runs with the CV freeze", seeds="0,1")
    p.add_argument("--variants", default="none,phase,phase-noupdate"); p.add_argument("--hours-pass", type=float, default=3.0, help="innovation criterion [mV]")
    a = p.parse_args(); runtime.setup(a.core)
    import numpy as np, pandas as pd
    from dfnekf import ageing as ag, ageing_tracker as at, params as prm
    from dfnekf.step import DFN
    ref = os.path.join(a.data, "aging_dyn_rptslow_full.npz"); P = ag.load(ref); R0 = np.load(ref); se = R0["step_end_t"]; rpt = np.where(R0["is_rpt"])[0]
    m = DFN(prm.base(), 20, 10, 20, 12, 2, at.DT, a.grid); DT = at.DT
    os.makedirs(os.path.join(a.results, "tables"), exist_ok=True); os.makedirs(os.path.join(a.results, "source_data"), exist_ok=True)
    tab = lambda sub: type("A", (), dict(results=os.path.join(a.results, sub), item=a.item, grid=a.grid))
    summ, md = [], []
    for var in [v.strip() for v in a.variants.split(",")]:
        Rs, runs = [], {}
        for s in cli.parse_ids(a.seeds):
            f = os.path.join(a.results, "traj", f"ageing_track_seed{s}{SUFFIX[var]}.npz")
            if os.path.exists(f): runs[s] = dict(np.load(f, allow_pickle=True))
        if not runs: print("variant", var, "not found"); continue
        for s, d in runs.items():
            R = at.evaluate_rpt(d, P, R0, m, f"seed{s}", variant=var, points=("precv", "cvend", "end")); R["seed"] = s
            R["C_ok"] = R.innov_rms_mV_chunk <= a.hours_pass; R["pass"] = R.ABZ_ok & R.C_ok; Rs.append(R)
        T = pd.concat(Rs, ignore_index=True); cli.write_csv(T, tab("tables"), f"freeze_rpt_{var}.csv", note=f"variant {var}: errors in % of the ageing range at precv / cvend / end", float_format="%.6g")
        out = {}
        for pt, nm in [("precv", "precv"), ("cvend", "cvend"), ("end", "end")]:
            for s in runs:
                q = T[(T.seed == s) & (T.point == pt) & (T.rpt >= 1)].sort_values("rpt")
                if pt == "precv" or pt == "end" or pt == "cvend": out.setdefault("rpt", q.rpt.values); out[f"seed{s}_{nm}"] = q.Rf_err.values.round(3)
        L = max(len(v) for v in out.values()); keep = {k: v for k, v in out.items() if len(v) == L}
        order = ["rpt", "seed0_precv", "seed1_precv", "seed0_end"] + [k for k in keep if k not in ("rpt", "seed0_precv", "seed1_precv", "seed0_end")]
        cli.write_csv(pd.DataFrame({k: keep[k] for k in order if k in keep}), tab("source_data"), f"trk_rf_{var}.csv", note=f"{var}: R_f error (% of range) of RPTs 1-10, format of figures/data/trk_rf.csv + cvend columns", float_format="%.6g")
        for s, d in runs.items():
            vals, li = d["vals"], d["li"]; Linv = at.inventory_loss(m, P, li); n10 = len(vals); nst = len(d["ps_innov"]); names = [str(c) for c in d["coords"]]
            k10 = lambda t: min(int(round(t / 10)), n10 - 1); st = lambda t: int(round(t / DT))
            row = dict(variant=var, seed=s, rpts_covered=int(T[(T.seed == s) & (T.point == "precv")].rpt.max()))
            for pt in ("precv", "cvend", "end"):
                q = T[(T.seed == s) & (T.point == pt) & (T.rpt >= 1)].sort_values("rpt"); ok = q["pass"].values; first_bad = int(np.argmax(~ok)) if (~ok).any() else len(ok)
                row[f"{pt}_rpts_passing"] = int(ok.sum()); row[f"{pt}_first_failing_rpt"] = int(q.rpt.values[first_bad]) if first_bad < len(ok) else -1
                row[f"{pt}_hour_all_pass"] = float(q.t_h.values[first_bad - 1]) if first_bad >= 1 else 0.0
                if first_bad < len(ok):
                    r_ = q.iloc[first_bad]; row[f"{pt}_first_failing_what"] = ",".join([n_ for n_, f_ in [("Rf", abs(r_.Rf_err) > 3.90), ("LAM", abs(r_.LAM_err) > 3.10), ("Ds", abs(r_.Ds_err) > 5.37), ("inv", abs(r_.LLI_inv_err) > 3.00), ("z", not r_.z_ok), ("edge", not r_.B_ok), ("innov", not r_.C_ok)] if f_])
            dch = {c: [] for c in [f"d_{n}_{ph}" for n in names + ["inventory"] for ph in ("CV", "rest")]}; inn, rej_cv, per = [], 0, []
            for kr, c in enumerate(rpt):
                t0, t1, t2 = se[c][16], se[c][17], se[c][18]
                if st(t2) > nst: continue
                a0, a1, a2 = k10(t0), k10(t1), k10(t2); sc = slice(st(t0), st(t1))
                inv_cv, inv_rest = 100 * (Linv[a1] - Linv[a0]) / 0.2, 100 * (Linv[a2] - Linv[a1]) / 0.2
                pr_ = dict(rpt=kr, innov_mean_CV_mV=1e3 * float(d["ps_innov"][sc].astype(float).mean()), guard_rejections_CV=int(d["ps_rejected"][sc].sum()))
                for i, n in enumerate(names): pr_[f"d_{n}_CV"] = 100 * (vals[a1, i] - vals[a0, i]) / RNG[n]; pr_[f"d_{n}_rest"] = 100 * (vals[a2, i] - vals[a1, i]) / RNG[n]
                pr_["d_inventory_CV"], pr_["d_inventory_rest"] = inv_cv, inv_rest; per.append(pr_)
            PR = pd.DataFrame(per); cli.write_csv(PR, tab("tables"), f"freeze_cv_changes_{var}_seed{s}.csv", note="change of each estimate during the CV and in the 30 min after it (points of range; inventory points of 0.2), mean CV innovation, guard rejections in the CV", float_format="%.6g")
            g = PR[(PR.rpt >= 3) & (PR.rpt <= 6)]
            if len(g):
                for c in PR.columns[1:]:
                    if c != "guard_rejections_CV": row[f"mean3_6_{c}"] = float(g[c].mean())
            row["guard_rejections_CV_total"] = int(PR.guard_rejections_CV.sum()) if len(PR) else 0; row["guard_rejections_run"] = int(d["n_rejected"].sum())
            row["innov_rms_max_mV"] = 1e3 * float(np.nanmax(d["innov_rms"])); row["frozen_frac_run"] = float((d["ps_frozen"] > 0).mean()) if "ps_frozen" in d else 0.0
            summ.append(row)
    S = pd.DataFrame(summ); cli.write_csv(S, tab("tables"), "freeze_summary.csv", note="summary per variant and seed (criteria: error within bias floor + 3 %, |z| <= 3, edge > 2 %, innovation RMS <= 3 mV per 20000-step chunk)", float_format="%.6g")
    with open(os.path.join(a.results, "freeze_cv_tables.md"), "w", encoding="utf8") as f:
        f.write(S.round(3).T.to_markdown() if hasattr(S, "to_markdown") and _has_tabulate() else S.round(3).T.to_string()); f.write("\n")
    print(S.round(3).T.to_string())


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
