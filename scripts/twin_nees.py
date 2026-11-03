"""Twin experiments and filter consistency (NEES): Fig. 6, Table V, Appendix D ('consistency at the fresh cell').

The truth is the grey-box model itself (``Case(twin=True)``, noise-free ROM voltage at the true parameters) plus 2 mV noise, so any inconsistency is due to the filter,
not to model error. Seeds 0-19 by default. NEES of (R_f, D_s) (2 dof, logit coordinates, covariance block of the filter), averaged over the steps of each run.
Filters: EKF, or ``--filter ukf`` (square-root UKF, ``--ukf-iters`` Newton iterations of the closure per sigma point, 1 = the 'full UKF' of the paper, ``conv`` = to convergence).
Appendix D variants (run on th1; use the same --results folder, file names carry the variant):
  --grid uniform           uniform electrode grid (the 15.08 vs 14.97 comparison)
  --p0-scale-rf 2          initial standard deviation of R_f doubled (variance x4)     -> 'wider initial covariance'
  --init-error relative    initial R_f error sampled relative to the true value (R_f,0 = R_f,true exp(u), |u| <= ln 2)
Per-run CSV: <results>/twin_<ekf|ukf<k>>[_uniform][_p0rf<s>][_relinit]_<state>_s<seed>.csv; trajectories in <results>/traj/*.npz.

``--aggregate`` writes in <results>/tables: tab_twin.csv (all variants found), tab_filters_m17.csv (EKF vs full UKF, uses the static runs of --static-results = th2 files of
static_mc.py and <results>/tables/tab_cost_jax_m17.csv if present), tab_appD.csv (numbers of Appendix D); in <results>/source_data: fig_nees.csv (Fig. 6), fig_nees_m17.csv.

  python scripts/twin_nees.py --grid 4:1 --states th1,th2 --seeds 0-19 --results results                   # EKF, ~6 min per run
  python scripts/twin_nees.py --grid 4:1 --filter ukf --ukf-iters 1 --states th2 --seeds 0-19 ...          # full UKF, ~46 min per run
  python scripts/twin_nees.py --grid 4:1 --states th1 --p0-scale-rf 2 ...;  --init-error relative ...;  --grid uniform --states th1 ...
  python scripts/twin_nees.py --aggregate --results results
  python scripts/twin_nees.py --quick
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

p = cli.parser("Fig. 6, Table V, Appendix D", "twin experiments: NEES consistency of the EKF and of the full UKF", seeds="0-19", states="th1,th2")
p.add_argument("--filter", default="ekf", choices=["ekf", "ukf"])
p.add_argument("--ukf-iters", default="1", help="Newton iterations of the closure per sigma point (integer) or 'conv'; UKF only")
p.add_argument("--p0-scale-rf", type=float, default=1.0, help="factor on the initial standard deviation of R_f (2 = wider initial covariance)")
p.add_argument("--init-error", default="absolute", choices=["absolute", "relative"], help="initial R_f error: +-40 %% of the range, or relative to the true value")
p.add_argument("--redo", action="store_true", help="recompute runs whose CSV already exists (default: skip them)")
p.add_argument("--aggregate", action="store_true", help="build the tables from the per-run files in --results")
p.add_argument("--static-results", default=None, help="folder with the static_mc.py runs (RMSE columns of tab_filters); default: --results")
args = p.parse_args()
runtime.setup(args.core)
import numpy as np, pandas as pd
from dfnekf.montecarlo import QUICK_STEPS, ci95, p90, parse_iters, iters_label

ITERS = parse_iters(args.ukf_iters)
STATIC = args.static_results or args.results
RANGE_W = {"Rf": 9.0, "LAM": 0.3, "Ds": 0.7}


def ftag_of(filt): return "ekf" if filt == "ekf" else f"ukf{iters_label(ITERS)}"


def variant_tag(grid, p0rf, init_error):
    return ("_uniform" if grid == "uniform" else "") + (f"_p0rf{p0rf:g}" if p0rf != 1.0 else "") + ("_relinit" if init_error == "relative" else "")


def run_jobs():
    from dfnekf.experiment import Case
    from dfnekf.montecarlo import run_one, truncate
    clock = cli.Clock(args, "twin_nees.py"); pre = "quick_" if args.quick else ""
    states = ["th2"] if args.quick else cli.states(args); seeds = [0] if args.quick else cli.parse_ids(args.seeds)
    vt = variant_tag(args.grid, args.p0_scale_rf, args.init_error)
    for tag in states:
        case = None
        for i in seeds:
            name = f"{pre}twin_{ftag_of(args.filter)}{vt}_{tag}_s{i}"; path = os.path.join(args.results, name + ".csv")
            if os.path.exists(path) and not args.redo: print("skip", name, flush=True); continue
            if case is None:
                case = Case(tag, args.grid, twin=True, data_dir=args.data)
                if args.quick: truncate(case, QUICK_STEPS)
            res, out = run_one(case, args.filter, i, closure_iters=ITERS, p0_scale_rf=args.p0_scale_rf, init_error=args.init_error)
            res.update(p0_scale_rf=args.p0_scale_rf, init_error=args.init_error)
            cli.write_csv(pd.DataFrame([res]), args, name + ".csv", note="twin experiment, one run")
            os.makedirs(os.path.join(args.results, "traj"), exist_ok=True); np.savez(os.path.join(args.results, "traj", name + ".npz"), **out)
            print(name, {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in res.items() if k in ("Rf_err_pct", "Ds_err_pct", "nees_RfDs_mean", "finite", "cpu_s")}, flush=True)
    clock.log(f"{args.filter} {states} {seeds} {vt}")


def read_runs(folder, prefix):
    fs = [os.path.join(folder, f"{prefix}_s{i}.csv") for i in range(1000) if os.path.exists(os.path.join(folder, f"{prefix}_s{i}.csv"))]
    return pd.concat([pd.read_csv(f, comment="#") for f in fs], ignore_index=True) if fs else None


def twin_stats(d, variant, tag):
    x = d.nees_RfDs_mean.values
    return dict(variant=variant, theta=tag, runs=len(d), NEES_RfDs_mean=float(x.mean()), NEES_ci95=ci95(x), NEES_median=float(np.median(x)), n_runs_gt10=int((x > 10).sum()), n_runs_gt3=int((x > 3).sum()),
                frac_runs_gt3=float((x > 3).mean()), NEES_in_1_3=bool(1 <= x.mean() <= 3), NEES_3dof_mean=float(d.nees_full_mean.mean()), p90_Rf=p90(d.Rf_err_pct), p90_LAM=p90(d.LAM_err_pct), p90_Ds=p90(d.Ds_err_pct),
                median_Rf_err_pct=float(d.Rf_err_pct.median()), median_Ds_err_pct=float(d.Ds_err_pct.median()), max_Rf_err_pct=float(d.Rf_err_pct.max()), max_Ds_err_pct=float(d.Ds_err_pct.max()),
                touch_edge_first60s=float(d.touch_edge_first60s.mean()), finite_all=bool(d.finite.all()), cpu_s_mean=float(d.cpu_s.mean()) if "cpu_s" in d else float("nan"))


def static_stats(d):
    r = {}
    for c in ("Rf", "LAM", "Ds"): r[f"p90_{c}"] = p90(d[f"{c}_err_pct"]); r[f"rmse_{c}"] = float(np.sqrt(np.mean(d[f"{c}_err_pct"] ** 2)))
    return r


def cost_ms(kind):
    f = os.path.join(args.results, "tables", "tab_cost_jax_m17.csv")
    if not os.path.exists(f): return np.nan
    c = pd.read_csv(f, comment="#"); c = c[c.grid.str.contains("20-10-20") & c.grid.str.contains("4:1") & (c.Nm == 12) & (c.kind == "per call")] if "kind" in c and (c.kind == "per call").any() else c[c.grid.str.contains("20-10-20 4:1") & (c.Nm == 12)]
    fl = c["filter"].str
    sel = c[fl.contains("EKF") & fl.contains("newton")] if kind == "ekf" else c[fl.startswith("UKF k=1") & fl.contains("completo|full", regex=True)]
    return float(sel.median_ms.iloc[0]) if len(sel) else np.nan


def aggregate():
    R = args.results; vs = [("ekf", "", "th1"), ("ekf", "", "th2"), (f"ukf{iters_label(ITERS)}", "", "th2"), ("ekf", "_uniform", "th1"), ("ekf", "_p0rf2", "th1"), ("ekf", "_relinit", "th1")]
    names = {"": "baseline", "_uniform": "uniform grid", "_p0rf2": "wider initial covariance (R_f std x2)", "_relinit": "relative initial error"}
    rows = []; D = {}
    for ft, vt, tag in vs:
        d = read_runs(R, f"twin_{ft}{vt}_{tag}")
        if d is None: continue
        D[(ft, vt, tag)] = d; rows.append(twin_stats(d, f"{ft}{vt}", tag))
    if not rows: sys.exit("no runs found in " + R)
    for sub in ("tables", "source_data"): os.makedirs(os.path.join(R, sub), exist_ok=True)
    tw = pd.DataFrame(rows); cli.write_csv(tw, args, os.path.join("tables", "tab_twin.csv"), note="twin statistics per variant", float_format="%.6g")
    ukf_key = (f"ukf{iters_label(ITERS)}", "", "th2"); ek2 = D.get(("ekf", "", "th2")); ek1 = D.get(("ekf", "", "th1")); uk = D.get(ukf_key)
    # ---- Fig. 6 data
    fn = []
    if ek2 is not None: fn += [dict(panel="a", theta="th2", filter="EKF", run=int(r), NEES_RfDs_time_mean=v) for r, v in zip(ek2.run, ek2.nees_RfDs_mean)]
    if uk is not None: fn += [dict(panel="a", theta="th2", filter="UKF full", run=int(r), NEES_RfDs_time_mean=v) for r, v in zip(uk.run, uk.nees_RfDs_mean)]
    if ek1 is not None: fn += [dict(panel="b", theta="th1", filter="EKF", run=int(r), NEES_RfDs_time_mean=v) for r, v in zip(ek1.run, ek1.nees_RfDs_mean)]
    cli.write_csv(pd.DataFrame(fn), args, os.path.join("source_data", "fig_nees.csv"), note="Fig. 6: time-averaged NEES (R_f, D_s) per run", float_format=None)
    fm = []
    if ek2 is not None: fm += [dict(filter="EKF", seed=int(r), NEES_RfDs_mean=v) for r, v in zip(ek2.run, ek2.nees_RfDs_mean)]
    if uk is not None: fm += [dict(filter="UKF completo", seed=int(r), NEES_RfDs_mean=v) for r, v in zip(uk.run, uk.nees_RfDs_mean)]
    cli.write_csv(pd.DataFrame(fm), args, os.path.join("source_data", "fig_nees_m17.csv"), note="Fig. 6a, th2", float_format="%.6g")
    # ---- Table V
    fr = []; c_e, c_u = cost_ms("ekf"), cost_ms("ukf")
    for lab, d, ft, cst in [("EKF strutturato", ek2, "ekf", c_e), ("UKF k=1 completo (Jacobiana per sigma-point), alpha = 1", uk, ukf_key[0], c_u)]:
        if d is None: continue
        t = twin_stats(d, "", "th2"); s = read_runs(STATIC, f"static_{ft}_th2"); st = static_stats(s) if s is not None else {}
        fr.append(dict(filter=lab, twin_runs=t["runs"], NEES_RfDs_mean=t["NEES_RfDs_mean"], NEES_ci95=t["NEES_ci95"], NEES_median=t["NEES_median"], NEES_3dof_mean=t["NEES_3dof_mean"], twin_p90_Rf=t["p90_Rf"], twin_p90_LAM=t["p90_LAM"], twin_p90_Ds=t["p90_Ds"],
                       NEES_in_1_3=t["NEES_in_1_3"], n_runs_NEES_gt10=t["n_runs_gt10"], n_runs_NEES_gt3=t["n_runs_gt3"], pybamm_runs=0 if s is None else len(s), pybamm_p90_Rf=st.get("p90_Rf", np.nan), pybamm_p90_LAM=st.get("p90_LAM", np.nan),
                       pybamm_p90_Ds=st.get("p90_Ds", np.nan), pybamm_rmse_Rf=st.get("rmse_Rf", np.nan), pybamm_rmse_LAM=st.get("rmse_LAM", np.nan), pybamm_rmse_Ds=st.get("rmse_Ds", np.nan), cost_ms_20_10_20_Nm12_N653=cst, cost_ratio_vs_EKF_N653=cst / c_e))
    cli.write_csv(pd.DataFrame(fr), args, os.path.join("tables", "tab_filters_m17.csv"), note="Table V; cost from tab_cost_jax_m17.csv", float_format="%.6g")
    # ---- Appendix D
    a = []
    def add(q, v, n="", note=""): a.append(dict(quantity=q, value=v, runs=n, note=note))
    def mean_of(key): return (D[key].nees_RfDs_mean.mean(), len(D[key])) if key in D else (np.nan, 0)
    for q, key, note in [("NEES_th1_nonuniform_grid", ("ekf", "", "th1"), "4:1 (14.97)"), ("NEES_th1_uniform_grid", ("ekf", "_uniform", "th1"), "uniform (15.08)"), ("NEES_th1_wider_P0", ("ekf", "_p0rf2", "th1"), "4:1, R_f std x2 (10.3)"),
                         ("NEES_th1_relative_init_error", ("ekf", "_relinit", "th1"), "4:1 (6.6)")]:
        v, n = mean_of(key); add(q, v, n, note)
    if ek1 is not None:
        add("th1_runs_NEES_gt10", int((ek1.nees_RfDs_mean > 10).sum()), len(ek1), "'six of 20 runs'"); add("th1_median_final_err_Rf_pct", ek1.Rf_err_pct.median(), len(ek1), "0.4"); add("th1_median_final_err_Ds_pct", ek1.Ds_err_pct.median(), len(ek1), "1.9")
        add("th1_twin_touch_edge_first60s_frac", ek1.touch_edge_first60s.mean(), len(ek1), "fraction of twin runs touching the logit edge")
    s1 = read_runs(STATIC, "static_ekf_th1")
    if s1 is not None: add("th1_pybamm_touch_edge_first60s_frac", s1.touch_edge_first60s.mean(), len(s1), "half of the PyBaMM runs (static_mc)")
    f = os.path.join(R, "tables", "tab_mc_m17.csv")
    if os.path.exists(f):
        m = pd.read_csv(f, comment="#"); m = m[(m.theta == "th1") & (m["filter"] == "EKF")]
        if len(m): add("th1_Ds_p90_minus_bias_pct", float(m.Ds_p90_minus_bias.iloc[0]), int(m.runs.iloc[0]), "3.4, from static_mc/bias")
    cli.write_csv(pd.DataFrame(a), args, os.path.join("tables", "tab_appD.csv"), note="Appendix D numbers", float_format="%.6g")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(tw[["variant", "theta", "runs", "NEES_RfDs_mean", "NEES_ci95", "NEES_median", "n_runs_gt10", "n_runs_gt3", "NEES_3dof_mean"]].round(3).to_string()); print(pd.DataFrame(a).to_string())


if __name__ == "__main__":
    aggregate() if args.aggregate else run_jobs()
