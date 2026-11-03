"""Static-protocol Monte Carlo of the EKF (and of the full UKF) against PyBaMM: Fig. 5, Sec. III-B (IV-B), RMSE columns of Table V.

Five ageing states (th1..th5) x seeds 0-9. Truth = PyBaMM P6Lx2 reference (40/20/40 mesh, dt 0.5 s, 4320 s); measurement noise default_rng(seed) with sigma_V = 2 mV;
initial offsets default_rng(200 + seed), +-40 % of the parameter range; EKF on (R_f, LAM, D_s) in logit coordinates, LLI fixed through the initial state.
``--filter ukf`` runs the same experiment with the square-root UKF (``--ukf-iters`` Newton iterations of the closure at every sigma point; 1 = the 'full UKF' of the paper,
``conv`` = Newton to convergence at every sigma point). Per-run CSV: <results>/static_<ekf|ukf<k>>_<state>_s<seed>.csv, trajectories <results>/traj/*.npz.

``--aggregate`` writes <results>/tables/tab_mc_m17.csv and <results>/source_data/{fig_mc_m17.csv, fig_mc.csv}: p90 and RMSE of the final error (% of range), batch-fit bias
(read from <results>/bias_<state>.csv written by bias.py), p90 and RMSE of the error after bias subtraction, fraction of runs touching the logit edge in the first minute.
Rows: EKF for the states found, and the UKF row for th2 if its runs are present.

  python scripts/static_mc.py --grid 4:1 --data data --results results --states th1 --seeds 0-9                  # one job: ~6 min per run, 1 thread
  python scripts/static_mc.py --grid 4:1 --filter ukf --ukf-iters 1 --states th2 --seeds 0-9 ...                 # full UKF: ~45 min per run
  python scripts/bias.py --grid 4:1 ...; python scripts/static_mc.py --aggregate --results results
  python scripts/static_mc.py --quick
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

p = cli.parser("Fig. 5, Sec. III-B, Table V (RMSE)", "static-protocol Monte Carlo of the EKF / UKF against PyBaMM", seeds="0-9")
p.add_argument("--filter", default="ekf", choices=["ekf", "ukf"])
p.add_argument("--ukf-iters", default="1", help="Newton iterations of the closure per sigma point (integer) or 'conv'; UKF only")
p.add_argument("--redo", action="store_true", help="recompute runs whose CSV already exists (default: skip them)")
p.add_argument("--aggregate", action="store_true", help="build the tables from the per-run files in --results")
args = p.parse_args()
runtime.setup(args.core)
import numpy as np, pandas as pd
from dfnekf.montecarlo import RANGE_W, QUICK_STEPS, ci95, p90, parse_iters, iters_label

ITERS = parse_iters(args.ukf_iters)
FILT_TAG = "ekf" if args.filter == "ekf" else f"ukf{iters_label(ITERS)}"
COORDS = ("Rf", "LAM", "Ds")
EKF_LABEL, UKF_LABEL = "EKF", "UKF k=1 completo"          # row labels of the published tables (kept so that the source data files are identical)


def run_jobs():
    from dfnekf.experiment import Case
    from dfnekf.montecarlo import run_one, truncate
    clock = cli.Clock(args, "static_mc.py"); pre = "quick_" if args.quick else ""
    states = ["th2"] if args.quick else cli.states(args); seeds = [0] if args.quick else cli.parse_ids(args.seeds)
    for tag in states:
        case = None
        for i in seeds:
            name = f"{pre}static_{FILT_TAG}_{tag}_s{i}"; path = os.path.join(args.results, name + ".csv")
            if os.path.exists(path) and not args.redo: print("skip", name, flush=True); continue
            if case is None:
                case = Case(tag, args.grid, twin=False, data_dir=args.data)
                if args.quick: truncate(case, QUICK_STEPS)
            res, out = run_one(case, args.filter, i, closure_iters=ITERS)
            cli.write_csv(pd.DataFrame([res]), args, name + ".csv", note="static protocol vs PyBaMM, one run")
            os.makedirs(os.path.join(args.results, "traj"), exist_ok=True); np.savez(os.path.join(args.results, "traj", name + ".npz"), **out)
            print(name, {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in res.items() if k in ("Rf_err_pct", "LAM_err_pct", "Ds_err_pct", "finite", "cpu_s")}, flush=True)
    clock.log(f"{args.filter} {states} {seeds}")


def read_runs(tag, ftag):
    fs = [os.path.join(args.results, f"static_{ftag}_{tag}_s{i}.csv") for i in range(1000) if os.path.exists(os.path.join(args.results, f"static_{ftag}_{tag}_s{i}.csv"))]
    return pd.concat([pd.read_csv(f, comment="#") for f in fs], ignore_index=True) if fs else None


def read_bias(tag):
    f = os.path.join(args.results, f"bias_{tag}.csv")
    if not os.path.exists(f): print(f"WARNING: {f} missing, bias columns are NaN (run bias.py)"); return None
    return pd.read_csv(f, comment="#").iloc[0]


def stats_row(d, tag, label):
    b = read_bias(tag); r = dict(theta=tag, runs=len(d), filter=label); extra = {}
    for c in COORDS:
        sg = 100 * (d[f"{c}_end"].values - d[f"{c}_true"].values) / RANGE_W[c]; bc = float(b[f"bias_{c}_pct"]) if b is not None else np.nan
        r[f"{c}_p90"] = p90(d[f"{c}_err_pct"]); r[f"{c}_rmse"] = float(np.sqrt(np.mean(d[f"{c}_err_pct"] ** 2))); r[f"bias_{c}"] = abs(bc); r[f"{c}_p90_minus_bias"] = p90(np.abs(sg - bc))
        extra[f"{c}_rmse_minus_bias"] = float(np.sqrt(np.mean((sg - bc) ** 2)))
    r["touch_edge_first60s"] = float(d.touch_edge_first60s.mean())
    r["pass_p90_le_10"] = bool(max(r["Rf_p90"], r["LAM_p90"], r["Ds_p90"]) <= 10); r["pass_p90_minus_bias_le_3"] = bool(max(r["Rf_p90_minus_bias"], r["LAM_p90_minus_bias"], r["Ds_p90_minus_bias"]) <= 3)
    r.update(extra); return r


def aggregate():
    rows = []
    for tag in ["th1", "th2", "th3", "th4", "th5"]:
        d = read_runs(tag, "ekf")
        if d is not None: rows.append(stats_row(d, tag, EKF_LABEL))
    d = read_runs("th2", f"ukf{iters_label(ITERS)}")
    if d is not None: rows.append(stats_row(d, "th2", UKF_LABEL if ITERS == 1 else f"UKF k={iters_label(ITERS)}"))
    if not rows: sys.exit("no runs found in " + args.results)
    tab = pd.DataFrame(rows)
    for sub in ("tables", "source_data"): os.makedirs(os.path.join(args.results, sub), exist_ok=True)
    cli.write_csv(tab, args, os.path.join("tables", "tab_mc_m17.csv"), note="error = absolute final error in % of the range; p90 over seeds", float_format="%.6g")
    cli.write_csv(tab, args, os.path.join("source_data", "fig_mc_m17.csv"), note="Fig. 5 d-f (bars)", float_format="%.6g")
    sc = []
    for tag in ["th1", "th2", "th3", "th4", "th5"]:
        d = read_runs(tag, "ekf")
        if d is None: continue
        for _, r in d.iterrows():
            sc.append(dict(theta=tag, seed=int(r.seed), a_Rf_true_over_Rf0=r.Rf_true, a_Rf_estimated_over_Rf0=r.Rf_end, b_LAM_true=r.LAM_true, b_LAM_estimated=r.LAM_end, c_Ds_true_over_Ds0=r.Ds_true, c_Ds_estimated_over_Ds0=r.Ds_end))
    cli.write_csv(pd.DataFrame(sc), args, os.path.join("source_data", "fig_mc.csv"), note="Fig. 5 a-c: final estimate vs true value (nominal fresh R_f and D_s = 1)", float_format=None)
    bb = []
    for tag in ["th1", "th2", "th3", "th4", "th5"]:
        b = read_bias(tag)
        if b is not None: bb.append(dict(theta=tag, bias_Rf_pct=b.bias_Rf_pct, bias_LAM_pct=b.bias_LAM_pct, bias_Ds_pct=b.bias_Ds_pct))
    if bb: cli.write_csv(pd.DataFrame(bb), args, os.path.join("tables", "bias.csv"), note="signed batch-fit bias, % of range", float_format=None)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(tab[["theta", "filter", "runs", "Rf_p90", "Rf_rmse", "LAM_rmse", "Ds_p90", "Ds_rmse", "Ds_p90_minus_bias", "touch_edge_first60s"]].round(4).to_string())


if __name__ == "__main__":
    aggregate() if args.aggregate else run_jobs()
