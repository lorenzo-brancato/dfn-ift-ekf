"""Pilot of the full UKF with the closure solved at every sigma point (Appendix B / Table IV, remark on the non-consistency of the UKF).

The manuscript's full UKF performs ONE Newton iteration of the closure per sigma point (``--closure-iters 1``). The pilot repeats four twin runs at th2 (seeds 0, 3, 14 and 5: the three
inconsistent runs of the 20 and one consistent run) with FOUR iterations, which brings the closure residual at the sigma points to machine precision in the median, and compares the
time-averaged NEES (R_f, D_s) and the parameter errors run by run with the one-iteration runs. It is a result of its own: the 30-run UKF set of the paper is NOT repeated at k = 4.

  python scripts/ukf_pilot.py --grid 4:1 --data data --results results/ukf_pilot_k4 --seeds 0,3,14,5     # runs the four filters, one process each is advised; about 8.7 h CPU per run
  python scripts/ukf_pilot.py --aggregate --results results/ukf_pilot_k4 --k1 results/4to1                 # table of NEES k=1 vs k=4 (the k=1 runs come from twin_nees.py --filter ukf --closure-iters 1)
  python scripts/ukf_pilot.py --quick --grid 4:1 --data data --results results/quick                       # 600 s of the protocol, k = 4

Outputs: twin_ukf4_th2_s<seed>.csv (per run, as twin_nees.py; resSigma_* = residual of the closure over all sigma points, maximum / median / p99 over the steps),
tables/tab_ukf_pilot.csv (aggregation), traj/*.npz (not committed).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

HERE = os.path.dirname(os.path.abspath(__file__))
K = "4"


def read_k1(path, seed):
    """k = 1 run of the same seed: from a folder with twin_ukf1_th2_s<seed>.csv (twin_nees.py) or from a CSV with columns seed, nees_RfDs_mean, Rf_err_pct, Ds_err_pct."""
    import pandas as pd
    f = os.path.join(path, f"twin_ukf1_th2_s{seed}.csv")
    if os.path.isdir(path): return pd.read_csv(f, comment="#").iloc[0] if os.path.exists(f) else None
    d = pd.read_csv(path, comment="#"); d = d[d.seed == seed]; return d.iloc[0] if len(d) else None


def aggregate(a):
    import pandas as pd
    rows = []
    for s in cli.parse_ids(a.seeds):
        f = os.path.join(a.results, f"twin_ukf{K}_th2_s{s}.csv")
        if not os.path.exists(f): continue
        n = pd.read_csv(f, comment="#").iloc[0]; o = read_k1(a.k1, s) if a.k1 else None
        r = dict(seed=s, NEES_k4=n.nees_RfDs_mean, Rf_err_pct_k4=n.Rf_err_pct, Ds_err_pct_k4=n.Ds_err_pct, resSigma_max_k4=n.get("resSigma_max"), resSigma_p99_k4=n.get("resSigma_p99"), resSigma_median_k4=n.get("resSigma_median"), cpu_s_k4=n.cpu_s)
        if o is not None: r.update(NEES_k1=o.nees_RfDs_mean, Rf_err_pct_k1=o.Rf_err_pct, Ds_err_pct_k1=o.Ds_err_pct)
        rows.append(r)
    if not rows: print("no pilot run found in", a.results); return
    d = pd.DataFrame(rows); print(d.to_string())
    os.makedirs(os.path.join(a.results, "tables"), exist_ok=True)
    cli.write_csv(d, a, os.path.join("tables", "tab_ukf_pilot.csv"), note="UKF pilot, closure iterations per sigma point 4 vs 1, twin th2", float_format="%.6g")


def main():
    p = cli.parser("Appendix B (UKF pilot, closure solved at the sigma points)", "four twin runs of the full UKF at th2 with 4 closure iterations per sigma point", seeds="0,3,14,5", states="th2")
    p.add_argument("--aggregate", action="store_true"); p.add_argument("--k1", default=None, help="folder with the k = 1 runs (twin_ukf1_th2_s*.csv) or a CSV with columns seed, nees_RfDs_mean, Rf_err_pct, Ds_err_pct")
    p.add_argument("--progress-chunk", type=int, default=0, help="print the progress every N steps (changes the NEES at the 1e-3 relative level, see twin_nees.py)")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    runtime.setup(a.core)
    cmd = [sys.executable, os.path.join(HERE, "twin_nees.py"), "--grid", a.grid, "--data", a.data, "--results", a.results, "--filter", "ukf", "--closure-iters", K, "--states", "th2", "--seeds", a.seeds,
           "--progress-chunk", str(a.progress_chunk)] + (["--quick"] if a.quick else []) + (["--core", str(a.core)] if a.core is not None else [])
    rc = subprocess.call(cmd)
    if rc == 0 and not a.quick: aggregate(a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
