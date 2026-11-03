"""Fig. 4, Sec. IV-A, Appendix B: computational cost of one filter step versus the state dimension N.

Medians (and p99, p99.9, worst) of the wall time of ONE call of the jitted step (Python dispatch included), one CPU thread, fixed input I = 5 A, y = 3.7 V:
  * EKF (structured, converged closure with warm start and cold fallback = the filter of the paper) and, for comparison, the EKF with a cold-started closure,
    >= 3600 calls per point; plus the amortized cost of 1000 steps inside one lax.scan at N = 653 (a mean, not a median);
  * the full UKF (the closure Newton-iterated --ukf-iters times at every sigma point, 1 as in the manuscript; 'none' = converged), 100 calls per point;
  * the UKF on the SPMe (N = 50 and 83), 600 calls.
Points: grids 10/5/10 with N_m = 4, 8, 12 (N = 168, 248, 328), 20/10/20 with N_m = 8, 12, 16 (N = 493, 653, 813) and 40/20/40 with N_m = 16 (N = 1623, EKF only), for the grid given by
--grid (4:1 -> file costgrid_4.csv; uniform -> costgrid_none.csv; Fig. 4 uses the 4:1 file, Table I the uniform one). Re-running a point replaces its rows in the file (the points can be timed one after the other).

THE TIMINGS ARE ONLY MEANINGFUL ON AN OTHERWISE IDLE MACHINE: close everything else, use --core to pin the process to one core, and run one timing process at a time. The script prints a warning
and records the number of other busy processes (needs psutil; skipped if it is not installed) and the machine description in <results>/cost_meta.csv.

Output: <results>/costgrid_<4|none>.csv with columns filter, grid, Nm, N, kind, median_ms, p99_ms, p999_ms, worst_ms, n_calls, note.
--aggregate writes <results>/tables/tab_cost_jax_m17.csv and <results>/source_data/fig_cost_m17.csv (both grids).

Commands (about 45 min per grid on the machine of the paper):
  python scripts/cost.py --grid 4:1 --core 2 --results results
  python scripts/cost.py --grid uniform --core 2 --results results
  python scripts/cost.py --grid 4:1 --points 20/10/20:12 --ukf-iters none --results results      (other UKF closure iterations; the file name gets a suffix)
  python scripts/cost.py --aggregate --results results
  python scripts/cost.py --quick --results /tmp/cost_quick
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

POINTS = [((10, 5, 10), 4), ((10, 5, 10), 8), ((10, 5, 10), 12), ((20, 10, 20), 8), ((20, 10, 20), 12), ((20, 10, 20), 16), ((40, 20, 40), 16)]
COLS = ["filter", "grid", "Nm", "N", "kind", "median_ms", "p99_ms", "p999_ms", "worst_ms", "n_calls", "note"]
EKF_LABEL = "EKF structured (JAX), {} closure"
SPME_LABEL = "UKF-SPMe (JAX, scan of 1 step)"


def ukf_label(k):
    return "UKF k=1 full (Jacobian at every sigma point)" if k == 1 else (f"UKF k={k} full (Jacobian at every sigma point)" if k is not None else "UKF converged-closure full (Jacobian at every sigma point)")


def parse_points(s):
    out = []
    for item in s.split(","):
        g, nm = item.split(":"); out.append((tuple(int(x) for x in g.split("/")), int(nm)))
    return out


def main():
    p = cli.parser("Fig. 4, Sec. IV-A, Appendix B (cost)", "median and p99 cost per filter step (EKF, full UKF, UKF-SPMe) versus the state dimension; IDLE MACHINE REQUIRED")
    p.add_argument("--points", default=None, help="comma-separated 'n/n/n:N_m' (default: the seven points of the paper)")
    p.add_argument("--parts", default="ekf,ukf,spme", help="comma-separated subset of ekf, ukf, spme")
    p.add_argument("--n-ekf", type=int, default=3600); p.add_argument("--n-ukf", type=int, default=100); p.add_argument("--n-spme", type=int, default=600)
    p.add_argument("--ukf-iters", default="1", help="Newton iterations of the closure at every sigma point (1 in the paper; 'none' = converged)")
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    runtime.setup(a.core)
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from dfnekf import params as prm, timing
    from dfnekf.step import DFN
    from dfnekf.ekf import EKF
    from dfnekf.ukf import UKF
    from dfnekf.spme import SPMe, UKFSPMe
    clock = cli.Clock(a, "cost.py"); parts = [s.strip() for s in a.parts.split(",")]
    kit = None if str(a.ukf_iters).lower() == "none" else int(a.ukf_iters)
    pts = parse_points(a.points) if a.points else POINTS
    n_ekf, n_ukf, n_spme = a.n_ekf, a.n_ukf, a.n_spme
    if a.quick: pts = [((10, 5, 10), 4)]; n_ekf, n_ukf, n_spme = 20, 3, 20
    n_other = timing.idle_warning(); info = timing.machine_info(); gtag = a.grid
    note = f"{info['processor']}; 1 thread; JAX with Python/XLA dispatch; grid {gtag}; other busy processes at start: {n_other}; not BMS hardware"
    rows = []

    def setup(grid, Nm, closure="newton"):
        m = DFN(prm.base(), *grid, Nm, 2, 0.5, a.grid); ek = EKF(m, closure=closure, coords=("Rf", "LAM", "Ds")); nel = m.nel
        z0 = np.asarray(m.init_state(prm.C0_N_50, prm.C0_P_50)); x0 = np.concatenate([z0, ek.coord_of(np.array([4.0, 0.1, 0.55]))])
        P0d = np.concatenate([np.full(nel, 10.0), np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0), np.full(3, 1.0)]); Qd = np.concatenate([np.zeros(m.nz), np.full(3, 1e-3)])
        return m, ek, x0, P0d, Qd

    def record(label, grid, Nm, N, f, args, n, kind="per call"):
        t = timing.time_calls(f, args, n); st = timing.stats(t); glab = "-".join(map(str, grid)) + f" {a.grid}"
        rows.append(dict(**{"filter": label}, grid=glab, Nm=Nm, N=N, kind=kind, median_ms=st["median_ms"], p99_ms=st["p99_ms"], p999_ms=st["p999_ms"], worst_ms=st["worst_ms"], n_calls=n, note=note)); print(rows[-1]["filter"], glab, Nm, N, f"median {st['median_ms']:.3f} ms, p99 {st['p99_ms']:.3f} ms ({n} calls)", flush=True)

    for grid, Nm in pts:
        if "ekf" in parts:
            for closure in ["newton", "cold"]:
                m, ek, x0, P0d, Qd = setup(grid, Nm, closure); step, scale = ek.make_step(x0, P0d, Qd, 4e-6)
                th_full = ek.th_of(jnp.asarray(x0[m.nz:])); g0 = ek.init_g(jnp.asarray(x0[:m.nz]), 5.0, th_full)
                carry = (jnp.zeros(ek.n), jnp.diag((jnp.asarray(P0d) / scale) ** 2), g0); f = jax.jit(lambda c, I, y: step(c, (I, y))[0])
                record(EKF_LABEL.format(closure), grid, Nm, ek.n, f, (carry, 5.0, 3.7), n_ekf)
                if closure == "newton" and grid == (20, 10, 20) and Nm == 12:
                    ns = 50 if a.quick else 1000
                    dt_ = timing.scan_mean_ms(step, carry, (jnp.full(ns, 5.0), jnp.full(ns, 3.7)), ns)
                    rows.append(dict(**{"filter": "EKF structured (JAX), amortized in lax.scan"}, grid="-".join(map(str, grid)) + f" {a.grid}", Nm=Nm, N=ek.n, kind=f"scan (mean over {ns} steps)", median_ms=dt_, n_calls=ns, note="mean, not median"))
                    print(rows[-1], flush=True)
        if "ukf" in parts and grid != (40, 20, 40):
            m, ek, x0, P0d, Qd = setup(grid, Nm); uk = UKF(ek, closure_iters=kit); step, c0_ = uk.run(x0, P0d, Qd, 4e-6, jnp.zeros(1), jnp.zeros(1), return_step=True)
            f = jax.jit(lambda c, I, y: step(c, (I, y))[0]); record(ukf_label(kit), grid, Nm, ek.n, f, (c0_, 5.0, 3.7), n_ukf)
        if "spme" in parts and ((grid == (10, 5, 10) and Nm == 8) or (grid == (20, 10, 20) and Nm == 12) or a.quick):
            m, ek, x0, P0d, Qd = setup(grid, Nm); sp = SPMe(m, ek); us = UKFSPMe(sp); xs = np.concatenate([np.asarray(sp.init(prm.C0_N_50, prm.C0_P_50)), ek.coord_of(np.array([4.0, 0.1, 0.55]))])
            P0s = np.concatenate([[10.0], np.full(Nm + 2, 1.0), [10.0], np.full(Nm + 2, 1.0), np.full(m.N, 3.0), np.full(3, 1.0)]); Qs = np.concatenate([np.zeros(sp.nzs), np.full(3, 1e-3)])
            fr = jax.jit(lambda I, y: us.run(xs, P0s, Qs, 4e-6, I, y)["vals"]); record(SPME_LABEL, grid, Nm, sp.n, fr, (jnp.full(1, 5.0), jnp.full(1, 3.7)), n_spme)

    new = pd.DataFrame(rows).reindex(columns=COLS)
    name = ("costgrid_4" if a.grid == "4:1" else "costgrid_none") + ("" if kit == 1 else f"_ukf{'none' if kit is None else kit}") + ("_quick" if a.quick else "") + ".csv"
    path = os.path.join(a.results, name)
    if os.path.exists(path):                                                  # merge: replace the rows of the re-timed points
        old = pd.read_csv(path, comment="#"); key = lambda d: d["filter"] + "|" + d.grid + "|" + d.Nm.astype(str) + "|" + d.kind
        new = pd.concat([old[~key(old).isin(set(key(new)))], new], ignore_index=True)
    cli.write_csv(new, a, name, note=f"timings on one thread, other busy processes at start: {n_other}", float_format="%.8g")
    meta = pd.DataFrame([dict(file=name, other_busy_processes=("unknown" if n_other is None else n_other), **info)])
    mpath = os.path.join(a.results, "cost_meta.csv"); meta.to_csv(mpath, mode="a", header=not os.path.exists(mpath), index=False)
    clock.log(f"other busy processes {n_other}")


def aggregate(a):
    import pandas as pd
    fs = [os.path.join(a.results, f) for f in ("costgrid_4.csv", "costgrid_none.csv") if os.path.exists(os.path.join(a.results, f))]
    if not fs: raise SystemExit(f"no costgrid_*.csv in {a.results}")
    d = pd.concat([pd.read_csv(f, comment="#") for f in fs], ignore_index=True).reindex(columns=COLS)
    note = "cost per call, median over >= 3600 calls (>= 100 for the UKF), JAX, one thread, idle machine; 4:1 and uniform grids (costgrid_4.csv, costgrid_none.csv written by scripts/cost.py)"
    for sub, nm in (("tables", "tab_cost_jax_m17.csv"), ("source_data", "fig_cost_m17.csv")):
        a2 = type("A", (), dict(item=a.item, grid="4:1 and uniform", results=os.path.join(a.results, sub)))(); print(cli.write_csv(d, a2, nm, note=note, float_format="%.6g"))
    e = d[d["filter"].str.contains("EKF") & d["filter"].str.contains("newton") & d.grid.str.endswith("4:1") & (d.kind == "per call")].sort_values("N")
    print(e[["grid", "Nm", "N", "median_ms", "p99_ms"]].round(3).to_string(index=False))
    u = d[d["filter"].str.contains("UKF") & d["filter"].str.contains("full") & d.grid.str.endswith("4:1")]; print(u[["filter", "grid", "N", "median_ms"]].round(2).to_string(index=False))
    s = d[d["filter"].str.contains("SPMe") & d.grid.str.endswith("4:1")]; print(s[["filter", "grid", "N", "median_ms"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
