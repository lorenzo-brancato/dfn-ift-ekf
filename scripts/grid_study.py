"""Table I and Fig. 3: effect of the spatial grid and of the number of particle modes on the model error and on the parameter bias, state th2.

Per grid row (n_neg/n_sep/n_pos cells, N_m modes, uniform or 4:1 electrode cells), at the ageing state th2 (R_f = 4, LAM = 0.1, LLI = 0.05, D_s = 0.55), dt = 0.5 s, converged closure:
  * E_V: RMS voltage error [mV] of the model driven by the reference current against the PyBaMM reference (40/20/40 uniform, 300 radial nodes), at 1C (P1_1C, discharge to cut-off)
    and at 3C for 30 s (P1_3C); lithium drift max |Li(t)/Li(0) - 1| and the largest closure residual;
  * bias of the batch estimate (R_f, LAM, D_s): Levenberg-Marquardt-damped Gauss-Newton fit of the model (sensitivities from the structured Jacobians) to the NOISE-FREE reference voltage
    of P6Lx2 at th2, started at the true values, in % of the ageing range (R_f 9, LAM 0.3, D_s 0.7), residual rms [mV].
The four rows of Table I: 10/5/10 uniform N_m = 8, 20/10/20 uniform N_m = 12, 40/20/40 uniform N_m = 16, 20/10/20 4:1 N_m = 12 (--rows all); --rows extended adds the other
four combinations measured for Fig. 3 (10/5/10 N_m = 8 with 4:1, 10/5/10 N_m = 12 uniform and 4:1, 40/20/40 N_m = 16 with 4:1). The grid of a row is given by the row (or by --grid-spec), NOT by --grid.
Per-row output: <results>/grid_<n>-<s>-<p>_Nm<Nm>_<uniform|nu4>.csv (columns as the earlier per-row tables).
--aggregate writes <results>/tables/tab_model_m17.csv and <results>/source_data/fig_grid_m17.csv; the EKF cost per step (columns cost_ms_EKF, cost_ms_EKF_p99, and N) is
taken from <results>/costgrid_4.csv and <results>/costgrid_none.csv written by scripts/cost.py (left empty if they are absent).

Commands (one process per row; wall time of the earlier runs, single thread: 10/5/10: 1.5 min, 20/10/20 uniform 4 min, 20/10/20 4:1 2.5 min, 40/20/40: 7-10 min):
  python scripts/grid_study.py --rows 10-5-10_Nm8_uniform --data data --results results            (names: see ROWS below)
  python scripts/grid_study.py --grid-spec 20/10/20:12:4:1 --data data --results results
  python scripts/grid_study.py --rows all --data data --results results                           (sequential, about 25 min)
  python scripts/grid_study.py --aggregate --results results
  python scripts/grid_study.py --quick --results /tmp/grid_quick
"""
import os
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

# name -> (n_neg, n_sep, n_pos, N_m, grid)
ROWS = {
    "10-5-10_Nm8_uniform": (10, 5, 10, 8, "uniform"), "20-10-20_Nm12_uniform": (20, 10, 20, 12, "uniform"), "40-20-40_Nm16_uniform": (40, 20, 40, 16, "uniform"), "20-10-20_Nm12_nu4": (20, 10, 20, 12, "4:1"),
    "10-5-10_Nm8_nu4": (10, 5, 10, 8, "4:1"), "10-5-10_Nm12_uniform": (10, 5, 10, 12, "uniform"), "10-5-10_Nm12_nu4": (10, 5, 10, 12, "4:1"), "40-20-40_Nm16_nu4": (40, 20, 40, 16, "4:1"),
}
TABLE_ROWS = ["10-5-10_Nm8_uniform", "20-10-20_Nm12_uniform", "40-20-40_Nm16_uniform", "20-10-20_Nm12_nu4"]
RNG = (9.0, 0.3, 0.7); LO = (0.5, -0.05, 0.15); HI = (12.0, 0.4, 1.3)           # ageing ranges of (R_f, LAM, D_s) and the clipping box of the fit
NOTE = "gate at th2 in closed loop with converged closure, batch-fit bias at th2; same procedure for every row"


def label_of(spec):
    nn, ns, npp, Nm, grid = spec
    return f"{nn}-{ns}-{npp}_Nm{Nm}_{'uniform' if grid == 'uniform' else 'nu4'}"


def parse_spec(s):
    """'20/10/20:12:4:1' or '20/10/20:12:uniform' -> (20, 10, 20, 12, '4:1' | 'uniform')."""
    g, nm, grid = s.split(":", 2)
    nn, ns, npp = (int(x) for x in g.split("/"))
    if grid not in ("uniform", "4:1"): raise SystemExit(f"unknown grid {grid!r} in --grid-spec")
    return nn, ns, npp, int(nm), grid


def run_row(a, spec, prefix="grid_"):
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from dfnekf import params as prm, refio
    from dfnekf.step import DFN
    from dfnekf.rom import simulate
    from dfnekf.sens import traj_sens
    nn, ns, npp, Nm, grid = spec; dt = 0.5; tag = "th2"; label = label_of(spec); c0 = time.process_time()
    m = DFN(prm.base(), nn, ns, npp, Nm, 2, dt, grid)
    row = dict(grid=f"{nn}/{ns}/{npp}", Nm=Nm, ratio=grid, N=m.nz + 3)
    for proto in (["P1_3C"] if a.quick else ["P1_1C", "P1_3C"]):
        r = refio.load(proto, tag, period=dt, data_dir=a.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]])
        out = jax.jit(lambda z0, I, th: simulate(m, z0, I, th, k=None))(z0, jnp.asarray(r["I"]), th); V = np.asarray(out["V"]); li = np.asarray(out["li"]); s = proto[3:]
        row[f"E_V_{s}_mV"] = 1e3 * float(np.sqrt(np.mean((V - r["V"]) ** 2))); row[f"lithium_drift_{s}"] = float(np.abs(li / li[0] - 1).max()); row[f"resF_max_{s}"] = float(np.max(out["resF"]))
    rng, lo, hi = np.array(RNG), np.array(LO), np.array(HI)
    f = jax.jit(lambda th, z0, I: traj_sens(m, th, z0, I, par=("Rf", "LAM", "Ds"), return_V=True))
    r = refio.load("P6Lx2", tag, period=dt, data_dir=a.data); true = np.array([r["Rf"], r["LAM"], r["Ds"]]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]])
    nsamp = 1200 if a.quick else len(r["I"]); I = jnp.asarray(r["I"][:nsamp]); Vref = r["V"][:nsamp]
    p = true.copy(); lam = 1e-3; hist = []; max_it = 3 if a.quick else 12

    def model(p):
        dV, V = f(jnp.array([p[0], 1 - p[1], p[2]]), z0, I); return np.asarray(dV), np.asarray(V)
    dV, V = model(p); cost = np.sum((V - Vref) ** 2)
    for it in range(max_it):                                          # Levenberg-Marquardt-damped Gauss-Newton on range-scaled parameters
        res = V - Vref; Jn = dV * rng[None, :]; H = Jn.T @ Jn; gv = Jn.T @ res; ok = False
        for _ in range(8):
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H)), gv) * rng; pn = np.clip(p + step, lo, hi); dVn, Vn = model(pn); cn = np.sum((Vn - Vref) ** 2)
            if cn < cost: ok = True; break
            lam *= 10
        if not ok: break
        conv = np.max(np.abs(pn - p) / rng) < 1e-6; p, dV, V, cost = pn, dVn, Vn, cn; lam = max(lam / 10, 1e-9); hist.append(cost)
        if conv: break
    bias = p - true; res = V - Vref
    row.update(theta_gate=tag, bias_Rf_pct=100 * bias[0] / rng[0], bias_LAM_pct=100 * bias[1] / rng[1], bias_Ds_pct=100 * bias[2] / rng[2], resid_rms_mV=1e3 * float(np.sqrt(np.mean(res ** 2))), iters=len(hist), cpu_s=time.process_time() - c0)
    a.grid = grid
    path = cli.write_csv(pd.DataFrame([row]), a, f"{prefix}{label}" + ("_quick" if a.quick else "") + ".csv", note=f"row {label}")
    print(row, "->", path, flush=True)
    return row


def main():
    p = cli.parser("Table I, Fig. 3 (grid study)", "model error and batch-fit bias versus grid and number of modes at th2")
    p.add_argument("--rows", default="all", help="'all' (the four rows of Table I), 'extended' (all eight combinations), 'modes' (8 against 12 modes on the 10/5/10 uniform grid, Sec. II-D) or a row name: " + ", ".join(ROWS))
    p.add_argument("--grid-spec", default=None, help="one custom row 'n_neg/n_sep/n_pos:N_m:grid', e.g. 20/10/20:12:4:1 (overrides --rows)")
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    runtime.setup(a.core)
    clock = cli.Clock(a, "grid_study.py")
    if a.rows == "modes" and not a.grid_spec and not a.quick: return modes(a, clock)
    if a.grid_spec: specs = [parse_spec(a.grid_spec)]
    elif a.quick: specs = [ROWS["10-5-10_Nm8_uniform"]]
    elif a.rows == "all": specs = [ROWS[k] for k in TABLE_ROWS]
    elif a.rows == "extended": specs = list(ROWS.values())
    elif a.rows in ROWS: specs = [ROWS[a.rows]]
    else: raise SystemExit(f"unknown row {a.rows!r}")
    for s in specs: run_row(a, s)
    clock.log()


def modes(a, clock):
    """Number of modes against grid (Sec. II-D): the coarsest grid (10/5/10, uniform) with 8 and with 12 modes, same procedure as Table I.
    Writes <results>/modes_10-5-10_uniform.csv (one row per N_m plus the change of the bias in percentage points); the per-row files use the prefix modes_, so the aggregate of Table I does not see them."""
    import pandas as pd
    rows = [run_row(a, ROWS[k], prefix="modes_") for k in ("10-5-10_Nm8_uniform", "10-5-10_Nm12_uniform")]
    D = pd.DataFrame(rows); a.grid = "uniform"
    for c in ("bias_Rf_pct", "bias_LAM_pct", "bias_Ds_pct", "E_V_1C_mV", "E_V_3C_mV"): D["d_" + c] = D[c] - D[c].iloc[0]
    print(D[["Nm", "E_V_1C_mV", "E_V_3C_mV", "bias_Rf_pct", "bias_LAM_pct", "bias_Ds_pct", "d_bias_Ds_pct"]].round(3).to_string())
    print(cli.write_csv(D, a, "modes_10-5-10_uniform.csv", note="10/5/10 uniform grid, 8 and 12 particle modes (d_* = change with respect to the first row)"))
    clock.log()


def aggregate(a):
    import numpy as np
    import pandas as pd
    files = sorted(f for f in glob.glob(os.path.join(a.results, "grid_*.csv")) if not f.endswith("_quick.csv"))
    if not files: raise SystemExit(f"no per-row grid_*.csv files in {a.results}")
    G = pd.concat([pd.read_csv(f, comment="#") for f in files], ignore_index=True)
    costs = [pd.read_csv(f, comment="#") for f in (os.path.join(a.results, "costgrid_4.csv"), os.path.join(a.results, "costgrid_none.csv")) if os.path.exists(f)]
    C = pd.concat(costs, ignore_index=True) if costs else None
    rows = []
    for _, g in G.iterrows():
        gl = "-".join(g.grid.split("/")); cost = np.nan; p99 = np.nan; N = g.N
        if C is not None:
            sel = C[C["filter"].str.contains("EKF") & C["filter"].str.contains("newton") & (C.grid == f"{gl} {g.ratio}") & (C.Nm == g.Nm)]
            if len(sel): cost, p99, N = float(sel.median_ms.iloc[0]), float(sel.p99_ms.iloc[0]), int(sel.N.iloc[0])
        rows.append(dict(table="grid", grid_label=f"{g.grid} {g.ratio} N_m={g.Nm}", grid=g.grid, ratio=g.ratio, Nm=g.Nm, N=int(N), E_V_1C_mV=g.E_V_1C_mV, E_V_3C_mV=g.E_V_3C_mV, bias_Rf_pct=g.bias_Rf_pct, bias_LAM_pct=g.bias_LAM_pct,
                         bias_Ds_pct=g.bias_Ds_pct, resid_rms_mV=g.resid_rms_mV, lithium_drift_3C=g.lithium_drift_3C, resF_max=max(g.resF_max_1C, g.resF_max_3C), cost_ms_EKF=cost, cost_ms_EKF_p99=p99, theta_gate="th2", note=NOTE))
    Mo = pd.DataFrame(rows); Mo["_o"] = Mo.grid.map({"10/5/10": 0, "20/10/20": 1, "40/20/40": 2}); Mo["_r"] = Mo.ratio.map({"4:1": 0, "uniform": 1})
    Mo = Mo.sort_values(["_o", "Nm", "_r"]).drop(columns=["_o", "_r"]).reset_index(drop=True)
    note = "4:1 grid: electrode cells in geometric progression, collector/separator cell-width ratio 4:1, uniform separator; N_m = number of particle modes; dt = 0.5 s"
    for sub, nm in (("tables", "tab_model_m17.csv"), ("source_data", "fig_grid_m17.csv")):
        a2 = type("A", (), dict(item=a.item, grid="see rows", results=os.path.join(a.results, sub)))(); print(cli.write_csv(Mo, a2, nm, note=note, float_format="%.6g"))
    pd.set_option("display.width", 220); print(Mo[["grid_label", "N", "E_V_1C_mV", "E_V_3C_mV", "bias_Rf_pct", "bias_LAM_pct", "bias_Ds_pct", "cost_ms_EKF"]].round(3).to_string())


if __name__ == "__main__":
    main()
