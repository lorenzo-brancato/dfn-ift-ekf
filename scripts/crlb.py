"""Cramer-Rao bounds of the ageing parameters (Fig. 5 grey bars, Sec. III-B, Appendix D).

Computes, for every ageing state and protocol, the Fisher information of the voltage record from the exact sensitivities of the grey-box model
(structured implicit-function-theorem Jacobians propagated along the trajectory), sigma_V = 2 mV, dt = 0.5 s, and writes three bounds per run:
  conditional_on_LLI   bound of (R_f, LAM, D_s) with LLI fixed = inverse of the 3 x 3 sub-block of the 4-parameter Fisher matrix = Schur complement
                       (the grey bars of Fig. 5; LAM 0.02-0.11 %, R_f 0.19-0.24 %, D_s up to 1.6 % of the range on P6Lx2)
  known_state          4-parameter marginal bound (R_f, LAM, LLI, D_s) and the correlation LAM-LLI (0.89-0.98 on pulses)
  unknown_soc          unknown initial SOC (extra nuisance parameters dc_n0, dc_p0, LLI removed): LAM at most 0.5 %, the others change by at most 0.12 %
The model uses the grid given by --grid (the paper's values are on 4:1, 20/10/20, N_m = 12).

Per-run output (one file per protocol and state, so the states can be run in separate processes):
  <results>/crlb_<protocol>_<state>.csv       rows (version x parameter), columns as the earlier table tab_crlb_m17.csv
  <results>/traj/crlb_dV_<protocol>_<state>.npz   sensitivity matrix dV/d(Rf, LAM, LLI, Ds, dc_n0, dc_p0) [n_samples, 6]
--aggregate writes <results>/tables/crlb_4par.csv (4-parameter table, versions known_state, unknown_soc), crlb_conditional.csv, tab_crlb_soc_m17.csv and <results>/source_data/fig_crlb_m17.csv.

Commands (about 15 s per state and protocol on one core):
  python scripts/crlb.py --grid 4:1 --states th1,th2,th3,th4,th5 --protocols P6Lx2 --data data --results results
  python scripts/crlb.py --grid 4:1 --protocols P6Lx2,P6L,P2,D1 --with-k0 --results results       (all protocols of the full table; P2 and D1 need PyBaMM for the initial state of D1)
  python scripts/crlb.py --aggregate --results results
  python scripts/crlb.py --quick --results /tmp/crlb_quick
"""
import os
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

COLS = ["protocol", "theta_tag", "theta", "version", "parameter", "sd", "range_width", "sd_pct_range", "corr_R_f", "corr_LAM_n", "corr_LLI", "corr_D_s", "corr_dc_n0", "corr_dc_p0", "corr_k0", "cond_fisher", "n_samples"]


def main():
    p = cli.parser("Fig. 5 (grey bars), Sec. III-B, Appendix D", "Cramer-Rao bounds of (R_f, LAM, D_s) conditioned on LLI, 4-parameter marginal, unknown initial SOC", states="th1,th2,th3,th4,th5")
    p.add_argument("--protocols", default="P6Lx2", help="comma-separated protocols (P6Lx2, P6L, P2, D1)")
    p.add_argument("--with-k0", action="store_true", help="also the 5-parameter version with the exchange-current constant k0 (finite differences; P6Lx2 and P6L only)")
    p.add_argument("--sigma", type=float, default=2e-3, help="measurement noise standard deviation [V]")
    p.add_argument("--aggregate", action="store_true", help="read the per-run files from --results and write the tables and the source data")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    runtime.setup(a.core)
    import json
    import numpy as np
    import pandas as pd
    from dfnekf import params as prm, crlb
    from dfnekf.experiment import TRUE
    from dfnekf.step import DFN
    clock = cli.Clock(a, "crlb.py"); dt = 0.5
    m = DFN(prm.base(), 20, 10, 20, 12, 2, dt, a.grid)
    protos = [s.strip() for s in a.protocols.split(",") if s.strip()]
    states = cli.states(a)[:1] if a.quick else cli.states(a)
    for tag in states:
        theta = TRUE[tag]
        for proto in protos:
            I = crlb.protocol_current(proto, dt)
            if a.quick: I = I[:600]                                       # 300 s of the protocol
            dV = crlb.sensitivities(m, theta, proto, tag, a.data, I=I)
            dk0 = crlb.k0_column(m, theta, proto, tag, a.data, I=I) if (a.with_k0 and proto in ("P6Lx2", "P6L")) else None
            rows = crlb.table_rows(dV, proto, tag, theta, sigma=a.sigma, dk0=dk0)
            suffix = "_quick" if a.quick else ""
            os.makedirs(os.path.join(a.results, "traj"), exist_ok=True)
            np.savez_compressed(os.path.join(a.results, "traj", f"crlb_dV_{proto}_{tag}{suffix}.npz"), dV=dV, sigma=a.sigma)
            df = pd.DataFrame(rows)
            path = cli.write_csv(df, a, f"crlb_{proto}_{tag}{suffix}.csv", note=f"protocol {proto}, state {tag}, sigma_V = {a.sigma:g} V, N = {len(dV)} samples")
            c = df[df.version == "conditional_on_LLI"]
            print(tag, proto, "conditional bound [% of range]:", {r.parameter: round(r.sd_pct_range, 3) for r in c.itertuples()}, "->", path, flush=True)
    clock.log()


def aggregate(a):
    import pandas as pd
    from dfnekf import crlb
    files = sorted(f for f in glob.glob(os.path.join(a.results, "crlb_*_th*.csv")) if not f.endswith("_quick.csv"))
    if not files: raise SystemExit(f"no per-run crlb_*.csv files in {a.results}")
    d = pd.concat([pd.read_csv(f, comment="#") for f in files], ignore_index=True)
    d = d.reindex(columns=COLS)
    order = {"known_state": 0, "conditional_on_LLI": 1, "unknown_soc": 2, "known_state+k0": 3}
    d["_o"] = d.version.map(order); d["_t"] = d.theta_tag.str[2:].astype(int); d = d.sort_values(["protocol", "_t", "_o"], kind="stable").drop(columns=["_o", "_t"]).reset_index(drop=True)
    os.makedirs(os.path.join(a.results, "tables"), exist_ok=True); os.makedirs(os.path.join(a.results, "source_data"), exist_ok=True)
    note = ("sd in the parameter's own units, sd_pct_range = 100 sd / range_width, corr_* = row of the correlation matrix of the estimator; versions: known_state (4-parameter marginal), "
            "conditional_on_LLI (bound of R_f, LAM, D_s with LLI fixed = Schur complement; the grey bars of Fig. 5), unknown_soc (dc_n0, dc_p0 extra, LLI removed), known_state+k0")
    t = os.path.join(a.results, "tables"); s = os.path.join(a.results, "source_data")
    a2 = type("A", (), dict(item=a.item, grid=a.grid, results=t))()
    cli.write_csv(d[d.version != "conditional_on_LLI"], a2, "crlb_4par.csv", note=note, float_format="%.8g")      # format of the earlier tab_crlb_m17.csv; versions known_state, unknown_soc, known_state+k0
    cond = []                                                                                                    # bound conditional on LLI next to the marginal and the unknown-SOC variant (independent of make_figdata)
    for (proto, tag), gg in d.groupby(["protocol", "theta_tag"], sort=False):
        for par, nm in (("R_f", "Rf"), ("LAM_n", "LAM"), ("D_s", "Ds")):
            get = lambda v: float(gg[(gg.version == v) & (gg.parameter == par)].sd_pct_range.iloc[0]) if len(gg[(gg.version == v) & (gg.parameter == par)]) else float("nan")
            cond.append(dict(protocol=proto, theta=tag, param=nm, sd_pct_range=get("conditional_on_LLI"), sd_pct_range_marginal_4par=get("known_state"), sd_pct_range_unknown_soc=get("unknown_soc")))
    cli.write_csv(pd.DataFrame(cond), a2, "crlb_conditional.csv", note="sd_pct_range = bound of (R_f, LAM, D_s) conditional on LLI (Schur complement), percent of range", float_format="%.6g")
    soc = crlb.soc_effect(d.to_dict("records")); cli.write_csv(soc, a2, "tab_crlb_soc_m17.csv", note="effect of the unknown initial SOC on the bound of (R_f, LAM_n, D_s), percent of range", float_format="%.6g")
    # source data of the grey bars: P6Lx2, one row per state and parameter
    g = d[d.protocol == "P6Lx2"]; rows = []
    for tag, gg in g.groupby("theta_tag", sort=False):
        for par, nm in (("R_f", "Rf"), ("LAM_n", "LAM"), ("D_s", "Ds")):
            get = lambda v: float(gg[(gg.version == v) & (gg.parameter == par)].sd_pct_range.iloc[0]) if len(gg[(gg.version == v) & (gg.parameter == par)]) else float("nan")
            rows.append(dict(state=tag, param=nm, crlb=get("conditional_on_LLI"), crlb_marginal_4par=get("known_state"), crlb_unknown_soc=get("unknown_soc")))
    a3 = type("A", (), dict(item=a.item, grid=a.grid, results=s))()
    cli.write_csv(pd.DataFrame(rows), a3, "fig_crlb_m17.csv", note="P6Lx2; crlb = conditional on LLI [% of range], the column of mc_budget.csv", float_format="%.6g")
    print(pd.DataFrame(rows).round(3).to_string()); print("written:", t, s)


if __name__ == "__main__":
    main()
