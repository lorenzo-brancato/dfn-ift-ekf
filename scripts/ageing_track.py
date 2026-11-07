"""Tracking of the ageing run: Fig. 8 (panels a-e) and Sec. III-D.

The guarded EKF with four parameters (Rf, LAM, Ds, LLI; state guard on, converged Newton closure, 20/10/20 cells, N_m = 12, dt = 0.5 s, sigma_V = 2 mV) is run once, without
interruption, over the 294 h ageing reference (``scripts/ageing_reference.py``). It is evaluated at every RPT at four read-out points (end of P6L, end of the slow
discharge, last sample of the slow charge before the CV = the main point, end of the RPT) against the true effective parameters, with the criteria of the paper:
error within the bias floor + 3 % of range (Rf 3.90, LAM 3.10, Ds 5.37, inventory 3.00 %), |z| <= 3, logit estimate more than 2 % from its boundary, innovation RMS < 3 mV
per 20000-step window. Panel e also needs the read-out at the end of the CV hold (``end``).

Seeds. seed 0 and seed 1 differ ONLY in the measurement-noise sequence ``default_rng(seed)``; initial state, initial parameters, tuning, guard, closure and read-out points are the same.
(The two campaign scripts that produced the paper results differed only in their output names; the filter of seed 1 was run with the default closure, without the smoothed surface
saturation that was tried and discarded in between.) Cost: about 13-14 h on one core per seed.

  python scripts/ageing_track.py --grid 4:1 --data data --results results/ageing --seeds 0 --core 0     # job 1
  python scripts/ageing_track.py --grid 4:1 --data data --results results/ageing --seeds 1 --core 1     # job 2 (in parallel)
  python scripts/ageing_track.py --aggregate --grid 4:1 --data data --results results/ageing --seeds 0,1
  python scripts/ageing_track.py --quick --data data --results results/quick                            # first 0.1 h, a minute or so
  options: --hours H (stop after H simulated hours), --resume (continue from the checkpoint written every 10 chunks), --no-guard (used by ablation_guard.py)
  CV freeze (default: none = the paper's run). --freeze voltage [--freeze-above 4.195]: parameter rows of the gain zeroed where the MEASURED voltage >= 4.195 V and the current is a charge
  (the first remedy of Sec. III-D, "freezing the parameter updates during the CV phase"); --freeze phase: the same during the constant-voltage segment of every RPT (known from the protocol
  schedule, as a BMS knows its charger state; --cv-scope all adds the CV of the ageing cycles); with --cv-update none no measurement update at all in that segment (prediction only for state
  and covariance), default --cv-update state (cell state updated, parameter rows of the gain zero). Outputs carry a suffix: _freeze-voltage, _freeze-phase, _freeze-phase-noupdate.
  python scripts/ageing_track.py --freeze phase --quick --data data --results results/quick      # mechanism check: the freeze code is applied to samples 200-299 of the first 0.1 h

Outputs (<results>): traj/ageing_track_seed<s>.npz (series every 10 s, innovation every step), ageing_track_seed<s>_rpt.csv, _cv.csv, _summary.csv, cpu_log.csv; with --aggregate:
tables/tab_aging_tracking.csv (one row per seed x RPT x point), tables/tab_aging_cv.csv (diagnostics of every CV hold), tables/tab_aging_summary.csv (gates and headline numbers),
source_data/fig_aging_dyn.csv (Fig. 8a-d), source_data/fig_tracking_rpt.csv (Fig. 8e).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

REF_NAME = "aging_dyn_rptslow_full.npz"
LABEL = {0: "A", 1: "B"}                       # column suffix of the two seeds in fig_aging_dyn.csv


def run_one(a, seed, P, R0):
    import numpy as np
    from dfnekf import ageing_tracker as at
    n = int(a.hours * 3600 / at.DT) + 1 if a.hours else (720 if a.quick else None)
    chunk = a.chunk or (360 if a.quick else at.CHUNK)
    tr = at.Tracker(P, a.grid, seed=seed, guard=not a.no_guard, nsteps=n, freeze=a.freeze, freeze_above=a.freeze_above, cv_update=a.cv_update, R0=R0, cv_scope=a.cv_scope)
    if a.quick and tr.fcode is not None:                       # mechanism test: the freeze code on a window of the pulses (the first CV hold is at 6 h)
        code = 1 if a.cv_update == "state" else 2; tr.fcode[:] = 0; tr.fcode[200:300] = code
    name = f"ageing_track_seed{seed}" + ("_noguard" if a.no_guard else "") + suffix(a)
    clock = cli.Clock(a, name)
    traj = os.path.join(a.results, "traj", name + ".npz"); ck = os.path.join(a.results, "traj", name + "_ckpt.pkl"); os.makedirs(os.path.dirname(traj), exist_ok=True)
    d = at.track(tr, ckpt=ck, resume=a.resume, stop_on_breakdown=False, chunk=chunk)
    at.save_run(traj, d)
    if a.quick and tr.fcode is not None:
        w = d["vals"][11:15]; print(f"freeze mechanism check (code {int(tr.fcode.max())} on samples 200-299): parameters constant over the window: {bool(np.allclose(w, w[0], rtol=0, atol=1e-12))}; frozen steps stored: {int((d['ps_frozen'] > 0).sum())}", flush=True)
    write_run_tables(a, name, f"seed{seed}", d, tr.m, P, R0)
    clock.log(f"seed {seed}, {tr.N} steps")
    if os.path.exists(ck) and not a.quick: os.remove(ck)
    print(f"seed {seed}: {tr.N} steps, rejected updates {int(d['n_rejected'].sum())}, wall {float(d['wall']):.0f} s -> {traj}", flush=True)


def suffix(a):
    """File-name suffix of a freeze variant ('' for the paper's run)."""
    return "" if a.freeze == "none" else f"_freeze-{a.freeze}" + ("-noupdate" if a.cv_update == "none" else "") + ("-allcv" if a.cv_scope == "all" else "")


def write_run_tables(a, name, label, d, m, P, R0):
    from dfnekf import ageing_tracker as at
    R = at.evaluate_rpt(d, P, R0, m, label, points=at.POINTS + (("cvend",) if a.freeze != "none" else ()))
    cli.write_csv(R, a, f"{name}_rpt.csv", note="read-out points of every RPT, errors in % of the ageing range")
    cli.write_csv(at.cv_table(d, P, R0, m, label), a, f"{name}_cv.csv", note="diagnostics of the constant-voltage holds")
    if len(R):
        import pandas as pd
        cli.write_csv(pd.DataFrame([{**at.summary(R, d, R0, label), **headline(R)}]), a, f"{name}_summary.csv", note="gates and counts")


def headline(R):
    """Numbers quoted in Sec. III-D from the read-out before the CV: last RPT passing the Rf criterion, worst LAM / Ds / inventory error up to RPT 8, Rf error at RPT 9, first RPT failing at the end of the CV."""
    import numpy as np
    pc = R[R.point == "precv"].set_index("rpt"); en = R[R.point == "end"].set_index("rpt"); h = {}
    ok = [r for r in pc.index if r >= 1 and pc.loc[r, "ABZ_ok"]]; bad = [r for r in pc.index if r >= 1 and not pc.loc[r, "ABZ_ok"]]
    h["precv_first_rpt_failing_ABZ"] = min(bad) if bad else -1
    h["precv_Rf_first_rpt_failing_A"] = next((r for r in sorted(pc.index) if r >= 1 and abs(pc.loc[r, "Rf_err"]) > 3.90), -1)
    sub = pc[(pc.index >= 1) & (pc.index <= 8)]
    for c in ("LAM", "Ds", "LLI_inv"): h[f"precv_rpt1_8_maxabs_{c}"] = float(sub[f"{c}_err"].abs().max()) if len(sub) else np.nan
    h["precv_rpt9_Rf_err"] = float(pc.loc[9, "Rf_err"]) if 9 in pc.index else np.nan
    h["end_first_rpt_failing_Rf_A"] = next((r for r in sorted(en.index) if r >= 1 and abs(en.loc[r, "Rf_err"]) > 3.90), -1)
    return h


def aggregate(a, P, R0):
    import numpy as np, pandas as pd
    from dfnekf import ageing_tracker as at
    from dfnekf.step import DFN
    from dfnekf import params as prm
    m = DFN(prm.base(), 20, 10, 20, 12, 2, at.DT, a.grid)
    seeds = [s for s in cli.parse_ids(a.seeds) if os.path.exists(os.path.join(a.results, "traj", f"ageing_track_seed{s}.npz"))]
    if not seeds: raise SystemExit("no trajectory in <results>/traj: run the tracker first")
    runs = {s: dict(np.load(os.path.join(a.results, "traj", f"ageing_track_seed{s}.npz"))) for s in seeds}
    tabs, cvs, sums = [], [], []
    for s, d in runs.items():
        lab = f"seed{s}"; R = at.evaluate_rpt(d, P, R0, m, lab); tabs.append(R); cvs.append(at.cv_table(d, P, R0, m, lab))
        if len(R): sums.append({**at.summary(R, d, R0, lab), **headline(R)})
    T = pd.concat(tabs, ignore_index=True); os.makedirs(os.path.join(a.results, "tables"), exist_ok=True); os.makedirs(os.path.join(a.results, "source_data"), exist_ok=True)
    tab_args = lambda sub: type("A", (), dict(results=os.path.join(a.results, sub), item=a.item, grid=a.grid))
    cols = ["label", "variant", "seed", "rpt", "point", "t_h", "innov_rms_mV_chunk"] + [f"{c}_{k}" for c in at.COORDS for k in ("true", "est", "err", "sd", "z", "edge_pct")] + ["LLI_inv_err", "LLI_inv_z", "corr_Rf_LAM", "corr_LAM_LLI", "corr_Rf_Ds", "A_ok", "B_ok", "z_ok", "ABZ_ok"]
    T["seed"] = T.label.str.replace("seed", "").astype(int)
    cli.write_csv(T[cols], tab_args("tables"), "tab_aging_tracking.csv", note="Table of the tracking criteria, one row per seed x RPT x read-out point", float_format="%.6g")
    cli.write_csv(pd.concat(cvs, ignore_index=True), tab_args("tables"), "tab_aging_cv.csv", note="diagnostics of the constant-voltage holds (Sec. III-D)")
    if sums: cli.write_csv(pd.DataFrame(sums), tab_args("tables"), "tab_aging_summary.csv", note="gates and headline numbers of Sec. III-D")
    # Fig. 8e: criteria rows of the four read-out points
    F = T[["label", "rpt", "point", "t_h", "Rf_err", "LAM_err", "Ds_err", "LLI_inv_err", "Rf_z", "LAM_z", "Ds_z", "LLI_inv_z", "A_ok", "B_ok", "z_ok", "ABZ_ok", "innov_rms_mV_chunk"]].rename(columns={"label": "run"})
    cli.write_csv(F, tab_args("source_data"), "fig_tracking_rpt.csv", note="Fig. 8e: error at the four read-out points of every RPT", float_format="%.6g")
    # Fig. 8a-d: truth, estimate, standard deviation and inventory every 60 s plus all read-out rows
    n = min([len(d["vals"]) for d in runs.values()] + [(len(P["Rf"]) - 1) // 20 + 1]); k = np.arange(n) * 20
    out = pd.DataFrame(dict(t_h=k * at.DT / 3600, Rf_true=P["Rf"][k], LAM_true=P["LAM"][k], Ds_true=1.0, LLI_true=P["LLI"][k])); mark = np.zeros(n, int); se = R0["step_end_t"]
    for c in np.where(R0["is_rpt"])[0]:
        for code, idx in [(1, 12), (2, 14), (3, 16), (4, None)]:
            t = np.nanmax(se[c]) if idx is None else se[c][idx]; mark[min(int(round(t / 10)), n - 1)] = code
    out["marker"] = mark
    for s, d in runs.items():
        if s not in LABEL: continue
        L = LABEL[s]
        for i, c in enumerate(["Rf", "LAM", "Ds"]): out[f"{c}_est_{L}"] = d["vals"][:n, i]; out[f"{c}_sd_{L}"] = d["sd_vals"][:n, i]
        out[f"inv_est_{L}"] = at.inventory_loss(m, P, d["li"][:n])
    cols_d = list(out.columns[:6]) + [f"{p}_{L}" for L in [LABEL[s] for s in runs if s in LABEL] for p in ("Rf_est", "Rf_sd", "LAM_est", "LAM_sd", "Ds_est", "Ds_sd", "inv_est")]
    sel = (np.arange(n) % 6 == 0) | (mark > 0)
    cli.write_csv(out.loc[sel, cols_d], tab_args("source_data"), "fig_aging_dyn.csv", note="Fig. 8a-d: A = seed 0, B = seed 1; every 60 s plus all read-out rows; marker 1 end P6L, 2 end slow discharge, 3 before the CV, 4 end of RPT; inv = lithium-inventory loss in units of the anode window", float_format="%.9g")
    print("aggregated:", [os.path.join(a.results, s) for s in ("tables", "source_data")])


def main():
    p = cli.parser("Fig. 8a-e, Sec. III-D", "ageing tracker over the 294 h reference and its evaluation", seeds="0,1")
    p.add_argument("--aggregate", action="store_true", help="build the tables and source data from the stored trajectories")
    p.add_argument("--hours", type=float, default=None, help="stop after this many simulated hours")
    p.add_argument("--resume", action="store_true", help="continue from the checkpoint of an interrupted run")
    p.add_argument("--no-guard", action="store_true", help="switch the state guard off (ablation)")
    p.add_argument("--freeze", default="none", choices=["none", "voltage", "phase"], help="CV freeze: none (paper), voltage (measured V >= --freeze-above in charge), phase (CV segment of the protocol)")
    p.add_argument("--freeze-above", type=float, default=4.195, help="threshold [V] of --freeze voltage")
    p.add_argument("--cv-update", default="state", choices=["state", "none"], help="in the frozen steps: state = cell state updated, parameter rows of the gain zero; none = no measurement update")
    p.add_argument("--cv-scope", default="rpt", choices=["rpt", "all"], help="--freeze phase: CV of the RPTs only (default) or also of the ageing cycles")
    p.add_argument("--chunk", type=int, default=None, help="steps per jitted chunk (default 20000, multiple of 20; the innovation window depends on it)")
    p.add_argument("--ref", default=None, help="reference file (default <data>/aging_dyn_rptslow_full.npz)")
    a = p.parse_args()
    runtime.setup(a.core)
    import numpy as np
    from dfnekf import ageing
    ref = a.ref or os.path.join(a.data, REF_NAME)
    P = ageing.load(ref); R0 = np.load(ref)
    if a.aggregate: return aggregate(a, P, R0)
    for s in ([0] if a.quick else cli.parse_ids(a.seeds)): run_one(a, s, P, R0)


if __name__ == "__main__":
    main()
