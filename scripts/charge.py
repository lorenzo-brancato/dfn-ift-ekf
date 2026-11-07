"""Charge experiments: grey-box DFN-EKF with state guard versus the SPMe unscented filter (Sec. III-C, Fig. 7, Tables III, IV, VI, Appendix D thresholds).

Protocol F6Q (F6Q3 for state th3): two P6L blocks, a 1C constant-current charge to nominal SOC 0.6 (1272 s; 372 s for F6Q3), a rest, and two 30 s 2C charge pulses
terminated at 4.2 V. Reference: PyBaMM DFN (mesh 40/20/40, dt = 0.5 s). Five ageing states x seeds 0-9. Both filters see the same initial parameter offsets
(``default_rng(200 + seed)``), noise (``default_rng(seed)``, 2 mV), P0 and Q. Grey-box filter: guarded EKF, converged-Newton closure, state guard on, estimating
(R_f, LAM_n, D_s) with LLI fixed. Baseline: SPMe model with an unscented filter (``experiment.run_spme``).

Plating potential: the code evaluates phi_pl = phi_s - phi_e - Rf0 * R_f * j at the anode/separator FACE, linearly extrapolated from the last two anode cells
(``step.DFN.interface`` for the filter, ``step.interface_ref`` for the reference), not in the single cell next to the separator. The same holds for eta_lit.
The SPMe baseline uses the electrode-average solid potential (see ``spme.py``).

Run (one job per state/seed set; every job writes per-run files, so jobs can be split over processes):
    python scripts/charge.py --grid 4:1 --data data/sim --results results --states th2 --seeds 0-4 [--core 3]
    python scripts/charge.py --quick --grid 4:1 --data data/sim --results results          # smoke test: first state, seed 0, first 400 s
Aggregate (reads the per-run files of --results, writes <results>/tables and <results>/source_data):
    python scripts/charge.py --aggregate --grid 4:1 --data data/sim --results results
Per-run outputs: <results>/traj/charge_<state>_s<seed>.npz (trajectories: t, I, V_ref, filter histories, spme_*), <results>/charge_<state>_s<seed>.csv (metrics per pulse x window
x filter, threshold detection), <results>/charge_info_<state>_s<seed>.csv (certificate: guard rejections, Newton residuals, final parameters, CPU).
Aggregate outputs: tables/tab_charge_m17.csv (Tables III and VI: one row per state x seed x charge pulse), tables/tab_cc_m17.csv (Table IV and Appendix D: thresholds on the 1C
charge), tables/charge_cc_runs.csv, tables/charge_summary.csv, tables/charge_text_numbers.csv, tables/charge_detection.csv, tables/charge_validity.csv,
source_data/fig_eta_charge.csv (Fig. 7a-c) and source_data/fig_charge_cc.csv (Fig. 7d-f).
"""
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime                                             # no jax import here
from dfnekf.metrics import pulses, windows, metrics, detect

THRS = (20, 10, 0)                                                          # phi_pl alarm thresholds, mV
QUICK_STEPS = 800                                                           # 400 s: contains the first -1C pulse
PHIPL_NOTE = ("phi_pl = phi_s - phi_e - Rf0*R_f*j evaluated at the anode/separator face, linearly extrapolated from the last two anode cells "
              "(not in the single cell next to the separator)")
NOTE_TAIL = "grid 20/10/20, N_m = 12, dt = 0.5 s; " + PHIPL_NOTE


def proto_of(th):
    return "F6Q3" if th == "th3" else "F6Q"


# ----------------------------------------------------------------------------------------------------------------------- runs
def run_jobs(args):
    runtime.setup(args.core)
    import numpy as np, pandas as pd
    from dfnekf.experiment import Case
    from dfnekf import protocol_runs as pr
    if args.quick: args.results = os.path.join(args.results, "quick")
    clock = cli.Clock(args, "charge.py")
    states = cli.states(args); seeds = cli.parse_ids(args.seeds)
    if args.quick: states, seeds = states[:1], seeds[:1]
    os.makedirs(os.path.join(args.results, "traj"), exist_ok=True)
    for th in states:
        proto = proto_of(th)
        case = Case(th, args.grid, proto=proto, data_dir=args.data)
        if args.quick: pr.truncate(case, QUICK_STEPS)
        r = case.r; N = case.N; t = r["t"][:N]; I = np.asarray(r["I"])[:N]
        runner = pr.GreyBoxRunner(case); ref = pr.reference_interface(case); ce_ref = r["ce"].min(1)[:N]
        for sd in seeds:
            traj = os.path.join(args.results, "traj", f"charge_{th}_s{sd}.npz")
            if args.resume and os.path.exists(traj): print(th, sd, "exists, skipped", flush=True); continue
            H, out_s, cpu_e, cpu_s = pr.run_pair(runner, sd)
            np.savez(traj, t=t, I=I, V_ref=np.asarray(r["V"])[:N], **{k: H[k] for k in pr.TRAJ_KEYS}, spme_eta=out_s["eta"], spme_ppl=out_s["ppl"], spme_ce_min=out_s["ce_min"], spme_vals=out_s["vals"])
            rows = []; pl = pulses(I)
            for pid, p in enumerate(pl):
                for wname, a, b in windows(I, p):
                    for f, d in [("EKF + guard", dict(eta=H["eta"], ppl=H["ppl"], ce_min=H["ce_min"])), ("UKF-SPMe", dict(eta=out_s["eta"], ppl=out_s["ppl"], ce_min=out_s["ce_min"]))]:
                        mm = metrics(ref[:, 0], ref[:, 1], ce_ref, d["eta"], d["ppl"], d["ce_min"], a, b)
                        if not mm: continue
                        row = dict(theta=th, seed=sd, pulse_id=pid + 1, pulse=p["label"], kind=p["kind"], window=wname, filter=f, t_start_s=a * case.dt, duration_s=(b - a) * case.dt, **mm)
                        if wname == "pulse" and p["kind"] == "pulse":
                            for thr in THRS:
                                dd = detect(ref[:, 1], d["ppl"], thr, t, a, b); row.update({f"thr{thr}_{kk}": v for kk, v in dd.items()})
                        rows.append(row)
            cli.write_csv(pd.DataFrame(rows), args, f"charge_{th}_s{sd}.csv", note=NOTE_TAIL, float_format=None)
            rej = H["rejected"].astype(bool); bad = np.where(~np.isfinite(H["eta"]) | (np.abs(H["eta"]) > 1.0))[0]; first_rej = int(np.where(rej)[0][0]) if rej.any() else -1
            cmax = case.P0["cmax_n"]; errs = pr.theta_end_errors(case, H)
            info = dict(theta=th, seed=sd, protocol=proto, finite=bool(np.all(np.isfinite(H["vals"]))), n_steps=N, rejected_n=int(rej.sum()), rejected_frac=float(rej.mean()), first_rejected_step=first_rej,
                        first_bad_eta_step=int(bad[0]) if len(bad) else -1, max_abs_eta_V=float(np.nanmax(np.abs(H["eta"]))),
                        F_newton_max_accepted=float(np.nanmax(H["F_newton"][~rej])) if (~rej).any() else np.nan, F_warm_max=float(np.nanmax(H["F_warm"])),
                        csn_max_ref_over_cmax=float(r["cs_n"][:N].max() / cmax), csn_max_est_over_cmax=float(np.nanmax(H["csn_max"]) / cmax),
                        Rf_end=float(H["vals"][-1, 0]), LAM_end=float(H["vals"][-1, 1]), Ds_end=float(H["vals"][-1, 2]), Rf_err_end_pct=float(errs[0]), LAM_err_end_pct=float(errs[1]), Ds_err_end_pct=float(errs[2]),
                        cpu_ekf_s=cpu_e, cpu_spme_s=cpu_s)
            for pid, p in enumerate(pl):
                sl = slice(p["a"], p["b"]); k = f"pulse{pid + 1}"
                info[f"{k}_label"] = p["label"]; info[f"{k}_Fwarm_max"] = float(np.nanmax(H["F_warm"][sl])); info[f"{k}_Fnewton_max"] = float(np.nanmax(H["F_newton"][sl])); info[f"{k}_rejected_n"] = int(rej[sl].sum())
                info[f"{k}_csn_ref_max_over_cmax"] = float(r["cs_n"][sl].max() / cmax); info[f"{k}_csn_est_max_over_cmax"] = float(np.nanmax(H["csn_max"][sl]) / cmax)
            if first_rej >= 0:
                lo = max(first_rej - 5, 0)
                info["last5_before_first_rejection_Fwarm"] = ";".join("%.3g" % x for x in H["F_warm"][lo:first_rej]); info["last5_before_first_rejection_Fnewton"] = ";".join("%.3g" % x for x in H["F_newton"][lo:first_rej])
                info["csn_ref_max_over_cmax_at_first_rejection"] = float(r["cs_n"][first_rej].max() / cmax)
            cli.write_csv(pd.DataFrame([info]), args, f"charge_info_{th}_s{sd}.csv", note="certificate of the guarded filter", float_format=None)
            print(th, sd, "done: EKF cpu", round(cpu_e), "s, SPMe cpu", round(cpu_s), "s", flush=True)
    clock.log()


# ----------------------------------------------------------------------------------------------------------------------- aggregation
def _read(pattern, seeds, states):
    import pandas as pd
    out = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f, comment="#")
        if len(d) and int(d.seed.iloc[0]) in seeds and d.theta.iloc[0] in states: out.append(d)
    if not out: raise SystemExit(f"no per-run files matching {pattern} for the selected states/seeds; run the jobs first")
    return pd.concat(out, ignore_index=True)


class RefCache:
    """Reference trajectories and their interface quantities (computed once per state)."""
    def __init__(self, args):
        from dfnekf import refio, params
        self.args, self.refio, self.P0, self.c = args, refio, params.base(), {}

    def get(self, th):
        if th not in self.c:
            import numpy as np
            from dfnekf.step import interface_ref
            r = self.refio.load_uniform(proto_of(th), th, period=0.5, data_dir=self.args.data)
            self.c[th] = (r, np.array([interface_ref(r, i, self.P0, r["Rf"]) for i in range(r["N"])]))
        return self.c[th]


def aggregate(args):
    runtime.single_thread()
    import numpy as np, pandas as pd
    seeds = set(cli.parse_ids(args.seeds)); states = cli.states(args)
    tab = os.path.join(args.results, "tables"); sdd = os.path.join(args.results, "source_data"); os.makedirs(tab, exist_ok=True); os.makedirs(sdd, exist_ok=True)
    out = lambda d, name: os.path.join(d, name)
    refs = RefCache(args)
    D = _read(out(args.results, "charge_th*_s*.csv"), seeds, states)
    D["filter"] = D["filter"].replace({"EKF + guard": "EKF", "UKF-SPMe": "SPMe"})
    info = _read(out(args.results, "charge_info_th*_s*.csv"), seeds, states)
    thr_cols = [c for c in D.columns if c.startswith("thr")]
    met = ["E_phipl_rms_mV", "E_phipl_max_mV", "phipl_err_signed_mean_mV", "frac_unsafe_gt5mV", "phipl_min_err_mV", "E_eta_rms_mV", "ce_min_relerr_max_pct"]

    # ---- Tables III and VI: one row per state x seed x charge pulse (window 'pulse'), plus the 90 s rest window
    P = D[D.window == "pulse"]; R_ = D[D.window == "rest90"]; rows = []
    for (th, sd, pid, pl), g in P.groupby(["theta", "seed", "pulse_id", "pulse"]):
        e = g[g["filter"] == "EKF"].iloc[0]; s = g[g["filter"] == "SPMe"].iloc[0]
        a = int(e.t_start_s / 0.5); b = a + int(round(e.duration_s / 0.5)); V = np.asarray(refs.get(th)[0]["V"])
        row = dict(theta=th, seed=sd, pulse_id=pid, pulse=pl, kind=e.kind, t_start_s=e.t_start_s, duration_s=e.duration_s, V_max_ref_V=float(V[a:b].max()), phipl_min_ref_mV=e.phipl_min_ref_mV)
        for m in met: row[f"{m}_EKF"] = e[m]; row[f"{m}_SPMe"] = s[m]
        row["margin_phipl"] = s.E_phipl_rms_mV / e.E_phipl_rms_mV; row["margin_eta"] = s.E_eta_rms_mV / e.E_eta_rms_mV; row["margin_ce"] = s.ce_min_relerr_max_pct / e.ce_min_relerr_max_pct
        rr = R_[(R_.theta == th) & (R_.seed == sd) & (R_.pulse_id == pid)]
        for f in ("EKF", "SPMe"):
            q = rr[rr["filter"] == f]; row[f"rest90_E_phipl_rms_mV_{f}"] = float(q.E_phipl_rms_mV.iloc[0]) if len(q) else np.nan
        for c in thr_cols:
            for f, ser in [("EKF", e), ("SPMe", s)]: row[f"{c}_{f}"] = ser[c]
        rows.append(row)
    T = pd.DataFrame(rows)
    cli.write_csv(T, args, "tables/tab_charge_m17.csv", note="Tables III and VI: per state, seed and charge pulse (kind 'pulse' = 2C pulses and P6L pulses, 'CC' = 1C charge). phi_pl errors in mV (RMS, signed mean, "
                  "fraction of samples more than 5 mV above the reference = unsafe side), eta RMS in mV, c_e,min error in %, margins = SPMe/EKF; thr*: detection of phi_pl crossings (delay + late, - early). " + NOTE_TAIL, float_format=None)

    # ---- the -2C pulses: summary per state and pulse (medians, worst runs), detection and validity of the guard
    P2 = P[P.pulse == "-2C"].copy(); ordn = {th: {pid: k + 1 for k, pid in enumerate(sorted(P2[P2.theta == th].pulse_id.unique()))} for th in P2.theta.unique()}
    P2["pulse_2C"] = [ordn[th][pid] for th, pid in zip(P2.theta, P2.pulse_id)]
    if len(P2) == 0: print('no 2C pulses in these runs (quick run?): 2C summary tables skipped')
    else:
        summ = []
        for (th, k), g in P2.groupby(["theta", "pulse_2C"]):
            e = g[g["filter"] == "EKF"]; s = g[g["filter"] == "SPMe"].set_index("seed").loc[e.seed]; e = e.set_index("seed")
            row = dict(theta=th, pulse_2C=k, runs=len(e), duration_s_median=e.duration_s.median(), phipl_min_ref_mV_median=e.phipl_min_ref_mV.median(), phipl_min_ref_mV_min=e.phipl_min_ref_mV.min())
            for nm, d in (("EKF", e), ("SPMe", s)):
                for c in ("E_phipl_rms_mV", "E_eta_rms_mV", "ce_min_relerr_max_pct", "frac_unsafe_gt5mV", "phipl_err_signed_mean_mV", "phipl_min_err_mV"):
                    row[f"{c}_{nm}_median"] = d[c].median(); row[f"{c}_{nm}_max"] = d[c].max()
            row["margin_phipl_ratio_of_medians"] = row["E_phipl_rms_mV_SPMe_median"] / row["E_phipl_rms_mV_EKF_median"]; row["margin_phipl_run_median"] = (s.E_phipl_rms_mV / e.E_phipl_rms_mV).median()
            row["margin_eta_ratio_of_medians"] = row["E_eta_rms_mV_SPMe_median"] / row["E_eta_rms_mV_EKF_median"]; row["margin_ce_ratio_of_medians"] = row["ce_min_relerr_max_pct_SPMe_median"] / row["ce_min_relerr_max_pct_EKF_median"]
            summ.append(row)
        S = pd.DataFrame(summ)
        cli.write_csv(S, args, "tables/charge_summary.csv", note="2C charge pulses per state and pulse (1 = first, 2 = second): medians and maxima over the seeds. frac_unsafe = fraction of samples with phi_pl estimate more than 5 mV above "
                      "the reference. " + NOTE_TAIL, float_format="%.6g")

        tn = []
        def add(q, state, pulse, stat, v, doc): tn.append(dict(quantity=q, state=state, pulse_2C=pulse, stat=stat, value=v, description=doc))
        for _, r_ in S.iterrows():
            th, k = r_.theta, int(r_.pulse_2C)
            add("frac_unsafe_gt5mV_SPMe", th, k, "median", r_.frac_unsafe_gt5mV_SPMe_median, "fraction of samples with SPMe phi_pl more than 5 mV above the reference, median over seeds")
            add("frac_unsafe_gt5mV_SPMe", th, k, "max", r_.frac_unsafe_gt5mV_SPMe_max, "same, worst run")
            add("E_phipl_rms_mV_EKF", th, k, "max", r_.E_phipl_rms_mV_EKF_max, "grey-box phi_pl RMS error (mV), worst run")
            add("phipl_min_ref_mV", th, k, "median", r_.phipl_min_ref_mV_median, "minimum of the reference phi_pl during the pulse (mV)")
        for th, g in S.groupby("theta"):
            add("frac_unsafe_gt5mV_SPMe", th, "any", "max", g.frac_unsafe_gt5mV_SPMe_max.max(), "worst run over both pulses")
            add("E_phipl_rms_mV_EKF", th, "any", "max", g.E_phipl_rms_mV_EKF_max.max(), "worst run over both pulses (mV)")
            add("phipl_min_ref_mV", th, "any", "min", g.phipl_min_ref_mV_min.min(), "lowest reference phi_pl over both pulses and seeds (mV)")
        add("E_phipl_rms_mV_EKF", "all", "any", "max_of_medians", S.E_phipl_rms_mV_EKF_median.max(), "largest per-pulse median grey-box phi_pl RMS (abstract: 0.6 mV)")
        add("E_phipl_rms_mV_EKF", "all", "any", "max", S.E_phipl_rms_mV_EKF_max.max(), "worst single run (mV)")
        add("E_phipl_rms_mV_SPMe", "all", "any", "max_of_medians", S.E_phipl_rms_mV_SPMe_median.max(), "largest per-pulse median SPMe phi_pl RMS (abstract: 23.8 mV)")
        add("frac_unsafe_gt5mV_SPMe", "all", "any", "max", S.frac_unsafe_gt5mV_SPMe_max.max(), "worst single run over states and pulses")
        add("margin_phipl_ratio_of_medians", "all", "any", "min", S.margin_phipl_ratio_of_medians.min(), "smallest SPMe/EKF margin over states and pulses")
        add("margin_phipl_ratio_of_medians", "all", "any", "max", S.margin_phipl_ratio_of_medians.max(), "largest SPMe/EKF margin over states and pulses")
        pe = P2[P2["filter"] == "EKF"]; dly = (pe.thr0_t_ref_s - pe.t_start_s).dropna()
        add("ref_phipl_zero_crossing_after_pulse_start_s", "all", "any", "max", dly.max(), "time from the pulse start to the first reference sample with phi_pl < 0 (s), all 2C pulses with a crossing")
        pd.DataFrame(tn).pipe(lambda d: cli.write_csv(d, args, "tables/charge_text_numbers.csv", note="numbers quoted in the text of Sec. III-C that were previously computed by hand; 2C pulses of " + NOTE_TAIL, float_format="%.6g"))

        det = []
        for thr in THRS:
            for (th, f), g in P2.groupby(["theta", "filter"]):
                rc = g[f"thr{thr}_ref_crossed"].astype(bool); dt_ = g[f"thr{thr}_est_crossed"].astype(bool)
                det.append(dict(threshold_mV=thr, theta=th, filter=f, pulses=len(g), ref_crossed=int(rc.sum()), detected=int((rc & dt_).sum()), missed=int(g[f"thr{thr}_missed"].astype(bool).sum()),
                                false_alarms=int(g[f"thr{thr}_false_alarm"].astype(bool).sum()), delay_median_s=float(g[f"thr{thr}_delay_s"].median()), delay_min_s=float(g[f"thr{thr}_delay_s"].min()),
                                delay_max_s=float(g[f"thr{thr}_delay_s"].max()), t_ref_median_s=float(g[f"thr{thr}_t_ref_s"].median())))
        cli.write_csv(pd.DataFrame(det), args, "tables/charge_detection.csv", note="Appendix D: threshold detection on the 2C pulses (both pulses of all seeds); delay + late, - early. " + NOTE_TAIL, float_format="%.6g")
    V = info.groupby("theta").agg(runs=("seed", "count"), finite_all=("finite", "all"), rejected_total=("rejected_n", "sum"), rejected_frac_max=("rejected_frac", "max"), F_newton_max_accepted=("F_newton_max_accepted", "max"),
                                  max_abs_eta_V=("max_abs_eta_V", "max"), csn_ref_max=("csn_max_ref_over_cmax", "max"), csn_est_max=("csn_max_est_over_cmax", "max"), Rf_err_end_max=("Rf_err_end_pct", "max"),
                                  Ds_err_end_max=("Ds_err_end_pct", "max")).reset_index()
    cli.write_csv(V, args, "tables/charge_validity.csv", note="guard rejections, Newton residual of accepted steps and anode surface concentration, per state", float_format="%.6g")

    # ---- Table IV and Appendix D: thresholds on the 1C constant-current charge (trajectories of the stored runs)
    cc = []
    for th in sorted(D.theta.unique()):
        r, ref = refs.get(th)
        for sd in sorted(D[D.theta == th].seed.unique()):
            z = np.load(os.path.join(args.results, "traj", f"charge_{th}_s{int(sd)}.npz")); t, I = z["t"], z["I"]
            pl = [p for p in pulses(I) if p["kind"] == "CC" and p["label"] == "-1C"]
            if not pl: continue
            _, a, b = windows(I, pl[0])[0]
            for f, ppl in [("EKF", z["ppl"]), ("UKF-SPMe", z["spme_ppl"])]:
                e_ = 1e3 * (ppl[a:b] - ref[a:b, 1])
                row = dict(theta=th, seed=int(sd), filter=f, t_start_s=a * 0.5, duration_s=(b - a) * 0.5, phipl_ref_min_mV=1e3 * ref[a:b, 1].min(), err_pos_max_mV=float(max(0.0, e_.max())), err_signed_mean_mV=float(e_.mean()),
                           E_phipl_rms_mV=float(np.sqrt(np.mean(e_ ** 2))))
                for thr in THRS:
                    dd = detect(ref[:, 1], ppl, thr, t, a, b); row.update({f"thr{thr}_{k}": v for k, v in dd.items()})
                cc.append(row)
    if cc:
        Rn = pd.DataFrame(cc); cli.write_csv(Rn, args, "tables/charge_cc_runs.csv", note="1C constant-current charge, per run. " + NOTE_TAIL, float_format="%.10g")
        o = []
        for (th, f), g in Rn.groupby(["theta", "filter"]):
            for thr in THRS:
                rc = g[f"thr{thr}_ref_crossed"].astype(bool); dl = g[f"thr{thr}_delay_s"]
                o.append(dict(theta=th, filter=f, threshold_mV=thr, runs=len(g), ref_crossed=int(rc.sum()), detected=int((rc & g[f"thr{thr}_est_crossed"].astype(bool)).sum()), missed=int(g[f"thr{thr}_missed"].astype(bool).sum()),
                              false_alarms=int(g[f"thr{thr}_false_alarm"].astype(bool).sum()), t0_ref_s_median=g[f"thr{thr}_t_ref_s"].median(), t0_ref_s_min=g[f"thr{thr}_t_ref_s"].min(), t0_ref_s_max=g[f"thr{thr}_t_ref_s"].max(),
                              delay_s_median=dl.median(), delay_s_min=dl.min(), delay_s_max=dl.max(), err_pos_max_mV_median=g.err_pos_max_mV.median(), err_pos_max_mV_min=g.err_pos_max_mV.min(),
                              err_pos_max_mV_max=g.err_pos_max_mV.max(), phipl_ref_min_mV=g.phipl_ref_min_mV.median(), E_phipl_rms_mV_max=g.E_phipl_rms_mV.max()))
        cli.write_csv(pd.DataFrame(o), args, "tables/tab_cc_m17.csv", note="Table IV and Appendix D: 1C constant-current charge of F6Q (starts at t_start_s), thresholds of phi_pl at +20/+10/0 mV, detection = first downward crossing "
                      "(delay + late, - early; false alarm = filter crosses, reference never does); err_pos_max = max(0, estimate - reference) over the charge; reference t0 = t0_ref_s. " + NOTE_TAIL, float_format="%.6g")

    # ---- source data of Fig. 7 (seed 0)
    parts = []
    for th in ("th1", "th2", "th4"):
        f0 = os.path.join(args.results, "traj", f"charge_{th}_s0.npz")
        if not os.path.exists(f0): continue
        r, ref = refs.get(th); d = np.load(f0); I = np.asarray(r["I"])
        if len(d["eta"]) != len(I): continue                                    # truncated (quick) run
        p = [q for q in pulses(I) if q["label"] == "-2C"][0]; a, b = p["a"], p["b"]
        idx = np.arange(a - 60, min(b + 240, len(I))); tt = (idx - a) * 0.5
        parts.append(pd.DataFrame(dict(theta=th, t_from_pulse_start_s=tt, current_A=I[idx], voltage_reference_V=r["V"][idx], eta_reference_mV=1e3 * ref[idx, 0], eta_EKF_mV=1e3 * d["eta"][idx], eta_UKF_SPMe_mV=1e3 * d["spme_eta"][idx],
                                       phi_pl_reference_mV=1e3 * ref[idx, 1], phi_pl_EKF_mV=1e3 * d["ppl"][idx], phi_pl_UKF_SPMe_mV=1e3 * d["spme_ppl"][idx])))
    if parts:
        cli.write_csv(pd.concat(parts), args, "source_data/fig_eta_charge.csv", note="Fig. 7a-c: first 2C pulse of F6Q (30 s, terminated at the 4.2 V event; the vertical line marks the end of the pulse), seed 0; eta_lit and phi_pl "
                      "at the anode/separator face (extrapolated from the last two anode cells); EKF with state guard and SPMe UKF against the PyBaMM reference; current = mean over the 0.5 s step. " + NOTE_TAIL, float_format="%.9g")
    parts = []
    for th in states:
        f0 = os.path.join(args.results, "traj", f"charge_{th}_s0.npz")
        if not os.path.exists(f0): continue
        r, ref = refs.get(th); d = np.load(f0); I = d["I"]; tt = d["t"]
        pl = [q for q in pulses(I) if q["kind"] == "CC" and q["label"] == "-1C"]
        if not pl: continue
        k0, k1 = pl[0]["a"], pl[0]["b"] - 1; idx = np.arange(max(k0 - 60, 0), min(k1 + 120, len(I) - 1) + 1)[::2]
        parts.append(pd.DataFrame(dict(theta=th, t_from_cc_start_s=tt[idx] - tt[k0], current_A=I[idx], voltage_reference_V=d["V_ref"][idx], phi_pl_reference_mV=1e3 * ref[idx, 1], phi_pl_EKF_mV=1e3 * d["ppl"][idx],
                                       phi_pl_UKF_SPMe_mV=1e3 * d["spme_ppl"][idx])))
    if parts:
        cli.write_csv(pd.concat(parts), args, "source_data/fig_charge_cc.csv", note="Fig. 7d-f: phi_pl during the 1C constant-current charge of F6Q (1272 s; 372 s for th3), seed 0; reference phi_pl at the anode/separator face "
                      "(extrapolated from the last two anode cells), every 2nd sample (1 s), from 30 s before to 60 s after the charge. " + NOTE_TAIL, float_format=None)
    print("aggregate written to", args.results)


def main():
    p = cli.parser("Sec. III-C, Fig. 7, Tables III, IV, VI, Appendix D", "grey-box EKF with state guard versus SPMe UKF on the charge protocol F6Q")
    p.add_argument("--aggregate", action="store_true", help="build the tables and source data from the per-run files in --results (no simulation)")
    p.add_argument("--resume", action="store_true", help="skip runs whose trajectory file already exists")
    args = p.parse_args()
    if args.aggregate: aggregate(args)
    else: run_jobs(args)


if __name__ == "__main__":
    main()
