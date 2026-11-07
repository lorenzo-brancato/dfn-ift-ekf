"""Discharge experiments: grey-box DFN-EKF with state guard versus the SPMe unscented filter on the 3C discharge pulse (Fig. 10, Table VII, Appendix D discharge numbers).

Protocols: F6 for state th2 (the last 3C discharge pulse of 30 s never reaches the 2.5 V cutoff) and F6C for th4 and th5 (the reference reaches 2.5 V during the last 3C pulse
and the current is cut: after 17.5 s for th4, 4 s for th5). Window '3C' = from the onset of the last 3C pulse (I = 15 A) to the cutoff / end of the pulse; a 90 s rest window
after the pulse is also scored. Seeds 0-9, same filters, offsets, noise, P0 and Q as in ``charge.py`` (guarded EKF with converged-Newton closure estimating R_f, LAM_n, D_s, LLI fixed;
SPMe baseline from ``experiment.run_spme``). The reference eta and phi_pl are evaluated at the anode/separator face, linearly extrapolated from the last two anode cells
(``step.interface_ref``; the filter uses the same extrapolation in ``step.DFN.interface``).

Run (one job per state/seed set):
    python scripts/discharge.py --grid 4:1 --data data/sim --results results --states th2 --seeds 0-4 [--core 3]
    python scripts/discharge.py --quick --grid 4:1 --data data/sim --results results       # smoke test: th2, seed 0, first 400 s (contains one 3C pulse)
Aggregate (reads the per-run files of --results; writes <results>/tables and <results>/source_data):
    python scripts/discharge.py --aggregate --grid 4:1 --data data/sim --results results
Per-run outputs: <results>/traj/discharge_<protocol>_<state>_s<seed>.npz (all filter histories, spme_*), <results>/discharge_<protocol>_<state>_s<seed>.csv (errors per window and filter),
<results>/discharge_info_<protocol>_<state>_s<seed>.csv (window times, guard certificate, final parameter errors, minimum reference phi_pl in the 3C window, CPU).
Aggregate outputs: tables/tab_baselines_m17.csv (Table VII), tables/discharge_runs.csv, tables/discharge_summary.csv, tables/discharge_text_numbers.csv (phi_pl RMS ranges, margins,
pulse durations, minimum reference phi_pl), source_data/fig_eta3C.csv (Fig. 10, seed 0).
"""
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime                                             # no jax import here

QUICK_STEPS = 800
PROTO = {"th2": "F6", "th4": "F6C", "th5": "F6C"}
WIN_PULSE, WIN_REST = "3C onset -> cutoff (pulse)", "90 s of rest after the cutoff"
FIG_PRE_S, FIG_POST_S = 10.0, 60.0
PHIPL_NOTE = ("eta_lit and phi_pl are evaluated at the anode/separator face, linearly extrapolated from the last two anode cells "
              "(not in the single cell next to the separator)")
NOTE_TAIL = "grid 20/10/20, N_m = 12, dt = 0.5 s; " + PHIPL_NOTE


def window_3c(I, t):
    """Indices of the last 3C discharge pulse (I >= 14.9 A): onset sample, last sample, and the masks of the pulse window and of the 90 s rest window."""
    import numpy as np
    k3 = np.where(I >= 14.9)[0]; starts = k3[np.r_[True, np.diff(k3) > 1]]; ends = k3[np.r_[np.diff(k3) > 1, True]]
    s_on, k_end = int(starts[-1]), int(ends[-1])
    win = (t >= t[s_on]) & (t <= t[k_end]); post = (t > t[k_end]) & (t <= t[k_end] + 90)
    return s_on, k_end, win, post


def load_reference(proto, th, data):
    """Reference as loaded by ``experiment.Case`` (F6C: resampled on the uniform grid because the cutoff event is off-grid)."""
    from dfnekf import refio
    return refio.load_uniform(proto, th, period=0.5, data_dir=data) if proto.startswith("F6C") else refio.load(proto, th, period=0.5, data_dir=data)


# ----------------------------------------------------------------------------------------------------------------------- runs
def run_jobs(args):
    runtime.setup(args.core)
    import numpy as np, pandas as pd
    from dfnekf.experiment import Case
    from dfnekf import protocol_runs as pr
    if args.quick: args.results = os.path.join(args.results, "quick")
    clock = cli.Clock(args, "discharge.py")
    states = cli.states(args); seeds = cli.parse_ids(args.seeds)
    if args.quick: states, seeds = states[:1], seeds[:1]
    os.makedirs(os.path.join(args.results, "traj"), exist_ok=True)
    for tag in states:
        proto = PROTO[tag]
        case = Case(tag, args.grid, proto=proto, data_dir=args.data)
        if args.quick: pr.truncate(case, QUICK_STEPS)
        r = case.r; N = case.N; t = r["t"][:N]; I = np.asarray(r["I"])[:N]; V = np.asarray(r["V"])[:N]
        runner = pr.GreyBoxRunner(case); eta_ref = pr.reference_interface(case); ce_ref = r["ce"].min(1)[:N]
        s_on, k_end, win, post = window_3c(I, t); t_on, t_cut = t[s_on], t[k_end + 1]
        for sd in seeds:
            traj = os.path.join(args.results, "traj", f"discharge_{proto}_{tag}_s{sd}.npz")
            if args.resume and os.path.exists(traj): print(tag, sd, "exists, skipped", flush=True); continue
            H, out_s, cpu_e, cpu_s = pr.run_pair(runner, sd)
            np.savez(traj, t=t, I=I, **H, spme_eta=out_s["eta"], spme_ppl=out_s["ppl"], spme_ce_min=out_s["ce_min"], spme_vals=out_s["vals"])
            rows = []
            for nm, msk in [(WIN_PULSE, win), (WIN_REST, post)]:
                for f, d_eta, d_pl, d_ce in [("EKF DFN + guard", H["eta"], H["ppl"], H["ce_min"]), ("UKF-SPMe", out_s["eta"], out_s["ppl"], out_s["ce_min"])]:
                    e = 1e3 * (d_eta[msk] - eta_ref[msk, 0]); ep = 1e3 * (d_pl[msk] - eta_ref[msk, 1]); ec = 100 * np.abs(d_ce[msk] - ce_ref[msk]) / ce_ref[msk]
                    rows.append(dict(theta=tag, protocol=proto, seed=sd, window=nm, filter=f, steps=int(msk.sum()), E_eta_rms_mV=float(np.sqrt(np.mean(e ** 2))), E_eta_max_mV=float(np.abs(e).max()),
                                     E_phipl_rms_mV=float(np.sqrt(np.mean(ep ** 2))), ce_min_relerr_max_pct=float(ec.max())))
            cli.write_csv(pd.DataFrame(rows), args, f"discharge_{proto}_{tag}_s{sd}.csv", note=NOTE_TAIL, float_format=None)
            errs = pr.theta_end_errors(case, H); last5 = np.arange(k_end - 4, k_end + 1)
            info = dict(theta=tag, protocol=proto, seed=sd, t_onset=t_on, t_cutoff=t_cut, V_at_end_of_pulse=float(V[k_end]), V_min_run=float(V.min()), cs_n_min_ref_in_pulse=float(r["cs_n"][:N][win].min()),
                        ce_min_ref_in_pulse=float(r["ce"][:N][win].min()), F_warm_last5=";".join("%.3g" % x for x in H["F_warm"][last5]), F_newton_last5=";".join("%.3g" % x for x in H["F_newton"][last5]),
                        F_warm_max_pulse=float(H["F_warm"][win].max()), F_newton_max_pulse=float(H["F_newton"][win].max()), F_next_max_pulse=float(H["F_next"][win].max()),
                        rejected_frac_run=float(H["rejected"].mean()), rejected_n_run=int(H["rejected"].sum()), rejected_n_pulse=int(H["rejected"][win].sum()), finite=bool(np.all(np.isfinite(H["vals"]))),
                        max_abs_eta_ekf_V=float(np.nanmax(np.abs(H["eta"]))), Rf_err_end_pct=float(errs[0]), LAM_err_end_pct=float(errs[1]), Ds_err_end_pct=float(errs[2]), cpu_ekf_s=cpu_e, cpu_spme_s=cpu_s,
                        phipl_min_ref_V_3C=float(eta_ref[win, 1].min()), phipl_min_ekf_V_3C=float(H["ppl"][win].min()), phipl_min_spme_V_3C=float(out_s["ppl"][win].min()))
            cli.write_csv(pd.DataFrame([info]), args, f"discharge_info_{proto}_{tag}_s{sd}.csv", note="window times, guard certificate, minimum of phi_pl in the 3C window", float_format=None)
            print(tag, proto, sd, "done: EKF cpu", round(cpu_e), "s, SPMe cpu", round(cpu_s), "s; E_eta GB/SPMe", round(rows[0]["E_eta_rms_mV"], 3), round(rows[1]["E_eta_rms_mV"], 3), flush=True)
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


def aggregate(args):
    runtime.single_thread()
    import numpy as np, pandas as pd
    from dfnekf import params
    from dfnekf.step import interface_ref
    seeds = set(cli.parse_ids(args.seeds)); states = cli.states(args)
    for d in ("tables", "source_data"): os.makedirs(os.path.join(args.results, d), exist_ok=True)
    S = _read(os.path.join(args.results, "discharge_F6*_th*_s*.csv"), seeds, states); I_ = _read(os.path.join(args.results, "discharge_info_F6*_th*_s*.csv"), seeds, states)
    S = S[S.window.str.startswith("3C")]; rows = []
    for (tg, sd), g in S.groupby(["theta", "seed"]):
        i = I_[(I_.theta == tg) & (I_.seed == sd)].iloc[0]; e = g[g["filter"].str.startswith("EKF")].iloc[0]; s = g[g["filter"] == "UKF-SPMe"].iloc[0]
        rows.append(dict(theta=tg, protocol=i.protocol, seed=sd, E_eta_rms_mV_EKF=e.E_eta_rms_mV, E_eta_rms_mV_SPMe=s.E_eta_rms_mV, E_eta_max_mV_EKF=e.E_eta_max_mV, E_phipl_rms_mV_EKF=e.E_phipl_rms_mV,
                         E_phipl_rms_mV_SPMe=s.E_phipl_rms_mV, ce_min_relerr_max_pct_EKF=e.ce_min_relerr_max_pct, ce_min_relerr_max_pct_SPMe=s.ce_min_relerr_max_pct, Rf_err_end_pct=i.Rf_err_end_pct,
                         LAM_err_end_pct=i.LAM_err_end_pct, Ds_err_end_pct=i.Ds_err_end_pct, pulse_duration_s=i.t_cutoff - i.t_onset, rejected_n_run=i.rejected_n_run, finite=i.finite,
                         F_newton_max_pulse=i.F_newton_max_pulse, V_min_run=i.V_min_run, phipl_min_ref_V_3C=i.phipl_min_ref_V_3C, phipl_min_ekf_V_3C=i.phipl_min_ekf_V_3C, phipl_min_spme_V_3C=i.phipl_min_spme_V_3C))
    D = pd.DataFrame(rows); cli.write_csv(D, args, "tables/discharge_runs.csv", note="3C window, one row per state and seed. " + NOTE_TAIL, float_format=None)
    cols = [c for c in D.columns if c not in ("theta", "protocol", "seed", "finite")]; out = []
    for tg, g in D.groupby("theta"):
        r = dict(theta=tg, protocol=g.protocol.iloc[0], runs=len(g), seeds=",".join(str(int(x)) for x in sorted(g.seed)), all_finite=bool(g.finite.all()))
        for c in cols: r[c + "_median"] = g[c].median(); r[c + "_min"] = g[c].min(); r[c + "_max"] = g[c].max()
        out.append(r)
    O = pd.DataFrame(out); cli.write_csv(O, args, "tables/discharge_summary.csv", note="3C window: median, min and max over seeds. " + NOTE_TAIL, float_format="%.6g")

    # ---- Table VII
    Bl = []
    for tg, g in D.groupby("theta"):
        eg, es = g.E_eta_rms_mV_EKF.median(), g.E_eta_rms_mV_SPMe.median(); n = len(g)
        Bl.append(dict(theta=tg, protocol=g.protocol.iloc[0], filter="EKF grey-box DFN + guard", runs=n, seeds=",".join(str(int(x)) for x in sorted(g.seed)), E_eta_3C_median_mV=eg, E_eta_3C_max_over_seeds_mV=g.E_eta_rms_mV_EKF.max(),
                       margin_E_eta=es / eg, ce_min_3C_err_median_pct=g.ce_min_relerr_max_pct_EKF.median(), theta_err_Rf_median_pct=g.Rf_err_end_pct.median(), theta_err_LAM_median_pct=g.LAM_err_end_pct.median(),
                       theta_err_Ds_median_pct=g.Ds_err_end_pct.median(), pulse_duration_s=g.pulse_duration_s.median()))
        Bl.append(dict(theta=tg, protocol=g.protocol.iloc[0], filter="UKF-SPMe", runs=n, seeds=",".join(str(int(x)) for x in sorted(g.seed)), E_eta_3C_median_mV=es, E_eta_3C_max_over_seeds_mV=g.E_eta_rms_mV_SPMe.max(),
                       margin_E_eta=es / eg, ce_min_3C_err_median_pct=g.ce_min_relerr_max_pct_SPMe.median(), pulse_duration_s=g.pulse_duration_s.median()))
    cli.write_csv(pd.DataFrame(Bl), args, "tables/tab_baselines_m17.csv", note="Table VII: 3C discharge pulse (th4, th5: F6C, truncated at the 2.5 V cutoff; th2: F6, the 3C does not reach the cutoff); medians over seeds; "
                  "margin = ratio of the medians of E_eta (SPMe / grey-box); the cost columns of the full table come from the timing script. " + NOTE_TAIL, float_format="%.6g")

    # ---- numbers quoted in the text
    tn = []
    def add(q, state, stat, v, doc): tn.append(dict(quantity=q, state=state, stat=stat, value=v, description=doc))
    for tg, g in D.groupby("theta"):
        for nm in ("EKF", "SPMe"):
            add(f"E_phipl_rms_mV_{nm}", tg, "median", g[f"E_phipl_rms_mV_{nm}"].median(), "phi_pl RMS error in the 3C window (mV)")
            add(f"E_phipl_rms_mV_{nm}", tg, "max", g[f"E_phipl_rms_mV_{nm}"].max(), "worst run")
        add("pulse_duration_s", tg, "median", g.pulse_duration_s.median(), "t_cutoff - t_onset")
        add("phipl_min_ref_V_3C", tg, "min_over_runs", g.phipl_min_ref_V_3C.min(), "minimum of the reference phi_pl in the 3C window (V)")
        add("phipl_min_ekf_V_3C", tg, "min_over_runs", g.phipl_min_ekf_V_3C.min(), "minimum of the grey-box phi_pl in the 3C window (V)")
        add("phipl_min_spme_V_3C", tg, "min_over_runs", g.phipl_min_spme_V_3C.min(), "minimum of the SPMe phi_pl in the 3C window (V)")
    for nm in ("EKF", "SPMe"):
        m = D.groupby("theta")[f"E_phipl_rms_mV_{nm}"].median()
        add(f"E_phipl_rms_mV_{nm}", "all", "range_of_medians_lo", m.min(), "lowest per-state median"); add(f"E_phipl_rms_mV_{nm}", "all", "range_of_medians_hi", m.max(), "highest per-state median")
        m = D.groupby("theta")[f"E_eta_rms_mV_{nm}"].median()
        add(f"E_eta_rms_mV_{nm}", "all", "range_of_medians_lo", m.min(), "lowest per-state median"); add(f"E_eta_rms_mV_{nm}", "all", "range_of_medians_hi", m.max(), "highest per-state median")
    lo = D.phipl_min_ref_V_3C.min(); add("phipl_min_ref_V_3C", "all", "min_over_states", lo, "minimum of the reference phi_pl over all states in the 3C window (V)")
    add("phipl_ref_stays_above_0p4V", "all", "bool", float(lo > 0.4), "1 if the reference phi_pl stays above 0.4 V in the 3C window of every state")
    cli.write_csv(pd.DataFrame(tn), args, "tables/discharge_text_numbers.csv", note="numbers quoted in the text of Sec. III-C (discharge) and Appendix D. " + NOTE_TAIL, float_format="%.6g")

    # ---- Fig. 10 source data (seed 0)
    P0 = params.base(); parts = []; cm = []
    for tg in states:
        proto = PROTO[tg]; f0 = os.path.join(args.results, "traj", f"discharge_{proto}_{tg}_s0.npz")
        if not os.path.exists(f0): continue
        r = load_reference(proto, tg, args.data); d = np.load(f0); t = r["t"]; I = r["I"]
        if len(d["eta"]) != len(t): continue                                    # truncated (quick) run
        s_on, k_end, _, _ = window_3c(I, t); t_on = t[s_on]
        idx = np.where((t >= t_on - FIG_PRE_S - 1e-6) & (t <= t[k_end] + FIG_POST_S))[0]       # 1e-6: the time stamps carry ~1e-12 jitter
        ref = np.array([interface_ref(r, i, P0, r["Rf"]) for i in idx]); tc = t[k_end + 1] - t_on
        parts.append(pd.DataFrame(dict(theta=tg, protocol=proto, t_from_3C_onset_s=t[idx] - t_on, current_A=I[idx], voltage_reference_V=r["V"][idx], eta_reference_mV=1e3 * ref[:, 0], eta_EKF_mV=1e3 * d["eta"][idx],
                                       eta_UKF_SPMe_mV=1e3 * d["spme_eta"][idx], phi_pl_reference_mV=1e3 * ref[:, 1], phi_pl_EKF_mV=1e3 * d["ppl"][idx], phi_pl_UKF_SPMe_mV=1e3 * d["spme_ppl"][idx])))
        cm.append(f"{tg}: reference {proto}" + (" (3C of 30 s, never reaches the 2.5 V cutoff)" if proto == "F6" else " (3C terminated at the 2.5 V cutoff)") + f", cutoff / end of pulse at t = {tc:.1f} s after the onset")
    if parts:
        cli.write_csv(pd.concat(parts), args, "source_data/fig_eta3C.csv", note="Fig. 10 (a, c, e: eta_lit; b, d, f: phi_pl): seed 0, EKF with state guard (voltage only) and SPMe UKF against the PyBaMM reference; "
                      + "; ".join(cm) + "; current_A = mean over the 0.5 s step (on F6C the off-grid cutoff event is rounded to the nearest grid time). " + NOTE_TAIL, float_format="%.9g")
    print("aggregate written to", args.results)


def main():
    p = cli.parser("Fig. 10, Table VII, Appendix D (discharge)", "grey-box EKF with state guard versus SPMe UKF on the 3C discharge pulse", states="th2,th4,th5")
    p.add_argument("--aggregate", action="store_true", help="build the tables and source data from the per-run files in --results (no simulation)")
    p.add_argument("--resume", action="store_true", help="skip runs whose trajectory file already exists")
    args = p.parse_args()
    if args.aggregate: aggregate(args)
    else: run_jobs(args)


if __name__ == "__main__":
    main()
