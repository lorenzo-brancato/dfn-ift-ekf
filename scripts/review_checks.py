"""Checks added after the external review of the manuscript.

Paper items: (spme) Appendix D, "SPMe in open loop with the true parameters": phi_pl error and 0 mV crossings of the SPMe without any estimation (3.7-20.7 mV RMS in the 2C pulses,
same missed / early / false crossings as the SPMe filter: the error is structural) and the parameter errors of both filters at the end of the P6L blocks;
(sei) Sec. III-D, "the SEI side-reaction current removes 13 % of the CV model error": the lithium-consuming reaction, added to the open-loop model from the lithium lost in the reference,
explains about 13 % of the 0.77 mV overestimate in the CV of RPT 3-6; (freeze) Sec. III-D, change of the estimates during and after the CV hold in a run made with ``ageing_track.py --freeze``.

Parts (--parts, default spme,sei):
  spme    SPMe OPEN LOOP with the TRUE parameters on the charge protocol F6Q (F6Q3 for th3), the five states, started from the true initial state at t = 0 (so it also runs
          through the two P6L blocks). phi_pl error against the PyBaMM reference at the anode/separator face, as in ``scripts/charge.py`` (same ``dfnekf.metrics``):
          RMS over each 2C pulse, and the +0/+10/+20 mV crossings of the 1C constant-current charge. Printed next to the SPMe-UKF filter (Tables III-IV, read from
          <ref-results>/tables). Also the parameter errors of the SPMe filter and of the grey-box EKF at the end of the two P6L blocks (8640 steps), from the stored
          charge trajectories (<ref-results>/traj/charge_th*_s*.npz).  Output: review_spme_open_loop.csv, review_spme_cc.csv, review_p6l_end_errors.csv.
  sei     Constant-voltage hold of RPT 3-6 of the ageing reference: how much of the model overestimate (Fig. 8f, 0.77 mV) the lithium-consuming SEI reaction explains.
          The side-reaction current is read from the reference (I_SEI = -F dLi_tot/dt, Savitzky-Golay over 120 s) and added to the open-loop model of
          ``scripts/cv_model_error.py`` (variant z1, uniform film) as a uniform extra anodic-side current density: j_n = j_intercalation + j_SEI, with the film drop R_f j_n on the total
          (as in the closure), the particles and the lithium accounting fed by the intercalation part only, the electrolyte source by the total. Three runs from the same state at the
          start of each CV: base (continuous realignment of the lithium inventory to the true Li_tot, as in cv_model_error.py), noSEI (no realignment in the hold: the reference with
          the side reaction switched off there), SEI (side reaction in the closure, no realignment). About 30 min (one pass over the reference up to RPT 6).
          Output: review_sei_cv.csv, review_sei_current.csv.
  freeze  Change of the estimates and of the estimated lithium inventory during the CV hold and in the 30 min of rest after it, for each RPT, from a run of
          ``ageing_track.py --freeze ...`` (--freeze-npz: its stored trajectory, default results/ageing/traj/ageing_track_seed0_freeze-voltage.npz; also valid for the paper's run).
          Output: review_freeze_after_cv_<run>.csv.
--quick: spme on the first 800 steps of th1; sei on the CV of RPT 0 (about 2 min); freeze on the RPTs that the stored trajectory covers.

  python scripts/review_checks.py --parts spme --grid 4:1 --data data --results results/review --ref-results results/4to1
  python scripts/review_checks.py --parts sei --grid 4:1 --data data --results results/review
  python scripts/review_checks.py --parts freeze --freeze-npz results/ageing/traj/ageing_track_seed0_freeze-voltage.npz --data data --results results/review
  python scripts/review_checks.py --quick --parts spme,sei --data data --results results/quick
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

DT = 0.5
FARADAY = 96485.33212


# ------------------------------------------------------------------------------------------------------------------------- SPMe open loop
def spme_part(a):
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from dfnekf.experiment import Case
    from dfnekf.spme import SPMe
    from dfnekf import protocol_runs as pr
    from dfnekf.protocols import PROTOCOLS
    from dfnekf.metrics import pulses, windows, metrics, detect
    rows, ccrows, trows = [], [], []
    n_p6l = int(round(sum(d for _, d in PROTOCOLS["P6Lx2"]["segs"]) / DT))
    for th in (cli.states(a)[:1] if a.quick else cli.states(a)):
        proto = "F6Q3" if th == "th3" else "F6Q"
        case = Case(th, a.grid, proto=proto, data_dir=a.data)
        if a.quick: pr.truncate(case, 800)
        sp = SPMe(case.m, case.ek); r = case.r; N = case.N; t = r["t"][:N]; I = np.asarray(r["I"])[:N]
        x0 = jnp.concatenate([sp.init(*[float(x) for x in r["c0_eff"]]), jnp.asarray(case.ek.coord_of(case.true_vals))])

        def body(x, Ik):
            xn, V, Et, _ = sp.f(x, Ik); return xn, (V, Et)
        _, (V, Et) = jax.jit(lambda x0_, I_: jax.lax.scan(body, x0_, I_))(x0, jnp.asarray(I))
        V, Et = np.asarray(V), np.asarray(Et)                                    # Et columns: eta, phi_pl, c_e,min (SPMe, uniform reaction current)
        ref = pr.reference_interface(case); ce_ref = r["ce"].min(1)[:N]
        print(th, proto, "SPMe open loop: voltage RMS error [mV]", round(1e3 * float(np.sqrt(np.mean((V - r["V"][:N]) ** 2))), 2), flush=True)
        k2 = 0
        for p in pulses(I):
            if p["kind"] == "pulse" and p["label"] == "-2C":
                k2 += 1; mm = metrics(ref[:, 0], ref[:, 1], ce_ref, Et[:, 0], Et[:, 1], Et[:, 2], p["a"], p["b"])
                rows.append(dict(theta=th, pulse_2C=k2, t_start_s=p["a"] * DT, duration_s=(p["b"] - p["a"]) * DT, **mm))
            if p["kind"] == "CC" and p["label"] == "-1C":
                e_ = 1e3 * (Et[p["a"]:p["b"], 1] - ref[p["a"]:p["b"], 1])
                row = dict(theta=th, t_start_s=p["a"] * DT, duration_s=(p["b"] - p["a"]) * DT, phipl_ref_min_mV=float(1e3 * ref[p["a"]:p["b"], 1].min()), E_phipl_rms_mV=float(np.sqrt(np.mean(e_ ** 2))),
                           err_signed_mean_mV=float(e_.mean()), err_pos_max_mV=float(max(0.0, e_.max())), err_neg_min_mV=float(min(0.0, e_.min())))
                for thr in (20, 10, 0):
                    dd = detect(ref[:, 1], Et[:, 1], thr, t, p["a"], p["b"]); row.update({f"thr{thr}_{k}": v for k, v in dd.items()})
                ccrows.append(row)
        # parameter errors of the two filters at the end of the two P6L blocks, from the stored charge runs
        trd = os.path.join(a.ref_results, "traj")
        for sd in cli.parse_ids(a.seeds):
            f = os.path.join(trd, f"charge_{th}_s{sd}.npz")
            if not os.path.exists(f): continue
            z = np.load(f); k = min(n_p6l, len(z["vals"])) - 1
            for lab, v in (("grey-box EKF", z["vals"]), ("SPMe UKF", z["spme_vals"])):
                e = 100 * (v[k] - case.true_vals) / case.range
                trows.append(dict(theta=th, seed=sd, filter=lab, step=k + 1, **{f"{c}_err_pct": e[i] for i, c in enumerate(case.coords)}, **{f"{c}_est": v[k][i] for i, c in enumerate(case.coords)}, **{f"{c}_true": case.true_vals[i] for i, c in enumerate(case.coords)}))
    R = pd.DataFrame(rows); C = pd.DataFrame(ccrows); T = pd.DataFrame(trows)
    cli.write_csv(R, a, "review_spme_open_loop.csv", note="SPMe open loop, true parameters, 2C pulses of F6Q: phi_pl and eta errors against the reference (mV), c_e,min error (%)")
    cli.write_csv(C, a, "review_spme_cc.csv", note="SPMe open loop, true parameters, 1C CC charge of F6Q: phi_pl RMS error and threshold crossings (delay + late, - early; false alarm = SPMe crosses, reference never does)")
    if len(T): cli.write_csv(T, a, "review_p6l_end_errors.csv", note="parameter errors (% of range, signed) at the end of the two P6L blocks (step 8640) in the stored charge runs")
    # comparison with the filters of Tables III-IV
    tab = os.path.join(a.ref_results, "tables")
    try:
        S = pd.read_csv(os.path.join(tab, "charge_summary.csv"), comment="#"); K = pd.read_csv(os.path.join(tab, "tab_cc_m17.csv"), comment="#")
        print("\n2C pulses: phi_pl RMS error [mV]  (open-loop SPMe, true parameters | SPMe UKF median of 10 runs | grey-box EKF median)")
        for _, r_ in R.iterrows():
            s = S[(S.theta == r_.theta) & (S.pulse_2C == r_.pulse_2C)].iloc[0]
            print(f"  {r_.theta} pulse {int(r_.pulse_2C)}: {r_.E_phipl_rms_mV:6.2f} | {s.E_phipl_rms_mV_SPMe_median:6.2f} | {s.E_phipl_rms_mV_EKF_median:5.2f}   (ref min phi_pl {r_.phipl_min_ref_mV:6.1f} mV)")
        print("\n1C charge: crossing of 0 mV, delay [s] (+ late, - early) (open-loop SPMe | SPMe UKF median [min, max] of 10 runs | grey-box EKF median)")
        for _, r_ in C.iterrows():
            kk = K[(K.theta == r_.theta) & (K.threshold_mV == 0)].set_index("filter")
            fmt = lambda d: "n/a" if not np.isfinite(d) else f"{d:+.0f}"
            s = kk.loc["UKF-SPMe"] if "UKF-SPMe" in kk.index else kk.loc["SPMe"]; e = kk.loc["EKF"]
            ol = "MISSED (never crosses)" if r_.thr0_missed else ("FALSE ALARM" if r_.thr0_false_alarm else fmt(r_.thr0_delay_s))
            print(f"  {r_.theta}: ref crosses {bool(r_.thr0_ref_crossed)} (t = {r_.thr0_t_ref_s:.0f} s) | open loop: {ol} | SPMe UKF {fmt(s.delay_s_median)} [{fmt(s.delay_s_min)}, {fmt(s.delay_s_max)}], false alarms {int(s.false_alarms)}/10 | EKF {fmt(e.delay_s_median)}")
    except Exception as ex:                                                  # the comparison tables are optional
        print("comparison tables not read:", ex)
    if len(T):
        g = T.groupby(["theta", "filter"])[[c + "_err_pct" for c in case.coords]].agg(["median", lambda x: x.abs().max()])
        print("\nparameter errors at the end of the P6L blocks (signed median / largest absolute over seeds), % of range\n", g.round(2).to_string())


# ------------------------------------------------------------------------------------------------------------------------- SEI in the CV
def sei_part(a):
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from scipy.signal import savgol_filter
    from dfnekf import ageing as ag, params as prm, ageing_tracker as at
    from dfnekf.step import DFN
    from dfnekf.closure import Alg
    from dfnekf.grid import ratio_of

    class AlgSEI(Alg):
        """Closure with a uniform side-reaction current density in the anode: th = [R_f, a_s, D_s, j_SEI] (j_SEI < 0 in charge, A/m2 of particle surface).
        Butler-Volmer applies to j_n - j_SEI; the film drop uses the total j_n (the Jacobian is unchanged because j_SEI is constant in the step)."""
        def coef(self, th):
            P = self.P
            return 3 * P["eps_sn"] * th[1] / P["Rn"], 3 * P["eps_sp"] / P["Rp"], P["Rf0"] * th[0]

        def residual(self, g, csn, csp, ce, I, th):
            return Alg.residual(self, g, csn, csp, ce, I, th).at[:self.Nn].add(-th[3])

    ref = os.path.join(a.data, "aging_dyn_rptslow_full.npz"); P = ag.load(ref); R0 = np.load(ref)
    cyc, sti = at.step_map(P, R0); rpt = np.where(R0["is_rpt"])[0]; blocks = [3, 4, 5, 6]
    win = {}
    for kr in blocks:
        idx = np.where((cyc == rpt[kr]) & (sti == 17))[0]; win[kr] = (int(idx[0]), int(idx[-1]) + 1)
    if a.quick: blocks = [0]; idx0 = np.where((cyc == rpt[0]) & (sti == 17))[0]; win = {0: (int(idx0[0]), int(idx0[-1]) + 1)}
    Ntot = len(P["t"]); Lt = np.asarray(P["Ltot"]); Isei = -FARADAY * savgol_filter(Lt, 241, 2, deriv=1, delta=DT)            # side-reaction current [A], from the lithium loss
    I = np.asarray(P["I"]); Vref = np.asarray(P["V"])
    P0 = prm.base(); m = DFN(P0, 20, 10, 20, 12, 2, DT, a.grid); A_ = m.A; Nn = m.Nn
    mS = DFN(P0, 20, 10, 20, 12, 2, DT, a.grid); AS = AlgSEI(P0, 20, 10, 20, "linear", "inv_mean", ratio_of(a.grid)); mS.A = AS
    z0 = m.init_state(float(P["cn_avg"][0]), float(P["cp_avg"][0])); th0 = jnp.array([1.0, 1.0, 1.0]); consts = m.part_consts(th0); E1, G1 = m.elec_mats(th0)
    n0 = m.lithium(z0, th0) * ag.AREA; Lt0 = float(Lt[0]); dxn = jnp.asarray(A_.dxn); Lj, Rj, Lnn = jnp.asarray(Lt), jnp.asarray(1.0), P0["Ln"]
    csn, csp, ce = m.surf(z0); g0 = A_.newton(A_.guess(csn, csp, jnp.asarray(I[0]), th0), csn, csp, ce, jnp.asarray(I[0]), th0)[0]
    asv = 1 - np.asarray(P["LAM"]); Rf = np.asarray(P["Rf"])

    def realign(z, th3, a_s, Ltk):
        nm = m.lithium(z, th3) * ag.AREA; target = n0 - (Lt0 - Ltk)
        return z.at[:Nn].add(-(nm - target) / (ag.AREA * P0["eps_sn"] * a_s * jnp.sum(dxn)))

    def body_base(carry, inp):                                                 # = scripts/cv_model_error.py, variant z1
        z, g = carry; Ik, rf, a_s, Ltk = inp; th = jnp.array([rf, a_s, 1.0]); em = (E1, G1.at[:, :Nn].multiply(a_s))
        z = realign(z, th, a_s, Ltk)
        csn, csp, ce = m.surf(z); g, res, _ = A_.solve_robust(g, csn, csp, ce, Ik, th, tol=1e-10); V = A_.V_of(g, Ik)
        return (m.advance(z, g, th, consts, em), g), V

    def make_body_sei(with_sei):                                                # no realignment: the lithium accounting is the model's own
        def body(carry, inp):
            z, g = carry; Ik, rf, a_s, Isk = inp
            js = -Isk / (P0["area"] * 3 * P0["eps_sn"] * a_s / P0["Rn"] * Lnn) if with_sei else 0.0 * Isk
            th4 = jnp.array([rf, a_s, 1.0, js]); th3 = th4[:3]; em = (E1, G1.at[:, :Nn].multiply(a_s))
            csn, csp, ce = mS.surf(z); g, res, _ = AS.solve_robust(g, csn, csp, ce, Ik, th4, tol=1e-10); V = AS.V_of(g, Ik)
            zi = mS.advance(z, g.at[:Nn].add(-js), th3, consts, em); zt = mS.advance(z, g, th3, consts, em)           # intercalation part for the particles, total for the electrolyte
            ci, ai, si, _ = mS.split(zi); _, _, _, cet = mS.split(zt)
            return (mS.join(ci, ai, si, cet), g), V
        return body
    runs = {k: jax.jit(lambda carry, xs, b=b: jax.lax.scan(b, carry, xs)) for k, b in (("base", body_base), ("noSEI", make_body_sei(False)), ("SEI", make_body_sei(True)))}

    # baseline pass up to the end of the last hold, snapshot of the state at the start of each hold
    stops = sorted({s for w in win.values() for s in w}); snaps, Vb = {}, np.full(Ntot, np.nan); carry = (z0, g0); pos = 0; CH = 100000
    xsb = lambda s, b: (jnp.asarray(I[s:b]), jnp.asarray(Rf[s:b]), jnp.asarray(asv[s:b]), jnp.asarray(Lt[s:b]))
    for stop in stops:
        while pos < stop:
            b = min(stop, pos + CH); carry, V = runs["base"](carry, xsb(pos, b)); Vb[pos:b] = np.asarray(V); pos = b; print("baseline", pos, "of", stops[-1], flush=True)
        for kr, (s0, s1) in win.items():
            if s0 == stop: snaps[kr] = carry
    rows, srows = [], []
    for kr, (s0, s1) in win.items():
        sl = slice(s0, s1); xs = (jnp.asarray(I[sl]), jnp.asarray(Rf[sl]), jnp.asarray(asv[sl]), jnp.asarray(Isei[sl]))
        out = {"base": Vb[sl]}
        for k in ("noSEI", "SEI"): out[k] = np.asarray(runs[k](snaps[kr], xs)[1]); print("block", kr, k, flush=True)
        err = {k: 1e3 * (v - Vref[sl]) for k, v in out.items()}
        row = dict(block=kr, n=s1 - s0, I_start_A=float(I[s0]), I_end_A=float(I[s1 - 1]), **{f"err_mean_mV_{k}": float(e.mean()) for k, e in err.items()}, **{f"err_rms_mV_{k}": float(np.sqrt((e ** 2).mean())) for k, e in err.items()},
                   V_effect_of_SEI_mean_mV=float(1e3 * (out["SEI"] - out["noSEI"]).mean()), V_effect_of_lithium_accounting_mean_mV=float(1e3 * (out["noSEI"] - out["base"]).mean()))
        rows.append(row)
        srows.append(dict(block=kr, I_sei_start_mA=1e3 * float(Isei[s0 + 120:s0 + 360].mean()), I_sei_mid_mA=1e3 * float(Isei[(s0 + s1) // 2 - 120:(s0 + s1) // 2 + 120].mean()), I_sei_end_mA=1e3 * float(Isei[s1 - 360:s1 - 120].mean()),
                          I_sei_mean_mA=1e3 * float(Isei[sl].mean()), charge_lost_to_SEI_mAh=float(Isei[sl].sum() * DT / 3.6), terminal_charge_mAh=float(-I[sl].sum() * DT / 3.6), frac_of_terminal_current_end_pct=100 * float(Isei[s1 - 360:s1 - 120].mean() / -I[s1 - 120:s1].mean())))
    R = pd.DataFrame(rows); S = pd.DataFrame(srows); w = R.n.values
    pooled = {k: float(np.average(R[f"err_mean_mV_{k}"], weights=w)) for k in ("base", "noSEI", "SEI")}
    R = pd.concat([R, pd.DataFrame([dict(block="3-6 pooled", n=int(w.sum()), **{f"err_mean_mV_{k}": v for k, v in pooled.items()})])], ignore_index=True)
    cli.write_csv(R, a, "review_sei_cv.csv", note="CV hold of RPT 3-6, open loop with the true parameters: mean model - reference voltage [mV] without side reaction and with continuous inventory realignment (base = cv_model_error z1), without side reaction and without realignment (noSEI), with the measured side-reaction current (SEI)")
    cli.write_csv(S, a, "review_sei_current.csv", note="side-reaction current in the CV holds, I_SEI = -F dLi_tot/dt (mA), and the charge it takes (mAh) against the terminal charge")
    print(R.round(4).to_string()); print(S.round(4).to_string())
    print(f"\nexplained by the side-reaction current: {100 * (1 - pooled['SEI'] / pooled['base']):.1f} % of the pooled model error ({pooled['base']:.3f} -> {pooled['SEI']:.3f} mV); without realignment and without the reaction: {pooled['noSEI']:.3f} mV")


# ------------------------------------------------------------------------------------------------------------------------- frozen-parameter ablation
def freeze_part(a):
    import numpy as np, pandas as pd
    from dfnekf import ageing as ag, ageing_tracker as at, params as prm
    from dfnekf.step import DFN
    ref = os.path.join(a.data, "aging_dyn_rptslow_full.npz"); P = ag.load(ref); R0 = np.load(ref); se = R0["step_end_t"]; rpt = np.where(R0["is_rpt"])[0]
    d = np.load(a.freeze_npz, allow_pickle=True); vals, li = d["vals"], d["li"]; n10 = len(vals); names = [str(c) for c in d["coords"]]
    nst = len(d["ps_innov"]); frozen = d["ps_frozen"] > 0 if "ps_frozen" in d.files else np.zeros(nst, bool); csn = d["ps_csn_max"].astype(float) / prm.base()["cmax_n"]
    rej = d["ps_rejected"].astype(bool); inn = d["ps_innov"].astype(float)
    m = DFN(prm.base(), 20, 10, 20, 12, 2, DT, a.grid); Linv = at.inventory_loss(m, P, li)
    rng = {"Rf": 9.0, "LAM": 0.3, "Ds": 0.7, "LLI": 0.2}; rows = []
    k = lambda t: min(int(round(t / 10)), n10 - 1); s = lambda t: int(round(t / DT))
    for kr, c in enumerate(rpt):
        t_cv0, t_cv1, t_end = se[c][16], se[c][17], (se[c][18] if len(se[c]) > 18 else np.nan)
        if not np.isfinite(t_end) or s(t_end) > nst: continue                       # RPT not covered by the run
        a0, a1, a2 = k(t_cv0), k(t_cv1), k(t_end)
        row = dict(rpt=kr, t_cv_start_h=t_cv0 / 3600, cv_s=t_cv1 - t_cv0, rest_s=t_end - t_cv1)
        for i, nm in enumerate(names):
            row[f"d_{nm}_CV_pts"] = 100 * (vals[a1, i] - vals[a0, i]) / rng[nm]; row[f"d_{nm}_after_CV_rest_pts"] = 100 * (vals[a2, i] - vals[a1, i]) / rng[nm]
        row["d_inventory_CV_pts"] = 100 * (Linv[a1] - Linv[a0]) / 0.2; row["d_inventory_after_CV_rest_pts"] = 100 * (Linv[a2] - Linv[a1]) / 0.2
        sc = slice(s(t_cv0), s(t_cv1)); sr = slice(s(t_cv1), s(t_end))
        row.update(frozen_frac_CV=float(frozen[sc].mean()), frozen_frac_rest=float(frozen[sr].mean()), innov_mean_CV_mV=1e3 * float(inn[sc].mean()), csn_max_over_cmax_CV=float(csn[sc].max()),
                   csn_max_over_cmax_end_of_rest=float(csn[sr][-1]), guard_rejections_CV=int(rej[sc].sum()), guard_rejections_rest=int(rej[sr].sum()))
        rows.append(row)
    D = pd.DataFrame(rows); cli.write_csv(D, a, f"review_freeze_after_cv_{os.path.splitext(os.path.basename(a.freeze_npz))[0]}.csv", note=f"{os.path.basename(a.freeze_npz)}: change of the estimates (points of range / of 0.2 for the inventory) during the CV and in the rest after it")
    print(D.round(3).T.to_string() if len(D) else "no RPT covered by this trajectory")


def main():
    p = cli.parser("review checks (not in the manuscript)", "SPMe open loop on F6Q, side-reaction current in the CV, frozen-parameter ablation after the CV")
    p.add_argument("--parts", default="spme,sei"); p.add_argument("--ref-results", default="results/4to1", help="results of charge.py (stored trajectories and tables) for the comparison")
    p.add_argument("--freeze-npz", default="results/ageing/traj/ageing_track_seed0_freeze-voltage.npz", help="trajectory of ageing_track.py (any variant) for --parts freeze")
    a = p.parse_args(); runtime.setup(a.core)
    for part in [s.strip() for s in a.parts.split(",")]:
        {"spme": spme_part, "sei": sei_part, "freeze": freeze_part}[part](a)


if __name__ == "__main__":
    main()
