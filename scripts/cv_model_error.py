"""Model error of the grey-box DFN in the constant-voltage holds of the ageing run: Fig. 8f and the 0.7 mV / 44 % / 0.15 mV numbers of Sec. IV-D.

The grey-box model is run in OPEN LOOP over the whole ageing reference with the TRUE ageing state: at every step the true Rf(t), LAM(t) of the reference are
imposed (Ds = 1), the true initial state is used, and the lithium inventory is realigned to the true Li_tot(t) (so that no estimation is involved: what remains is model error).
The voltage error model - reference is averaged over each phase of each cycle block.
  z1: uniform film resistance (what the filter assumes).
  z2: film resistance distributed along the electrode with the true SEI profile of the reference (per-cell Rf; the closure accepts th = [Rf, a_s, Ds, w_1..w_Nn]).
Numbers produced (``tables/tab_cv_model_error.csv``, aggregate mode):
  * mean model overestimate in the CV of RPT 3-10 with the uniform film (text: 0.7 mV);
  * the same with the distributed film and the relative reduction (text: 44 %);
  * mean innovation (measured - predicted voltage) of the tracker in the CV of RPT 3-6 (text: 0.15 mV), read from ``<tracking>/tables/tab_aging_cv.csv``. NB: this window
    (RPT 3-6) differs from the one of the 0.7 mV (RPT 3-10); the table also gives the model error of z1 in RPT 3-6 and the tracker innovation in RPT 3-10.

  python scripts/cv_model_error.py --variants z1,z2 --grid 4:1 --data data --results results/cv_model_error      # about 0.2 h per variant, one job each
  python scripts/cv_model_error.py --aggregate --results results/cv_model_error --tracking results/ageing
  python scripts/cv_model_error.py --quick --data data --results results/quick                               # first 1000 s of the run, both variants

Outputs: cv_model_error_<variant>_phase.csv (columns as the campaign tables: variant, block, phase, n, err_mean_mV, err_rms_mV, csn_max_over_cmax, n_trunc, excess_max_mol_m3),
cv_model_error_<variant>_summary.csv, traj/cv_model_error_<variant>.npz; aggregate: source_data/fig_tracking_z.csv (Fig. 8f), tables/tab_cv_model_error.csv.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

REF_NAME = "aging_dyn_rptslow_full.npz"
PHASES = ["P6L", "slow discharge", "slow charge", "CV RPT", "CC P4", "CV P4", "C/2 discharge P4", "rest", "RPT discharge to 3.65 V"]


def phases(cyc, sti):
    """Phase name of every sample from (cycle, step-in-cycle): RPT cycles are multiples of 10 (19 steps), the others are ageing cycles (5 steps)."""
    import numpy as np
    isrpt = (cyc % 10 == 0); ph = np.empty(len(cyc), dtype=object)
    mk = isrpt
    ph[mk & (sti == 0)] = "RPT discharge to 3.65 V"; ph[mk & (sti >= 2) & (sti <= 12)] = "P6L"; ph[mk & (sti == 14)] = "slow discharge"; ph[mk & (sti == 16)] = "slow charge"
    ph[mk & (sti == 17)] = "CV RPT"; ph[mk & np.isin(sti, [1, 13, 15, 18])] = "rest"
    mk = ~isrpt
    ph[mk & (sti == 0)] = "C/2 discharge P4"; ph[mk & (sti == 2)] = "CC P4"; ph[mk & (sti == 3)] = "CV P4"; ph[mk & np.isin(sti, [1, 4])] = "rest"
    return ph, np.minimum(cyc // 10, 10), isrpt


def run_variant(a, var, P, R0, nmax=None):
    import time
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from dfnekf import ageing as ag, params as prm, ageing_tracker as at
    from dfnekf.step import DFN
    c0 = time.process_time(); t0 = time.time()
    P0 = prm.base(); m = DFN(P0, 20, 10, 20, 12, 2, 0.5, a.grid); A_ = m.A; Nn = m.Nn; Ntot = len(P["t"]); N = Ntot if nmax is None else nmax; tiny = 1e-6; cmx = P0["cmax_n"]
    cyc, sti = at.step_map(P, R0)
    I = jnp.asarray(P["I"]); Rf = jnp.asarray(P["Rf"]); asv = jnp.asarray(1 - P["LAM"]); Lt = jnp.asarray(P["Ltot"]); Vref = P["V"]
    z0 = m.init_state(float(P["cn_avg"][0]), float(P["cp_avg"][0])); th0 = jnp.array([1.0, 1.0, 1.0]); consts = m.part_consts(th0); E1, G1 = m.elec_mats(th0)
    n0 = m.lithium(z0, th0) * ag.AREA; Lt0 = Lt[0]; dxn = jnp.asarray(A_.dxn)
    csn, csp, ce = m.surf(z0); g0 = A_.newton(A_.guess(csn, csp, I[0], th0), csn, csp, ce, I[0], th0)[0]
    if var == "z2":
        pt = R0["prof_t"]; prof = R0["prof_Negative_SEI_thickness"] / 5e-9                      # (40, n_prof) in units of the nominal film resistance, index 0 = collector
        xc = np.cumsum(np.asarray(A_.dxn)) - 0.5 * np.asarray(A_.dxn); xr = xc / np.sum(np.asarray(A_.dxn)); xp = (np.arange(prof.shape[0]) + 0.5) / prof.shape[0]
        Wp = np.stack([np.interp(xr, xp, prof[:, j]) for j in range(prof.shape[1])], 0)         # (n_prof, Nn): profile mapped by relative position onto the anode cells
        tt = P["t"]; jj = np.clip(np.searchsorted(pt, tt) - 1, 0, len(pt) - 2); fr = np.clip((tt - pt[jj]) / (pt[jj + 1] - pt[jj]), 0, 1)

    def body(carry, inp):
        z, g = carry
        if var == "z1":
            Ik, rf, a_s, Ltk = inp; th = jnp.array([rf, a_s, 1.0])
        else:
            Ik, rf, a_s, Ltk, w = inp; th = jnp.concatenate([jnp.array([1.0, a_s, 1.0]), w])
        em = (E1, G1.at[:, :Nn].multiply(a_s)); nm = m.lithium(z, th[:3]) * ag.AREA; target = n0 - (Lt0 - Ltk)
        dc = (nm - target) / (ag.AREA * P0["eps_sn"] * a_s * jnp.sum(dxn)); z = z.at[:Nn].add(-dc)       # inventory realigned to the true total lithium
        csn, csp, ce = m.surf(z); g, res, _ = A_.solve_robust(g, csn, csp, ce, Ik, th, tol=1e-10); V = A_.V_of(g, Ik); z2 = m.advance(z, g, th, consts, em)
        return (z2, g), (V, jnp.max(csn), jnp.maximum(jnp.max(csn) - (cmx - tiny), 0.0), res)

    run = jax.jit(lambda carry, xs: jax.lax.scan(body, carry, xs))
    CH = 100000; Vm, cmax_m, exc, resF = [], [], [], []; carry = (z0, g0)
    for s in range(0, N, CH):
        b = min(N, s + CH)
        if var == "z1": xs = (I[s:b], Rf[s:b], asv[s:b], Lt[s:b])
        else:
            w = (1 - fr[s:b, None]) * Wp[jj[s:b]] + fr[s:b, None] * Wp[jj[s:b] + 1]; xs = (I[s:b], Rf[s:b], asv[s:b], Lt[s:b], jnp.asarray(w))
        carry, (V, cm, ex, rs) = run(carry, xs); Vm.append(np.asarray(V)); cmax_m.append(np.asarray(cm)); exc.append(np.asarray(ex)); resF.append(np.asarray(rs)); print(var, b, "of", N, flush=True)
    Vm, cmax_m, exc, resF = map(np.concatenate, (Vm, cmax_m, exc, resF)); err = 1e3 * (Vm - Vref[:N])
    ph, blk, isrpt = phases(cyc[:N], sti[:N]); rows = []
    for b_ in range(11):
        for p_ in PHASES:
            mk = (blk == b_) & (ph == p_)
            if mk.sum() > 0:
                rows.append(dict(variant=var, block=b_, phase=p_, n=int(mk.sum()), err_mean_mV=float(err[mk].mean()), err_rms_mV=float(np.sqrt((err[mk] ** 2).mean())), csn_max_over_cmax=float(cmax_m[mk].max() / cmx),
                                 n_trunc=int((exc[mk] > 0).sum()), excess_max_mol_m3=float(exc[mk].max())))
    D = pd.DataFrame(rows); tag = "cv_model_error_" + var + ("_quick" if a.quick else "")
    cli.write_csv(D, a, tag + "_phase.csv", note=f"variant {var}: mean / RMS voltage error model - reference per phase and cycle block (open loop, true ageing parameters)")
    cv = (ph == "CV RPT") & (blk >= 3); old = (blk >= 7) & ~np.isin(ph, ["CV RPT", "CV P4"])
    S = dict(variant=var, mu_CV_mV=float(err[cv].mean()) if cv.any() else float("nan"), n_CV=int(cv.sum()), mu_old_mV=float(err[old].mean()) if old.any() else float("nan"), n_old=int(old.sum()),
             rms_all_mV=float(np.sqrt((err ** 2).mean())), max_resF=float(resF.max()), cpu_s=time.process_time() - c0, wall_s=time.time() - t0)
    cli.write_csv(pd.DataFrame([S]), a, tag + "_summary.csv", note="mu_CV: mean error in the CV of RPT 3-10; mu_old: all phases of blocks 7-10 except the CVs")
    os.makedirs(os.path.join(a.results, "traj"), exist_ok=True)
    np.savez(os.path.join(a.results, "traj", tag + ".npz"), err_mV=err.astype(np.float32), csn_max=cmax_m.astype(np.float32), excess=exc.astype(np.float32), res=resF.astype(np.float32),
             cycle=cyc[:N].astype(np.int16), step=sti[:N].astype(np.int8), block=blk.astype(np.int8))
    print(S, flush=True)


def aggregate(a):
    import numpy as np, pandas as pd
    def phase(v): return pd.read_csv(os.path.join(a.results, f"cv_model_error_{v}_phase.csv"), comment="#")
    def summ(v): return pd.read_csv(os.path.join(a.results, f"cv_model_error_{v}_summary.csv"), comment="#").iloc[0]
    p1, p2 = phase("z1"), phase("z2"); c1 = p1[p1.phase == "CV RPT"].set_index("block"); c2 = p2[p2.phase == "CV RPT"].set_index("block")
    Z = pd.DataFrame(dict(block=c1.index, err_mean_mV_z1=c1.err_mean_mV.values, err_rms_mV=c1.err_rms_mV.values, n_trunc=c1.n_trunc.values, err_mean_mV_z2=c2.err_mean_mV.reindex(c1.index).values))
    sa = lambda sub: type("A", (), dict(results=os.path.join(a.results, sub), item=a.item, grid=a.grid))
    cli.write_csv(Z, sa("source_data"), "fig_tracking_z.csv", note="Fig. 8f: mean model error in the CV hold of each RPT block, uniform (z1) and distributed (z2) film", float_format="%.16g")
    s1, s2 = summ("z1"), summ("z2"); rows = []
    rows.append(dict(quantity="model_overestimate_uniform_film_mV", value=s1.mu_CV_mV, window="CV of RPT 3-10, pooled samples", paper_text="0.7 mV"))
    rows.append(dict(quantity="model_overestimate_distributed_film_mV", value=s2.mu_CV_mV, window="CV of RPT 3-10, pooled samples", paper_text=""))
    rows.append(dict(quantity="reduction_by_distributed_film_pct", value=100 * (1 - s2.mu_CV_mV / s1.mu_CV_mV), window="CV of RPT 3-10", paper_text="44 %"))
    b36 = c1.loc[3:6]; rows.append(dict(quantity="model_overestimate_uniform_film_blocks3_6_mV", value=float(np.average(b36.err_mean_mV, weights=b36.n)), window="CV of RPT 3-6, pooled", paper_text="(same window as the 0.15 mV)"))
    trk = os.path.join(a.tracking, "tables", "tab_aging_cv.csv")
    if os.path.exists(trk):
        T = pd.read_csv(trk, comment="#"); T = T[T.kind == "RPT"]
        for lab in sorted(T.label.unique()):
            for lo, hi, txt in [(3, 6, "0.15 mV"), (3, 10, "")]:
                g = T[(T.label == lab) & (T.block >= lo) & (T.block <= hi)]
                rows.append(dict(quantity=f"tracker_mean_innovation_{lab}_mV", value=float(g.innov_mean_mV.mean()), window=f"CV of RPT {lo}-{hi}, mean of the per-CV means (measured - predicted)", paper_text=txt))
                rows.append(dict(quantity=f"tracker_mean_innovation_{lab}_weighted_mV", value=float(np.average(g.innov_mean_mV, weights=g.n_steps)), window=f"CV of RPT {lo}-{hi}, sample-weighted", paper_text=""))
    else: print(f"no tracker table at {trk}: the innovation number is skipped")
    cli.write_csv(pd.DataFrame(rows), sa("tables"), "tab_cv_model_error.csv", note="numbers of Sec. IV-D on the model error in the CV")
    print(pd.DataFrame(rows).to_string())


def main():
    p = cli.parser("Fig. 8f, Sec. IV-D", "open-loop model error with the true parameters in the CV hold of each RPT")
    p.add_argument("--variants", default="z1,z2", help="z1 = uniform film, z2 = distributed film")
    p.add_argument("--aggregate", action="store_true"); p.add_argument("--tracking", default="results/ageing", help="results folder of ageing_track.py (innovation number)")
    p.add_argument("--ref", default=None, help="reference file (default <data>/aging_dyn_rptslow_full.npz)")
    a = p.parse_args()
    runtime.setup(a.core)
    if a.aggregate: return aggregate(a)
    import numpy as np
    from dfnekf import ageing
    ref = a.ref or os.path.join(a.data, REF_NAME); P = ageing.load(ref); R0 = np.load(ref)
    for v in [s.strip() for s in a.variants.split(",")]: run_variant(a, v, P, R0, nmax=2000 if a.quick else None)


if __name__ == "__main__":
    main()
