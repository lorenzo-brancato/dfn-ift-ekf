"""Why the fitted model over-predicts the voltage in the CV phase: the Expt 5 cycle 0 discharge (1C) followed by the
cycle 1 CC-CV charge (0.3C to 4.2 V, CV to 50 mA), driven by the measured current and the measured surface temperature.

    python diag_cv.py [--fit results/exp_LGM50/fit_m50_v4/best.json] [--cells 5D,5E,5F,5A]

Per cell: the error in the CC part above 4.0 V and in the CV, against
  - the window OCV at the coulomb-counted charge, OCV(q): measured and model overpotential V - OCV(q)
    (both runs see the same q, so a gap here is overpotential, not window);
  - variants with one loss switched off (x100 faster diffusion / kinetics, fast electrolyte, no contact resistance):
    the drop of the model CV voltage under each variant is the share of that loss.
Output: results/exp_LGM50/diag_cv/{summary.json, cv_<cell>.png}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
os.environ.setdefault("LP_CELL", "lgm50")
sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
import m50data as M  # noqa: E402


def cycle_profile(cell):
    """Cycle 0 discharge + cycle 1 charge of cell '5X' as one test (step 0 discharge, 1 charge), V_rest at the CV end."""
    r0, r1 = M.read_cycle(cell, 0), M.read_cycle(cell, 1)
    i0 = int(np.flatnonzero(r0["ns"] == 1)[0])
    k1 = r1["ns"] == 0
    t0 = r0["t"][i0:] - r0["t"][i0]
    t1 = r1["t"][k1] - r1["t"][k1][0] + t0[-1] + 1.0
    cat = lambda a, b: np.r_[a, b]
    d = dict(t=cat(t0, t1), V=cat(r0["V"][i0:], r1["V"][k1]), I=cat(r0["I"][i0:], r1["I"][k1]),
             T_meas=M.clean_T(cat(r0["T_meas"][i0:], r1["T_meas"][k1])),
             step=cat(np.zeros(len(t0), int), np.ones(int(k1.sum()), int)), names=[os.path.basename(r0["file"])],
             test=cell + "_CYCCV", V_rest=float(r0["V"][i0 - 1]), file=r0["file"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(M.RES, "fit_m50_v4", "best.json"))
    ap.add_argument("--cells", default="5D,5E,5F,5G,5A")
    a = ap.parse_args()
    import common as C
    from lpmodel import Model, prepare
    m = Model()
    p0 = json.load(open(a.fit))["params"]
    out = os.path.join(M.RES, "diag_cv"); os.makedirs(out, exist_ok=True)
    variants = {
        "fitted": {},
        "Ds_p x100, no drop": dict(Ds_p=lambda p: p["Ds_p"] * 100, drop=lambda p: 0.0),
        "Ds_p drop off": dict(drop=lambda p: 0.0),
        "Ds_n x100": dict(Ds_n=lambda p: p["Ds_n"] * 100),
        "k0_n x100": dict(k0_n=lambda p: p["k0_n"] * 100),
        "k0_p x100": dict(k0_p=lambda p: p["k0_p"] * 100),
        "electrolyte fast (tau/20)": dict(tau=lambda p: p["tau"] / 20),
        "R_contact 0": dict(R_contact=lambda p: 1e-6),
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    summ = {}
    for c in a.cells.split(","):
        d = cycle_profile(c)
        pr = prepare(d); pr["V_rest"] = d["V_rest"]
        T = M.AGE_T[c[-1]]
        pk = C.params_for(p0, dict(win=c))
        dq0 = m.dq_from_ocv(C.model_params(pk), T, d["V_rest"])
        t = np.cumsum(pr["dt"]); I = pr["I"]; Vm = pr["V"]
        q = dq0 + np.cumsum(I * pr["dt"]) / 3600.0
        ocv_q = m.ocv(C.model_params(pk), 25.0, q)
        ch = t > t[np.flatnonzero(I > 1)[-1]]                           # charge part
        cv = ch & (Vm > 4.195) & (I < -0.02)
        cc = ch & (Vm > 4.0) & (Vm < 4.19) & (I < -1.0)
        res = {}
        Vs = {}
        for name, mod in variants.items():
            pv = dict(pk, **{k: f(pk) for k, f in mod.items()})
            V, F, xs, ys = m.simulate(C.model_params(pv), pr, T, dq=dq0, full=True)
            Vs[name] = V
            e = 1e3 * (V - Vm)
            res[name] = dict(cc_err_mV=float(np.nanmean(e[cc])), cv_err_mV=float(np.nanmean(e[cv])),
                             cv_err_end_mV=float(np.nanmean(e[cv][-20:])), cv_err_start_mV=float(np.nanmean(e[cv][:20])),
                             bad=int(np.sum(~np.isfinite(V)) + np.sum(F > 1e-6)))
            if name == "fitted":
                xs0, ys0 = xs, ys
        eta_meas = 1e3 * (Vm - ocv_q); eta_mod = 1e3 * (Vs["fitted"] - ocv_q)
        # surface OCV of the model (mean surface stoichiometries)
        Usurf = np.asarray(m.ocp_p(ys0)) - np.asarray(m.ocp_n(xs0))
        summ[c] = dict(T_C=T, dq0=float(dq0), q_end_charge=float(q[-1]), cv_hours=float(pr["dt"][cv].sum() / 3600),
                       I_cv_start=float(-I[cv][0]), I_cv_end=float(-I[cv][-1]),
                       ocv_q_at_cv_end=float(ocv_q[cv][-1]), V_meas_cv_end=float(Vm[cv][-1]),
                       eta_meas_cv_mean_mV=float(np.mean(eta_meas[cv])), eta_model_cv_mean_mV=float(np.mean(eta_mod[cv])),
                       eta_meas_cc_mV=float(np.mean(eta_meas[cc])), eta_model_cc_mV=float(np.mean(eta_mod[cc])),
                       model_surface_minus_bulk_cv_mV=float(1e3 * np.mean(Usurf[cv] - ocv_q[cv])),
                       variants=res)
        print(c, json.dumps(summ[c], indent=1), flush=True)
        th = t / 3600; s = ch
        fig, ax = plt.subplots(3, 1, figsize=(170 * mm, 130 * mm), layout="constrained", sharex=True)
        ax[0].plot(th[s], Vm[s], "k", lw=1, label="measured")
        for name, V in Vs.items():
            ax[0].plot(th[s], V[s], lw=0.8, ls="-" if name == "fitted" else "--", label=name)
        ax[0].plot(th[s], ocv_q[s], color="0.5", lw=0.8, ls=":", label="OCV(q)")
        ax[0].set_ylim(3.9, 4.3); ax[0].set_ylabel("V [V]"); ax[0].legend(frameon=False, fontsize=5, ncol=2)
        ax[0].set_title("%s (%d °C): cycle 0 discharge + cycle 1 CC-CV charge, fit %s" % (c, T, os.path.basename(os.path.dirname(a.fit))), fontsize=7)
        ax[1].plot(th[s], eta_meas[s], "k", lw=0.8, label="measured V - OCV(q)")
        ax[1].plot(th[s], eta_mod[s], "C3", lw=0.8, label="model V - OCV(q)")
        ax[1].set_ylabel("overpotential [mV]"); ax[1].legend(frameon=False, fontsize=6); ax[1].set_ylim(-50, 250)
        ax[2].plot(th[s], -I[s], "0.3", lw=0.8); ax[2].set_ylabel("charge current [A]"); ax[2].set_xlabel("t [h]")
        fig.savefig(os.path.join(out, "cv_%s.png" % c), dpi=200); plt.close(fig)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
