"""OCV of a fitted set against the C/20 discharge of the fresh sister cell (and the 1-h rests of ID01's HPPCs).

    LP_GROUPS=1 LP_THERMAL=1 LP_ANODE=Ai2020 python ocv_check_c20.py --fit <results>/fit_g7/best.json --tag g7

The C/20 cell is not ID01: its electrode window (x100, y100, Q_n, Q_p) can differ, its electrode curves cannot. So:
1. the set as is (ID01's window): the C/20 current simulated from the rested top (start by OCV inversion), and the
   set's OCV along the same charge;
2. the set's electrode curves with the window refitted to the C/20 cell: pseudo-OCV = V + the set's overpotential
   along the discharge (eta = OCV at the mean stoichiometries - V), fitted with x100, y100, Q_n, Q_p and a start
   offset. A residual that stays after this fit is in the electrode curves (OCPs), not in the window;
3. differential voltage dV/dq of the recording and of both model curves (graphite stage peaks);
4. ID01's HPPC rests (end of 1 h) against the set's OCV at their own temperature.
Output: <results>/ocv_check/ocv_c20_<tag>.{png,pdf,json}.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

import common as C
from dfnekf.datasets import lp_c20


def dvdq(q, V, w=0.02):
    """dV/dq on a uniform charge grid, Savitzky-Golay over w A.h."""
    g = np.arange(q[0], q[-1], 0.001)
    v = np.interp(g, q, V)
    n = max(5, int(w / 0.001) | 1)
    return g, savgol_filter(v, n, 2, deriv=1, delta=0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--tag", default="g7")
    a = ap.parse_args()
    from lpmodel import Model, prepare
    best = json.load(open(a.fit)); p = C.model_params(best["params"])
    m = Model()
    d = lp_c20()
    i0 = d["i_dis"]
    T_C = float(np.mean(d["T"][i0:]))
    V_rest = float(d["V"][i0 - 1])
    rec = dict(t=d["t"][i0:], V=d["V"][i0:], I=d["I"][i0:], step=np.zeros(len(d["t"]) - i0, int), names=["C20"], test="C20")
    pr = prepare(rec, dts=(1.0, 5.0, 20.0))
    q0 = m.dq_from_ocv(p, T_C, V_rest)
    Vm, res, xs, ys = (np.asarray(v) for v in m.simulate(p, pr, T_C, dq=q0, full=True))
    q = np.cumsum(pr["I"] * pr["dt"]) / 3600.0
    Vd = np.asarray(pr["V"])
    U_set = np.asarray(m.ocv(p, T_C, q0 + q))                          # the set's OCV, ID01 window, same charge
    # pseudo-OCV of the recording: V + the set's overpotential (OCV at the mean stoichiometries - V of the model)
    U_mean = np.asarray(m.ocp_p(ys)) - np.asarray(m.ocp_n(xs))
    if m.dudt is not None:
        U_mean = U_mean + (T_C - 20.0) * np.asarray(m.dudt(xs, T_C - 20.0))
    eta = U_mean - Vm
    ok = np.isfinite(eta) & np.isfinite(Vm) & (Vd > 3.0)
    pseudo = Vd + np.interp(q, q[ok], eta[ok])

    # window of the C/20 cell with the set's electrode curves
    def ocv_w(w, qv):
        x100, y100, Qn, Qp, dq0 = w
        x = x100 - (qv + dq0) / Qn; y = y100 + (qv + dq0) / Qp
        U = np.asarray(m.ocp_p(y)) - np.asarray(m.ocp_n(x))
        if m.dudt is not None:
            U = U + (T_C - 20.0) * np.asarray(m.dudt(x, T_C - 20.0))
        return U
    sel = ok & (Vd > 3.05)
    w0 = np.array([p["x100"], p["y100"], p["Q_n"], p["Q_p"], q0])
    fit = least_squares(lambda w: ocv_w(w, q[sel]) - pseudo[sel], w0, x_scale=np.array([0.01, 0.01, 0.05, 0.05, 0.01]))
    U_w = ocv_w(fit.x, q)
    r_set = 1e3 * (Vd - Vm)[ok]; r_w = 1e3 * (pseudo - U_w)[sel]
    out = dict(fit=a.fit, T_C=T_C, q0=float(q0), cap_meas_3V=float(q[np.argmax(Vd < 3.0)] if np.any(Vd < 3.0) else q[-1]),
               cap_model_3V=float(q[np.argmax(Vm < 3.0)] if np.any(Vm < 3.0) else np.nan),
               rms_sim_mV=float(np.sqrt(np.mean(r_set ** 2))), rms_refit_window_mV=float(np.sqrt(np.mean(r_w ** 2))),
               max_refit_window_mV=float(np.max(np.abs(r_w))),
               window_ID01=dict(zip(["x100", "y100", "Q_n", "Q_p", "dq0"], map(float, w0))),
               window_C20=dict(zip(["x100", "y100", "Q_n", "Q_p", "dq0"], map(float, fit.x))))
    # ID01 rests
    from ocv_temperature import rests, TESTS
    dq = {"HPPC20": 0.0, "HPPC40": best["params"]["dq_40"], "HPPC00": best["params"]["dq_00"]}   # model_params drops the dq_
    rest = {}
    for k, T in TESTS.items():
        rr = rests(C.ALL[k]["test"])
        rest[k] = [(r["q"] + dq[k], 1e3 * (r["V"] - float(m.ocv(p, T, r["q"] + dq[k])))) for r in rr]
    out["rests_rms_mV"] = {k: float(np.sqrt(np.mean([e ** 2 for _, e in v]))) for k, v in rest.items()}
    od = os.path.join(C.RES, "ocv_check"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "ocv_c20_%s.json" % a.tag), "w"), indent=1)
    print(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(C.HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(2, 2, figsize=(180 * mm, 125 * mm), layout="constrained")
    ax[0, 0].plot(q, Vd, "k", lw=1.0, label="C/20 measured (sister cell)")
    ax[0, 0].plot(q, Vm, "C1--", lw=0.9, label="%s simulated, ID01 window" % a.tag)
    ax[0, 0].plot(q, U_w, "C0:", lw=0.9, label="%s OCPs, window refitted" % a.tag)
    ax[0, 0].set_xlabel("q discharged [Ah]"); ax[0, 0].set_ylabel("V [V]"); ax[0, 0].set_ylim(2.9, 4.25)
    ax[0, 0].legend(frameon=False, fontsize=6.5); ax[0, 0].set_title("(a) C/20 discharge, %.1f °C" % T_C, fontsize=7)
    ax[0, 1].plot(q[ok], 1e3 * (Vm - Vd)[ok], "C1", lw=0.8, label="simulated - measured (ID01 window): rms %.1f mV" % out["rms_sim_mV"])
    ax[0, 1].plot(q[sel], 1e3 * (U_w - pseudo)[sel], "C0", lw=0.8, label="OCV - pseudo-OCV (window refitted): rms %.1f mV" % out["rms_refit_window_mV"])
    ax[0, 1].axhline(0, color="0.5", lw=0.5); ax[0, 1].set_ylim(-60, 60)
    ax[0, 1].set_xlabel("q [Ah]"); ax[0, 1].set_ylabel("error [mV]"); ax[0, 1].legend(frameon=False, fontsize=6.5)
    ax[0, 1].set_title("(b) error along the discharge", fontsize=7)
    for (qq, VV, st, lab) in ((q[ok], Vd[ok], "k", "measured"), (q[ok], Vm[ok], "C1--", "simulated, ID01 window"),
                              (q[sel], U_w[sel], "C0:", "OCV, window refitted")):
        g, dv = dvdq(qq, VV)
        ax[1, 0].plot(g, -dv, st, lw=0.9, label=lab)
    ax[1, 0].set_ylim(0, 1.5); ax[1, 0].set_xlabel("q [Ah]"); ax[1, 0].set_ylabel("-dV/dq [V/Ah]")
    ax[1, 0].legend(frameon=False, fontsize=6.5); ax[1, 0].set_title("(c) differential voltage", fontsize=7)
    col = {"HPPC00": "C0", "HPPC20": "k", "HPPC40": "C3"}
    for k, v in rest.items():
        ax[1, 1].plot([a_ for a_, _ in v], [b_ for _, b_ in v], "o-", ms=3, lw=0.6, color=col[k],
                      label="%s: rms %.1f mV" % (k, out["rests_rms_mV"][k]))
    ax[1, 1].axhline(0, color="0.5", lw=0.5); ax[1, 1].set_xlabel("q from the top of HPPC20 [Ah]")
    ax[1, 1].set_ylabel("rest - OCV [mV]"); ax[1, 1].legend(frameon=False, fontsize=6.5)
    ax[1, 1].set_title("(d) ID01 rests (end of 1 h) against the %s OCV" % a.tag, fontsize=7)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(od, "ocv_c20_%s.%s" % (a.tag, e)), dpi=200)

    # incremental capacity dQ/dV against V: (a) the recorded voltage against the simulated one (both with the C/20
    # overpotential), (b) the pseudo-OCV of the recording against the set's OCV with the window refitted (equilibrium)
    def ica(qq, VV, w=0.03):
        g = np.arange(qq[0], qq[-1], 0.001)
        n = max(5, int(w / 0.001) | 1)
        v = savgol_filter(np.interp(g, qq, VV), n, 2)
        dv = savgol_filter(np.interp(g, qq, VV), n, 2, deriv=1, delta=0.001)
        k = dv < -1e-3
        return v[k], -1.0 / dv[k]
    fig2, bx = plt.subplots(1, 2, figsize=(180 * mm, 70 * mm), layout="constrained")
    for (qq, VV, st, lab) in ((q[ok], Vd[ok], "k", "C/20 measured"), (q[ok], Vm[ok], "C1--", "%s simulated, ID01 window" % a.tag)):
        v_, i_ = ica(qq, VV); bx[0].plot(v_, i_, st, lw=0.9, label=lab)
    for (qq, VV, st, lab) in ((q[sel], pseudo[sel], "k", "pseudo-OCV of the recording"), (q[sel], U_w[sel], "C0--", "%s OCV, window refitted" % a.tag)):
        v_, i_ = ica(qq, VV); bx[1].plot(v_, i_, st, lw=0.9, label=lab)
    for b_, t_ in zip(bx, ("(a) C/20 voltage", "(b) equilibrium (overpotential removed)")):
        b_.set_xlabel("V [V]"); b_.set_ylabel("dQ/dV [Ah/V]"); b_.set_xlim(3.3, 4.2); b_.set_ylim(0, 8)
        b_.legend(frameon=False, fontsize=6.5, loc="upper right"); b_.set_title(t_, fontsize=7)
    for e in ("png", "pdf"):
        fig2.savefig(os.path.join(od, "ica_c20_%s.%s" % (a.tag, e)), dpi=200)


if __name__ == "__main__":
    main()
