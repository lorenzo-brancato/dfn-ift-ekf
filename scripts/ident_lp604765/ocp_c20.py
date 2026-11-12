"""Cathode OCP of the LP604765 by subtraction on the C/20 discharge of a fresh cell of the same type (BLE board,
2026-11-09/10, about 26 C; 2.133 Ah, the nominal capacity).

    python ocp_c20.py --fit <results>/fit_v7s1/best.json --ocp-model <results>/ocp/ocp_p_s1.json --tag c20

1. Pseudo-OCV: the C/20 voltage plus the model's own overpotential along the discharge (the fitted set simulated on
   the recording, eta = OCV at the electrode mean stoichiometries - V), referred to 20 C with the entropic terms of the set.
2. Window of the recorded cell (another cell than ID01: its window differs, the electrode curves do
   not): x100, y100, Q_n, Q_p and a start offset fitted once to the pseudo-OCV with the literature graphite U_n and the
   lm3c cathode U_p (the graphite stages fix x(q)).
3. U_p(y) = OCV(q) + U_n(x(q)) on the measured range; beyond it (above the rested top, below the 3.0 V cut-off) the
   lm3c table continues, offset-matched at the ends. No window/OCP alternation: on a dense curve the window is set by
   the shape, not by U_p.
Output: <results>/ocp/ocp_p_<tag>.json (same format as ocp_sub.py) and a figure.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
import common as C  # noqa: E402
from dfnekf.datasets import lp_c20  # noqa: E402
T_REF_C = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="fitted set used for the overpotential correction (its OCP: --ocp-model)")
    ap.add_argument("--ocp-model", default=None, help="the cathode table that set was fitted with")
    ap.add_argument("--tag", default="c20")
    ap.add_argument("--no-eta", action="store_true", help="no overpotential correction (raw C/20 voltage)")
    a = ap.parse_args()
    if a.ocp_model:
        os.environ["LP_OCP_P"] = os.path.abspath(a.ocp_model)
    from lpmodel import Model, prepare
    best = json.load(open(a.fit)); p = C.model_params(best["params"])
    d = lp_c20()
    i0 = d["i_dis"]
    T_C = float(np.mean(d["T"][i0:]))
    V_rest = float(d["V"][i0 - 1])
    # 1. the model's overpotential along the discharge (1-s grid of the discharge; start by OCV inversion at the rest)
    rec = dict(t=d["t"][i0:], V=d["V"][i0:], I=d["I"][i0:], step=np.zeros(len(d["t"]) - i0, int), names=["C20"], test="C20")
    m = Model()
    pr = prepare(rec, dts=(1.0, 1.0, 1.0))
    q0 = m.dq_from_ocv(p, T_C, V_rest)
    Vm, res, xs, ys = m.simulate(p, pr, T_C, dq=q0, full=True)
    U_mod = np.asarray(m.ocp_p(ys)) - np.asarray(m.ocp_n(xs)) + (T_C - T_REF_C) * (p["dUdT0"] + p["dUdT_n"] * (xs - 0.5))
    eta = np.where(np.isfinite(Vm), U_mod - Vm, np.nan)              # > 0 on discharge
    q = np.cumsum(pr["I"] * pr["dt"]) / 3600.0                      # A.h discharged
    ok = np.isfinite(eta) & (pr["V"] > 3.0)
    eta_c = np.interp(q, q[ok], eta[ok]) if not a.no_eta else np.zeros_like(q)
    # referred to 20 C (entropic term of the set at the model's anode stoichiometry)
    ocv = pr["V"] + eta_c - (T_C - T_REF_C) * (p["dUdT0"] + p["dUdT_n"] * (np.interp(q, q[ok], xs[ok]) - 0.5))
    qq = np.r_[0.0, q]; oo = np.r_[V_rest - (T_C - T_REF_C) * (p["dUdT0"] + p["dUdT_n"] * (xs[0] - 0.5)), ocv]
    # 2. window of the recorded cell (lm3c U_p, literature U_n): q measured from the rested start, offset dq0 free
    lm = m.cs["ocp"]["p"]; yb, Ub = np.asarray(lm["y"]), np.asarray(lm["U"])
    xn, Un = np.asarray(m.cs["ocp"]["n"]["x"]), np.asarray(m.cs["ocp"]["n"]["U"])
    qs = np.linspace(0.0, qq[-1], 1500); os_ = np.interp(qs, qq, oo)

    def model_ocv(w, qv):
        x100, y100, Qn, Qp, dq0 = w
        x = x100 - (qv + dq0) / Qn; y = y100 + (qv + dq0) / Qp
        return np.interp(y, yb, Ub) - np.interp(np.clip(x, xn[0], xn[-1]), xn, Un), x, y
    w0 = np.array([p["x100"], p["y100"], p["Q_n"], p["Q_p"], 0.0])
    lo = np.array([0.4, 0.3, 2.0, 2.0, -0.3]); hi = np.array([1.0, 0.6, 7.0, 6.0, 0.3])
    best_s = None
    rng = np.random.default_rng(0)
    for k in range(60):
        st = w0 if k == 0 else lo + rng.random(5) * (hi - lo)
        s = least_squares(lambda w: (model_ocv(w, qs)[0] - os_) * 1e3, st, bounds=(lo, hi), x_scale=hi - lo)
        if best_s is None or s.cost < best_s.cost:
            best_s = s
    w = best_s.x
    r = model_ocv(w, qs)[0] - os_
    print("window of the recorded cell: x100 %.4f y100 %.4f Q_n %.3f Q_p %.3f dq0 %+.4f Ah; lm3c+graphite vs pseudo-OCV: rms %.1f mV, max %.1f mV"
          % (*w, 1e3 * np.sqrt(np.mean(r ** 2)), 1e3 * np.max(np.abs(r))))
    # 3. subtraction on the measured range
    _, x, y = model_ocv(w, qs)
    Up = os_ + np.interp(np.clip(x, xn[0], xn[-1]), xn, Un)
    o = np.argsort(y); y, Up = y[o], Up[o]
    off_top = Up[0] - np.interp(y[0], yb, Ub); off_bot = Up[-1] - np.interp(y[-1], yb, Ub)
    top, bot = yb < y[0], yb > y[-1]
    yy = np.r_[yb[top], y, yb[bot]]; UU = np.r_[Ub[top] + off_top, Up, Ub[bot] + off_bot]
    Uo = np.interp(yb, yy, UU)
    dlt = Up - np.interp(y, yb, Ub)
    out = os.path.join(C.RES, "ocp"); os.makedirs(out, exist_ok=True)
    res_ = dict(y=yb.tolist(), U=Uo.tolist(), tag=a.tag, method="C/20 discharge (BLE board), model overpotential correction, "
                "window fitted once with lm3c U_p + Mohtat2020 graphite, U_p = OCV + U_n", source=d["file"],
                T_C=T_C, window_recorded=dict(zip(["x100", "y100", "Q_n", "Q_p", "dq0"], map(float, w))),
                window_fit_rms_mV=float(1e3 * np.sqrt(np.mean(r ** 2))), y_measured=[float(y[0]), float(y[-1])],
                ends_delta_mV=[1e3 * off_top, 1e3 * off_bot], eta_mV=dict(start=float(1e3 * eta_c[0]), median=float(1e3 * np.median(eta_c)),
                                                                          at_3V3=float(1e3 * np.interp(3.3, pr["V"][::-1], eta_c[::-1]))),
                diff_vs_lm3c_mV=dict(rms=float(1e3 * np.sqrt(np.mean(dlt ** 2))), max=float(1e3 * np.max(np.abs(dlt)))),
                capacity_Ah=float(q[-1]), fit=a.fit)
    json.dump(res_, open(os.path.join(out, "ocp_p_%s.json" % a.tag), "w"), indent=1)
    print("eta: start %.1f mV, median %.1f mV, at 3.3 V %.1f mV; U_p on y %.3f-%.3f, vs lm3c rms %.1f mV max %.1f mV; end offsets %.1f / %.1f mV"
          % (res_["eta_mV"]["start"], res_["eta_mV"]["median"], res_["eta_mV"]["at_3V3"], y[0], y[-1],
             res_["diff_vs_lm3c_mV"]["rms"], res_["diff_vs_lm3c_mV"]["max"], *res_["ends_delta_mV"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 3, figsize=(170 * mm, 62 * mm), layout="constrained")
    ax[0].plot(q, pr["V"], color="0.6", lw=0.8, label="C/20 discharge")
    ax[0].plot(qq, oo, "C0", lw=1.0, label="pseudo-OCV (20 °C)")
    ax[0].plot(qs, model_ocv(w, qs)[0], "C3", lw=0.8, ls="--", label="lm3c + graphite, fitted window")
    ax[0].set_xlabel("q [A h]"); ax[0].set_ylabel("V [V]"); ax[0].legend(fontsize=6.5)
    ax[1].plot(q, 1e3 * eta_c, "C0", lw=1.0); ax[1].set_xlabel("q [A h]"); ax[1].set_ylabel(r"model overpotential $\eta$ [mV]")
    ax[2].plot(yb, Ub, color="0.6", lw=0.8, label="lm3c")
    ax[2].plot(yb, Uo, "C3", lw=1.0, label="LP604765, C/20 subtraction")
    ax[2].axvspan(y[0], y[-1], color="C3", alpha=0.08, lw=0)
    ax[2].set_xlim(max(0.2, y[0] - 0.08), min(1.0, y[-1] + 0.06)); ax[2].set_ylim(3.0, 4.5)
    ax[2].set_xlabel("y"); ax[2].set_ylabel(r"$U_p$ [V]"); ax[2].legend(fontsize=6.5)
    for A in ax:
        A.spines[["top", "right"]].set_visible(False)
    f = os.path.join(out, "ocp_p_%s" % a.tag)
    fig.savefig(f + ".png"); fig.savefig(f + ".pdf")
    print("wrote", f + ".json/.png")


if __name__ == "__main__":
    main()
