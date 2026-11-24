"""Cathode OCP of the LG M50 by subtraction: U_p = OCV + U_n, on the C/10 discharge of one cell (default D, 25 C =
T_REF) corrected to rest with that cell's GITT rests.

    python ocp_sub.py [--cell D] [--ocp-n results/exp_LGM50/ocp_neg/ocp_neg_Chen2020.json]

1. Pseudo-OCV on the discharge branch, no model: the overpotential of the C/10 discharge at each GITT rest,
   eta_g = V_C10(q_g - delta) - V_rest,g (V_rest,g the voltage at the end of the 1 h rest after the g-th 288 s pulse,
   q_g the charge discharged since the GITT's own rested top), interpolated smoothly in q and added to the C/10 voltage.
   delta: the charge between the two rested tops (the CV ends differ), from the difference of the rested top voltages
   over the OCV slope of the first GITT rests.
2. Window of the cell (x100, y100, Q_n, Q_p, offset) fitted once to the pseudo-OCV with the selected U_n and the
   Chen2020 U_p (the graphite stages fix x(q)).
3. U_p(y) = OCV(q) + U_n(x(q)) on the measured range (where the C/10 overpotential is below --eta-max: in the knee the
   interpolated correction is not reliable; from there to the last GITT rest, a monotone PCHIP through the rests);
   beyond the measured range the Chen2020 table, offset-matched at the ends.
Output: results/exp_LGM50/ocp/ocp_p_c10_<cell>.json (y, U) and a figure.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
import m50data as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="D")
    ap.add_argument("--eta-max", type=float, default=0.08, help="[V] largest C/10 overpotential kept (the knee is cut)")
    ap.add_argument("--y-knee", type=float, default=0.85, help="cathode stoichiometry of the knee: below it the residual goes to U_n")
    ap.add_argument("--ocp-n", default=os.path.join(M.RES, "ocp_neg", "ocp_neg_Chen2020.json"))
    a = ap.parse_args()
    base = json.load(open(os.path.join(M.RES, "base_chen2020.json"), encoding="utf-8"))
    yb, Ub = np.asarray(base["ocp"]["p"]["y"]), np.asarray(base["ocp"]["p"]["U"])
    tn = json.load(open(a.ocp_n, encoding="utf-8")); xn, Un = np.asarray(tn["x"]), np.asarray(tn["U"])
    # C/10 discharge from its rested top
    d = M.cut("%s_C10" % a.cell)
    s = np.flatnonzero(d["I"] > 0.3); br = np.flatnonzero(np.diff(s) > 1); s = s[: br[0] + 1] if len(br) else s
    t, V, I = d["t"][s], d["V"][s], d["I"][s]
    q = np.r_[0.0, np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)) / 3600]
    # GITT rests
    g = M.cut("%s_GITT" % a.cell)
    R = np.array(M.rests("%s_GITT" % a.cell))                          # q, V, duration
    qg, Vg = np.r_[0.0, R[:, 0]], np.r_[g["V_rest"], R[:, 1]]
    slope = (Vg[2] - Vg[0]) / (qg[2] - qg[0])                        # [V/A.h] OCV slope at the top (< 0)
    delta = (d["V_rest"] - g["V_rest"]) / slope                       # C/10 top below the GITT top [A.h]
    keep = (qg - delta >= q[0]) & (qg - delta <= q[-1]) & (qg > 0)
    eta = np.interp(qg[keep] - delta, q, V) - Vg[keep]                 # < 0 on discharge (V below OCV)
    sp = UnivariateSpline(qg[keep] - delta, eta, k=3, s=len(eta) * (1.5e-3) ** 2)
    inr = (q >= (qg[keep] - delta).min()) & (q <= (qg[keep] - delta).max()) & (np.abs(sp(q)) <= a.eta_max)   # knee: too uncertain
    qo, ocv = q[inr], V[inr] - sp(q[inr])
    # beyond the knee cut: the GITT rests themselves (true rested points down to the last one), joined monotonically
    tail = (qg - delta) > qo[-1] + 0.02
    if tail.any():
        from scipy.interpolate import PchipInterpolator
        qk = np.r_[qo[-50::10], qg[tail] - delta]; vk = np.r_[ocv[-50::10], Vg[tail]]
        qt = np.arange(qo[-1] + 0.002, qk[-1], 0.002)
        qo, ocv = np.r_[qo, qt], np.r_[ocv, PchipInterpolator(qk, vk)(qt)]
    print("cell %s: C/10 %.3f Ah, %d GITT rests, top offset delta %.1f mAh (slope %.0f mV/Ah); eta %.1f..%.1f mV"
          % (a.cell, q[-1], keep.sum(), 1e3 * delta, 1e3 * slope, 1e3 * eta.min(), 1e3 * eta.max()))
    # window of the cell with U_n and the Chen2020 U_p (q from the C/10 top)
    qs = np.linspace(qo[0], qo[-1], 1500); os_ = np.interp(qs, qo, ocv)

    def model_ocv(w, qv):
        x100, y100, Qn, Qp, dq0 = w
        x = x100 - (qv + dq0) / Qn; y = y100 + (qv + dq0) / Qp
        return np.interp(y, yb, Ub) - np.interp(np.clip(x, xn[0], xn[-1]), xn, Un), x, y
    lo = np.array([0.6, 0.15, 4.5, 4.5, -0.3]); hi = np.array([1.0, 0.45, 8.0, 10.0, 0.3])
    w0 = np.array([0.9014, 0.27, 5.83, 8.73, 0.0])
    rng = np.random.default_rng(0); best = None
    for k in range(80):
        st = w0 if k == 0 else lo + rng.random(5) * (hi - lo)
        r_ = least_squares(lambda w: (model_ocv(w, qs)[0] - os_) * 1e3, st, bounds=(lo, hi), x_scale=hi - lo)
        if best is None or r_.cost < best.cost:
            best = r_
    w = best.x; r = model_ocv(w, qs)[0] - os_
    print("window (Chen2020 U_p + U_n vs pseudo-OCV): x100 %.4f y100 %.4f Q_n %.3f Q_p %.3f dq0 %+.4f; rms %.1f mV, max %.1f mV"
          % (*w, 1e3 * np.sqrt(np.mean(r ** 2)), 1e3 * np.max(np.abs(r))))
    _, x, y = model_ocv(w, qs)
    Up = os_ + np.interp(np.clip(x, xn[0], xn[-1]), xn, Un)
    o = np.argsort(y); y, Up, x, ocv_s = y[o], Up[o], x[o], os_[o]
    # Below the knee (y > --y-knee) the subtraction gives a non-monotone U_p: there U_n is steep (graphite end, Si),
    # so the residual of the literature U_n lands on U_p. There U_p keeps the Chen2020 shape (offset-matched at the
    # knee, continuous) and the anode takes the rest: U_n(x) = U_p(y) - OCV(q) on x below the knee, down to the
    # last GITT rest; beyond it the literature U_n shape, offset-matched. Above the knee nothing changes.
    from scipy.optimize import isotonic_regression
    kk = y > a.y_knee
    if kk.any():
        i0 = np.flatnonzero(~kk)[-1]
        Up[kk] = np.interp(y[kk], yb, Ub) + (Up[i0] - np.interp(y[i0], yb, Ub))
        xk, Unk = x[kk][::-1], (Up[kk] - ocv_s[kk])[::-1]                 # x ascending
        Unk = isotonic_regression(Unk, increasing=False).x
        Un_lit = np.interp(xn, np.clip(xn, 0, 1), Un)
        lo_, hi_ = xn < xk[0], xn > x[i0]
        off_lo = Unk[0] - np.interp(xk[0], xn, Un)
        xx = np.r_[xn[lo_], xk, x[i0], xn[hi_]]; UUn = np.r_[Un[lo_] + off_lo, Unk, np.interp(x[i0], xn, Un), Un[hi_]]
        o2 = np.argsort(xx); Un_new = isotonic_regression(np.interp(xn, xx[o2], UUn[o2]), increasing=False).x
        d_n = Un_new - Un_lit
        on = dict(name="Chen2020_knee_%s" % a.cell, x=xn.tolist(), U=Un_new.tolist(), base=os.path.abspath(a.ocp_n),
                  method="literature U_n (DVA choice) above the knee; below it (x < %.4f) U_n = U_p - OCV with U_p on the "
                         "Chen2020 shape, from the C/10 + GITT pseudo-OCV of cell %s; Chen2020 shape offset-matched beyond "
                         "the last GITT rest (x < %.4f, offset %.0f mV)" % (x[i0], a.cell, xk[0], 1e3 * off_lo),
                  x_knee=float(x[i0]), y_knee=float(a.y_knee), x_measured_min=float(xk[0]),
                  max_change_mV=float(1e3 * np.max(np.abs(d_n))))
        fn = os.path.join(M.RES, "ocp_neg", "ocp_neg_knee_%s.json" % a.cell)
        json.dump(on, open(fn, "w"), indent=1)
        print("anode below the knee (x %.4f..%.4f): max change vs literature %.0f mV, offset beyond %.0f mV -> %s"
              % (xk[0], x[i0], on["max_change_mV"], 1e3 * off_lo, fn))
        for xv in (0.0, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15):
            print("   x %.2f  U_n lit %.3f  new %.3f" % (xv, np.interp(xv, xn, Un), np.interp(xv, xn, Un_new)))
    Up_raw = Up.copy()
    Up = isotonic_regression(Up, increasing=False).x
    smin = 0.2                                                          # [V per unit y] minimum |dU_p/dy|
    for i in range(1, len(Up)):
        Up[i] = min(Up[i], Up[i - 1] - smin * (y[i] - y[i - 1]))
    print("monotone projection of U_p: max change %.1f mV, rms %.1f mV" % (1e3 * np.max(np.abs(Up - Up_raw)),
                                                                          1e3 * np.sqrt(np.mean((Up - Up_raw) ** 2))))
    off_top = Up[0] - np.interp(y[0], yb, Ub); off_bot = Up[-1] - np.interp(y[-1], yb, Ub)
    top, bot = yb < y[0], yb > y[-1]
    yy = np.r_[yb[top], y, yb[bot]]; UU = np.r_[Ub[top] + off_top, Up, Ub[bot] + off_bot]
    Uo = np.interp(yb, yy, UU)
    dlt = Up - np.interp(y, yb, Ub)
    out = os.path.join(M.RES, "ocp"); os.makedirs(out, exist_ok=True)
    tag = "c10_%s" % a.cell
    res = dict(y=yb.tolist(), U=Uo.tolist(), tag=tag, cell=a.cell, ocp_n=os.path.abspath(a.ocp_n),
               method="C/10 discharge corrected to rest at the GITT rests (data, no model); window fitted once with Chen2020 U_p "
                      "+ the selected U_n; U_p = OCV + U_n; Chen2020 beyond the measured range",
               delta_mAh=1e3 * delta, eta_mV=dict(min=1e3 * eta.min(), max=1e3 * eta.max()),
               window_recorded=dict(zip(["x100", "y100", "Q_n", "Q_p", "dq0"], map(float, w))),
               window_fit_rms_mV=float(1e3 * np.sqrt(np.mean(r ** 2))), y_measured=[float(y[0]), float(y[-1])],
               ends_delta_mV=[1e3 * off_top, 1e3 * off_bot],
               diff_vs_chen_mV=dict(rms=float(1e3 * np.sqrt(np.mean(dlt ** 2))), max=float(1e3 * np.max(np.abs(dlt)))))
    json.dump(res, open(os.path.join(out, "ocp_p_%s.json" % tag), "w"), indent=1)
    print("U_p on y %.3f-%.3f; vs Chen2020 rms %.1f mV, max %.1f mV; end offsets %.1f / %.1f mV"
          % (y[0], y[-1], res["diff_vs_chen_mV"]["rms"], res["diff_vs_chen_mV"]["max"], 1e3 * off_top, 1e3 * off_bot))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 3, figsize=(170 * mm, 62 * mm), layout="constrained")
    ax[0].plot(q, V, color="0.6", lw=0.8, label="C/10 discharge")
    ax[0].plot(qo, ocv, "C0", lw=1.0, label="pseudo-OCV")
    ax[0].plot(qg - delta, Vg, "k.", ms=3, label="GITT rests (1 h)")
    ax[0].plot(qs, model_ocv(w, qs)[0], "C3", lw=0.8, ls="--", label="Chen2020, fitted window")
    ax[0].set_xlabel("q [A h]"); ax[0].set_ylabel("V [V]"); ax[0].legend(fontsize=6)
    ax[1].plot(qg[keep] - delta, -1e3 * eta, "k.", ms=3); ax[1].plot(qo, -1e3 * sp(qo), "C0", lw=1.0)
    ax[1].set_xlabel("q [A h]"); ax[1].set_ylabel(r"C/10 overpotential [mV]")
    ax[2].plot(yb, Ub, color="0.6", lw=0.8, label="Chen2020")
    ax[2].plot(yb, Uo, "C3", lw=1.0, label="cell %s, subtraction" % a.cell)
    ax[2].axvspan(y[0], y[-1], color="C3", alpha=0.08, lw=0)
    ax[2].set_xlim(max(0.0, y[0] - 0.08), min(1.0, y[-1] + 0.06)); ax[2].set_ylim(3.3, 4.4)
    ax[2].set_xlabel("y"); ax[2].set_ylabel(r"$U_p$ [V]"); ax[2].legend(fontsize=6.5)
    for A in ax:
        A.spines[["top", "right"]].set_visible(False)
    f = os.path.join(out, "ocp_p_%s" % tag)
    fig.savefig(f + ".png", dpi=200); fig.savefig(f + ".pdf")
    print("wrote", f + ".json/.png")


if __name__ == "__main__":
    main()
