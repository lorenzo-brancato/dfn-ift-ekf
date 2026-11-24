"""OCV step of the LG M50: the electrode window (x100, y100, Q_n, Q_p, and the ends x0, y0) of every cell.

    python ocv_window.py [--ocp-p results/exp_LGM50/ocp/ocp_p_c10_D.json]

U_p: the cathode table by subtraction (ocp_sub.py); U_n: the selected anode table (Chen2020). Model (lpmodel.Model.ocv):
    V(q) = U_p(y100 + q / Q_p) - U_n(x100 - q / Q_n)       (entropic terms 0: every point is at 21-26 C)
Points, per cell (each cell has its own window; the fit is per cell, 4 parameters):
  Expt 4 cells A-H: the end of every 1 h rest of the GITT above --vfit-min (2.5 V: all 25 points, discharge branch; the anode is corrected below the knee), q from the GITT's rested top
    (the window's top, dq = 0). The top itself (after the CC-CV, charge branch) is left out.
  Expt 5 cells 5A-5H: no GITT before the ageing; the RPT0 C/10 discharge corrected to rest with the C/10 overpotential
    measured on the Expt 4 fit cells (eta(q) = V_C10 - V_rest at their GITT rests, mean of A, D, F; same cell type,
    same current), kept where |eta| <= --eta-max. q from the C/10 start (the window's top).
Window ends at the cut-offs: x0, y0 at --vmin (2.5 V), top at 4.2 V.
Output: results/exp_LGM50/windows.json (read by common.py with LP_CELL=lgm50), fig_windows.png
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
os.environ.setdefault("LP_CELL", "lgm50")
sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
import m50data as M  # noqa: E402

WN = ["x100", "y100", "Q_n", "Q_p"]
WLO, WHI = np.array([0.6, 0.15, 4.5, 4.5]), np.array([1.0, 0.45, 8.0, 10.0])


def c10_discharge(cell):
    d = M.cut("%s_C10" % cell)
    s = np.flatnonzero(d["I"] > 0.3); br = np.flatnonzero(np.diff(s) > 1); s = s[: br[0] + 1] if len(br) else s
    t, V, I = d["t"][s], d["V"][s], d["I"][s]
    return np.r_[0.0, np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)) / 3600], V, d["V_rest"]


def eta_curve(cell):
    """C/10 overpotential of an Expt 4 cell at its GITT rests, against q from the C/10 start: (q, eta [V], < 0)."""
    q, V, Vr = c10_discharge(cell)
    g = M.cut("%s_GITT" % cell); R = np.array(M.rests("%s_GITT" % cell))
    qg, Vg = np.r_[0.0, R[:, 0]], np.r_[g["V_rest"], R[:, 1]]
    delta = (Vr - g["V_rest"]) / ((Vg[2] - Vg[0]) / (qg[2] - qg[0]))
    k = (qg - delta >= q[0]) & (qg - delta <= q[-1]) & (qg > 0)
    return qg[k] - delta, np.interp(qg[k] - delta, q, V) - Vg[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocp-p", default=os.path.join(M.RES, "ocp", "ocp_p_c10_D.json"))
    ap.add_argument("--cells", default="A,B,C,D,E,F,G,H,5A,5B,5C,5D,5E,5F,5G,5H")
    ap.add_argument("--out", default="windows.json")
    ap.add_argument("--vmin", type=float, default=2.5)
    ap.add_argument("--eta-max", type=float, default=0.08)
    # below the knee the split of the OCV between the electrodes is not known (U_n steep: graphite end, Si; the
    # monotone U_p cannot follow the last rests); the Expt 5 points stop there too (eta cut, about 3.2 V)
    ap.add_argument("--vfit-min", type=float, default=2.5)
    ap.add_argument("--ocp-n", default=os.path.join(M.RES, "ocp_neg", "ocp_neg_knee_D.json"),
                    help="anode table (literature, corrected below the knee by ocp_sub.py)")
    a = ap.parse_args()
    os.environ["LP_OCP_P"] = os.path.abspath(a.ocp_p)
    os.environ["LP_OCP_N"] = os.path.abspath(a.ocp_n)
    import common as C  # noqa: F401  (sets the LG M50 environment of lpmodel)
    from lpmodel import Model
    m = Model()
    # mean C/10 overpotential of the Expt 4 fit cells
    qe = np.linspace(0.2, 4.7, 400); E = []
    for c in ("A", "D", "F"):
        qc, ec = eta_curve(c)
        sp = UnivariateSpline(qc, ec, k=3, s=len(ec) * (1.5e-3) ** 2)
        E.append(np.where((qe >= qc.min()) & (qe <= qc.max()), sp(qe), np.nan))
    eta_mean = np.nanmean(E, 0); eta_sd = np.nanstd(E, 0)
    okE = np.isfinite(eta_mean)
    print("C/10 overpotential of A, D, F: spread between the cells (sd) median %.1f mV, max %.1f mV (where |eta| <= %.0f mV)"
          % (1e3 * np.nanmedian(eta_sd[okE & (np.abs(eta_mean) <= a.eta_max)]), 1e3 * np.nanmax(eta_sd[okE & (np.abs(eta_mean) <= a.eta_max)]), 1e3 * a.eta_max))
    out = dict(ocp_p=os.path.abspath(a.ocp_p), ocp_n=os.path.abspath(a.ocp_n), dUdT0=0.0, dUdT_n=0.0, vmin=a.vmin, cells={})
    sp0 = json.load(open(a.ocp_p))["window_recorded"]
    w0 = np.array([sp0["x100"], sp0["y100"], sp0["Q_n"], sp0["Q_p"]])
    P = lambda w: dict(zip(WN, w), dUdT0=0.0, dUdT_n=0.0)
    curves = {}
    for c in a.cells.split(","):
        if c.startswith("5"):
            q, V, _ = c10_discharge(c)
            k = (q >= qe[okE][0]) & (q <= qe[okE][-1])
            eq = np.interp(q[k], qe[okE], eta_mean[okE])
            k2 = np.abs(eq) <= a.eta_max
            qp, Vp = q[k][k2][::20], (V[k] - eq)[k2][::20]
            src = "RPT0 C/10 discharge + mean C/10 overpotential of A, D, F"
        else:
            R = np.array(M.rests("%s_GITT" % c)); R = R[R[:, 1] >= a.vfit_min]; qp, Vp = R[:, 0], R[:, 1]
            src = "GITT rests (1 h) above %.2f V" % a.vfit_min
        res = lambda w: (m.ocv(P(w), 25.0, qp) - Vp) * 1e3
        best = None
        for st in [w0] + [WLO + np.random.default_rng(i).random(4) * (WHI - WLO) for i in range(15)]:
            s = least_squares(res, st, bounds=(WLO, WHI), x_scale=WHI - WLO)
            if best is None or s.cost < best.cost:
                best = s
        w = best.x; r = res(w); p = P(w)
        qq = np.linspace(-0.3, 6.5, 6801); v = m.ocv(p, 25.0, qq)
        inw = (v <= 4.2) & (v >= a.vmin)
        q_top, q_bot = float(qq[inw].min()), float(qq[inw].max())
        role = "fit" if c in M.FIT_CELLS else ("held out" if c in M.VAL_CELLS else "spare")
        o = dict({k_: float(p[k_]) for k_ in WN}, x0=p["x100"] - q_bot / p["Q_n"], y0=p["y100"] + q_bot / p["Q_p"],
                 x_top=p["x100"] - q_top / p["Q_n"], y_top=p["y100"] + q_top / p["Q_p"], Q_window_Ah=q_bot - q_top, q_top=q_top,
                 rms_mV=float(np.sqrt(np.mean(r ** 2))), max_mV=float(np.max(np.abs(r))), n_points=int(len(r)), points=src,
                 experiment=5 if c.startswith("5") else 4, age_T_C=M.AGE_T[c[-1]], role=role,
                 at_bound=[k_ for k_, v_, l_, h_ in zip(WN, w, WLO, WHI) if v_ <= l_ + 1e-6 or v_ >= h_ - 1e-6])
        out["cells"][c] = o; curves[c] = (qp, Vp, m.ocv(p, 25.0, qp))
        print("  %-3s (Expt %d, %-8s) x %.4f..%.4f  y %.4f..%.4f  Q_n %.3f  Q_p %.3f | window %.3f Ah | rms %.1f max %.1f mV (%d pts) %s"
              % (c, o["experiment"], role, o["x0"], o["x_top"], o["y_top"], o["y0"], o["Q_n"], o["Q_p"], o["Q_window_Ah"],
                 o["rms_mV"], o["max_mV"], o["n_points"], o["at_bound"]))
    json.dump(out, open(os.path.join(M.RES, a.out), "w"), indent=1)
    print("wrote", os.path.join(M.RES, a.out))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 65 * mm), layout="constrained")
    for i, (c, (qp, Vp, Vm)) in enumerate(curves.items()):
        col = "C0" if not c.startswith("5") else "C1"
        ax[0].plot(qp, Vp, ".", ms=1.5, color=col); ax[0].plot(qp, Vm, "-", lw=0.4, color="k")
        ax[1].plot(qp, 1e3 * (Vm - Vp), "-" if c.startswith("5") else ".-", lw=0.6, ms=2, color=col, alpha=0.7)
    ax[0].set_xlabel("q [A h]"); ax[0].set_ylabel("OCV [V]"); ax[0].set_title("points (blue Expt 4 GITT, orange Expt 5 C/10) and model", fontsize=7)
    ax[1].set_xlabel("q [A h]"); ax[1].set_ylabel("model - data [mV]"); ax[1].axhline(0, color="0.5", lw=0.5)
    f = os.path.join(M.RES, "fig_windows")
    fig.savefig(f + ".png", dpi=200); fig.savefig(f + ".pdf")


if __name__ == "__main__":
    main()
