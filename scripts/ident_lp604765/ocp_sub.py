"""Cathode OCP of the LP604765 by subtraction: U_p(y) = OCV_cell(q) + U_n(x(q)), with the literature graphite U_n
(Mohtat2020) and the electrode window x(q) = x100 - q / Q_n, y(q) = y100 + q / Q_p of a fitted set.

    python ocp_sub.py --fit <results>/fit_v4b/best.json --tag s1 [--cycle AGE22C2]

OCV_cell(q). Until a C/20 (or GITT) test exists, a pseudo-OCV of this cell at 20 C:
  1. the rested points (last sample of every rest >= 30 min: HPPC 20 C, EIS 20 C), q from the top of charge;
  2. a full 1C cycle (default AGE22C2: from the rested top, discharge, 5 min rest, CC-CV charge): the discharge curve
     V_d(q) and the CC part of the charge V_c(q); where both exist, the overpotential is taken as half their gap,
     eta = (V_c - V_d) / 2 (it cancels the symmetric ohmic and kinetic parts);
  3. eta is corrected so that V_d + eta passes through the rested points exactly: c(q) = OCV_rest - V_d - eta at the
     rest points, interpolated linearly in q (held constant beyond the first / last point); where the charge has no CC
     part (the top of the window), eta is the rest-point value interpolated directly.
  OCV_cell = V_d + eta on the discharge's q range.
U_p is then tabulated on the y range of that q range; beyond it (top: y < y100, bottom: past the end of the 1C
discharge) the borrowed lm3c cathode table continues, offset-matched at the ends (the slight extrapolation of the knee).
The window and U_p depend on each other: iterate (subtract with the current window, refit the dynamics with
--ocp-p, subtract again). A C/20 curve, when available, replaces steps 1-3.
Output: <results>/ocp/ocp_p_<tag>.json (y, U, provenance), ocp_p_<tag>.png/.pdf.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
import common as C  # noqa: E402
from step1_ocv import ocv_points  # noqa: E402
from dfnekf.datasets import lp_cut  # noqa: E402

REST_TESTS = ["20261011_HPPC_20°C", "20261019_EIS_20°C_BoL"]


def cycle_curves(key, i_cc=1.5):
    """Discharge V_d(q) and the CC part of the charge V_c(q) of a full cycle starting at the rested top (q from it)."""
    d = lp_cut(key)
    t, V, I = d["t"], d["V"], d["I"]
    dt = np.r_[np.diff(t), 0.0]
    q = np.cumsum(I * dt) / 3600.0
    dis = I > 0.5
    chg = I < -i_cc                                    # CC charge only (the CV taper is not near equilibrium)
    order = np.argsort(q[dis]); qd, Vd = q[dis][order], V[dis][order]
    order = np.argsort(q[chg]); qc, Vc = q[chg][order], V[chg][order]
    return qd, Vd, qc, Vc


def pseudo_ocv(rest_q, rest_V, qd, Vd, qc, Vc, ng=2000):
    qg = np.linspace(qd.min(), qd.max(), ng)
    vd = np.interp(qg, qd, Vd)
    has_c = (qg >= qc.min()) & (qg <= qc.max())
    eta_half = np.where(has_c, 0.5 * (np.interp(qg, qc, Vc) - vd), np.nan)
    # rest-point anchors: eta_rest = OCV_rest - V_d at the rest q
    ok = (rest_q >= qd.min()) & (rest_q <= qd.max())
    rq, rV = rest_q[ok], rest_V[ok]
    eta_rest = rV - np.interp(rq, qg, vd)
    eh_r = np.interp(rq, qg[has_c], eta_half[has_c]) if has_c.any() else np.zeros_like(rq)
    corr = np.interp(qg, rq, eta_rest - eh_r)                   # constant beyond the first / last rest point
    eta = np.where(has_c, eta_half + corr, np.interp(qg, rq, eta_rest))
    return qg, vd + eta, dict(rest_q=rq.tolist(), rest_V=rV.tolist(), cc_charge_q=[float(qc.min()), float(qc.max())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="best.json whose window (x100, y100, Q_n, Q_p) is used")
    ap.add_argument("--cycle", default="AGE22C2")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    p = json.load(open(a.fit))["params"]
    from lpmodel import Model
    m = Model()
    lm = m.cs["ocp"]["p"]; yb, Ub = np.asarray(lm["y"]), np.asarray(lm["U"])
    xn, Un = np.asarray(m.cs["ocp"]["n"]["x"]), np.asarray(m.cs["ocp"]["n"]["U"])
    pts = np.vstack([ocv_points(t) for t in REST_TESTS])
    rq, rV = pts[:, 0], pts[:, 1]
    o = np.argsort(rq); rq, rV = rq[o], rV[o]
    # points closer than 30 mAh (an HPPC 1-h rest and an EIS 3-h rest at nearly the same charge): one point, averaged
    grp = np.r_[0, np.cumsum(np.diff(rq) > 0.03)]
    rq = np.array([rq[grp == g].mean() for g in np.unique(grp)]); rV = np.array([rV[grp == g].mean() for g in np.unique(grp)])
    qd, Vd, qc, Vc = cycle_curves(a.cycle)
    qg, ocv, info = pseudo_ocv(rq, rV, qd, Vd, qc, Vc)             # diagnostic only (figure)
    # subtraction at the rested points: delta_i = OCV_i + U_n(x_i) - U_p,lm3c(y_i); U_p = U_p,lm3c + PCHIP(delta)(y),
    # delta held constant beyond the first / last point (the slight extrapolation at the ends of the window)
    from scipy.interpolate import PchipInterpolator
    xr = p["x100"] - rq / p["Q_n"]; yr = p["y100"] + rq / p["Q_p"]
    Upr = rV + np.interp(xr, xn, Un)                            # 20 C: no entropic term (T = T_ref)
    delta = Upr - np.interp(yr, yb, Ub)
    dfun = PchipInterpolator(yr, delta, extrapolate=False)
    yo = yb.copy()
    dd = np.where(yo < yr[0], delta[0], np.where(yo > yr[-1], delta[-1], np.nan_to_num(dfun(np.clip(yo, yr[0], yr[-1])))))
    Uo = Ub + dd
    y = yr; Up = Upr; off_top, off_bot = 1e3 * delta[0], 1e3 * delta[-1]
    out_dir = os.path.join(C.RES, "ocp"); os.makedirs(out_dir, exist_ok=True)
    res = dict(y=yo.tolist(), U=Uo.tolist(), tag=a.tag, window_from=a.fit, method="rested points, PCHIP correction of lm3c",
               rest_tests=REST_TESTS, rest_y=yr.tolist(), rest_delta_mV=(1e3 * delta).tolist(),
               y_measured=[float(y[0]), float(y[-1])], ends_delta_mV=[off_top, off_bot],
               diff_vs_lm3c_mV=dict(rms=float(1e3 * np.sqrt(np.mean(delta ** 2))), max=float(1e3 * np.max(np.abs(delta)))),
               pseudo_ocv_diagnostic=info)
    json.dump(res, open(os.path.join(out_dir, "ocp_p_%s.json" % a.tag), "w"), indent=1)
    print("rested points on y %.3f-%.3f; U_p - lm3c there: %s mV (rms %.1f)"
          % (y[0], y[-1], " ".join("%.1f" % v for v in 1e3 * delta), res["diff_vs_lm3c_mV"]["rms"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 70 * mm), layout="constrained")
    ax[0].plot(qd, Vd, color="0.6", lw=0.8, label="1C discharge")
    ax[0].plot(qc, Vc, color="0.3", lw=0.8, label="1C charge (CC)")
    ax[0].plot(qg, ocv, "C0", lw=0.8, ls=":", label="1C pseudo-OCV (diagnostic)")
    ax[0].plot(rq, rV, "kD", ms=4, label="rested points")
    ax[0].set_xlabel("q from the top [A h]"); ax[0].set_ylabel("V [V]"); ax[0].legend(fontsize=7)
    ax[1].plot(yb, Ub, color="0.6", lw=0.8, label="lm3c (Mohtat cells)")
    ax[1].plot(yo, Uo, "C3", lw=1.0, label="LP604765: lm3c + PCHIP correction")
    ax[1].plot(yr, Upr, "kD", ms=4, label="subtraction at the rested points")
    ax[1].axvspan(y[0], y[-1], color="C3", alpha=0.08, lw=0)
    ax[1].set_xlim(max(0.2, y[0] - 0.1), min(1.0, y[-1] + 0.08)); ax[1].set_ylim(3.0, 4.5)
    ax[1].set_xlabel("cathode stoichiometry y"); ax[1].set_ylabel(r"$U_p$ [V]"); ax[1].legend(fontsize=7)
    for A in ax:
        A.spines[["top", "right"]].set_visible(False)
    f = os.path.join(out_dir, "ocp_p_%s" % a.tag)
    fig.savefig(f + ".png"); fig.savefig(f + ".pdf")
    print("wrote", f + ".json/.png")


if __name__ == "__main__":
    main()
