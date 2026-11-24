"""Cathode OCP by subtraction on ID01's own electrode window (instead of the window of the C/20 sister cell).

    LP_GROUPS=1 LP_THERMAL=1 LP_ANODE=Ai2020 python ocp_c20_id01.py --fit <results>/fit_g7/best.json --tag id01_an_Ai2020

Why: U_p = OCV + U_n carries the image of any error of the literature U_n at the window where the subtraction is done.
Done at the sister cell's window (ocp_c20.py) and used at ID01's window, the anode features and their image no longer
cancel: spurious peaks in dQ/dV (3.48 V with Ai2020, 3.82 V with Mohtat2020). Here:
1. pseudo-OCV of the C/20 recording = V + the set's overpotential along the discharge (as ocv_check_c20.py);
2. ID01's OCV is taken as that curve, stretched and shifted in charge, q_C20 = d + s q_ID01, with s and d fitted to
   ID01's 18 HPPC rests (each referred to the recording's temperature with the measured dU/dT at ID01's x);
3. U_p(y(q)) = OCV_ID01(q, 20 C) + U_n(x(q)) with ID01's window (step 1 of the set: x100, y100, Q_n, Q_p); outside the
   measured range the previous table continues, offset-matched at the ends.
So the model with ID01's window reproduces the (stretched) C/20 curve exactly and the image of U_n cancels.
Output: <results>/ocp/ocp_p_c20_<tag>.json (+ png).
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

import common as C
from dfnekf.datasets import lp_c20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--tag", default="id01_an_Ai2020")
    ap.add_argument("--q-bend", type=float, default=1.7, help="ID01 charge beyond which the second stretch acts [A.h]")
    a = ap.parse_args()
    from lpmodel import Model, prepare
    from ocv_temperature import rests, TESTS
    best = json.load(open(a.fit)); P = best["params"]; p = C.model_params(P)
    m = Model()
    d = lp_c20()
    i0 = d["i_dis"]; T_C = float(np.mean(d["T"][i0:])); V_rest = float(d["V"][i0 - 1])
    rec = dict(t=d["t"][i0:], V=d["V"][i0:], I=d["I"][i0:], step=np.zeros(len(d["t"]) - i0, int), names=["C20"], test="C20")
    pr = prepare(rec, dts=(1.0, 5.0, 20.0))
    q0 = m.dq_from_ocv(p, T_C, V_rest)
    Vm, res, xs, ys = (np.asarray(v) for v in m.simulate(p, pr, T_C, dq=q0, full=True))
    qc = np.r_[0.0, np.cumsum(pr["I"] * pr["dt"]) / 3600.0]
    U_mean = np.asarray(m.ocp_p(ys)) - np.asarray(m.ocp_n(xs)) + (T_C - 20.0) * np.asarray(m.dudt(xs, T_C - 20.0))
    eta = U_mean - Vm
    ok = np.isfinite(eta) & np.isfinite(Vm) & (np.asarray(pr["V"]) > 3.0)
    pseudo = np.r_[V_rest, np.asarray(pr["V"]) + np.interp(qc[1:], qc[1:][ok], eta[ok])]
    qmax = qc[1:][ok][-1]                                             # last sample above 3.0 V with a finite overpotential
    keep = np.r_[True, ok] & (qc <= qmax)
    qc, pseudo = qc[keep], pseudo[keep]

    x100, y100, Qn, Qp = (P[k] for k in ("x100", "y100", "Q_n", "Q_p"))
    xq = lambda q: x100 - q / Qn
    yq = lambda q: y100 + q / Qp
    dq = {"HPPC20": 0.0, "HPPC40": P["dq_40"], "HPPC00": P["dq_00"]}
    R = []
    for k, T in TESTS.items():
        for r in rests(C.ALL[k]["test"]):
            qi = r["q"] + dq[k]
            R.append((qi, r["V"] + (T_C - T) * float(m.dudt(xq(qi), T_C - T)), k))
    qr = np.array([r[0] for r in R]); vr = np.array([r[1] for r in R])

    # rests in the steep end (below 3.45 V) carry the window ends, not the shape: left out of the fit, reported
    use = vr > 3.45

    def resid(w):
        s_, d_, c_ = w
        return np.interp(d_ + s_ * qr, qc, pseudo, left=np.nan, right=np.nan) + c_ - vr
    # c: a constant voltage offset between the C/20 pseudo-OCV of the sister cell and ID01's rests (the C/20
    # polarisation the model does not carry, cell-to-cell OCV spread); q_C20 = d + s q_ID01, d ~ -q0 to start
    f = least_squares(lambda w: np.nan_to_num(resid(w)[use], nan=0.0), [1.0, -q0, 0.0])
    s, dsh, cof = f.x
    rr = 1e3 * resid(f.x)
    print("stretch s %.4f, shift d %+.4f Ah, offset c %+.1f mV; rests rms %.2f mV (fitted), max %.2f mV"
          % (s, dsh, 1e3 * cof, np.sqrt(np.nanmean(rr[use] ** 2)), np.nanmax(np.abs(rr[use]))))
    for k in TESTS:
        e = rr[[r[2] == k for r in R]]
        print("  %s: rms %.2f mV  %s" % (k, np.sqrt(np.nanmean(e ** 2)), np.round(e, 1)))

    # ID01's OCV at 20 C on its own charge axis, then U_p = OCV + U_n
    # bottom of the window: the sister cell's end does not map onto ID01's with one stretch. A second stretch s2 beyond
    # q_b (smoothed kink, width 0.04 A.h) is fitted to ID01's deepest rests (V < 3.45 V); the shape stays the C/20's
    q_b = a.q_bend
    bend = lambda q: 0.04 * np.logaddexp(0.0, (q - q_b) / 0.04)
    qc_of = lambda q, s2: dsh + s * q + (s2 - s) * bend(q)
    deep = ~use
    if deep.any():                           # charge where the recording reaches each deep rest's voltage, then s2 in closed form
        q_need = np.interp(vr[deep] - cof, pseudo[::-1], qc[::-1])
        s2 = float(np.mean(s + (q_need - (dsh + s * qr[deep])) / bend(qr[deep])))
    else:
        s2 = s
    rd = 1e3 * (np.interp(qc_of(qr, s2), qc, pseudo, right=np.nan) + cof - vr)
    print("bottom: stretch s2 %.4f beyond q %.2f A.h; all rests rms %.2f mV; deepest %s mV" % (s2, q_b, np.sqrt(np.nanmean(rd ** 2)), np.round(rd[deep], 1)))
    # ID01's charge grid -> the recording's charge, OCV at 20 C, U_p = OCV + U_n
    qg = np.arange(-0.05, x100 * Qn - 1e-3, 0.0005)
    qcg = qc_of(qg, s2)
    inside = (qcg >= qc[0]) & (qcg <= qc[-1])
    qg = qg[inside]; qcg = qcg[inside]
    o20 = np.interp(qcg, qc, pseudo) + cof - (T_C - 20.0) * np.asarray(m.dudt(xq(qg), T_C - 20.0))
    y_m = yq(qg); U_m = o20 + np.asarray(m.ocp_n(xq(qg)))
    old = json.load(open(os.environ["LP_OCP_P"]))
    yo, Uo = np.asarray(old["y"]), np.asarray(old["U"])
    lo, hi = y_m[0], y_m[-1]
    off_lo = U_m[0] - np.interp(lo, yo, Uo); off_hi = U_m[-1] - np.interp(hi, yo, Uo)
    yb = yo[yo < lo]; ya = yo[yo > hi]
    y_all = np.r_[yb, y_m, ya]; U_all = np.r_[np.interp(yb, yo, Uo) + off_lo, U_m, np.interp(ya, yo, Uo) + off_hi]
    # resample on the table grid of the old file (uniform), keeping the measured part dense enough
    yg = np.linspace(1e-5, 0.99999, 4001)
    Ug = np.interp(yg, y_all, U_all)
    out = dict(y=list(yg), U=list(Ug), tag=a.tag, T_C=20.0, source=d["file"], fit=a.fit,
               method="C/20 pseudo-OCV (overpotential of the set removed), stretched/shifted in charge to ID01's rests "
                      "(q_C20 = d + s q_ID01), U_p = OCV + U_n at ID01's window (step 1); old table beyond, offset-matched; beyond q_bend a second stretch s2 fitted to ID01's deepest rests",
               window_ID01=dict(x100=x100, y100=y100, Q_n=Qn, Q_p=Qp), stretch=float(s), shift_Ah=float(dsh), offset_mV=1e3 * float(cof),
               rests_rms_mV=float(np.sqrt(np.nanmean(rr[use] ** 2))), y_measured=[float(lo), float(hi)],
               ends_delta_mV=[1e3 * float(off_lo), 1e3 * float(off_hi)], q_bend=q_b, stretch_bottom=s2)
    fn = os.path.join(C.RES, "ocp", "ocp_p_c20_%s.json" % a.tag)
    json.dump(out, open(fn, "w"), indent=1)
    print("y measured %.4f-%.4f, end offsets %.1f / %.1f mV -> %s" % (lo, hi, 1e3 * off_lo, 1e3 * off_hi, fn))


if __name__ == "__main__":
    main()
