"""Selection of the anode OCP U_n(x) by differential voltage analysis (dV/dQ) of the C/20 discharge.

    python ocp_neg_select.py [--knots 6] [--out <results>/ocp_neg]

The C/20 discharge (datasets.lp_c20, fresh cell) is modelled as
    V(q) = S(q) - U_n(x100 - q / Q_n)
with S a SMOOTH cathode term: a cubic B-spline in q with --knots interior knots (spacing much wider than the graphite
staging peaks, so it cannot reproduce them). The cathode by subtraction (ocp_c20.py) cannot be used here: it absorbs
any U_n and every candidate would fit equally well. For each candidate U_n (every graphite OCP in the PyBaMM parameter
sets, plus the cell set's current table) and each window (x100, Q_n) on a grid, S is solved by linear least squares on
V and dV/dQ together (variable projection; dV/dq = S'(q) + U_n'(x) / Q_n); then the best window is refined.
Score: RMS of the dV/dQ residual on the discharge (ends trimmed), the measure of how well the staging peaks are placed.
Output: <out>/ocp_neg_select.csv, ocp_neg_<name>.json (x, U of each candidate), fig_dvdq_select.png/.pdf
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import datasets as D  # noqa: E402
from dfnekf import cellset  # noqa: E402


def c20_curve(dq=0.002, win_ah=0.04):
    """C/20 discharge on a uniform charge grid: q [A.h], V [V], dV/dQ [V/A.h] (Savitzky-Golay, window win_ah)."""
    d = D.lp_c20()
    t, V, I = d["t"], d["V"], d["I"]
    s = np.flatnonzero(I > 0.05)
    s = s[s >= d["i_dis"]]
    t, V, I = t[s], V[s], I[s]
    q = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)) / 3600])
    qg = np.arange(0.0, q[-1], dq)
    Vg = np.interp(qg, q, V)
    n = int(round(win_ah / dq)) | 1
    dV = savgol_filter(Vg, n, 3, deriv=1, delta=dq)
    Vs = savgol_filter(Vg, n, 3)
    return qg, Vs, dV, float(q[-1]), float(np.mean(I))


def candidates():
    """Every distinct graphite OCP of the PyBaMM parameter sets, as tables on x in [0, 1]: {name: (x, U)}."""
    import pybamm
    xs = np.linspace(0.0, 1.0, 2001)
    out = {}
    for name in sorted(pybamm.parameter_sets.keys()):
        try:
            pv = pybamm.ParameterValues(name)
            if "Negative electrode OCP [V]" not in pv.keys():
                continue
            sto = pybamm.Vector(xs)
            sym = pybamm.FunctionParameter("Negative electrode OCP [V]", {"Negative particle stoichiometry": sto})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U = np.asarray(pv.process_symbol(sym).evaluate()).ravel()
        except Exception as e:  # noqa: BLE001
            print("  skip %s: %s" % (name, str(e).splitlines()[0][:80]))
            continue
        if U.shape != xs.shape or not np.all(np.isfinite(U[50:-50])):
            continue
        if U[1000] > 0.6 or U[1000] < 0.03:              # not a graphite anode (e.g. silicon blends, LTO)
            continue
        if any(np.max(np.abs(U[100:-100] - u[100:-100])) < 2e-3 for _, u in out.values()):   # duplicate curve
            continue
        out[name] = (xs, U)
    return out


def fit_one(x, U, q, V, dV, knots, w, q_lo, q_hi, qn_max=None, span_min=0.0):
    """Best (x100, Q_n) and smooth S for one candidate: grid + Nelder-Mead on the variable-projection objective."""
    k = 3
    t_in = np.linspace(q[0], q[-1], knots + 2)[1:-1]
    tk = np.r_[[q[0]] * (k + 1), t_in, [q[-1]] * (k + 1)]
    nb = len(tk) - k - 1
    B = np.stack([BSpline(tk, np.eye(nb)[i], k)(q) for i in range(nb)], 1)
    dB = np.stack([BSpline(tk, np.eye(nb)[i], k).derivative()(q) for i in range(nb)], 1)
    dU = np.gradient(U, x)
    sel = (q >= q_lo) & (q <= q_hi)

    def solve(p):
        x100, Qn = p
        xq = x100 - q / Qn
        if xq.min() < x[0] + 1e-3 or xq.max() > x[-1] - 1e-3 or Qn <= 0:
            return None
        if (qn_max and Qn > qn_max) or (xq.max() - xq.min()) < span_min:      # physical window (--qn-max, --span-min)
            return None
        Un = np.interp(xq, x, U); dUn = np.interp(xq, x, dU) / Qn
        A = np.vstack([B, w * dB[sel]]); b = np.r_[V + Un, w * (dV[sel] - dUn[sel])]
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        Vm = B @ c - Un; dVm = dB @ c + dUn
        return c, Vm, dVm

    def cost(p):
        r = solve(p)
        if r is None:
            return 1e3
        _, Vm, dVm = r
        return float(np.sqrt(np.mean((dVm[sel] - dV[sel]) ** 2)) + 0.2 * np.sqrt(np.mean((Vm - V) ** 2)))

    best = (1e9, None)
    for x100 in np.linspace(0.45, 1.0, 23):
        for Qn in np.linspace(q[-1] * 1.02, (qn_max or q[-1] * 2.5), 30):
            c_ = cost((x100, Qn))
            if best is None or c_ < best[0]:
                best = (c_, (x100, Qn))
    if best[1] is None:
        return None
    res = minimize(cost, best[1], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7, maxiter=600))
    p = res.x if res.fun <= best[0] else best[1]
    r = solve(p)
    if r is None:
        return None
    c, Vm, dVm = r
    return dict(x100=float(p[0]), Q_n=float(p[1]), x0=float(p[0] - q[-1] / p[1]),
                dvdq_rms=float(np.sqrt(np.mean((dVm[sel] - dV[sel]) ** 2))), v_rms_mV=float(1e3 * np.sqrt(np.mean((Vm - V) ** 2))),
                Vm=Vm, dVm=dVm, S=B @ c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knots", type=int, default=6, help="interior knots of the smooth cathode spline")
    ap.add_argument("--qn-max", type=float, default=None, help="[A.h] largest anode capacity allowed")
    ap.add_argument("--span-min", type=float, default=0.0, help="smallest anode stoichiometry span used by the discharge")
    ap.add_argument("--tag", default="")
    ap.add_argument("--w", type=float, default=0.05, help="weight of dV/dQ [V/A.h] against V [V] in the linear solve")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp_LP604765", "ocp_neg"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q, V, dV, qmax, Imean = c20_curve()
    q_lo, q_hi = 0.05, qmax - 0.08                          # ends trimmed (relaxation at the start, knee at the end)
    print("C/20 discharge: %.3f A.h at %.3f A mean, %d points" % (qmax, Imean, len(q)))
    C = candidates()
    cs = cellset.load(os.path.join(ROOT, "results", "exp_LP604765", "fit_v12", "lp604765_cellset.json"))
    C["current (cell set: Mohtat2020 table)"] = (np.asarray(cs["ocp"]["n"]["x"]), np.asarray(cs["ocp"]["n"]["U"]))
    print("%d candidate graphite OCPs: %s" % (len(C), ", ".join(C)))
    rows, fits = [], {}
    for name, (x, U) in C.items():
        r = fit_one(x, U, q, V, dV, a.knots, a.w, q_lo, q_hi, a.qn_max, a.span_min)
        if r is None:
            print("  %-40s no admissible window" % name); continue
        fits[name] = r
        rows.append(dict(candidate=name, dvdq_rms=r["dvdq_rms"], v_rms_mV=r["v_rms_mV"], x100=r["x100"], x0=r["x0"], Q_n=r["Q_n"]))
        print("  %-40s dV/dQ rms %.4f V/Ah, V rms %.2f mV, x %.3f..%.3f, Q_n %.3f Ah" % (name, r["dvdq_rms"], r["v_rms_mV"], r["x0"], r["x100"], r["Q_n"]))
        json.dump(dict(name=name, x=list(map(float, x)), U=list(map(float, U))), open(os.path.join(a.out, "ocp_neg_%s.json" % "".join(ch if ch.isalnum() else "_" for ch in name)[:40]), "w"))
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("dvdq_rms")
    df.to_csv(os.path.join(a.out, "ocp_neg_select_k%d%s.csv" % (a.knots, a.tag)), index=False)
    print(df.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    top = list(df.candidate[:4])
    if "current (cell set: Mohtat2020 table)" not in top:
        top.append("current (cell set: Mohtat2020 table)")
    fig, ax = plt.subplots(2, 1, figsize=(170 * mm, 120 * mm), sharex=True, layout="constrained", gridspec_kw=dict(height_ratios=[2, 1]))
    sel = (q >= q_lo) & (q <= q_hi)
    ax[0].plot(q[sel], dV[sel], "k", lw=1.2, label="measured C/20")
    for i, n in enumerate(top):
        r = fits[n]
        ax[0].plot(q[sel], r["dVm"][sel], lw=0.9, color="C%d" % i, label="%s (%.3f V/Ah)" % (n, r["dvdq_rms"]))
        ax[1].plot(q[sel], 1e3 * (r["Vm"] - V)[sel], lw=0.8, color="C%d" % i)
    ax[0].set_ylabel("dV/dQ [V/Ah]"); ax[0].legend(fontsize=6.5)
    ax[0].set_ylim(np.percentile(dV[sel], 1) * 1.3, 0)
    ax[1].set_ylabel("V error [mV]"); ax[1].set_xlabel("charge discharged [Ah]"); ax[1].axhline(0, color="0.5", lw=0.5)
    ax[0].set_title("C/20 discharge, smooth cathode (%d knots): anode OCP candidates ranked by dV/dQ fit" % a.knots, fontsize=8)
    out = os.path.join(a.out, "fig_dvdq_select_k%d%s" % (a.knots, a.tag))
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
