"""Anode OCP selection for the LG M50 by differential voltage analysis of the C/10 discharge (fit cells A, D, F).

    python dva_select.py [--cells A,D,F] [--knots 6]

Same method as scripts/ident_lp604765/ocp_neg_select.py (its fit_one and candidates are reused): V(q) = S(q) - U_n(x100 - q/Q_n),
S a smooth cubic B-spline cathode term, each literature graphite U_n (PyBaMM sets) scored by the RMS of the dV/dQ residual
with its best window. Chen2020 (= OKane2022) was measured on this cell type, so it is the expected answer: the test of the
method. Output: results/exp_LGM50/ocp_neg/dva_<cell>.csv, ocp_neg_<name>.json, fig_dva_<cell>.png/.pdf, dva_summary.csv
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
sys.path.insert(0, ROOT)
from ocp_neg_select import candidates, fit_one  # noqa: E402
import m50data as M  # noqa: E402


def c10_curve(cell, dq=0.002, win_ah=0.04):
    """C/10 discharge of `cell` on a uniform charge grid: q [A.h], V, dV/dQ (Savitzky-Golay)."""
    d = M.cut("%s_C10" % cell)
    t, V, I = d["t"], d["V"], d["I"]
    s = np.flatnonzero(I > 0.3)
    s = s[: np.flatnonzero(np.diff(s) > 1)[0] + 1] if np.any(np.diff(s) > 1) else s    # first discharge only
    t, V, I = t[s], V[s], I[s]
    q = np.r_[0.0, np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)) / 3600]
    qg = np.arange(0.0, q[-1], dq)
    Vg = np.interp(qg, q, V)
    n = int(round(win_ah / dq)) | 1
    return qg, savgol_filter(Vg, n, 3), savgol_filter(Vg, n, 3, deriv=1, delta=dq), float(q[-1]), float(np.mean(I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(M.FIT_CELLS))
    ap.add_argument("--knots", type=int, default=6)
    ap.add_argument("--w", type=float, default=0.05)
    ap.add_argument("--qn-max", type=float, default=None, help="[A.h] largest anode capacity (physical window)")
    ap.add_argument("--span-min", type=float, default=0.0, help="smallest anode stoichiometry span of the discharge")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.join(M.RES, "ocp_neg"); os.makedirs(out, exist_ok=True)
    Cd = candidates()
    print("%d candidate graphite OCPs: %s" % (len(Cd), ", ".join(Cd)))
    for name, (x, U) in Cd.items():
        json.dump(dict(name=name, x=list(map(float, x)), U=list(map(float, U))),
                  open(os.path.join(out, "ocp_neg_%s.json" % "".join(ch if ch.isalnum() else "_" for ch in name)[:40]), "w"))
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    allrows = []
    for cell in a.cells.split(","):
        q, V, dV, qmax, Im = c10_curve(cell)
        q_lo, q_hi = 0.1, qmax - 0.2
        print("cell %s (%g C): C/10 discharge %.3f Ah at %.3f A" % (cell, M.CELL_T[cell], qmax, Im))
        rows, fits = [], {}
        for name, (x, U) in Cd.items():
            r = fit_one(x, U, q, V, dV, a.knots, a.w, q_lo, q_hi, a.qn_max, a.span_min)
            if r is None:
                print("  %-22s no admissible window" % name); continue
            fits[name] = r
            rows.append(dict(cell=cell, candidate=name, dvdq_rms=r["dvdq_rms"], v_rms_mV=r["v_rms_mV"], x100=r["x100"], x0=r["x0"], Q_n=r["Q_n"]))
        df = pd.DataFrame(rows).sort_values("dvdq_rms")
        df["rank"] = np.arange(1, len(df) + 1)
        df.to_csv(os.path.join(out, "dva_%s%s.csv" % (cell, a.tag)), index=False)
        print(df.round(4).to_string(index=False))
        allrows.append(df)
        top = list(df.candidate[:4]) + (["Chen2020"] if "Chen2020" not in list(df.candidate[:4]) and "Chen2020" in fits else [])
        fig, ax = plt.subplots(2, 1, figsize=(170 * mm, 120 * mm), sharex=True, layout="constrained", gridspec_kw=dict(height_ratios=[2, 1]))
        sel = (q >= q_lo) & (q <= q_hi)
        ax[0].plot(q[sel], dV[sel], "k", lw=1.2, label="measured C/10, cell %s" % cell)
        for i, n in enumerate(top):
            r = fits[n]
            ax[0].plot(q[sel], r["dVm"][sel], lw=0.9, color="C%d" % i, label="%s (%.3f V/Ah)" % (n, r["dvdq_rms"]))
            ax[1].plot(q[sel], 1e3 * (r["Vm"] - V)[sel], lw=0.8, color="C%d" % i)
        ax[0].set_ylabel("dV/dQ [V/Ah]"); ax[0].legend(fontsize=6.5)
        ax[0].set_ylim(np.percentile(dV[sel], 1) * 1.3, 0)
        ax[1].set_ylabel("V error [mV]"); ax[1].set_xlabel("charge discharged [Ah]"); ax[1].axhline(0, color="0.5", lw=0.5)
        f = os.path.join(out, "fig_dva_%s%s" % (cell, a.tag))
        fig.savefig(f + ".png", dpi=200); fig.savefig(f + ".pdf"); plt.close(fig)
    S = pd.concat(allrows)
    summ = S.groupby("candidate").agg(mean_rank=("rank", "mean"), dvdq_rms=("dvdq_rms", "mean"), v_rms_mV=("v_rms_mV", "mean")).sort_values("mean_rank")
    summ.to_csv(os.path.join(out, "dva_summary%s.csv" % a.tag))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
