"""Ageing replay of the board (tools/replay.py --out CSV, Rf + LLI filter) against the measured ground truth, with the
analysis of scripts/ident_lp604765/soh_lp.py (SOH from the OCV window holding the state's lithium; SOC on that window).

    python embedded/tools/board_vs_gt.py --board results/embedded_board_AGE30_RfLLI.csv --test AGE30 \
        [--python results/exp_LP604765/soh/q1e-4/ekf_AGE30_s-1.npz]

Ground truth: the measured 1C discharge capacity of every cycle (datasets.lp_discharge_capacities; SOH reference = the
block's first cycle) and Coulomb counting re-anchored at 100 % at the end of every CC-CV charge (that cycle's 1C capacity
as the full scale). The board log carries LLI, Rf and the cathode read-out SOC; ybar follows from the SOC read-out, xbar
from the lithium balance x Q_n + y Q_p = x100 Q_n + y100 Q_p (exact in the model; the filter's lithium pseudo-measurement
holds the state at the beginning-of-life lithium). Output: <board>_gt.png and a summary on stdout.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
from dfnekf import cellset, datasets as D  # noqa: E402
import soh_lp  # noqa: E402


def board_npz(path, test, cs, cell, out):
    r = list(csv.DictReader(open(path)))
    g = lambda k: np.array([float(x[k]) for x in r])
    c = cs["cells"][cell]
    Q_ref = D.LP_Q_REF
    soc_cat = g("soc")
    ybar = c["y100"] + (1 - soc_cat) * Q_ref / c["Q_p"]
    Li = c["x100"] * c["Q_n"] + c["y100"] * c["Q_p"]
    xbar = (Li - ybar * c["Q_p"]) / c["Q_n"]
    dxw = cellset.params_for(cs, cell)["dx_window_n"]
    t = g("t")
    np.savez(out, cell=cell, coords=np.array(["Rf", "LLI"]), t=t, dt=float(np.median(np.diff(t))), I=g("I"),
             vals=np.c_[g("Rf"), g("LLI")], sd=np.c_[g("sd_Rf"), g("sd_LLI")], dx_window_n=dxw, xbar=xbar, ybar=ybar,
             innov=g("V_meas") - g("V_pred"), test=test)
    return out


def summary(name, r):
    es = 100 * (r["soc"] - r["soc_ref"])
    ok = np.isfinite(es)
    qd = r["qd"]
    fade_meas = 100 * (qd[-1] / qd[0] - 1)
    soh = r["soh"]
    fade_ekf = 100 * (soh[-1] / soh[np.isfinite(soh)][0] - 1)
    # SOH at the measured cycles (nearest estimate)
    soh_at = np.interp(r["te"], r["tk"], soh)
    rel_meas = qd / qd[0]
    rel_ekf = soh_at / soh_at[0]
    print("%-26s SOC err vs Coulomb counting: rms %.2f %%, max %.2f %% | capacity fade over the block: measured %.2f %%, "
          "EKF %.2f %% | per-cycle relative capacity rms diff %.2f %% | LLI end %.0f +- %.1f mAh | Rf end %.2f | innov rms %.1f mV"
          % (name, np.sqrt(np.mean(es[ok] ** 2)), np.max(np.abs(es[ok])), fade_meas, fade_ekf,
             100 * np.sqrt(np.mean((rel_ekf - rel_meas) ** 2)), 1e3 * r["lli_Ah"][-1], 1e3 * r["lli_sd_Ah"][-1],
             r["Rf"][-1], 1e3 * np.sqrt(np.mean(r["innov"] ** 2))))
    return es


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True)
    ap.add_argument("--test", default="AGE30")
    ap.add_argument("--python", default=None, help="a Python run of scripts/ekf_exp.py (npz) to draw alongside")
    ap.add_argument("--every-s", type=float, default=300.0)
    a = ap.parse_args()
    cs = D.cellset("exp_LP604765")
    cell = D.default_cell("exp_LP604765")
    f = board_npz(a.board, a.test, cs, cell, os.path.splitext(a.board)[0] + "_analysis.npz")
    runs = [("board (10/6/10, 3+1, 5 s)", soh_lp.analyse(f, cs, a.every_s))]
    if a.python:
        runs.append(("Python (%s)" % os.path.basename(os.path.dirname(a.python)), soh_lp.analyse(a.python, cs, a.every_s)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(4, 1, figsize=(170 * mm, 170 * mm), layout="constrained", sharex=True)
    for i, (name, r) in enumerate(runs):
        es = summary(name, r)
        col = "C%d" % i
        th, tk = r["t"] / 3600, r["tk"] / 3600
        ax[0].plot(tk, 100 * r["soh"] / r["soh"][np.isfinite(r["soh"])][0], color=col, lw=1.1, label="EKF, " + name)
        ax[1].plot(th, 1e3 * r["lli_Ah"], color=col, lw=1.0)
        ax[2].plot(th, es, color=col, lw=0.4)
        ax[3].plot(th, r["Rf"], color=col, lw=0.6)
    r0 = runs[0][1]
    ax[0].plot(r0["te"] / 3600, 100 * r0["qd"] / r0["qd"][0], "k.", ms=3, label="measured 1C capacity (per cycle)")
    ax[0].set_ylabel("capacity / first cycle [%]"); ax[0].legend(fontsize=7)
    ax[1].set_ylabel("LLI [mAh]")
    ax[2].set_ylabel("SOC − Coulomb [%]"); ax[2].axhline(0, color="k", lw=0.5)
    ax[3].set_ylabel("Rf multiplier"); ax[3].set_xlabel("t [h]")
    ax[0].set_title("%s: EKF on the XIAO nRF52840 against the measurement" % a.test, fontsize=9)
    out = os.path.splitext(a.board)[0] + "_gt.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
