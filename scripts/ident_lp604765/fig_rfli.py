"""EKF runs with R_f and LLI (scripts/ekf_exp.py --coords Rf,LLI): SOC error, R_f multiplier and LLI against time, from the
placed start and from SOC 50 %, one column per test.

    python fig_rfli.py --runs <results>/rfli_v12 --tests CYC20,AGE22V,AGE22,AGE27,AGE30 --q-window 2.220

SOC error against fig_soh_nom.soc_reference (100 % at the end of every CV, full scale --q-window). A run that diverged
is drawn up to its last finite step and marked with an x. Output: <runs>/fig_rfli.png/.pdf
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, HERE)
from fig_soh_nom import soc_reference  # noqa: E402

plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tests", default="CYC20,AGE22V,AGE22,AGE27,AGE30")
    ap.add_argument("--q-window", type=float, default=2.220)
    ap.add_argument("--q-window-run", action="store_true", help="SOC reference full scale = the run's own OCV window at each anchor")
    ap.add_argument("--vb", action="store_true", help="a fourth row: the voltage bias Vb [mV]")
    a = ap.parse_args()
    tests = a.tests.split(",")
    nr = 4 if a.vb else 3
    fig, ax = plt.subplots(nr, len(tests), figsize=(180 * mm, 36 * nr * mm), sharex="col", sharey="row", layout="constrained", squeeze=False)
    for j, test in enumerate(tests):
        for f in sorted(glob.glob(os.path.join(a.runs, "ekf_%s_e*.npz" % test))):
            d = np.load(f, allow_pickle=True)
            placed = "e+0.00" in f
            col, lab = ("C0", "placed start") if placed else ("C3", "start at SOC 50 %")
            th = d["t"] / 3600
            ok = np.isfinite(d["soc"])
            last = np.flatnonzero(ok)[-1]
            ref, anc, _ = soc_reference(d["t"], d["I"], q_window=d["Q_window_Ah"] if (a.q_window_run and "Q_window_Ah" in d) else a.q_window)
            co = list(map(str, d["coords"]))
            rf = d["vals"][:, co.index("Rf")]; lli = 1e3 * d["lli_Ah"]
            ax[0, j].plot(th[ok], 100 * (d["soc"] - ref)[ok], col, lw=0.6, label=lab)
            ax[1, j].plot(th[ok], rf[ok], col, lw=0.7)
            ax[2, j].plot(th[ok], lli[ok], col, lw=0.7)
            if a.vb and "Vb" in co:
                ax[3, j].plot(th[ok], 1e3 * d["vals"][ok, co.index("Vb")], col, lw=0.5)
            if last < len(th) - 1:                        # diverged
                for r, v in ((0, 100 * (d["soc"] - ref)[last]), (1, rf[last]), (2, lli[last])):
                    ax[r, j].plot(th[last], v, "x", color=col, ms=6, mew=1.5)
                ax[0, j].text(th[last], 9, " diverged", color=col, fontsize=6, va="top")
        ax[0, j].set_title(test, fontsize=8)
        ax[0, j].axhline(0, color="0.5", lw=0.5); ax[1, j].axhline(1, color="0.5", lw=0.5); ax[2, j].axhline(0, color="0.5", lw=0.5)
        ax[-1, j].set_xlabel("t [h]")
        if a.vb:
            ax[3, j].axhline(0, color="0.5", lw=0.5)
    ax[0, 0].set_ylabel("SOC error [%]"); ax[1, 0].set_ylabel(r"$R_f$ multiplier"); ax[2, 0].set_ylabel("LLI [mAh]")
    ax[0, 0].set_ylim(-10, 10); ax[1, 0].set_yscale("log")
    if a.vb:
        ax[3, 0].set_ylabel("voltage bias [mV]")
    ax[0, 0].legend(fontsize=6, loc="lower right")
    out = os.path.join(a.runs, "fig_rfli" + ("_qwrun" if a.q_window_run else ""))
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
