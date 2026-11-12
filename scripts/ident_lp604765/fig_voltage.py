"""Voltage of SOC-only EKF runs against the measurement: model-plant mismatch.

    python fig_voltage.py --runs <dir>[,<dir>...] --labels a[,b...] --tests CYC20,AGE22V [--open <dir>] [--out <png>]

Per test (columns): measured V, the open-loop model from the placed start (open_<test>_e+0.00.npy in --open: the pure
model-plant mismatch, no correction) and the EKF's predicted voltage of each run (before its update); below, the errors
model - measured [mV], with the instants of the EKF's voltage updates; bottom, the current.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--tests", default="CYC20,AGE22V")
    ap.add_argument("--tag", default="e+0.00")
    ap.add_argument("--open", default=None, help="folder with open_<test>_e+0.00.npy (open loop)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = a.runs.split(","); labels = a.labels.split(",") if a.labels else [os.path.basename(r) for r in runs]
    tests = a.tests.split(",")
    fig, ax = plt.subplots(3, len(tests), figsize=(180 * mm, 130 * mm), sharex="col", sharey="row", layout="constrained",
                           squeeze=False, gridspec_kw=dict(height_ratios=[2, 2, 1]))
    for j, test in enumerate(tests):
        d0 = np.load(os.path.join(runs[0], "ekf_%s_%s.npz" % (test, a.tag)), allow_pickle=True)
        th, y = d0["t"] / 3600, d0["y"]
        ax[0, j].plot(th, y, "k", lw=1.0, label="measured")
        if a.open:
            f = os.path.join(a.open, "open_%s_e+0.00.npy" % test)
            if os.path.exists(f):
                Vo = np.load(f)
                r = 1e3 * np.sqrt(np.nanmean((Vo - y) ** 2))
                ax[0, j].plot(th, Vo, color="0.55", lw=0.8, ls="--", label="open loop")
                ax[1, j].plot(th, 1e3 * (Vo - y), color="0.55", lw=0.7, label="open loop (RMS %.0f mV)" % r)
        for i, (r_, lab) in enumerate(zip(runs, labels)):
            d = np.load(os.path.join(r_, "ekf_%s_%s.npz" % (test, a.tag)), allow_pickle=True)
            e = 1e3 * (d["V"] - d["y"])
            ax[0, j].plot(d["t"] / 3600, d["V"], "C%d" % i, lw=0.7, label="EKF " + lab)
            ax[1, j].plot(d["t"] / 3600, e, "C%d" % i, lw=0.6, label="EKF %s (RMS %.0f mV)" % (lab, np.sqrt(np.nanmean(e ** 2))))
            if i == 0:
                upd = np.isfinite(d["innov"])
                ax[1, j].plot(d["t"][upd] / 3600, np.full(upd.sum(), ax[1, j].get_ylim()[0] if False else -95), "|", color="C3", ms=3)
        ax[1, j].axhline(0, color="0.5", lw=0.5)
        ax[2, j].plot(th, d0["I"], "0.3", lw=0.7)
        ax[0, j].set_title(test, fontsize=8); ax[2, j].set_xlabel("t [h]")
        ax[1, j].legend(fontsize=5.5, loc="upper right")
    ax[0, 0].set_ylabel("V [V]"); ax[1, 0].set_ylabel("model - measured [mV]"); ax[2, 0].set_ylabel("I [A]")
    ax[1, 0].set_ylim(-100, 100)
    ax[0, 0].legend(fontsize=6, loc="lower left")
    fig.savefig(a.out, dpi=200); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
