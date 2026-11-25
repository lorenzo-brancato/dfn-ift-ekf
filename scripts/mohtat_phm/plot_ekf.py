"""Figures of the PHM EKF runs (ekf_batch.py): voltage, SOC estimate vs reference with +-2 sd, SOC error, LDP.

    .venv/Scripts/python.exe scripts/mohtat_phm/plot_ekf.py --name soc_heldout [--cells C01,C02]

One PNG per cell in results/mohtat_phm/ekf/<name>/fig_<cell>.png (runs started at the reference charge).
The LDP is phi_pl_n linearly extrapolated to the anode/separator face from the last two anode cells (as step.DFN.interface).
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def ldp_sep(d):
    x = d["x_n"]
    dx = np.zeros_like(x)
    dx[0] = 2 * x[0]
    for i in range(1, len(x)):
        dx[i] = 2 * (x[i] - x[i - 1]) - dx[i - 1]
    xb = dx.sum()
    p = d["phi_pl_n"]
    return p[:, -1] + (p[:, -1] - p[:, -2]) * (xb - x[-1]) / (x[-1] - x[-2])


def plot_cell(out, cell, files):
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.2, 1, 1]))
    for k, f in enumerate(sorted(files)):
        d = np.load(f)
        e = float(d["err"])
        th = d["t"] / 3600
        col = "C1" if k else "C0"
        lab = "start at the reference SOC" if abs(e) < 1e-9 else "start %+.2f A.h from the reference" % e
        if k == 0:
            ax[0].plot(th, d["y"], "k", lw=1, label="measured")
            ax0b = ax[0].twinx()
            ax0b.plot(th, d["I"], color="0.7", lw=0.6)
            ax0b.set_ylabel("I [A]", color="0.5")
            ax[1].plot(th, 100 * d["soc_ref"], "k", lw=1, label="reference (Coulomb count, 100 % at CV ends)")
        ax[0].plot(th, d["V"], col, lw=0.8, ls="--" if k else "-", label="EKF, " + lab)
        s, sd = 100 * d["soc"], 100 * d["sd_soc"]
        ax[1].plot(th, s, col, lw=0.8, ls="--" if k else "-", label="EKF, " + lab)
        ax[1].fill_between(th, s - 2 * sd, s + 2 * sd, color=col, alpha=0.2, lw=0)
        er = s - 100 * d["soc_ref"]
        ax[2].plot(th, er, col, lw=0.8, label=lab)
        ax[2].fill_between(th, er - 2 * sd, er + 2 * sd, color=col, alpha=0.2, lw=0)
        ldp = ldp_sep(d) * 1e3
        ax[3].plot(th, ldp, col, lw=0.8, ls="--" if k else "-", label=lab)
    ax[0].set_ylabel("V [V]"); ax[0].legend(fontsize=8, loc="lower left")
    ax[1].set_ylabel("SOC [%]"); ax[1].legend(fontsize=8)
    ax[2].set_ylabel("SOC error [%]\n(band: +-2 sd)"); ax[2].axhline(0, color="k", lw=0.5)
    lim = max(3.0, np.nanpercentile(np.abs(er), 99) * 1.3)
    ax[2].set_ylim(-lim, lim); ax[2].legend(fontsize=8)
    ax[3].set_ylabel("LDP at separator [mV]"); ax[3].axhline(0, color="r", lw=0.8)
    ax[3].set_ylim(-50, 300); ax[3].set_xlabel("t [h]")
    ax[0].set_title("%s, fit B, SOC-only EKF, %s (measured T, 2-s steps)" % (cell.replace("_", " "), os.path.basename(out)))
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "fig_%s.png" % cell)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="soc_heldout")
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "mohtat_phm", "ekf", a.name)
    by = {}
    for f in glob.glob(os.path.join(out, "ekf_*_e*.npz")):
        by.setdefault(re.search(r"ekf_(C\d+_[A-Z0-9]+)_e", f).group(1), []).append(f)
    by = {c: sorted(fs)[:1] for c, fs in by.items()}    # one run per cell (the first start error: e+0.00 when there)
    cells = a.cells.split(",") if a.cells else sorted(by)
    for c in cells:
        if c in by:
            print(plot_cell(out, c, by[c]))


if __name__ == "__main__":
    main()
