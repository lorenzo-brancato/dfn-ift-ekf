"""Figure of the EKF runs of scripts/ekf_exp.py --dataset exp_LP604765 (--start soc): voltage, SOC against Coulomb counting, SOC error with the +-2 sd band, Rf.

    python plot_ekf.py --test CYC20
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="CYC20")
    ap.add_argument("--err-lim", type=float, default=25.0, help="SOC error axis [+- %]")
    ap.add_argument("--no-open", action="store_true", help="do not draw the open-loop runs")
    ap.add_argument("--electrodes", action="store_true", help="also the SOC of each electrode (anode, cathode) and their errors")
    ap.add_argument("--q-window", type=float, default=None,
                    help="[A.h] OCV-window capacity of the set at BOL: reference = Coulomb counting re-anchored at 100 %% at the end "
                         "of every CV, full scale = this x the measured capacity fade (soc_ref.reference). Default: the run's soc_ref")
    ap.add_argument("--results", default=os.path.join(HERE, os.pardir, os.pardir, "results", "exp_LP604765", "ekf"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.results, "ekf_%s_e*.npz" % a.test)), key=lambda f: float(np.load(f)["err"]))
    runs = [dict(np.load(f)) for f in files]
    mm = 1 / 25.4
    fig, ax = plt.subplots(4, 1, figsize=(170 * mm, 175 * mm), layout="constrained", sharex=True,
                           gridspec_kw=dict(height_ratios=[1.2, 1.2, 1, 0.7]))
    th = runs[0]["t"] / 3600
    ax[0].plot(th, runs[0]["y"], "k", lw=0.9, label="measured")
    cols = {r["err"].item(): "C%d" % i for i, r in enumerate(runs)}
    if a.q_window:
        from soc_ref import reference
        ref = 100 * reference(runs[0]["t"], runs[0]["I"], a.test, a.q_window)[0]
        ref_lab = "reference (CC, 100 % at the end of each CV)"
    else:
        ref = runs[0]["soc_ref"] * 100; ref_lab = "Coulomb counting"
    ax[1].plot(th, ref, "k", lw=1.2, label=ref_lab)
    for r in runs:
        e = r["err"].item(); c = cols[e]
        lab = "" if len(runs) == 1 else r", $\Delta q_0 = %+.1f$ A h" % e      # one run: plain "EKF"
        op = os.path.join(a.results, "open_%s_e%+.2f.npy" % (a.test, e))
        if abs(e) > 0 and os.path.exists(op) and not a.no_open:
            ax[0].plot(th, np.load(op), color=c, lw=0.6, ls=":", label="open loop" + lab)
        ax[0].plot(th, r["V"], color=c, lw=0.6, label="EKF" + lab)
        soc, sd = 100 * (r["soc"] if "soc" in r else r["soc_p"]), 100 * r["sd_soc"]
        ax[1].plot(th, soc, color=c, lw=0.8, label="EKF, mean of the electrodes" + lab if a.electrodes else "EKF" + lab)
        if a.electrodes and "soc_n" in r:
            for key, ls, nm in (("soc_n", "--", "anode"), ("soc_p", ":", "cathode")):
                ax[1].plot(th, 100 * r[key], color="#D55E00" if key == "soc_n" else "#009E73", lw=0.9, ls=ls, label="EKF, %s%s" % (nm, lab))
                ax[2].plot(th, 100 * r[key] - ref, color="#D55E00" if key == "soc_n" else "#009E73", lw=0.9, ls=ls)
        err = soc - ref
        ax[2].fill_between(th, err - 2 * sd, err + 2 * sd, color=c, alpha=0.25, lw=0)
        ax[2].plot(th, err, color=c, lw=0.8)
        if r["vals"].shape[1] == 0:
            ax[3].plot(th, 1e3 * r["innov"], color=c, lw=0.5)
            continue
        ax[3].plot(th, r["vals"][:, 0], color=c, lw=0.8)
        ax[3].fill_between(th, r["vals"][:, 0] - 2 * r["sd_vals"][:, 0], r["vals"][:, 0] + 2 * r["sd_vals"][:, 0],
                           color=c, alpha=0.12, lw=0)
    ax[0].set_ylabel("V [V]"); ax[0].set_ylim(2.6, 4.35); ax[0].legend(ncol=3, fontsize=6, loc="lower left")
    ax[1].set_ylabel("SOC [%]"); ax[1].legend(ncol=4, fontsize=6, loc="lower left")
    ax[2].set_ylabel(r"SOC error [%]" "\n" r"(error $\pm 2\sigma$)"); ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylim(-a.err_lim, a.err_lim)
    if runs[0]["vals"].shape[1] == 0:
        ax[3].set_ylabel("innovation [mV]"); ax[3].set_ylim(-80, 80)
    else:
        ax[3].set_ylabel(r"$R_f$ multiplier"); ax[3].axhline(1.0, color="k", lw=0.5, ls="--")
    ax[3].set_xlabel("t [h]")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    out = os.path.join(a.results, "ekf_%s" % a.test)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    # zoom on the first 30 min: convergence
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 60 * mm), layout="constrained")
    k = th <= 0.5
    for r in runs:
        e = r["err"].item(); c = cols[e]
        sk = r["soc"] if "soc" in r else r["soc_p"]
        ax[0].plot(60 * th[k], 100 * sk[k] - ref[k], color=c, lw=0.9, label="EKF" if len(runs) == 1 else r"$\Delta q_0 = %+.1f$ A h" % e)
        ek = 100 * sk[k] - ref[k]
        ax[0].fill_between(60 * th[k], ek - 200 * r["sd_soc"][k], ek + 200 * r["sd_soc"][k], color=c, alpha=0.25, lw=0)
        ax[1].plot(60 * th[k], 1e3 * r["innov"][k], color=c, lw=0.6)
    ax[0].set_xlabel("t [min]"); ax[0].set_ylabel("SOC error [%]"); ax[0].legend(fontsize=6); ax[0].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("t [min]"); ax[1].set_ylabel("innovation [mV]")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.savefig(out + "_zoom.png"); fig.savefig(out + "_zoom.pdf")
    print("wrote", out + ".png", out + "_zoom.png")


if __name__ == "__main__":
    main()
