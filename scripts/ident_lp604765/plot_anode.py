"""Anode potential estimated by the EKF (scripts/ekf_exp.py --start soc, which stores the closure's anode profile):
the plating potential phi_s - phi_e - R_f j_n [V vs Li/Li+] at each anode node, extrapolated to the separator.

    python plot_anode.py --results ../../results/exp_LP604765/soc_anode --run ekf_CYC20_e+0.97.npz

fig_anode_time   (a) voltage, (b) current, (c) SOC (+-2 sd) against Coulomb counting, (d) min over x of the potential
fig_anode_x      colour map of the potential over (t, x) and five profiles along x at instants of the second cycle
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4


def with_interface(x, v, L):
    """Profile at the cell centres plus the value at the anode/separator interface x = L (linear from the last two)."""
    vL = v[:, -1] + (v[:, -1] - v[:, -2]) * (L - x[-1]) / (x[-1] - x[-2])
    return np.r_[x, L], np.c_[v, vL]


def instants(t, I):
    """Five instants of the second cycle (discharge start at the second rest-to-discharge transition): mid discharge, end
    of discharge, early CC charge, end of CC (minimum of the potential), end of CV."""
    dis = I > 0.5; chg = I < -0.02
    starts = np.where(dis[1:] & ~dis[:-1])[0] + 1
    k0 = starts[1] if len(starts) > 1 else (starts[0] if len(starts) else 0)
    k_end_dis = k0 + np.argmax(~dis[k0:]) - 1
    k_chg = k_end_dis + np.argmax(chg[k_end_dis:])
    k_end_chg = k_chg + np.argmax(~chg[k_chg:]) - 1
    cc = np.where(chg[k_chg:k_end_chg] & (np.abs(I[k_chg:k_end_chg]) > 0.95 * np.abs(I[k_chg])))[0]
    k_end_cc = k_chg + (cc[-1] if len(cc) else 0)
    k_mid = (k0 + k_end_dis) // 2
    k_early = k_chg + np.argmax(t[k_chg:] > t[k_chg] + 600)
    labs = ["mid discharge", "end of discharge", "CC charge, +10 min", "end of CC", "end of CV"]
    return [k_mid, k_end_dis, k_early, k_end_cc, k_end_chg], labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "exp_LP604765", "soc_anode"))
    ap.add_argument("--run", default="ekf_CYC20_e+0.97.npz")
    ap.add_argument("--Ln-um", type=float, default=None, help="anode thickness [um] (default: from the cell set via x_n)")
    a = ap.parse_args()
    d = np.load(os.path.join(a.results, a.run))
    t, th = d["t"], d["t"] / 3600
    x = d["x_n"]
    # cell widths are not stored: L = x[-1] + w_last / 2, the widths from the centres (w_i + w_{i+1} = 2 (x_{i+1} - x_i))
    w = np.zeros(len(x)); w[0] = 2 * x[0]
    for i in range(1, len(x)):
        w[i] = 2 * (x[i] - x[i - 1]) - w[i - 1]
    L = a.Ln_um * 1e-6 if a.Ln_um else x[-1] + 0.5 * w[-1]
    xs, phi = with_interface(x, d["phi_pl_n"], L)
    phi_mV = 1e3 * phi
    skip = t > 60                                     # the first minute: the filter jumps from its 50 % start
    soc_ref = 100 * d["soc_ref"]; soc = 100 * d["soc_p"]; sd = 100 * d["sd_soc"]

    # ---- figure 1: time traces
    fig, ax = plt.subplots(4, 1, figsize=(170 * mm, 160 * mm), layout="constrained", sharex=True)
    ax[0].plot(th, d["y"], "k", lw=0.9, label="measured")
    ax[0].plot(th[skip], d["V"][skip], color="#0072B2", lw=0.7, label="EKF (predicted)")
    ax[0].set_ylabel(r"$V$ [V]"); ax[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    ax[1].plot(th, d["I"], "k", lw=0.8); ax[1].set_ylabel(r"$I$ [A]" "\n(+ discharge)")
    ax[2].plot(th, soc_ref, "k", lw=1.0, label="Coulomb counting")
    ax[2].fill_between(th, soc - 2 * sd, soc + 2 * sd, color="#0072B2", alpha=0.25, lw=0)
    ax[2].plot(th, soc, color="#0072B2", lw=0.8, label=r"EKF $\pm 2\sigma$"); ax[2].set_ylim(-5, 110)
    ax[2].set_ylabel("SOC [%]"); ax[2].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    mn = phi_mV.min(1); kx = phi_mV.argmin(1)
    ax[3].plot(th[skip], mn[skip], color="#D55E00", lw=0.9)
    ax[3].axhline(0, color="k", lw=0.6, ls="--")
    ax[3].set_ylabel(r"$\min_x\,(\phi_s-\phi_e-R_f j_n)$" "\n[mV vs Li/Li$^+$]"); ax[3].set_xlabel(r"$t$ [h]")
    for k, s in enumerate("abcd"):
        ax[k].text(-0.09, 1.0, "(%s)" % s, transform=ax[k].transAxes, va="top", fontsize=9)
        ax[k].spines[["top", "right"]].set_visible(False)
    print("min over the run (after 1 min): %.1f mV at t = %.2f h, x = %.1f um" % (mn[skip].min(), th[skip][np.argmin(mn[skip])],
          1e6 * xs[kx[skip][np.argmin(mn[skip])]]))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.results, "fig_anode_time." + ext))
    plt.close(fig)

    # ---- figure 2: colour map and profiles
    ks, labs = instants(t, d["I"])
    fig = plt.figure(figsize=(170 * mm, 95 * mm), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 0.32])
    axm = fig.add_subplot(gs[0, 0]); axp = fig.add_subplot(gs[:, 1]); axi = fig.add_subplot(gs[1, 0], sharex=axm)
    T, X = np.meshgrid(th[skip], 1e6 * xs, indexing="ij")
    pm = axm.pcolormesh(T, X, phi_mV[skip], shading="gouraud", cmap="viridis")
    cb = fig.colorbar(pm, ax=axm, pad=0.01); cb.set_label(r"$\phi_s-\phi_e-R_f j_n$ [mV vs Li/Li$^+$]")
    axm.set_ylabel(r"$x$ [$\mu$m] (0: current collector)"); axm.axhline(1e6 * L, color="w", lw=0.5, ls=":")
    cols = plt.cm.plasma(np.linspace(0.05, 0.85, len(ks)))
    for k, lab, c in zip(ks, labs, cols):
        axm.axvline(th[k], color=c, lw=1.0, ls="--")
        axi.axvline(th[k], color=c, lw=1.0, ls="--")
        axp.plot(1e6 * xs, phi_mV[k], "-o", color=c, ms=2.5, lw=1.0, label="%s (%.2f h)" % (lab, th[k]))
    axp.axhline(0, color="k", lw=0.6, ls="--")
    axp.set_xlabel(r"$x$ [$\mu$m]"); axp.set_ylabel(r"$\phi_s-\phi_e-R_f j_n$ [mV]")
    axp.set_ylim(-10, None)
    axp.legend(fontsize=6, frameon=False, loc="lower left")
    axi.plot(th, d["I"], "k", lw=0.7); axi.set_ylabel(r"$I$ [A]"); axi.set_xlabel(r"$t$ [h]")
    plt.setp(axm.get_xticklabels(), visible=False)
    for ax_ in (axp, axi):
        ax_.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.results, "fig_anode_x." + ext))
    print("wrote", os.path.join(a.results, "fig_anode_time.png"), os.path.join(a.results, "fig_anode_x.png"))


if __name__ == "__main__":
    main()
