"""Lithium deposition potential (LDP) estimated by the EKF: phi_s - phi_e - R_f j_n at the anode/separator interface
[V vs Li/Li+] (plating when < 0), from the closure's anode profile each EKF run stores (ekf_exp.py anode_profile,
extrapolated linearly from the last two anode cells to the interface).

    python fig_ldp.py --runs <results>/rfli_final_eps --test AGE22V --blocks AGE22,AGE27,AGE30

(a) LDP and current over the test (--test), with the margin band below --margin-mV shaded;
(b) zoom on one charge (the second): LDP and voltage;
(c) per cycle of the ageing blocks (in order): minimum LDP during each CC charge (and at its end), with the estimated R_f
    multiplier: the plating margin as the cell ages.
Output: <runs>/fig_ldp.png/.pdf, ldp_per_cycle.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4


def ldp_of(d):
    """LDP at the anode/separator interface: linear extrapolation of phi_pl from the last two anode cell centres."""
    x, v = d["x_n"], d["phi_pl_n"]
    L = x[-1] + 0.5 * (x[-1] - x[-2])                     # interface (last cell's half width)
    return v[:, -1] + (v[:, -1] - v[:, -2]) * (L - x[-1]) / (x[-1] - x[-2])


def cc_charges(I, i_min=1.5):
    """(start, end) indices of every CC charge (|I| >= i_min while charging)."""
    cc = I < -i_min
    e = np.flatnonzero(np.diff(cc.astype(int)))
    s = np.r_[0, e + 1]; t = np.r_[e, len(I) - 1]
    return [(a, b) for a, b in zip(s, t) if cc[a] and b > a + 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--test", default="AGE22V")
    ap.add_argument("--blocks", default="AGE22,AGE27,AGE30")
    ap.add_argument("--tag", default="e+0.00")
    ap.add_argument("--margin-mV", type=float, default=20.0)
    a = ap.parse_args()
    fig = plt.figure(figsize=(180 * mm, 120 * mm), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax_t = fig.add_subplot(gs[0, :2]); ax_z = fig.add_subplot(gs[0, 2]); ax_c = fig.add_subplot(gs[1, :])

    d = np.load(os.path.join(a.runs, "ekf_%s_%s.npz" % (a.test, a.tag)), allow_pickle=True)
    th, ldp = d["t"] / 3600, 1e3 * ldp_of(d)
    ax_t.axhspan(-50, a.margin_mV, color="C3", alpha=0.10, lw=0)
    ax_t.plot(th, ldp, "C0", lw=0.8, label="LDP (EKF)")
    ax_t.axhline(0, color="C3", lw=0.8)
    ax_t.set_ylim(-20, 300); ax_t.set_ylabel("LDP [mV vs Li/Li$^+$]"); ax_t.set_xlabel("t [h]")
    ax_t.set_title("%s: LDP at the anode/separator interface (shaded: < %g mV)" % (a.test, a.margin_mV), fontsize=8)
    axI = ax_t.twinx(); axI.plot(th, d["I"], color="0.6", lw=0.6); axI.set_ylabel("I [A]", color="0.5")
    ch = cc_charges(d["I"])
    if len(ch) >= 2:
        s, e = ch[1]
        k1 = min(len(th) - 1, e + int(1.0 * 3600 / max(np.median(np.diff(d["t"])), 1)))
        sl = slice(max(0, s - 30), k1)
        ax_z.plot(th[sl], ldp[sl], "C0", lw=1.0)
        ax_z.axhline(0, color="C3", lw=0.8); ax_z.axhspan(-50, a.margin_mV, color="C3", alpha=0.10, lw=0)
        ax_z.axvline(th[e], color="0.5", lw=0.6, ls=":"); ax_z.text(th[e], 250, " end of CC", fontsize=6, color="0.4")
        ax_z.set_ylim(-20, 300); ax_z.set_xlabel("t [h]"); ax_z.set_title("second charge (CC-CV)", fontsize=8)
        axV = ax_z.twinx(); axV.plot(th[sl], d["y"][sl], color="0.6", lw=0.6); axV.set_ylabel("V [V]", color="0.5")

    rows, x0 = [], 0
    for b in a.blocks.split(","):
        f = os.path.join(a.runs, "ekf_%s_%s.npz" % (b, a.tag))
        if not os.path.exists(f):
            continue
        db = np.load(f, allow_pickle=True)
        lb = 1e3 * ldp_of(db)
        co = list(map(str, db["coords"]))
        rf = db["vals"][:, co.index("Rf")] if "Rf" in co else np.ones(len(lb))
        for k, (s, e) in enumerate(cc_charges(db["I"])):
            rows.append(dict(block=b, cycle=x0 + k + 1, ldp_min_mV=float(np.nanmin(lb[s:e + 1])), ldp_end_cc_mV=float(lb[e]),
                             Rf=float(rf[e])))
        x0 += len(cc_charges(db["I"]))
    if rows:
        df = pd.DataFrame(rows); df.to_csv(os.path.join(a.runs, "ldp_per_cycle.csv"), index=False)
        for i, (b, g) in enumerate(df.groupby("block", sort=False)):
            ax_c.plot(g.cycle, g.ldp_min_mV, "o", color="C%d" % i, ms=3, label="%s: min LDP of each CC charge" % b)
        ax_c.axhline(0, color="C3", lw=0.8); ax_c.axhspan(-50, a.margin_mV, color="C3", alpha=0.10, lw=0)
        ax_c.set_ylim(min(-5, df.ldp_min_mV.min() - 5), max(60, df.ldp_min_mV.max() + 10))
        ax_c.set_xlabel("1C cycle (ageing blocks in order)"); ax_c.set_ylabel("min LDP [mV]")
        axR = ax_c.twinx(); axR.plot(df.cycle, df.Rf, color="0.4", lw=0.8, ls="--"); axR.set_ylabel(r"$R_f$ multiplier (EKF)", color="0.4")
        ax_c.legend(fontsize=6, loc="lower left")
        ax_c.set_title("plating margin per cycle as the cell ages (EKF with $R_f$, LLI)", fontsize=8)
    out = os.path.join(a.runs, "fig_ldp")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
