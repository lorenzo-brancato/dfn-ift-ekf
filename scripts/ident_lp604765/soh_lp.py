"""SOH and SOC of the LP604765 ageing blocks from the EKF runs of scripts/ekf_exp.py (--start params, coords Rf,LLI).

    python soh_lp.py --runs <results>/soh/q1e-4 [--runs2 <results>/soh/q1e-6] [--blocks AGE22,AGE27,AGE30]

SOH (estimated)  OCV capacity between 4.2 and 2.75 V of the estimated window / that of the identified window (beginning
                 of life). Window from the filter state: lithium Li = x_eff Q_n + y Q_p (x_eff = anode mean lowered by
                 LLI dx_window Q_n), cathode capacity Q_p, anode Q_n (LAM fixed at 0). Every --every-s.
SOH (reference)  measured 1C discharge capacity of every cycle in the blocks, and the mean 1C capacity at the capacity
                 checks, / the first 1C discharge of AGE22. The 1C capacity also contains resistance growth, so it is an
                 upper bound on the equilibrium loss; compare the fade, not the absolute level.
SOC              (y_2.75 - y) / (y_2.75 - y_4.2) on the estimated window. Reference: Coulomb counting re-anchored at
                 100 % at the end of every CC-CV charge, with that cycle's measured 1C capacity as the full scale.
Blocks are drawn on one time axis in their real order (gaps between them not to scale).
Output: <runs>/soh_<blocks>.png/.pdf, soh_summary.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import datasets as D  # noqa: E402

plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4
VMAX, VMIN = 4.2, 2.75
CAP_DATE = {"AGE22": "2026-10-26", "AGE27": "2026-10-29", "AGE30": "2026-11-02"}    # check right after each block


def window(cs, cell, xbar, ybar, vmax=VMAX, vmin=VMIN, ny=3000):
    """(capacity [A.h], y at vmax, y at vmin) of the OCV window holding the lithium of the state (xbar, ybar)."""
    c = cs["cells"][cell]
    xn, Un = np.asarray(cs["ocp"]["n"]["x"]), np.asarray(cs["ocp"]["n"]["U"])
    yp, Up = np.asarray(cs["ocp"]["p"]["y"]), np.asarray(cs["ocp"]["p"]["U"])
    Qn, Qp = c["Q_n"], c["Q_p"]
    Li = xbar * Qn + ybar * Qp
    y = np.linspace(0.2, 0.999, ny)
    x = (Li - y * Qp) / Qn
    okx = (x > 1e-4) & (x < 0.999)
    V = np.where(okx, np.interp(y, yp, Up) - np.interp(np.clip(x, 1e-5, 1), xn, Un), np.nan)
    inwin = (V <= vmax) & (V >= vmin)
    if not np.any(inwin):
        return np.nan, np.nan, np.nan
    return Qp * (y[inwin].max() - y[inwin].min()), y[inwin].min(), y[inwin].max()


def analyse(f, cs, every_s):
    d = np.load(f, allow_pickle=True)
    cell = str(d["cell"]); c = cs["cells"][cell]
    co = [str(x) for x in d["coords"]]
    t = d["t"]; dt = float(d["dt"]); I = d["I"]
    lli = d["vals"][:, co.index("LLI")]; lli_sd = d["sd"][:, co.index("LLI")]
    dxw = float(d["dx_window_n"])
    x_eff = d["xbar"] - lli * dxw
    k = np.arange(0, len(t), max(1, int(every_s / dt)))
    W = np.array([window(cs, cell, x_eff[i], d["ybar"][i]) for i in k])
    Q0 = window(cs, cell, c["x100"], c["y100"])[0]
    soh = W[:, 0] / Q0
    # SOH band from the LLI sd (one-sided linearisation: d capacity / d LLI by a finite step)
    Wp = np.array([window(cs, cell, x_eff[i] - lli_sd[i] * dxw, d["ybar"][i])[0] for i in k])
    soh_sd = np.abs(Wp - W[:, 0]) / Q0
    yt = np.interp(np.arange(len(t)), k, W[:, 1]); yb = np.interp(np.arange(len(t)), k, W[:, 2])
    soc = (yb - d["ybar"]) / (yb - yt)
    # reference SOC: Coulomb counting from the end of every CC-CV charge (last sample of each charge run)
    ch = I < -0.02
    e = np.flatnonzero(np.diff(ch.astype(int)) != 0) + 1
    tops = [s[-1] for s in np.split(np.arange(len(I)), e) if ch[s[0]] and len(s) > 600]
    te, qd = D.lp_discharge_capacities(t, I)
    qc = np.cumsum(I) * dt / 3600
    soc_ref = np.full(len(t), np.nan)
    for j, kt in enumerate(tops):
        kn = tops[j + 1] if j + 1 < len(tops) else len(t)
        nxt = np.flatnonzero(te > t[kt])
        Qc = qd[nxt[0]] if len(nxt) else qd[-1]
        soc_ref[kt:kn] = 1 - (qc[kt:kn] - qc[kt]) / Qc
    return dict(t=t, tk=t[k], soh=soh, soh_sd=soh_sd, lli_Ah=lli * dxw * c["Q_n"], lli_sd_Ah=lli_sd * dxw * c["Q_n"],
                Rf=d["vals"][:, co.index("Rf")], soc=soc, soc_ref=soc_ref, innov=d["innov"], te=te, qd=qd,
                q_th=None, test=str(d["test"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder of the ekf_<block>_s-1.npz runs")
    ap.add_argument("--runs2", default=None, help="a second set of runs (other q), drawn dashed")
    ap.add_argument("--blocks", default="AGE22,AGE27,AGE30")
    ap.add_argument("--every-s", type=float, default=300.0)
    ap.add_argument("--zoom-h", type=float, default=9.0, help="zoom figure: the first hours of the first block")
    a = ap.parse_args()
    cs = D.cellset("exp_LP604765")
    blocks = a.blocks.split(",")
    sets = [(a.runs, "-", os.path.basename(os.path.normpath(a.runs)))]
    if a.runs2:
        sets.append((a.runs2, "--", os.path.basename(os.path.normpath(a.runs2))))
    Qref = None
    fig, ax = plt.subplots(4, 1, figsize=(170 * mm, 175 * mm), layout="constrained", sharex=True,
                           gridspec_kw=dict(height_ratios=[1.6, 1.0, 1.0, 1.0]))
    rows = []
    off = 0.0; gap = 6.0
    for bi, b in enumerate(blocks):
        span = None
        for si, (folder, ls, lab) in enumerate(sets):
            f = os.path.join(folder, "ekf_%s_s-1.npz" % b)
            if not os.path.exists(f):
                print("missing", f); continue
            r = analyse(f, cs, a.every_s)
            if Qref is None:
                Qref = r["qd"][0]
            th = (r["t"] / 3600) + off; tk = r["tk"] / 3600 + off
            span = th[-1] - off
            col = "C0" if si == 0 else "C1"
            lbl = lambda s: s if bi == 0 else None
            ax[0].fill_between(tk, 100 * (r["soh"] - 2 * r["soh_sd"]), 100 * (r["soh"] + 2 * r["soh_sd"]), color=col, alpha=0.15, lw=0)
            ax[0].plot(tk, 100 * r["soh"], ls, color=col, lw=1.2, label=lbl("EKF, %s (OCV window, ±2σ)" % lab))
            if si == 0:
                ax[0].plot(r["te"] / 3600 + off, 100 * r["qd"] / Qref, "k.", ms=3, label=lbl("measured 1C capacity, every cycle"))
            ax[1].plot(th, 1e3 * r["lli_Ah"], ls, color=col, lw=1.0, label=lbl(lab))
            ax[1].fill_between(th, 1e3 * (r["lli_Ah"] - 2 * r["lli_sd_Ah"]), 1e3 * (r["lli_Ah"] + 2 * r["lli_sd_Ah"]), color=col, alpha=0.15, lw=0)
            es = 100 * (r["soc"] - r["soc_ref"])
            ax[2].plot(th, es, ls, color=col, lw=0.5, label=lbl(lab))
            ax[3].plot(th, r["Rf"], ls, color=col, lw=0.6)
            ok = np.isfinite(es)
            if not ok.any():
                es = np.array([np.nan]); ok = np.array([True])
            nf = np.flatnonzero(~np.isfinite(r["Rf"]))
            if len(nf):
                ax[0].plot([th[nf[0]]], [100], "x", color=col, ms=7)
                ax[0].annotate("%s diverged" % lab, (th[nf[0]], 100), xytext=(3, -12), textcoords="offset points", fontsize=6.5, color=col)
            # end-of-block SOH against the last measured cycles (mean of the last 3)
            rows.append(dict(block=b, run=lab, SOH_est_end=100 * float(np.mean(r["soh"][-12:])),
                             SOH_1C_end=100 * float(np.mean(r["qd"][-3:]) / Qref),
                             fade_est=100 * float(np.mean(r["soh"][-12:]) - np.mean(r["soh"][:12])),
                             fade_1C=100 * float((np.mean(r["qd"][-3:]) - np.mean(r["qd"][:3])) / Qref),
                             LLI_end_mAh=1e3 * float(r["lli_Ah"][-1]), LLI_sd_end_mAh=1e3 * float(r["lli_sd_Ah"][-1]),
                             SOC_err_rms_pct=float(np.sqrt(np.mean(es[ok] ** 2))), SOC_err_p95_pct=float(np.percentile(np.abs(es[ok]), 95)),
                             innov_rmse_mV=float(1e3 * np.sqrt(np.nanmean(r["innov"] ** 2))),
                             diverged_at_h=float(r["t"][nf[0]] / 3600) if len(nf) else np.nan))
        if b in CAP_DATE and span is not None:
            ax[0].plot([off + span + gap / 2], [100 * D.LP_CAPCHECK[CAP_DATE[b]] / Qref], "D", color="C3", ms=5,
                       label="capacity check (1C)" if bi == 0 else None)
        for A in ax:
            A.axvspan(off, off + (span or 0), color="0.95", lw=0, zorder=-1)
        ax[0].text(off + (span or 0) / 2, 101.2, b, ha="center", fontsize=7)
        off += (span or 66) + gap
    ax[0].set_ylabel("SOH [%]"); ax[0].legend(fontsize=7, loc="lower left")
    ax[1].set_ylabel("LLI [mAh]"); ax[1].legend(fontsize=7, loc="upper left")
    ax[2].set_ylabel("SOC error [%]"); ax[2].axhline(0, color="k", lw=0.5); ax[2].set_ylim(-10, 10)
    ax[3].set_ylabel(r"$R_f$ multiplier"); ax[3].set_xlabel("time in the ageing blocks [h] (gaps not to scale)")
    for A in ax:
        A.spines[["top", "right"]].set_visible(False)
    out = os.path.join(a.runs, "soh_%s" % "_".join(blocks))
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.runs, "soh_summary.csv"), index=False)
    print(df.round(3).to_string(index=False))
    print("wrote", out + ".png")
    zoom(sets, blocks[0], a.zoom_h, os.path.join(a.runs, "soh_zoom_%s" % blocks[0]))


def zoom(sets, block, hours, out):
    """First cycles of a block: measured vs filtered voltage, innovation, LLI and Rf, CV phases shaded."""
    fig, ax = plt.subplots(4, 1, figsize=(170 * mm, 150 * mm), layout="constrained", sharex=True,
                           gridspec_kw=dict(height_ratios=[1.5, 1.0, 1.0, 1.0]))
    for si, (folder, ls, lab) in enumerate(sets):
        f = os.path.join(folder, "ekf_%s_s-1.npz" % block)
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        co = [str(x) for x in d["coords"]]
        th = d["t"] / 3600; m = th <= hours
        col = "C0" if si == 0 else "C1"
        if si == 0:
            ax[0].plot(th[m], d["y"][m], "k", lw=1.0, label="measured")
            # shaded: the masked samples of the run (no voltage update), else the CV phases (charge at V >= 4.19 V)
            cv = np.asarray(d["masked"], bool) if "masked" in d and np.any(d["masked"]) else (d["I"] < -0.02) & (d["y"] >= 4.19)
            shade = "masked: no voltage update" if "masked" in d and np.any(d["masked"]) else "CV phase at 4.2 V"
            e = np.flatnonzero(np.diff(cv[m].astype(int)) != 0) + 1
            for s in np.split(np.arange(m.sum()), e):
                if len(s) and cv[m][s[0]]:
                    for A in ax:
                        A.axvspan(th[m][s[0]], th[m][s[-1]], color="C3", alpha=0.1, lw=0)
        ax[0].plot(th[m], d["V"][m], ls, color=col, lw=0.9, label="EKF, %s" % lab)
        ax[1].plot(th[m], 1e3 * d["innov"][m], ls, color=col, lw=0.6)
        dxw = float(d["dx_window_n"])
        ax[2].plot(th[m],
                   1e3 * d["vals"][m, co.index("LLI")] * dxw * D.cellset("exp_LP604765")["cells"]["LP01"]["Q_n"], ls, color=col, lw=1.0)
        ax[3].plot(th[m], d["vals"][m, co.index("Rf")], ls, color=col, lw=1.0)
    ax[0].set_ylabel("V [V]"); ax[0].legend(fontsize=7, loc="lower right")
    ax[1].set_ylabel("innovation [mV]"); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylim(-60, 60)
    ax[2].set_ylabel("LLI [mAh]")
    ax[3].set_ylabel(r"$R_f$ multiplier"); ax[3].set_xlabel("t [h] (%s; shaded: %s)" % (block, shade))
    for A in ax:
        A.spines[["top", "right"]].set_visible(False)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
