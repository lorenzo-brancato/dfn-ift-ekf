"""Figures of a Mohtat EKF run (scripts/ekf_exp.py --dataset exp_Mohtat .npz): voltage, SOC, SOH (capacity), LLI, LAM, parameters.

    python scripts/mohtat_ekf_figs.py --set data/exp_Mohtat/params/mohtat_lm3c.json [--ref per_checkup_window.csv] run.npz ...

Estimated window. From the filter's state and parameters at time t: anode capacity Q_n (1 - LAM), lithium
    Li = x_eff Q_n (1 - LAM) + y Q_p (x_eff = the anode mean the closure sees, lowered by LLI; conserved by the model),
    OCV U_p(y) - U_n(x) on a y grid -> y at 4.2 V and at 3.0 V. Every --every-s.
SOC     (y_3.0 - y) / (y_3.0 - y_4.2) on that estimated window, +- 2 sd from the cathode mean's sd.
        Reference: coulomb counting. BOL prepared tests: from the identification's initial charge on the C/20 window.
        Ageing cycles: anchored at 100 % at the end of charge (the voltage maximum), capacity = SOH_ref x the BOL C/20 one.
SOH     OCV capacity between 4.2 and 3.0 V of the estimated window / the same of the identified window (BOL).
        Reference: the file's SOH_ref (C/20 check-up anchors).
LLI     estimated in units of the anode window, shown in A.h (LLI dx Q_n). Reference (--ref): the electrode-window fit
        of the nearest check-up (ParamIdentification-independent: SOC SOH Coestimation runs/electrode_soh_window_1), its
        lithium loss n_Li(0) - n_Li [A.h] and LAM_n, LAM_p (the DFN has no LAM_p).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def window(cs, cell, xbar, ybar, lam, vmax=4.2, vmin=3.0, ny=3000):
    """(capacity [A.h], y at vmax, y at vmin) of the OCV window of the state (xbar, ybar) with anode loss lam."""
    c = cs["cells"][cell]
    xn, Un = np.asarray(cs["ocp"]["n"]["x"]), np.asarray(cs["ocp"]["n"]["U"])
    yp, Up = np.asarray(cs["ocp"]["p"]["y"]), np.asarray(cs["ocp"]["p"]["U"])
    Qn = c["Q_n"] * (1 - lam); Qp = c["Q_p"]
    Li = xbar * Qn + ybar * Qp
    y = np.linspace(0.25, 0.999, ny)
    x = (Li - y * Qp) / Qn
    okx = (x > 1e-4) & (x < 0.999)
    V = np.where(okx, np.interp(y, yp, Up) - np.interp(np.clip(x, 1e-5, 1), xn, Un), np.nan)
    inwin = (V <= vmax) & (V >= vmin)
    if not np.any(inwin):
        return np.nan, np.nan, np.nan
    return Qp * (y[inwin].max() - y[inwin].min()), y[inwin].min(), y[inwin].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ref", default=None, help="per_checkup_window.csv (electrode-window fits at the check-ups)")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every-s", type=float, default=300.0)
    a = ap.parse_args()
    cs = json.load(open(a.set, encoding="utf-8"))
    ref = pd.read_csv(a.ref) if a.ref else None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for f in a.runs:
        d = np.load(f, allow_pickle=True)
        cell, test, twin = str(d["cell"]), str(d["test"]), bool(d["twin"])
        cycle = int(d["cycle"]) if "cycle" in d else -1
        c = cs["cells"][cell]
        t = d["t"] / 3600; I = d["I"]; dt = float(d["dt"])
        coords = [str(s) for s in d["coords"]]
        vals, sd = d["vals"], d["sd"]
        lam = vals[:, coords.index("LAM")] if "LAM" in coords else np.zeros(len(t))
        lli = vals[:, coords.index("LLI")] if "LLI" in coords else np.zeros(len(t))
        lli_sd = sd[:, coords.index("LLI")] if "LLI" in coords else np.zeros(len(t))
        dxw = float(d["dx_window_n"]) if "dx_window_n" in d else c["x100"] - c["x0"]
        x_eff = d["xbar"] - lli * dxw
        ybar = d["ybar"]
        k = np.arange(0, len(t), max(1, int(a.every_s / dt)))
        W = np.array([window(cs, cell, x_eff[i], ybar[i], lam[i]) for i in k])
        Qnom = window(cs, cell, c["x100"], c["y100"], 0.0)[0]
        y_top = np.interp(np.arange(len(t)), k, W[:, 1]); y_bot = np.interp(np.arange(len(t)), k, W[:, 2])
        soc = (y_bot - ybar) / (y_bot - y_top)
        soc_sd = d["sd_ybar"] / (y_bot - y_top)
        qc = np.cumsum(I) * dt / 3600.0
        soh_ref = float(d["SOH_ref"]) if "SOH_ref" in d else np.nan
        if cycle >= 0:
            k_top = int(np.argmax(d["y"]))
            Q_age = soh_ref / 100 * c.get("q_ocv_Ah", Qnom)
            soc_ref = 1 - (qc - qc[k_top]) / Q_age
        else:
            soc_ref = 1 - (float(d["q0"]) + qc) / c.get("q_ocv_Ah", Qnom)
        r = None
        if ref is not None:
            rc = ref[ref.cell == int(cell[1:])]
            if len(rc):
                r = rc.iloc[int(np.argmin(np.abs(rc.cycle.values - max(cycle, 0))))]
                nLi0 = float(rc[rc.cycle == rc.cycle.min()].n_Li.iloc[0])
                lli_ref_Ah = nLi0 - float(r.n_Li)
        ok = d["eis"] == 0
        e = 100 * (soc - soc_ref)
        lli_Ah = lli * dxw * c["Q_n"]
        fig, ax = plt.subplots(6, 1, figsize=(8.5, 14), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.2, 0.7, 1, 1, 1]))
        ax[0].plot(t, d["y"], "k", lw=1, label="measured" if not twin else "twin data")
        ax[0].plot(t, d["V"], "C3", lw=0.8, ls="--", label="EKF prediction")
        ax[0].set_ylabel("V [V]"); ax[0].legend(frameon=False, loc="lower left")
        ax[1].fill_between(t, 100 * (soc - 2 * soc_sd), 100 * (soc + 2 * soc_sd), color="C0", alpha=0.25, lw=0, label="EKF +- 2 sd")
        ax[1].plot(t, 100 * soc, "C0", lw=1.2, label="EKF (estimated window)")
        ax[1].plot(t, 100 * soc_ref, "k", lw=0.8, ls="--", label="coulomb counting")
        ax[1].set_ylabel("SOC [%]"); ax[1].legend(frameon=False, loc="lower left", fontsize=8)
        ax[2].plot(t, e, "C0", lw=0.9); ax[2].axhline(0, color="0.5", lw=0.5); ax[2].set_ylabel("SOC err. [%]")
        soh = 100 * W[:, 0] / Qnom
        ax[3].plot(t[k], soh, "C2", lw=1.2, marker=".", ms=3, label="EKF (OCV capacity, 4.2-3.0 V)")
        if np.isfinite(soh_ref):
            ax[3].axhline(soh_ref, color="k", lw=0.9, ls="--", label="reference (C/20 check-up) %.1f %%" % soh_ref)
        ax[3].set_ylabel("SOH [%]"); ax[3].legend(frameon=False, fontsize=8)
        ax[4].fill_between(t, lli_Ah - 2 * lli_sd * dxw * c["Q_n"], lli_Ah + 2 * lli_sd * dxw * c["Q_n"], color="C1", alpha=0.25, lw=0)
        ax[4].plot(t, lli_Ah, "C1", lw=1.2, label="LLI EKF")
        ax[4].plot(t, lam * c["Q_n"], "C4", lw=1.2, label="LAM_n EKF (x Q_n)")
        if r is not None:
            ax[4].axhline(lli_ref_Ah, color="C1", ls="--", lw=0.9, label="LLI ref. %.2f A.h" % lli_ref_Ah)
            ax[4].axhline(float(r.LAM_n) * c["Q_n"], color="C4", ls="--", lw=0.9, label="LAM_n ref. %.1f %%" % (100 * r.LAM_n))
        ax[4].set_ylabel("loss [A.h]"); ax[4].legend(frameon=False, fontsize=7, ncol=2)
        for i, cn in enumerate(coords):
            if cn in ("LLI", "LAM"):
                continue
            ax[5].plot(t, vals[:, i], lw=1.1, label={"Rf": "Rf / Rf,BOL", "Ds": "Ds,p / Ds,p,BOL"}[cn])
        ax[5].axhline(1, color="k", lw=0.8, ls="--"); ax[5].set_yscale("log")
        ax[5].set_ylabel("Rf, Ds\n(BOL = 1)"); ax[5].legend(frameon=False, fontsize=8); ax[5].set_xlabel("t [h]")
        name = "%s %s" % (cell, test if cycle < 0 else "cycle %d" % cycle)
        ax[0].set_title("%s%s  |  SOC err RMS %.2f %%  |  SOH end %.1f %% (ref %.1f)  |  LLI end %.2f A.h (ref %s)"
                        % (name, " (twin)" if twin else " (measured)", np.sqrt(np.mean(e[ok] ** 2)), soh[-1], soh_ref,
                           lli_Ah[-1], "%.2f" % lli_ref_Ah if r is not None else "-"), fontsize=8.5)
        fig.tight_layout()
        out = os.path.splitext(f)[0] + "_soc_soh.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        rows.append(dict(run=os.path.basename(f), cycle=cycle, SOH_ref=soh_ref, SOH_end=soh[-1], SOC_rms=np.sqrt(np.mean(e[ok] ** 2)),
                         LLI_Ah=lli_Ah[-1], LLI_ref_Ah=lli_ref_Ah if r is not None else np.nan, LAM_n=lam[-1],
                         LAM_n_ref=float(r.LAM_n) if r is not None else np.nan, LAM_p_ref=float(r.LAM_p) if r is not None else np.nan,
                         **{"%s_end" % cn: vals[-1, i] for i, cn in enumerate(coords) if cn in ("Rf", "Ds")}))
        print(out)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
