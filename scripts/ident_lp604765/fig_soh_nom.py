"""SOC and SOH of the LP604765 ageing blocks from the closed-loop EKF runs of scripts/ekf_exp.py (--start soc,
--coords Rf,LLI, --soc-vlim 4.2,3.0).

    python fig_soh_nom.py --runs <results>/soh_v12 [--blocks AGE22,AGE27,AGE30] [--tag e+0.00]

SOC (estimated)  from the stoichiometry of the state on the OCV window 4.2-3.0 V moved by the estimated LLI:
                 mean of (y_0 - y) / (y_0 - y_100) and (x - x_0) / (x_100 - x_0). No charge counting, no capacity.
SOC (reference)  charge counting re-anchored at 100 % at the end of every CC-CV charge, full scale = the measured
                 capacity of the discharge that follows (per cycle, so it ages with the cell). It does not enter the
                 filter. At the 1C cut-off it reads 0 % while the stoichiometric SOC keeps the charge left behind by the
                 1C overpotential, so a few % positive error at the end of each discharge is expected.
SOH              OCV-window capacity (4.2-3.0 V) of the estimated state / nominal capacity 2.133 A.h, in %.
                 Reference: measured 1C discharge capacity of each cycle / 2.133 A.h (a lower bound of the OCV capacity:
                 it also contains the 1C overpotential).
Output: <runs>/fig_soh_nom_<tag>.png/.pdf
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
sys.path.insert(0, ROOT)
from dfnekf import datasets as D  # noqa: E402

plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4


def soc_reference(t, I, min_charge_s=600.0, q_window=None):
    """Charge-counting SOC reference re-anchored every cycle: 100 % at the end of every CC-CV charge (a charging run
    longer than min_charge_s), full scale = the charge of the discharge that follows it (that cycle's measured capacity).
    Before the first anchor: back from it with the first cycle's scale. Returns (soc_ref [0-1], anchors, capacities [A.h])."""
    dts = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    q = np.cumsum(I * dts) / 3600.0                       # charge removed since the start [A.h] (+ = discharge)
    chg = I < -0.01
    edges = np.flatnonzero(np.diff(chg.astype(int)))      # run boundaries
    starts = np.r_[0, edges + 1]; ends = np.r_[edges, len(I) - 1]
    anchors = [e for s, e in zip(starts, ends) if chg[s] and t[e] - t[s] >= min_charge_s and e < len(I) - 1]   # not cut by the record's end
    caps = []
    for i, k in enumerate(anchors):
        nxt = anchors[i + 1] if i + 1 < len(anchors) else len(I) - 1
        seg = slice(k + 1, nxt + 1)
        stop = np.flatnonzero(chg[seg]); stop = (k + 1 + stop[0]) if len(stop) else nxt + 1   # until charging resumes
        caps.append(float(np.sum(np.clip(I[k + 1:stop], 0, None) * dts[k + 1:stop]) / 3600.0))
    for i in range(1, len(caps)):                         # a discharge cut short by the end of the record: previous scale
        if caps[i] < 0.8 * caps[i - 1]:
            caps[i] = caps[i - 1]
    if q_window is not None and np.ndim(q_window) == 1:   # per-sample window (aged: the run's Q_window_Ah) read at each anchor
        caps = [float(q_window[k]) for k in anchors]
    elif q_window:                                        # full scale = the OCV window (stoichiometric SOC), not the 1C capacity
        caps = [q_window] * len(caps)
    ref = np.full(len(I), np.nan)
    if not anchors:
        return ref, anchors, caps
    bounds = [0] + anchors[1:] + [len(I)]
    for i, k in enumerate(anchors):
        sl = slice(bounds[i], bounds[i + 1])
        ref[sl] = 1.0 - (q[sl] - q[k]) / caps[i]
    return ref, anchors, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--blocks", default="AGE22,AGE27,AGE30")
    ap.add_argument("--tag", default="e+0.00", help="run of each block: e+0.00 (placed start) or e.g. e-0.90 (SOC 50 %% start)")
    ap.add_argument("--soc-only", action="store_true", help="runs without parameter states (no SOH row)")
    ap.add_argument("--q-window", type=float, default=None,
                    help="[A.h] reference full scale = this OCV-window capacity (4.2-3.0 V of the cell set) instead of each "
                         "cycle's measured 1C capacity: consistent with the stoichiometric SOC (0 %% at the 3.0 V OCV, not at the 1C cut-off)")
    a = ap.parse_args()
    blocks = a.blocks.split(",")
    nr = 2 if a.soc_only else 3
    fig, ax = plt.subplots(nr, len(blocks), figsize=(180 * mm, 40 * nr * mm), sharex="col", sharey="row", layout="constrained", squeeze=False)
    Qn = D.LP_Q_NOMINAL
    for j, b in enumerate(blocks):
        fs = [f for f in os.listdir(a.runs) if f.startswith("ekf_%s_" % b) and f.endswith(".npz")]
        f = next((x for x in fs if a.tag in x), None) if a.tag != "s50" else next((x for x in fs if "e+0.00" not in x), None)
        if f is None:
            ax[0, j].set_title("%s: no run" % b); continue
        d = np.load(os.path.join(a.runs, f), allow_pickle=True)
        th = d["t"] / 3600
        r, anc, caps = soc_reference(d["t"], d["I"], q_window=a.q_window)
        soc, sd, ref = 100 * d["soc"], 100 * d["sd_soc"], 100 * r
        print("%s: %d cycles, capacity %.3f..%.3f A.h" % (b, len(anc), min(caps), max(caps)))
        ax[0, j].fill_between(th, soc - 2 * sd, soc + 2 * sd, color="C0", alpha=0.25, lw=0)
        ax[0, j].plot(th, ref, "k", lw=0.7, label="reference")
        ax[0, j].plot(th, soc, "C0", lw=0.7, label="EKF (stoichiometry)")
        es = soc - ref
        half = len(es) // 2
        ax[1, j].fill_between(th, -2 * sd, 2 * sd, color="C0", alpha=0.25, lw=0)
        ax[1, j].plot(th, es, "C3", lw=0.6)
        ax[1, j].axhline(0, color="0.5", lw=0.5)
        ax[1, j].text(0.02, 0.95, "RMS (2nd half) %.1f %%" % np.sqrt(np.mean(es[half:] ** 2)), transform=ax[1, j].transAxes, va="top", fontsize=6)
        ax[0, j].set_title(b, fontsize=8)
        ax[-1, j].set_xlabel("t [h]")
        if a.soc_only:
            continue
        sn, ssd = 100 * d["soh_nom"], 100 * d["sd_soh_nom"]
        ax[2, j].fill_between(th, sn - 2 * ssd, sn + 2 * ssd, color="C0", alpha=0.25, lw=0)
        ax[2, j].plot(th, sn, "C0", lw=0.8, label="EKF: OCV window / 2.133 A.h")
        ct, cq = d["cap_1C_t"], d["cap_1C"]
        if len(cq):
            ax[2, j].plot(np.asarray(ct) / 3600, 100 * np.asarray(cq) / Qn, "ks", ms=2.5, label="measured 1C capacity / 2.133 A.h")
        ax[2, j].text(0.02, 0.05, "end %.1f %% (LLI %.0f mAh)" % (sn[-1], 1e3 * d["lli_Ah"][-1]), transform=ax[2, j].transAxes, fontsize=6)
    ax[0, 0].set_ylabel("SOC [%]"); ax[1, 0].set_ylabel("SOC error [%]")
    ax[1, 0].set_ylim(-15, 15)
    ax[0, 0].legend(fontsize=6, loc="lower left")
    if not a.soc_only:
        ax[2, 0].set_ylabel("SOH [% of 2.133 A.h]"); ax[2, 0].legend(fontsize=6, loc="lower left")
    out = os.path.join(a.runs, ("fig_soc_%s" % a.tag if a.soc_only else "fig_soh_nom_%s" % a.tag) + ("_qw" if a.q_window else ""))
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
