"""Figure of the group screening (sens_groups.py): per group, the median elementary effect on each calibration test
(normal operation, both runs >= 2.7 V) and the 1 mV discard threshold of the ApEn draft.

    python fig_morris_groups.py   -> results/exp_LP604765/sens_groups/morris.{png,pdf}
"""
import json
import os

import numpy as np

import common as C


def main():
    od = os.path.join(C.RES, "sens_groups")
    d = json.load(open(os.path.join(od, "morris.json")))
    keys = d["keys"]
    rows = sorted(d["rows"], key=lambda r: np.nanmax([r["med_%s_all_N" % k] for k in keys]))
    names = [r["param"] for r in rows]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(C.HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(figsize=(120 * mm, 85 * mm), layout="constrained")
    y = np.arange(len(names))
    mk = {"HPPC00": ("C0", "o"), "HPPC20": ("k", "o"), "HPPC40": ("C3", "o"), "CAP00F": ("C0", "s"), "CAP40F": ("C3", "s"),
          "AGE22C2": ("k", "s")}
    for j, k in enumerate(keys):
        v = np.array([r["med_%s_all_N" % k] for r in rows])
        v = np.where(v > 0, v, np.nan)
        ax.plot(v, y + (j - 2.5) * 0.08, ls="none", marker=mk[k][1], color=mk[k][0], ms=3.5,
                mfc=mk[k][0] if k.startswith("HPPC") else "none", label=k)
    ax.axvline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(y, names); ax.set_xscale("log"); ax.set_xlim(0.5, 2e3)
    ax.set_xlabel("median elementary effect [mV], V $\geq$ 2.7 V")
    ax.set_title("Morris screening of the groups (r = %d, %d levels, literature boxes); dashed: 1 mV" % (d["r"], d["levels"]), fontsize=7)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(od, "morris." + e), dpi=200)
    print(os.path.join(od, "morris.png"))


if __name__ == "__main__":
    main()
