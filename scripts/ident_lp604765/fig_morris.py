"""Figure of the Morris screening: mu* in normal operation (V >= 2.7 V) per parameter, mean and min-max over the tests.

    python fig_morris.py [--res results/exp_LGM50/sens]
Reads <res>/morris.json (sens_morris.py); writes <res>/morris.{png,pdf}.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(ROOT, "results", "exp_LGM50", "sens"))
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.res, "morris.json")))
    rows = sorted(d["rows"], key=lambda r: r["muN_mean"])
    names = [r["param"] for r in rows]
    keys = sorted({k[4:] for k in rows[0] if k.startswith("muN_") and k != "muN_mean"})
    muN = np.array([[r["muN_" + k] for k in keys] for r in rows])
    mu = np.array([[r["mu_" + k] for k in keys] for r in rows])
    sd = np.array([[r["sd_" + k] for k in keys] for r in rows])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 1, figsize=(85 * mm, 70 * mm), layout="constrained"); ax = [ax]
    y = np.arange(len(names))
    ax[0].barh(y, muN.mean(1), color="C0", height=0.6)
    ax[0].plot(muN.min(1), y, "|", color="k", ms=5); ax[0].plot(muN.max(1), y, "|", color="k", ms=5)
    ax[0].hlines(y, muN.min(1), muN.max(1), color="k", lw=0.5)
    ax[0].set_yticks(y, names); ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\mu^*$, V $\geq$ 2.7 V [mV per unit range]")
    ax[0].set_title("Morris, normal operation (bar: mean of %d tests, line: min-max)" % len(keys), fontsize=7)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(a.res, "morris." + e), dpi=200)
    print(os.path.join(a.res, "morris.png"))


if __name__ == "__main__":
    main()
