"""Figure of the Morris screening (results/sens/morris.csv): mu* per HPPC on samples where both runs stay >= 2.7 V,
and mu* on the whole test (end-of-discharge collapses included), parameters sorted by the first."""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))
OUT = os.path.join(HERE, os.pardir, os.pardir, "results", "exp_LP604765", "sens")
mm = 1 / 25.4

d = pd.read_csv(os.path.join(OUT, "morris.csv")).sort_values("muN_mean", ascending=False)
tests = [("HPPC00", "0 °C"), ("HPPC20", "20 °C"), ("HPPC40", "40 °C")]
y = np.arange(len(d))
fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 90 * mm), layout="constrained", sharey=True)
for i, (k, lab) in enumerate(tests):
    ax[0].barh(y + (i - 1) * 0.27, d["muN_" + k].clip(lower=0.05), height=0.27, label=lab)
    ax[1].barh(y + (i - 1) * 0.27, d["mu_" + k].clip(lower=0.05), height=0.27, label=lab)
ax[0].set_yticks(y, d.param); ax[0].invert_yaxis()
ax[0].set_xscale("log"); ax[1].set_xscale("log")
ax[0].set_xlabel(r"$\mu^*$, $V \geq 2.7$ V [mV / box]"); ax[1].set_xlabel(r"$\mu^*$, whole test [mV / box]")
ax[0].legend(frameon=False, loc="lower right")
for a in ax:
    a.axvline(10, color="0.6", lw=0.5, ls=":")
    a.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(OUT, "morris.png")); fig.savefig(os.path.join(OUT, "morris.pdf"))
