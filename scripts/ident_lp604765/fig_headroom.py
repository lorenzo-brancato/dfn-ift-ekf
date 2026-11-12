"""Charging headroom from the EKF (scripts/ekf_exp.py --headroom-every-s): the largest constant charging current that
keeps the anode lithium deposition potential LDP >= 0 over the next horizon (plating limit), and V <= 4.2 V (voltage
limit), from the estimated state every few minutes.

    python fig_headroom.py --runs <results>/headroom --tests CYC20,AGE22V

Rows: (a) charging current: measured (black) against the plating limit (red) and the voltage limit (grey); the shaded
band is the safe envelope (below both); (b) LDP at the measured current and at rest [mV vs Li/Li+]; (c) SOC.
Output: <runs>/fig_headroom.png/.pdf
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tests", default="CYC20,AGE22V")
    ap.add_argument("--tag", default="e+0.00")
    ap.add_argument("--c-rate-ah", type=float, default=1.893, help="[A] 1C, for the right axis")
    a = ap.parse_args()
    tests = a.tests.split(",")
    fig, ax = plt.subplots(3, len(tests), figsize=(180 * mm, 120 * mm), sharex="col", sharey="row", layout="constrained",
                           squeeze=False, gridspec_kw=dict(height_ratios=[2, 1.4, 1]))
    for j, test in enumerate(tests):
        d = np.load(os.path.join(a.runs, "ekf_%s_%s.npz" % (test, a.tag)), allow_pickle=True)
        th = d["t"] / 3600; ht = d["hr_t"] / 3600; imax = float(d["hr_imax"])
        bad = ~np.isfinite(d["hr_ldp_rest"]) | (np.abs(d["hr_ldp_rest"]) > 1.5)   # forward charge failed (near full): dropped
        Ipl = np.where(bad, np.nan, np.where(np.isinf(d["hr_I_pl"]), imax, d["hr_I_pl"]))
        IV = np.where(bad, np.nan, np.where(np.isinf(d["hr_I_V"]), imax, d["hr_I_V"]))
        env = np.minimum(Ipl, IV)
        ax[0, j].fill_between(ht, 0, env, step="mid", color="C2", alpha=0.18, lw=0, label="safe envelope")
        ax[0, j].step(ht, Ipl, "C3", where="mid", lw=1.0, label="plating limit (LDP $\\geq$ 0, 5 min)")
        ax[0, j].step(ht, IV, color="0.5", where="mid", lw=0.8, ls="--", label="voltage limit (4.2 V, 5 min)")
        ax[0, j].plot(th, np.clip(-d["I"], 0, None), "k", lw=0.8, label="measured charging current")
        ax[0, j].set_ylim(0, imax * 1.02)
        ldp, ldp0 = 1e3 * np.where(bad, np.nan, d["hr_ldp"]), 1e3 * np.where(bad, np.nan, d["hr_ldp_rest"])
        ax[1, j].plot(ht, ldp0, color="0.6", lw=0.8, label="min over 5 min at rest")
        ax[1, j].plot(ht, ldp, "C0", lw=0.9, label="at the measured current")
        ax[1, j].axhline(0, color="C3", lw=0.8)
        ax[1, j].text(0.99, 0.04, "min %.0f mV; %d of %d states dropped" % (np.nanmin(ldp), bad.sum(), len(bad)), transform=ax[1, j].transAxes, ha="right", fontsize=6)
        ax[2, j].fill_between(th, 100 * (d["soc"] - 2 * d["sd_soc"]), 100 * (d["soc"] + 2 * d["sd_soc"]), color="C0", alpha=0.25, lw=0)
        ax[2, j].plot(th, 100 * d["soc"], "C0", lw=0.8)
        ax[0, j].set_title(test, fontsize=8); ax[2, j].set_xlabel("t [h]")
        sec = ax[0, j].secondary_yaxis("right", functions=(lambda i: i / a.c_rate_ah, lambda c: c * a.c_rate_ah))
        if j == len(tests) - 1:
            sec.set_ylabel("C-rate")
    ax[0, 0].set_ylabel("charging current [A]"); ax[1, 0].set_ylabel("LDP [mV vs Li]"); ax[2, 0].set_ylabel("SOC [%]")
    ax[1, 0].set_ylim(-20, 260)
    ax[0, 0].legend(fontsize=5.5, loc="upper left"); ax[1, 0].legend(fontsize=5.5, loc="upper left")
    out = os.path.join(a.runs, "fig_headroom")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
