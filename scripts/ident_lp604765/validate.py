"""Open-loop evaluation of a fitted LP604765 set on every dataset (fit: HPPC 0/20/40 C; held out: 1C cycling at 20 C,
1C capacity discharges at 0/20/40 C) and the figures.

    python validate.py --fit results/fit_v1/best.json

Reported per test: time-weighted RMSE over the whole test and over V >= 3.0 V, max |error|, mean error.
Output: <fit folder>/validation.csv, val_<test>.png/.pdf.
"""
import argparse
import json
import os

import numpy as np

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(C.RES, "fit_v1", "best.json"))
    ap.add_argument("--tests", default=",".join(C.ALL))
    a = ap.parse_args()
    out = os.path.dirname(os.path.abspath(a.fit))
    best = json.load(open(a.fit))
    p = best["params"]
    from lpmodel import Model
    m = Model()
    keys = a.tests.split(",")
    prof = C.profiles(keys)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    rows = []
    for k in keys:
        ds, pr = C.ALL[k], prof[k]
        V, res, xs, ys = m.simulate(C.model_params(p), pr, ds["T"], dq=C.dq_of(p, ds, pr, m), full=True)
        e = (V - pr["V"]) * 1e3
        w = pr["dt"] / pr["dt"].sum()
        ok = np.isfinite(e)
        hi = ok & (pr["V"] >= 3.0)
        row = dict(test=k, role="fit" if k in C.FIT else "held out", T_C=ds["T"], hours=pr["dt"].sum() / 3600,
                   dq_Ah=C.dq_of(p, ds, pr, m),
                   rmse_mV=float(np.sqrt(np.sum(w[ok] * e[ok] ** 2) / np.sum(w[ok]))),
                   rmse_above3V_mV=float(np.sqrt(np.sum(w[hi] * e[hi] ** 2) / np.sum(w[hi]))),
                   max_abs_mV=float(np.nanmax(np.abs(e))), mean_mV=float(np.sum(w[ok] * e[ok]) / np.sum(w[ok])),
                   bad_steps=int(np.sum(~np.isfinite(V)) + np.sum(res > 1e-6)))
        rows.append(row)
        print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
        np.savez(os.path.join(out, "val_%s.npz" % k), t=pr["t"], I=pr["I"], V_meas=pr["V"], V_model=V, x_s=xs, y_s=ys)
        th = pr["t"] / 3600
        fig, ax = plt.subplots(3, 1, figsize=(170 * mm, 110 * mm), layout="constrained", sharex=True,
                               gridspec_kw=dict(height_ratios=[2.2, 1.2, 1]))
        ax[0].plot(th, pr["V"], "k", lw=0.9, label="measured")
        ax[0].plot(th, V, color="C3", lw=0.9, ls="--", label="DFN")
        ax[0].set_ylabel(r"$V$ [V]"); ax[0].legend(frameon=False, loc="lower left")
        ax[0].set_title("%s (%s, %.0f °C): RMSE %.1f mV, above 3 V %.1f mV"
                        % (k, row["role"], ds["T"], row["rmse_mV"], row["rmse_above3V_mV"]), fontsize=8)
        ax[1].plot(th, e, color="C0", lw=0.7); ax[1].axhline(0, color="0.6", lw=0.5)
        lim = max(20.0, np.nanpercentile(np.abs(e), 99) * 1.2)
        ax[1].set_ylim(-lim, lim); ax[1].set_ylabel(r"error [mV]")
        ax[2].plot(th, pr["I"], color="0.3", lw=0.7); ax[2].set_ylabel(r"$I$ [A]"); ax[2].set_xlabel(r"$t$ [h]")
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        fig.savefig(os.path.join(out, "val_%s.png" % k)); fig.savefig(os.path.join(out, "val_%s.pdf" % k))
        plt.close(fig)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "validation.csv"), index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
