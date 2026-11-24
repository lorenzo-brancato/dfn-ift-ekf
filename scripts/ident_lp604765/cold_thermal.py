"""Cold tests (0 C): the identified set in open loop with and without its lumped thermal state, to see how much of the
model error at 0 C (EKF innovation) comes from running the model isothermal at the chamber temperature.

    python cold_thermal.py --fit ../../results/exp_LP604765/fit_an_Mohtat2020/best.json --tests CAP00A,CAP00V,HPPC00

Per test: voltage RMSE [mV] isothermal and with the thermal state (lpmodel, LP_THERMAL=1, the fitted th_C / th_hA),
and the cell temperature of the thermal run, rebuilt with the model's own recursion
(T+ = T0 + (T - T0) e + Q_h / hA (1 - e), Q_h = I (U - V) - I T dU/dT, e = exp(-hA dt / C)). Writes
cold_thermal_<test>.png/pdf and cold_thermal.csv next to --fit.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))
mm = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--tests", default="CAP00A,CAP00V,HPPC00")
    a = ap.parse_args()
    best = json.load(open(a.fit))
    argv = best.get("argv", {})
    if argv.get("ocp_p"):
        os.environ["LP_OCP_P"] = argv["ocp_p"]
    p = best["params"]
    from lpmodel import Model
    os.environ.pop("LP_THERMAL", None); m_iso = Model()
    os.environ["LP_THERMAL"] = "1"; m_th = Model()
    keys = a.tests.split(",")
    prof = C.profiles(keys)
    out = os.path.dirname(os.path.abspath(a.fit))
    rows = []
    for k in keys:
        ds, pr = C.ALL[k], prof[k]
        pk = C.model_params(C.params_for(p, ds))
        dq = C.dq_of(p, ds, pr, m_iso)
        Vi = m_iso.simulate(pk, pr, ds["T"], dq=dq)
        Vt, _, xs, ys = m_th.simulate(pk, pr, ds["T"], dq=dq, full=True)
        # cell temperature of the thermal run (the model's recursion, from its own V and stoichiometries)
        T0 = ds["T"] + 273.15; Cth, hA = p.get("th_C", 40.0), p.get("th_hA", 0.1)
        U = np.asarray(m_th.ocp_p(np.asarray(ys))) - np.asarray(m_th.ocp_n(np.asarray(xs)))
        dUdT = p.get("dUdT0", 0.0) + p.get("dUdT_n", 0.0) * (np.asarray(xs) - 0.5)
        Tc = np.empty(len(Vt)); T = T0
        for i in range(len(Vt)):
            Tc[i] = T
            Qh = pr["I"][i] * (U[i] + (T - 293.15) * dUdT[i] - Vt[i]) - pr["I"][i] * T * dUdT[i]
            ed = np.exp(-hA * pr["dt"][i] / Cth)
            T = T0 + (T - T0) * ed + Qh / hA * (1.0 - ed)
        w = pr["dt"] / pr["dt"].sum()
        rm = lambda V: float(np.sqrt(np.nansum(w * (1e3 * (V - pr["V"])) ** 2) / np.sum(w[np.isfinite(V)])))
        row = dict(test=k, T_C=ds["T"], rmse_iso_mV=rm(Vi), rmse_thermal_mV=rm(Vt), dT_max_K=float(np.max(Tc) - T0),
                   dT_mean_K=float(np.sum(w * (Tc - T0))))
        rows.append(row); print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
        th = pr["t"] / 3600
        fig, ax = plt.subplots(3, 1, figsize=(170 * mm, 120 * mm), layout="constrained", sharex=True,
                               gridspec_kw=dict(height_ratios=[1.3, 1, 0.8]))
        ax[0].plot(th, pr["V"], "k", lw=0.9, label="measured")
        ax[0].plot(th, Vi, color="#0072B2", lw=0.7, label="isothermal (%.1f mV)" % row["rmse_iso_mV"])
        ax[0].plot(th, Vt, color="#D55E00", lw=0.7, label="thermal state (%.1f mV)" % row["rmse_thermal_mV"])
        ax[0].set_ylabel("V [V]"); ax[0].legend(fontsize=7, frameon=False, loc="lower left")
        ax[1].plot(th, 1e3 * (Vi - pr["V"]), color="#0072B2", lw=0.7); ax[1].plot(th, 1e3 * (Vt - pr["V"]), color="#D55E00", lw=0.7)
        ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("model - measured [mV]"); ax[1].set_ylim(-150, 150)
        ax[2].plot(th, Tc - 273.15, color="#D55E00", lw=0.9); ax[2].axhline(ds["T"], color="k", lw=0.5, ls="--")
        ax[2].set_ylabel("cell T [°C]\n(thermal model)"); ax[2].set_xlabel("t [h]")
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        ax[0].set_title("%s at %g °C: identified set, isothermal vs lumped thermal state" % (k, ds["T"]), fontsize=9)
        for e in ("png", "pdf"):
            fig.savefig(os.path.join(out, "cold_thermal_%s.%s" % (k, e)))
        plt.close(fig)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out, "cold_thermal.csv"), index=False)


if __name__ == "__main__":
    main()
