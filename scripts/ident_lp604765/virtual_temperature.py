"""Virtual temperature sensor for the LP604765 tests (no thermocouple on the cell): the uniform cell temperature from a
lumped energy balance driven by the MEASURED current and voltage,

    C dT/dt = Q_gen - hA (T - T_chamber),   Q_gen = I (U(q, T) - V_meas) - I T dU/dT

U the OCV of the cell set at the charge q below the top (q from the OCV-inverted start + Coulomb counting), dU/dT its
entropic coefficient; C and hA those of the fit (lpmodel lumped thermal state). Exact step of the linear ODE with Q_gen
held over each sample. The dynamic model is not used, so its errors do not enter the temperature.

    python virtual_temperature.py --set <cellset.json> --fit <best.json> --tests CAP00A,CAP00V,HPPC00,CYC20,AGE22V

Writes <out>/vtemp_<test>.npz (t [s], T_C [degC], Q_gen [W]) on the dataset's 1-s grid, and fig_vtemp.png/pdf, with the
temperature of lpmodel's own thermal state for comparison (--compare).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import cellset, datasets  # noqa: E402


def cell_temperature(t, I, V, cs, cell, T_ch_C, q0, C=40.0, hA=0.136):
    """Cell temperature [degC] per sample from the measured I [A, discharge > 0] and V [V], and the heat Q_gen [W]."""
    c = cs["cells"][cell]; tp = cs.get("temperature", {})
    dt = np.diff(t, prepend=t[0])
    q = q0 + np.cumsum(I * dt) / 3600.0
    x = c["x100"] - q / c["Q_n"]
    dUdT = tp.get("dUdT0", 0.0) + tp.get("dUdT_n", 0.0) * (x - 0.5)
    U_ref = datasets.ocv(cs, cell, q, cs["T_ref_K"])                 # OCV at T_ref, the entropic term added at T
    T0 = T_ch_C + 273.15
    T = np.empty(len(t)); Q = np.empty(len(t)); Tk = T0
    for k in range(len(t)):
        T[k] = Tk
        U = U_ref[k] + (Tk - cs["T_ref_K"]) * dUdT[k]
        Q[k] = I[k] * (U - V[k]) - I[k] * Tk * dUdT[k]
        e = np.exp(-hA * (dt[k + 1] if k + 1 < len(t) else dt[k]) / C)
        Tk = T0 + (Tk - T0) * e + Q[k] / hA * (1.0 - e)
    return T - 273.15, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(ROOT, "results", "exp_LP604765", "fit_an_Mohtat2020", "lp604765_cellset.json"))
    ap.add_argument("--fit", default=os.path.join(ROOT, "results", "exp_LP604765", "fit_an_Mohtat2020", "best.json"),
                    help="best.json of the fit: th_C, th_hA (and the lpmodel thermal run for --compare)")
    ap.add_argument("--tests", default="CAP00A,CAP00V,HPPC00,CYC20,AGE22V")
    ap.add_argument("--compare", action="store_true", help="also lpmodel's thermal state (the model's own V)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp_LP604765", "vtemp"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cs = cellset.load(a.set); cell = "LP01"
    best = json.load(open(a.fit)); p = best["params"]
    Cth, hA = float(p.get("th_C", 40.0)), float(p.get("th_hA", 0.136))
    print("lumped thermal: C %.1f J/K, hA %.4f W/K, time constant %.0f s" % (Cth, hA, Cth / hA))
    comp = {}
    if a.compare:
        if best.get("argv", {}).get("ocp_p"):
            os.environ["LP_OCP_P"] = best["argv"]["ocp_p"]
        os.environ["LP_THERMAL"] = "1"
        sys.path.insert(0, HERE)
        import common as C_
        from lpmodel import Model
        m = Model(); prof = C_.profiles(a.tests.split(","))
        for k in a.tests.split(","):
            ds, pr = C_.ALL[k], prof[k]
            pk = C_.model_params(C_.params_for(p, ds))
            Vt, _, xs, ys = m.simulate(pk, pr, ds["T"], dq=C_.dq_of(p, ds, pr, m), full=True)
            dUdT = p.get("dUdT0", 0.0) + p.get("dUdT_n", 0.0) * (np.asarray(xs) - 0.5)
            U = np.asarray(m.ocp_p(np.asarray(ys))) - np.asarray(m.ocp_n(np.asarray(xs)))
            T0 = ds["T"] + 273.15; Tm = np.empty(len(Vt)); T = T0
            for i in range(len(Vt)):
                Tm[i] = T
                Qh = pr["I"][i] * (U[i] + (T - 293.15) * dUdT[i] - Vt[i]) - pr["I"][i] * T * dUdT[i]
                e = np.exp(-hA * pr["dt"][i] / Cth); T = T0 + (T - T0) * e + Qh / hA * (1.0 - e)
            comp[k] = (pr["t"], Tm - 273.15)
            os.makedirs(a.out + "_model", exist_ok=True)              # the model's own temperature, same format
            np.savez(os.path.join(a.out + "_model", "vtemp_%s.npz" % k), t=pr["t"], T_C=Tm - 273.15, T_chamber=ds["T"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(ROOT, "journal.mplstyle")); mm = 1 / 25.4
    tests = a.tests.split(",")
    fig, ax = plt.subplots(len(tests), 1, figsize=(170 * mm, 38 * mm * len(tests)), layout="constrained", squeeze=False)
    for j, k in enumerate(tests):
        d = datasets.load("exp_LP604765", k, dt=1.0)
        q0 = datasets.initial_charge("exp_LP604765", d, cs)
        Tc, Q = cell_temperature(d["t"], d["I"], d["V"], cs, cell, d["T_C"], q0, Cth, hA)
        np.savez(os.path.join(a.out, "vtemp_%s.npz" % k), t=d["t"], T_C=Tc, Q_gen=Q, T_chamber=d["T_C"], C=Cth, hA=hA)
        print("%-7s chamber %4.1f C: virtual cell T max %+.2f K, mean %+.2f K; heat max %.2f W" % (
            k, d["T_C"], Tc.max() - d["T_C"], Tc.mean() - d["T_C"], Q.max()), flush=True)
        x = ax[j, 0]; th = d["t"] / 3600
        x.plot(th, Tc, color="#D55E00", lw=1.0, label="virtual sensor (measured I, V)")
        if k in comp:
            x.plot(comp[k][0] / 3600, comp[k][1], color="#0072B2", lw=0.8, ls="--", label="lpmodel thermal state (model V)")
        x.axhline(d["T_C"], color="k", lw=0.5, ls=":")
        x.set_ylabel("T [°C]"); x.set_title(k, fontsize=8, loc="left"); x.spines[["top", "right"]].set_visible(False)
        x2 = x.twinx(); x2.plot(th, d["I"], color="0.7", lw=0.5); x2.set_ylabel("I [A]", color="0.5")
    ax[0, 0].legend(fontsize=7, frameon=False, loc="upper right"); ax[-1, 0].set_xlabel("t [h]")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "fig_vtemp." + e))


if __name__ == "__main__":
    main()
