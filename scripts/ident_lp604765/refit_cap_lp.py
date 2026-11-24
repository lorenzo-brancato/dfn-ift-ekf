"""LP604765 ID01: capacity calibrated on one 1C discharge of a held-out test, then the rest of the test with it.

    python refit_cap_lp.py [--fit results/exp_LP604765/fit_v12/best.json] [--thermal]

The held-out tests are later in the life of the cell than the fit set (CAP20: 2026-10-09, before the temperature tests,
1C capacity 1.894 Ah; AGE22V: cycles 5-7 of the 2026-10-22 ageing block). One factor s scales both electrode capacities
(Q_n, Q_p -> s Q_n, s Q_p; x100, y100 kept), so the window width changes and the stoichiometry range does not; the start
follows from the new window (OCV inversion of the rested voltage). Dynamic parameters: the fit's, fixed.
  CAP20F: steps 2-5 of the capacity test (1C discharge from the rested top, 60 s rest, CC-CV charge, 1 h rest);
          calibration = the discharge and the 60 s rest; checked on the CC-CV charge and the rest.
  AGE22V: three 1C cycles; calibration = the first discharge and its rest; checked on the two cycles after it.
s minimises the time-weighted RMSE of the calibration part only.
Output: results/exp_LP604765/refit_cap/<fit tag>/{summary.json, cal_<test>.png}
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=None)
    ap.add_argument("--thermal", action="store_true")
    a = ap.parse_args()
    if a.thermal:
        os.environ["LP_THERMAL"] = "1"
    import common as C
    from dfnekf import datasets as DS
    fit = a.fit or os.path.join(C.RES, "fit_v12", "best.json")
    best = json.load(open(fit))
    if best.get("argv", {}).get("ocp_p"):
        os.environ["LP_OCP_P"] = os.path.abspath(os.path.join(C.HERE, best["argv"]["ocp_p"]))
    from lpmodel import Model, prepare
    from scipy.optimize import minimize_scalar
    m = Model()
    p0 = best["params"]
    DS.LP_TESTS["CAP20F"] = dict(DS.LP_TESTS["CAP20"], end_step=6)
    out = os.path.join(C.RES, "refit_cap", os.path.basename(os.path.dirname(os.path.abspath(fit)))); os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(C.HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    summ = {}
    for key in ("CAP20F", "AGE22V"):
        ds = DS.LP_TESTS[key]
        d = DS.lp_cut(key)
        pr = prepare(d, start_step=ds.get("start_step", 0), end_step=ds.get("end_step")); pr["V_rest"] = d["V_rest"]
        T = ds["T"]
        t = np.cumsum(pr["dt"]); I = pr["I"]
        n_cal = int(np.flatnonzero(I < -0.1)[0])             # up to the first charge: the first 1C discharge and its rest
        n = len(I)
        pc = {k: (v[:n_cal] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in pr.items()}

        def run(s, prof):
            p = dict(p0, Q_n=p0["Q_n"] * s, Q_p=p0["Q_p"] * s)
            dq = m.dq_from_ocv(C.model_params(p), T, prof["V_rest"])
            V, F, _, _ = m.simulate(C.model_params(p), prof, T, dq=dq, full=True)
            return V, F, dq

        def rmse(e, w, k):
            ok = k & np.isfinite(e)
            return float(np.sqrt(np.sum(w[ok] * e[ok] ** 2) / np.sum(w[ok]))) if ok.sum() > 0.98 * k.sum() else 1e3

        def cal_cost(s):
            V, F, _ = run(s, pc)
            return rmse((V - pc["V"]) * 1e3, pc["dt"], np.ones(n_cal, bool))

        sol = minimize_scalar(cal_cost, bounds=(0.85, 1.15), method="bounded", options=dict(xatol=5e-4))
        cal = np.arange(n) < n_cal
        res = {}
        Vs = {}
        for name, s in (("before", 1.0), ("after", float(sol.x))):
            V, F, dq = run(s, pr)
            e = (V - pr["V"]) * 1e3
            Vs[name] = V
            res[name] = dict(s=s, dq_Ah=float(dq), rmse_all_mV=rmse(e, pr["dt"], np.ones(n, bool)),
                             rmse_cal_mV=rmse(e, pr["dt"], cal), rmse_check_mV=rmse(e, pr["dt"], ~cal),
                             rmse_check_above3V_mV=rmse(e, pr["dt"], ~cal & (pr["V"] >= 3.0)),
                             bad=int(np.sum(~np.isfinite(V)) + np.sum(F > 1e-6)))
        Qw = lambda s: float(np.diff(np.interp([4.2, 3.0], m.ocv(dict(p0, Q_n=p0["Q_n"] * s, Q_p=p0["Q_p"] * s), 20.0,
                                                                np.linspace(-0.3, 3.0, 3301))[::-1],
                                                np.linspace(-0.3, 3.0, 3301)[::-1]))[0])
        summ[key] = dict(s=float(sol.x), window_3_42V_Ah_before=Qw(1.0), window_3_42V_Ah_after=Qw(float(sol.x)),
                         cal_hours=float(pr["dt"][:n_cal].sum() / 3600), check_hours=float(pr["dt"][n_cal:].sum() / 3600), **res)
        print(key, json.dumps(summ[key], indent=1), flush=True)
        th = t / 3600
        fig, ax = plt.subplots(2, 1, figsize=(170 * mm, 100 * mm), layout="constrained", sharex=True)
        ax[0].plot(th, pr["V"], "k", lw=0.9, label="measured")
        ax[0].plot(th, Vs["before"], "C0", lw=0.8, ls="--", label="fit capacity (s = 1)")
        ax[0].plot(th, Vs["after"], "C3", lw=0.8, label="calibrated (s = %.4f)" % sol.x)
        ax[0].axvspan(0, th[n_cal - 1], color="0.92", lw=0); ax[0].set_ylabel("V [V]"); ax[0].legend(frameon=False, fontsize=6)
        ax[0].set_title("%s: capacity calibrated on the shaded 1C discharge, checked on the rest" % key, fontsize=7)
        ax[1].plot(th, 1e3 * (Vs["before"] - pr["V"]), "C0", lw=0.7, ls="--")
        ax[1].plot(th, 1e3 * (Vs["after"] - pr["V"]), "C3", lw=0.7); ax[1].axhline(0, color="0.6", lw=0.5)
        ax[1].set_ylim(-150, 150); ax[1].set_ylabel("model - data [mV]"); ax[1].set_xlabel("t [h]")
        fig.savefig(os.path.join(out, "cal_%s.png" % key), dpi=200); plt.close(fig)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
