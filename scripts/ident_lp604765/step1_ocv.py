"""Step 1 for the LP604765: rested OCV points and the electrode window for candidate OCP pairs.

OCV points: the last sample of every rest >= 1800 s (HPPC 1 h, EIS 3 h), with q [A.h] discharged since the end of the
first rest at the top of charge (CC-CV to C/20). Window: V_ocv(q) = U_p(y100 + q/Q_p) - U_n(x100 - q/Q_n).

    python step1_ocv.py                  # fit every candidate on the 20 C points, write results/step1/*.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

from lpdata import load_test, temperature_of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, os.pardir, "results", "exp_LP604765", "step1")
TESTS = ["20261011_HPPC_20°C", "20261019_EIS_20°C_BoL", "20261014_HPPC_40°C", "20261021_HPPC_00°C"]


def ocv_points(test, min_rest=1800.0):
    d = load_test(test)
    t, V, I, st = d["t"], d["V"], d["I"], d["step"]
    dt = np.r_[np.diff(t), 0.0]
    q = np.cumsum(I * dt) / 3600.0
    rows = []
    first = None
    for k, n in enumerate(d["names"]):
        m = np.where(st == k)[0]
        if len(m) == 0 or "Relaxation" not in n:
            continue
        if t[m[-1]] - t[m[0]] < min_rest:
            continue
        if first is None:
            first = q[m[-1]]
        rows.append((q[m[-1]] - first, V[m[-1]], t[m[-1]] - t[m[0]]))
    return np.array(rows)


def ocp_candidates():
    import pybamm
    cands = {}
    for name in ["Mohtat2020", "Ai2020", "Chen2020", "Ecker2015", "Marquis2019"]:
        p = pybamm.ParameterValues(name)
        x = np.linspace(1e-4, 1 - 1e-4, 2001)

        def tab(f):
            v = f(pybamm.Vector(x))
            v = v.evaluate() if hasattr(v, "evaluate") else v
            return np.broadcast_to(np.asarray(v, dtype=float).ravel(), x.shape).copy()
        cands[name] = dict(x=x, Un=tab(p["Negative electrode OCP [V]"]), Up=tab(p["Positive electrode OCP [V]"]))
    return cands


def ocv_model(c, w, q):
    x100, y100, Qn, Qp = w
    x = np.clip(x100 - q / Qn, c["x"][0], c["x"][-1])
    y = np.clip(y100 + q / Qp, c["x"][0], c["x"][-1])
    return np.interp(y, c["x"], c["Up"]) - np.interp(x, c["x"], c["Un"])


def fit(c, q, V, q_full=1.95, n_start=200, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    lo = np.array([0.5, 0.0, q_full * 1.0, q_full * 1.0])
    hi = np.array([1.0, 0.7, q_full * 2.5, q_full * 3.5])

    def res(w):
        r = (ocv_model(c, w, q) - V) * 1e3
        # the window must stay inside both electrodes over the full discharge
        x_end, y_end = w[0] - q_full / w[2], w[1] + q_full / w[3]
        pen = [max(0.0, 1e-3 - x_end) * 1e4, max(0.0, y_end - 0.999) * 1e4]
        return np.r_[r, pen]
    for _ in range(n_start):
        w0 = lo + rng.random(4) * (hi - lo)
        try:
            s = least_squares(res, w0, bounds=(lo, hi), x_scale=hi - lo)
        except ValueError:
            continue
        if best is None or s.cost < best.cost:
            best = s
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    pts = {}
    for test in TESTS:
        p = ocv_points(test)
        pts[test] = p
        print(test, "T=%s" % temperature_of(test))
        for r in p:
            print("   q %.4f Ah  V %.4f  (rest %.0f s)" % tuple(r))
    q20 = np.r_[pts[TESTS[0]][:, 0], pts[TESTS[1]][:, 0]]
    V20 = np.r_[pts[TESTS[0]][:, 1], pts[TESTS[1]][:, 1]]
    cands = ocp_candidates()
    summary = {}
    for name, c in cands.items():
        s = fit(c, q20, V20)
        w = s.x
        r = (ocv_model(c, w, q20) - V20) * 1e3
        summary[name] = dict(x100=w[0], y100=w[1], Q_n=w[2], Q_p=w[3], rmse_mV=float(np.sqrt(np.mean(r ** 2))),
                             max_mV=float(np.max(np.abs(r))), x0=w[0] - 1.95 / w[2], y0=w[1] + 1.95 / w[3])
        print("%-12s rmse %.1f mV max %.1f  x100 %.3f y100 %.3f Qn %.3f Qp %.3f  x_end %.3f y_end %.3f"
              % (name, summary[name]["rmse_mV"], summary[name]["max_mV"], *w, summary[name]["x0"], summary[name]["y0"]))
    json.dump(dict(points={k: v.tolist() for k, v in pts.items()}, fits=summary),
              open(os.path.join(OUT, "step1_candidates.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "ocp_tables.npz"), **{f"{k}_{f}": v[f] for k, v in cands.items() for f in ("x", "Un", "Up")})


if __name__ == "__main__":
    main()
