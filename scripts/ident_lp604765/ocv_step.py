"""OCV step of the two-step identification of ID01: the electrode window and the entropic terms from the rested points.

    python ocv_step.py --ocp-p <results>/ocp/ocp_p_c20.json --tag c20

U_p(y): the cathode table by subtraction on the C/20 discharge (ocp_c20.py), U_n(x): Mohtat2020 graphite. Points: the
last sample of every rest >= 30 min of the three HPPCs (1 h rests, 0 / 20 / 40 C), q [A.h] from the first rest at the
top of charge of each test. Model (lpmodel.Model.ocv):
    V(q, T) = U_p(y100 + (q + dq_T) / Q_p) - U_n(x100 - (q + dq_T) / Q_n) + (T - 20 C)(dUdT0 + dUdT_n (x - 0.5))
with dq_20 = 0 (the window's top is the rested top of HPPC20) and dq_40, dq_00 the start of the other two tests.
Fitted: x100, y100, Q_n, Q_p, dUdT0, dUdT_n, dq_40, dq_00 (least squares, multi-start); Gauss-Newton standard errors
from the residual's RMS. Window ends at the cut-off voltages (4.2 V, 3.0 V) reported as x0, y0 (from 3.0 V).
Output: <results>/step1_ocv_<tag>.json (params in lpmodel names, fixed by fit.py --fix in the dynamic step).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
import common as C  # noqa: E402
from step1_ocv import ocv_points  # noqa: E402

TESTS = {"HPPC20": 20.0, "HPPC40": 40.0, "HPPC00": 0.0}
NAMES = ["x100", "y100", "Q_n", "Q_p", "dUdT0", "dUdT_n", "dq_40", "dq_00"]
LO = np.array([0.4, 0.3, 2.0, 2.0, -3e-3, -3e-3, -0.3, -0.3])
HI = np.array([1.0, 0.6, 7.0, 6.0, 3e-3, 3e-3, 0.3, 0.4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocp-p", required=True)
    ap.add_argument("--tag", default="c20")
    ap.add_argument("--vmin", type=float, default=3.0, help="cut-off for the window ends x0, y0 [V]")
    ap.add_argument("--fix", default="", help="groups held at a value, e.g. dUdT_n=0.285e-3 (not identifiable on the rests)")
    ap.add_argument("--discharge-branch", action="store_true",
                    help="only the rests after a discharge step (consistent with the C/20 discharge U_p; OCV hysteresis)")
    a = ap.parse_args()
    global NAMES, LO, HI
    fixed = {k: float(v) for k, v in (it.split("=") for it in a.fix.split(",") if it)}
    keep = [i for i, n in enumerate(NAMES) if n not in fixed]
    NAMES, LO, HI = [NAMES[i] for i in keep], LO[keep], HI[keep]
    os.environ["LP_OCP_P"] = os.path.abspath(a.ocp_p)
    from lpmodel import Model
    m = Model()
    pts = []
    for k, T in TESTS.items():
        for q, V, dur in ocv_points(C.ALL[k]["test"]):
            if a.discharge_branch and abs(q) < 1e-6:
                continue                     # the first rest (top of charge, after CC-CV) is on the charge branch
            pts.append((k, T, q, V))
    key = np.array([p[0] for p in pts]); T = np.array([p[1] for p in pts]); q = np.array([p[2] for p in pts]); V = np.array([p[3] for p in pts])

    def unpack(w):
        return {**fixed, **dict(zip(NAMES, w))}

    def model(w):
        p = unpack(w)
        dq = np.where(key == "HPPC40", p["dq_40"], np.where(key == "HPPC00", p["dq_00"], 0.0))
        return np.array([m.ocv(p, Ti, qi + di) for Ti, qi, di in zip(T, q, dq)])

    res = lambda w: (model(w) - V) * 1e3
    rng = np.random.default_rng(0)
    best = None
    for k in range(120):
        w0 = np.clip(np.array([dict(x100=0.65, y100=0.40, Q_n=4.0, Q_p=3.9, dUdT0=0.0, dUdT_n=0.0, dq_40=0.0, dq_00=0.05)[n] for n in NAMES]), LO, HI) if k == 0 else LO + rng.random(len(LO)) * (HI - LO)
        try:
            s = least_squares(res, w0, bounds=(LO, HI), x_scale=HI - LO)
        except ValueError:
            continue
        if best is None or s.cost < best.cost:
            best = s
    w = best.x; r = res(w)
    rms = float(np.sqrt(np.mean(r ** 2)))
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * rms ** 2 * len(r) / max(len(r) - len(w), 1)
        sd = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sd = np.full(len(w), np.nan)
    p = unpack(w)
    # window ends at the cut-offs (20 C): q range of the OCV between vmin and 4.2 V
    qq = np.linspace(-0.3, 3.0, 6601)
    v = m.ocv(p, 20.0, qq)
    inw = (v <= 4.2) & (v >= a.vmin)
    q_top, q_bot = float(qq[inw].min()), float(qq[inw].max())
    out = dict(params={k_: float(v_) for k_, v_ in p.items()}, fixed=fixed, sd={k_: float(s_) for k_, s_ in zip(NAMES, sd)},
               at_bound=[n for n, x, lo, hi in zip(NAMES, w, LO, HI) if x <= lo + 1e-9 or x >= hi - 1e-9],
               rms_mV=rms, per_test_rms_mV={k_: float(np.sqrt(np.mean(r[key == k_] ** 2))) for k_ in TESTS},
               n_points=int(len(r)), ocp_p=os.path.abspath(a.ocp_p),
               window_ends=dict(vmin=a.vmin, q_top=q_top, q_bot=q_bot, Q_window_Ah=q_bot - q_top,
                                x_top=p["x100"] - q_top / p["Q_n"], y_top=p["y100"] + q_top / p["Q_p"],
                                x0=p["x100"] - q_bot / p["Q_n"], y0=p["y100"] + q_bot / p["Q_p"]),
               residuals_mV=[dict(test=k_, T=float(t_), q=float(q_), r=float(rr)) for k_, t_, q_, rr in zip(key, T, q, r)])
    f = os.path.join(C.RES, "step1_ocv_%s.json" % a.tag)
    json.dump(out, open(f, "w"), indent=1)
    print("OCV step (%d rested points): rms %.1f mV (%s); at bound %s" % (len(r), rms, ", ".join("%s %.1f" % kv for kv in out["per_test_rms_mV"].items()), out["at_bound"]))
    for n, x, s_ in zip(NAMES, w, sd):
        print("  %-7s %.5g  (sd %.2g)" % (n, x, s_))
    print("  window at 20 C between %.1f and 4.2 V: %.3f Ah (x %.3f..%.3f, y %.3f..%.3f)" % (a.vmin, q_bot - q_top, out["window_ends"]["x0"], out["window_ends"]["x_top"], out["window_ends"]["y_top"], out["window_ends"]["y0"]))
    print("wrote", f)


if __name__ == "__main__":
    main()
