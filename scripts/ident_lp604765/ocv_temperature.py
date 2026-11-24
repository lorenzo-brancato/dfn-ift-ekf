"""OCV of ID01 against temperature from the rests of the three HPPCs (0 / 20 / 40 C), and the entropic coefficient.

    python ocv_temperature.py [--step1 <results>/step1_ocv_c20_an_Mohtat2020.json]

Points: every 1 h rest of HPPC00, HPPC20, HPPC40 (step1_ocv.ocv_points), q [A.h] discharged since the rested top of the
test. The tests start at slightly different states (each top is reached by a CC-CV at its own temperature), so q is
aligned with the start offsets dq_T of the step-1 fit (dq_20 = 0); the variant dq_T = 0 shows how much the alignment
matters. Relaxation: each rest's tail (last 30 min) is fitted with V = V_inf + b / sqrt(t) (diffusion-limited tail),
which gives the rate at the end of the rest and the voltage still to come.
Entropic coefficient per point: e_i = (V_T,i - V_20(q_i)) / (T - 20 C), with V_20 the 20 C points interpolated along
the shape of the step-1 OCV (model at 20 C + PCHIP of its residuals at the 20 C points); two estimates per state of
charge (0 -> 20 C and 20 -> 40 C), compared with the step-1 law dUdT0 + dUdT_n (x - 0.5).
Output: results/exp_LP604765/ocv_T/{ocv_T.json, ocv_T.png, ocv_T.pdf}.
"""
import argparse
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

import common as C
from lpdata import load_test

TESTS = {"HPPC00": 0.0, "HPPC20": 20.0, "HPPC40": 40.0}
COL = {"HPPC00": "C0", "HPPC20": "k", "HPPC40": "C3"}


def rests(test, min_rest=1800.0, tail=1800.0):
    """Rested points with the relaxation tail: q, V_end, rest length, V_inf, rate at the end [mV/h]."""
    d = load_test(test)
    t, V, I, st = d["t"], d["V"], d["I"], d["step"]
    q = np.cumsum(I * np.r_[np.diff(t), 0.0]) / 3600.0
    out, first = [], None
    for k, n in enumerate(d["names"]):
        m = np.where(st == k)[0]
        if len(m) == 0 or "Relaxation" not in n or t[m[-1]] - t[m[0]] < min_rest:
            continue
        if first is None:
            first = q[m[-1]]
        tr = t[m] - t[m[0]] + 1.0
        sel = tr > tr[-1] - tail
        try:
            (vinf, b), _ = curve_fit(lambda x, a_, b_: a_ + b_ / np.sqrt(x), tr[sel], V[m][sel], p0=(V[m][-1], 0.0))
        except RuntimeError:
            vinf, b = np.nan, np.nan
        rate = -0.5 * b * tr[-1] ** -1.5 * 3600 * 1e3       # dV/dt at the end of the rest [mV/h]
        out.append(dict(q=float(q[m[-1]] - first), V=float(V[m][-1]), rest_s=float(tr[-1]), V_inf=float(vinf),
                        rate_mV_h=float(rate), still_mV=float(1e3 * (vinf - V[m][-1]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", default=os.path.join(C.RES, "step1_ocv_c20_an_Mohtat2020.json"))
    a = ap.parse_args()
    s1 = json.load(open(a.step1))
    os.environ.setdefault("LP_OCP_P", s1["ocp_p"])
    if "an_Mohtat2020" in a.step1:
        os.environ.setdefault("LP_OCP_N", os.path.join(C.RES, "ocp_neg", "ocp_neg_Mohtat2020.json"))
    from lpmodel import Model
    m = Model()
    p = s1["params"]
    dq = {"HPPC20": 0.0, "HPPC40": p["dq_40"], "HPPC00": p["dq_00"]}
    pts = {k: rests(C.ALL[k]["test"]) for k in TESTS}
    out = dict(step1=a.step1, dq=dq, points={})
    for k, T in TESTS.items():
        for r in pts[k]:
            r["q_al"] = r["q"] + dq[k]
            r["x"] = p["x100"] - r["q_al"] / p["Q_n"]
            r["ocv_model"] = float(m.ocv(p, T, r["q_al"]))
        out["points"][k] = pts[k]

    # the 20 C reference along the model's shape
    def ref20(which, shift0=False):
        r20 = pts["HPPC20"]
        qq = np.array([r["q"] for r in r20]); vv = np.array([r[which] for r in r20])
        res = vv - np.array([float(m.ocv(p, 20.0, x)) for x in qq])
        f = PchipInterpolator(qq, res, extrapolate=False)

        def V20(q):
            rr = f(np.clip(q, qq[0], qq[-1]))
            return float(m.ocv(p, 20.0, q)) + float(rr)
        return V20

    ent = {}
    for which in ("V", "V_inf"):
        V20 = ref20(which)
        for align in (True, False):
            lab = "%s_%s" % (which, "aligned" if align else "dq0")
            ent[lab] = {}
            for k in ("HPPC00", "HPPC40"):
                T = TESTS[k]
                rows = []
                for r in pts[k]:
                    qa = r["q"] + (dq[k] if align else 0.0)
                    rows.append(dict(q=qa, x=p["x100"] - qa / p["Q_n"], e_mV_K=1e3 * (r[which] - V20(qa)) / (T - 20.0)))
                ent[lab][k] = rows
    out["entropic"] = ent
    out["law"] = dict(dUdT0=p["dUdT0"], dUdT_n=p["dUdT_n"])
    od = os.path.join(C.RES, "ocv_T"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "ocv_T.json"), "w"), indent=1)
    # the measured entropic coefficient as a model input (lpmodel LP_DUDT_TABLE): dOCV/dT against the anode
    # stoichiometry, the 0 -> 20 C estimate below 20 C and the 20 -> 40 C one above (aligned rests, end of the 1 h rest)
    tab = {}
    for side, k in (("lo", "HPPC00"), ("hi", "HPPC40")):
        rows = sorted(ent["V_aligned"][k], key=lambda r: r["x"])
        tab["x_" + side] = [r["x"] for r in rows]; tab["dUdT_" + side] = [1e-3 * r["e_mV_K"] for r in rows]
    # one curve for all temperatures: the mean of the two estimates (each linearly interpolated in x, held constant
    # outside its range) on the union of their x points; the two separate estimates are kept as x_lo/x_hi_sep
    xg = np.unique(np.r_[tab["x_lo"], tab["x_hi"]])
    mean = 0.5 * (np.interp(xg, tab["x_lo"], tab["dUdT_lo"]) + np.interp(xg, tab["x_hi"], tab["dUdT_hi"]))
    tab.update({k_ + "_sep": v_ for k_, v_ in tab.items()})
    tab.update(x_lo=list(xg), dUdT_lo=list(mean), x_hi=list(xg), dUdT_hi=list(mean))
    tab.update(T_ref_C=20.0, units="V/K", kind="mean of the 0->20 C and 20->40 C estimates",
               source="ocv_temperature.py, HPPC00/20/40 rests aligned with the step-1 dq", step1=a.step1)
    json.dump(tab, open(os.path.join(od, "dUdT_table.json"), "w"), indent=1)

    # ---- print
    print("rests (end of 1 h): V, rate at the end, voltage still to come (V_inf - V)")
    for k in TESTS:
        print(k, "  ".join("q %.2f V %.4f (%.1f mV/h, +%.1f mV)" % (r["q_al"], r["V"], r["rate_mV_h"], r["still_mV"]) for r in pts[k]))
    for lab, d in ent.items():
        print("entropic [mV/K], %s:" % lab)
        for k, rows in d.items():
            print("  %s: %s" % (k, "  ".join("x %.3f %+.3f" % (r["x"], r["e_mV_K"]) for r in rows)))
    print("step-1 law: dUdT0 %+.3f mV/K, dUdT_n %+.3f mV/K per unit x" % (1e3 * p["dUdT0"], 1e3 * p["dUdT_n"]))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(C.HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(2, 2, figsize=(170 * mm, 120 * mm), layout="constrained")
    qq = np.linspace(-0.05, 2.15, 400)
    for k, T in TESTS.items():
        q_ = [r["q_al"] for r in pts[k]]
        ax[0, 0].plot(q_, [r["V"] for r in pts[k]], "o", ms=3.5, color=COL[k], label="%d °C" % T)
        ax[0, 0].plot(qq, m.ocv(p, T, qq), "-", lw=0.7, color=COL[k])
        ax[0, 1].plot(q_, [1e3 * (r["V"] - r["ocv_model"]) for r in pts[k]], "o-", ms=3, lw=0.6, color=COL[k], label="%d °C" % T)
        ax[1, 1].plot(q_, [r["rate_mV_h"] for r in pts[k]], "o-", ms=3, lw=0.6, color=COL[k], label="%d °C" % T)
    ax[0, 0].set_xlabel("q from the top of HPPC20 [Ah]"); ax[0, 0].set_ylabel("rested voltage [V]")
    ax[0, 0].legend(frameon=False, fontsize=7); ax[0, 0].set_title("(a) end of the 1 h rests; lines: step-1 OCV(q, T)", fontsize=7)
    ax[0, 1].axhline(0, color="0.6", lw=0.5); ax[0, 1].set_xlabel("q [Ah]"); ax[0, 1].set_ylabel("data - step-1 OCV [mV]")
    ax[0, 1].set_title("(b) residual of the step-1 OCV (linear entropic law)", fontsize=7)
    xx = np.linspace(0.05, 0.7, 50)
    ax[1, 0].plot(xx, 1e3 * (p["dUdT0"] + p["dUdT_n"] * (xx - 0.5)), "-", color="0.4", lw=0.8, label="step-1 law")
    for lab, ls, mk in (("V_aligned", "-", "o"), ("V_dq0", ":", "x")):
        for k in ("HPPC00", "HPPC40"):
            rows = ent[lab][k]
            ax[1, 0].plot([r["x"] for r in rows], [r["e_mV_K"] for r in rows], ls=ls, marker=mk, ms=3, lw=0.6, color=COL[k],
                          label="%s, %s" % ("0→20 °C" if k == "HPPC00" else "20→40 °C",
                                            {"V_aligned": "aligned", "V_dq0": "no alignment", "V_inf_aligned": "relaxed (V∞)"}[lab]))
    tb_ = json.load(open(os.path.join(od, "dUdT_table.json")))
    ax[1, 0].plot(tb_["x_lo"], 1e3 * np.array(tb_["dUdT_lo"]), "-", color="C2", lw=1.6, label="mean (model input)")
    ax[1, 0].axhline(0, color="0.6", lw=0.5)
    ax[1, 0].set_xlabel("anode stoichiometry x"); ax[1, 0].set_ylabel("dOCV/dT [mV/K]"); ax[1, 0].invert_xaxis()
    ax[1, 0].legend(frameon=False, fontsize=5.5, ncol=2); ax[1, 0].set_title("(c) entropic coefficient per rest", fontsize=7)
    ax[1, 1].set_xlabel("q [Ah]"); ax[1, 1].set_ylabel("dV/dt at the end of the rest [mV/h]")
    ax[1, 1].set_title("(d) relaxation still going at 1 h", fontsize=7); ax[1, 1].axhline(0, color="0.6", lw=0.5)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(od, "ocv_T." + e), dpi=200)
    print("wrote", os.path.join(od, "ocv_T.png"))


if __name__ == "__main__":
    main()
