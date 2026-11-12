"""Charging current at which the anode plating potential (phi_s - phi_e - R_f j_n at the anode/separator interface,
V vs Li/Li+; step.DFN.interface) reaches 0, with the identified LP604765 set, in open loop.

    python plating_threshold.py --set ../../results/exp_LP604765/fit_v6b_r2/lp604765_cellset.json --T 20,10,0

Protocol per temperature and charging current (as the ageing block): rested at the top of the window, 1C discharge
(1.893 A) for 3600 s or to 2.75 V, 300 s rest, CC charge at the current to 4.2 V. Two estimates:
  full charge    minimum of the plating potential over each CC charge; the threshold is where it crosses 0
  instantaneous  the state frozen at the end of the reference CC charge (--i-ref), current increased until it crosses 0
Writes plating_threshold.csv and fig_plating_threshold.png/pdf in --out.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import runtime  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dfnekf import cellset  # noqa: E402
from dfnekf.step import DFN  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4
Q_REF = 1.95       # [A.h] the window of the identification (1C)


def model(cs, cell, T_C, soft_eps, dims=(10, 6, 10, 12)):
    P = cellset.params_for(cs, cell, T_C + 273.15, film_share=0.5, ds_on="p")
    P["soft_eps"] = soft_eps
    m = DFN(P, *dims, 2, 1.0, "4:1")
    th1 = jnp.array([1.0, 1.0, 1.0])
    A_ = m.A

    @jax.jit
    def step(z, g, I, th=th1):
        csn, csp, ce = m.surf(z)
        imp = m.imp_coeffs(z, th)
        g, _, _ = A_.solve_robust(g, csn, csp, ce, I, th, tol=1e-10, imp=imp)
        V = A_.V_of(g, I, csn, csp)
        _, ppl = m.interface(g, csn, th)
        dphi = ppl + P["Rf0"] * th[0] * m.extrap_right(g[:m.Nn])        # phi_s - phi_e, the film drop included
        return m.advance(z, g, th), g, V, jnp.stack([ppl, dphi])

    @jax.jit
    def frozen(z, g, I):                     # closure only: potential at the state z for the current I
        th = th1
        csn, csp, ce = m.surf(z)
        g, _, _ = A_.solve_robust(g, csn, csp, ce, I, th, tol=1e-10, imp=m.imp_coeffs(z, th))
        return g, m.interface(g, csn, th)[1], A_.V_of(g, I, csn, csp)

    def g0(z, I, th=th1):
        csn, csp, ce = m.surf(z)
        return A_.newton(A_.guess(csn, csp, I, th), csn, csp, ce, I, th, imp=m.imp_coeffs(z, th))[0]
    return m, step, frozen, g0


def protocol(m, step, g0, z, I_chg, v_max=4.2, v_min=2.75, i_dis=1.893, rf=1.0):
    """Discharge (<= 3600 s), rest 300 s, CC charge at I_chg [A] to v_max, with the film resistance R_f0 x rf. Returns the
    charge's t [s], V, phi_pl = phi_s - phi_e - R_f j_n and dphi = phi_s - phi_e at the separator, q [A.h], and the state
    at the end of CC."""
    th = jnp.array([rf, 1.0, 1.0])
    g = g0(z, i_dis, th)
    for k in range(3600):
        z, g, V, _ = step(z, g, i_dis, th)
        if float(V) <= v_min:
            break
    for k in range(300):
        z, g, V, _ = step(z, g, 0.0, th)
    t, Vs, ph, dp = [], [], [], []
    k = 0
    while k < int(3 * 3600 * 1.95 / I_chg) + 600:
        z2, g2, V, pp = step(z, g, -I_chg, th)
        if not np.isfinite(float(V)) or float(V) >= v_max:
            break
        t.append(k); Vs.append(float(V)); ph.append(float(pp[0])); dp.append(float(pp[1])); z, g = z2, g2; k += 1
    t = np.array(t, float)
    return dict(t=t, V=np.array(Vs), ppl=np.array(ph), dphi=np.array(dp), q=I_chg * t / 3600), z, g


def crossing(x, y):
    """First x where y (decreasing in x) crosses 0, by linear interpolation; nan if it does not."""
    s = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if not len(s):
        return np.nan
    i = s[0]
    return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(ROOT, "results", "exp_LP604765", "fit_v6b_r2", "lp604765_cellset.json"))
    ap.add_argument("--cell", default="LP01")
    ap.add_argument("--T", default="20,10,0", help="temperatures [C]")
    ap.add_argument("--rates", default="1,1.5,2,2.5,3,4,5", help="CC charging rates [C of 1.95 A.h]")
    ap.add_argument("--i-ref", type=float, default=1.92, help="[A] reference CC charge (the ageing block's)")
    ap.add_argument("--soft-eps", type=float, default=1e-2)
    ap.add_argument("--rf-mults", default="1,2,3,5,8,12,20,30,50", help="film resistance multipliers for the R_f sweep at --i-ref")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "exp_LP604765", "plating"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cs = cellset.load(a.set)
    c = cs["cells"][a.cell]
    rates = [float(r) for r in a.rates.split(",")]
    rows, curves, inst = [], {}, {}
    for T in [float(v) for v in a.T.split(",")]:
        m, step, frozen, g0 = model(cs, a.cell, T, a.soft_eps)
        ztop = m.init_state(*cellset.c0_at(cs, a.cell, 0.0))
        mins = []
        for r in rates + [a.i_ref / Q_REF]:
            I = r * Q_REF
            ch, zend, gend = protocol(m, step, g0, ztop, I)
            k = int(np.argmin(ch["ppl"])) if len(ch["ppl"]) else 0
            row = dict(T_C=T, rate_C=r, I_A=I, cc_s=float(ch["t"][-1]) if len(ch["t"]) else 0.0, q_cc_Ah=float(ch["q"][-1]) if len(ch["q"]) else 0.0,
                       ppl_min_mV=1e3 * float(ch["ppl"][k]) if len(ch["ppl"]) else np.nan, t_min_s=float(ch["t"][k]) if len(ch["t"]) else np.nan,
                       ppl_end_cc_mV=1e3 * float(ch["ppl"][-1]) if len(ch["ppl"]) else np.nan)
            print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
            if r == a.i_ref / Q_REF:                 # instantaneous: state at the end of the reference CC charge
                Is = np.linspace(a.i_ref, 15 * a.i_ref, 141)
                g = gend; ph = []
                for Ik in Is:
                    g, p, _ = frozen(zend, g, -Ik)
                    ph.append(float(p))
                inst[T] = (Is, 1e3 * np.array(ph))
                row["kind"] = "reference"
            else:
                mins.append(row["ppl_min_mV"]); curves[(T, r)] = ch; row["kind"] = "rate"
            rows.append(row)
        I_full = crossing(np.array(rates) * Q_REF, np.array(mins))
        I_inst = crossing(*inst[T])
        print("T %g C: plating potential 0 at about %.2f A (%.2f C) over a full CC charge; %.2f A (%.2f C) as an instantaneous "
              "step at the end of the %.2f A CC charge" % (T, I_full, I_full / Q_REF, I_inst, I_inst / Q_REF, a.i_ref), flush=True)
        rows.append(dict(T_C=T, kind="threshold", I_full_A=I_full, I_inst_A=I_inst))
        # film resistance growth at the reference current: minimum over the CC charge of both definitions
        rfm = [float(v) for v in a.rf_mults.split(",")]
        mp, md = [], []
        for rf in rfm:
            ch, _, _ = protocol(m, step, g0, ztop, a.i_ref, rf=rf)
            mp.append(1e3 * ch["ppl"].min() if len(ch["ppl"]) else np.nan); md.append(1e3 * ch["dphi"].min() if len(ch["dphi"]) else np.nan)
            rows.append(dict(T_C=T, kind="Rf", rf_mult=rf, Rf_ohm_m2=rf * m.P["Rf0"], I_A=a.i_ref, cc_s=float(ch["t"][-1]) if len(ch["t"]) else 0.0,
                             ppl_min_mV=mp[-1], dphi_min_mV=md[-1]))
            print("T %g C, Rf x %g: min phi_pl %.1f mV, min (phi_s - phi_e) %.1f mV, CC %.0f s" % (T, rf, mp[-1], md[-1], rows[-1]["cc_s"]), flush=True)
        rf_p, rf_d = crossing(np.array(rfm), np.array(mp)), crossing(np.array(rfm), np.array(md))
        print("T %g C: R_f multiplier to reach 0 at %.2f A: %.1f (phi_pl), %.1f (phi_s - phi_e); R_f0 = %.3g ohm m2" % (T, a.i_ref, rf_p, rf_d, m.P["Rf0"]), flush=True)
        rows.append(dict(T_C=T, kind="Rf_threshold", rf_ppl=rf_p, rf_dphi=rf_d, Rf0_ohm_m2=m.P["Rf0"]))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "plating_threshold.csv"), index=False)

    # ---- figure: (a) potential during the CC charge vs charged capacity, per rate (first temperature); (b) min vs rate
    Ts = sorted({k[0] for k in curves}, reverse=True)
    fig, ax = plt.subplots(1, 3, figsize=(170 * mm, 62 * mm), layout="constrained")
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(rates)))
    for r, cc in zip(rates, cols):
        ch = curves.get((Ts[0], r))
        if ch is not None and len(ch["t"]):
            ax[0].plot(ch["q"], 1e3 * ch["ppl"], color=cc, lw=0.9, label="%.1fC" % r)
    ax[0].axhline(0, color="k", lw=0.6, ls="--"); ax[0].set_xlabel("charged [A h]"); ax[0].set_ylabel(r"$\phi_s-\phi_e-R_f j_n$ at the separator [mV]")
    ax[0].set_title("(a) CC charge to 4.2 V, %g °C" % Ts[0], fontsize=8); ax[0].legend(fontsize=6, frameon=False, ncol=2)
    tc = plt.cm.coolwarm(np.linspace(1, 0, len(Ts)))
    for T, cT in zip(Ts, tc):
        mn = [r_["ppl_min_mV"] for r_ in rows if r_.get("kind") == "rate" and r_["T_C"] == T]
        ax[1].plot(rates, mn, "-o", color=cT, ms=3, lw=0.9, label="%g °C" % T)
        Is, ph = inst[T]
        ax[2].plot(Is / Q_REF, ph, color=cT, lw=0.9, label="%g °C" % T)
    for k in (1, 2):
        ax[k].axhline(0, color="k", lw=0.6, ls="--"); ax[k].legend(fontsize=6, frameon=False)
    ax[1].set_xlabel("CC charging rate [C]"); ax[1].set_ylabel("minimum over the charge [mV]")
    ax[1].set_title("(b) full CC charge", fontsize=8)
    ax[2].set_xlabel("charging current [C]"); ax[2].set_ylabel("at the end-of-CC state [mV]")
    ax[2].set_title("(c) instantaneous, state at the end of\nthe %.2f A CC charge" % a.i_ref, fontsize=8)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "fig_plating_threshold." + ext))
    print("wrote", os.path.join(a.out, "fig_plating_threshold.png"))


if __name__ == "__main__":
    main()
