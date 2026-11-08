"""Figures of the parameter-group selection for the paper (reads saved results only; rerun as results arrive).

    python fig_ident.py [--ref fit_v2b] [--old fit_v2]

  fig_ident_screening   (a) Morris mu* per group (prior box), (b) orthogonal ranking at the reference optimum:
                        sensitivity delta_j, orthogonal residual, collinearity index of the ranked prefixes.
                        Bars coloured by the decision of IDENTIFIABILITY.md.
  fig_ident_residual    the fit residual is model error: (a) residual of HPPC20 in time, (b) its autocorrelation per
                        test with the integrated correlation time tau (-> N_eff, the profile threshold).
  fig_ident_profiles    profile likelihood: Delta RMSE vs the fixed value of each group, converged refits at the
                        reference (filled) and the capped refits at fit_v2 (open), threshold sqrt(1 + chi2/N_eff) - 1.
  fig_ident_fits        RMSE per test (fit: HPPC 0/20/40 C; held out: CYC20, CAP 0/20/40 C), old vs reference set.
Output: results/exp_LP604765/figs/<name>.png/.pdf.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(ROOT, "results", "exp_LP604765")
plt.style.use(os.path.join(ROOT, "journal.mplstyle"))
mm = 1 / 25.4

LABEL = {"y100": r"$y_{100}$", "x100": r"$x_{100}$", "Q_p": r"$Q_p$", "Q_n": r"$Q_n$", "drop": r"$d_p$",
         "shape": r"$s_p$", "Ds_p": r"$D_{s,p}$", "Ds_n": r"$D_{s,n}$", "k0_p": r"$k_{0,p}$", "k0_n": r"$k_{0,n}$",
         "tau": r"$\tau_e$", "R_contact": r"$R_c$", "Ea_Dp": r"$E_{a,Dp}$", "Ea_Dn": r"$E_{a,Dn}$", "Ea_r": r"$E_{a,r}$",
         "Ea_s": r"$E_{a,s}$", "dUdT0": r"$\partial_T U_0$", "dUdT_n": r"$\partial_T U_n$", "dq_40": r"$\Delta q_{40}$",
         "dq_00": r"$\Delta q_{0}$"}
# decision categories (IDENTIFIABILITY.md, Step 2); colours of the style cycle
CAT = {"not identifiable": (["Ds_n", "Ea_Dn", "k0_n"], "#999999"),
       "redundant": (["Ds_p"], "#CC79A7"),
       "weak (profile)": (["k0_p", "shape", "Ea_s", "dq_40"], "#E69F00"),
       "identifiable, bound-active": (["Ea_r", "dUdT_n"], "#D55E00"),
       "identifiable": ([], "#0072B2")}


def cat_of(p):
    for c, (ps, col) in CAT.items():
        if p in ps:
            return c, col
    return "identifiable", CAT["identifiable"][1]


def save(fig, name):
    out = os.path.join(RES, "figs")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png")); fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".png"))


def legend_cats(ax, **kw):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col, label=c) for c, (_, col) in CAT.items()], fontsize=7, **kw)


def fig_screening(ref):
    m = pd.read_csv(os.path.join(RES, "sens", "morris.csv")).set_index("param")
    idn = json.load(open(os.path.join(RES, ref, "ident_%s.json" % ref.replace("fit_", ""))))
    rank, resid = idn["orth_rank"], idn["orth_resid_mV"]
    delta = idn["delta_mV"]
    gam = [idn["largest_identifiable"]["prefix_gamma"][p] for p in rank]
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 110 * mm), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.35]))
    # (a) Morris, sorted by mu*
    order = m["mu_mean"].sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))
    ax[0].barh(y, m.loc[order, "mu_mean"].clip(lower=0.03), color=[cat_of(p)[1] for p in order], height=0.7)
    ax[0].set_yticks(y, [LABEL.get(p, p) for p in order]); ax[0].invert_yaxis()
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"Morris $\mu^*$ [mV]")
    ax[0].set_title("(a) global screening, prior box", fontsize=9, loc="left")
    # (b) orthogonal ranking at the reference
    y = np.arange(len(rank))
    ax[1].barh(y, [delta[p] for p in rank], color="0.85", height=0.75, label=r"sensitivity $\delta_j$")
    ax[1].barh(y, resid, color=[cat_of(p)[1] for p in rank], height=0.45, label="orthogonal residual")
    ax[1].set_yticks(y, [LABEL.get(p, p) for p in rank]); ax[1].invert_yaxis()
    ax[1].set_xscale("log"); ax[1].set_xlabel("mV per unit relative change")
    ax[1].set_title("(b) local, at the reference optimum (%.1f mV)" % idn["rmse_mV"], fontsize=9, loc="left")
    tw = ax[1].twiny()
    tw.plot(gam, y, "k.-", lw=0.8, ms=3)
    tw.axvline(15, color="k", lw=0.6, ls="--")
    tw.set_xscale("log"); tw.set_xlabel(r"collinearity index $\gamma$ of the first $k$ groups (dots)", fontsize=8)
    from matplotlib.patches import Patch
    h = [Patch(color=col, label=c) for c, (_, col) in CAT.items()]
    h += [Patch(color="0.85", label=r"(b) sensitivity $\delta_j$"), Patch(color="k", alpha=0.0, label=r"(b) dots: $\gamma$, dashed: 15")]
    fig.legend(handles=h, fontsize=7, loc="outside lower center", ncol=4)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    tw.spines[["right"]].set_visible(False)
    save(fig, "fig_ident_screening")


def fig_residual(ref):
    import common as C
    import fit as Fm
    from ident import tau_int
    z = np.load(os.path.join(RES, ref, "ident_%s.npz" % ref.replace("fit_", "")))
    r0 = z["r0"]
    prof = C.profiles(Fm.TESTS)
    nk = [len(prof[k]["V"]) for k in Fm.TESTS]; cut = np.cumsum([0] + nk)
    lab = {"HPPC00": "0 °C", "HPPC20": "20 °C", "HPPC40": "40 °C"}
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 62 * mm), layout="constrained", gridspec_kw=dict(width_ratios=[1.5, 1]))
    for i, k in enumerate(Fm.TESTS):
        w = C.weights(prof[k]) / np.sqrt(len(Fm.TESTS))
        e = r0[cut[i]:cut[i + 1]] / w
        if k == "HPPC20":
            ax[0].plot(prof[k]["t"] / 3600, e, lw=0.5, color="C0")
        x = e[prof[k]["dt"] == 1.0]; x = x - x.mean()
        n = len(x); f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
        lag = np.arange(n)
        sel = lag <= 1500
        ax[1].plot(lag[sel], ac[sel], lw=1.0, label=r"%s, $\tau$ = %.0f s" % (lab[k], tau_int(e[prof[k]["dt"] == 1.0])))
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("t [h]"); ax[0].set_ylabel("residual [mV]")
    ax[0].set_title("(a) HPPC 20 °C: residual of the reference fit", fontsize=9, loc="left")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("lag [s] (1-s samples)"); ax[1].set_ylabel("autocorrelation")
    idn = json.load(open(os.path.join(RES, ref, "ident_%s.json" % ref.replace("fit_", ""))))
    ax[1].set_title(r"(b) $N_\mathrm{eff}$ = %.0f, threshold +%.2f %%" % (idn["N_eff"], 100 * idn["profile_threshold_rel"]), fontsize=9, loc="left")
    ax[1].legend(fontsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ident_residual")


def fig_profiles(ref, old, with_old=False):
    f = os.path.join(RES, ref, "profile_%s.json" % ref.replace("fit_", ""))
    if not os.path.exists(f):
        print("no profiles yet at", f); return
    P = json.load(open(f))
    pref = json.load(open(os.path.join(RES, ref, "best.json")))["params"]
    fo = os.path.join(RES, old, "profile_%s.json" % old.replace("fit_", ""))
    O = json.load(open(fo)) if (with_old and os.path.exists(fo)) else None
    pold = json.load(open(os.path.join(RES, old, "best.json")))["params"] if O else None
    names = list(dict.fromkeys(q["param"] for q in P["points"]))
    if O:
        names += [n for n in dict.fromkeys(q["param"] for q in O["points"]) if n not in names]
    nc = 4; nr = int(np.ceil(len(names) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(170 * mm, 42 * mm * nr), layout="constrained", squeeze=False)
    import common as C
    for i, nm in enumerate(names):
        A = ax[i // nc][i % nc]
        for D, p0, sty, lab in ((P, pref, dict(marker="o", ls="-", color=cat_of(nm)[1]), "converged, reference"),
                                (O, pold, dict(marker="o", ls=":", mfc="none", color="0.55"), "capped, fit_v2")):
            if D is None:
                continue
            q = sorted([p for p in D["points"] if p["param"] == nm], key=lambda p: p["value"])
            if not q:
                continue
            fx, xl = xform(nm, pref)
            x = fx(np.array([p["value"] for p in q] + [p0[nm]])); yv = np.array([100 * p["rel_increase"] for p in q] + [0.0])
            o = np.argsort(x)
            A.plot(x[o], yv[o], ms=3.5, lw=1.0, label=lab, **sty)
        fx, xl = xform(nm, pref)
        A.plot(fx(np.array([pref[nm]])), [0.0], "k*", ms=7)
        A.axhline(100 * P["threshold_rel"], color="k", lw=0.6, ls="--")
        lit = LIT.get(nm)
        if lit is not None:
            A.axvspan(*fx(np.array(lit)), color="C2", alpha=0.12, lw=0)
        if C.BOUNDS[nm][2] == "log":
            from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
            A.set_xscale("log", base=2)
            A.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v)); A.xaxis.set_minor_formatter(NullFormatter())
        A.set_xlabel(xl)
        A.set_ylim(bottom=min(-0.1, A.get_ylim()[0]))
        if i % nc == 0:
            A.set_ylabel(r"$\Delta$RMSE [%]")
        A.spines[["top", "right"]].set_visible(False)
    for j in range(len(names), nr * nc):
        ax[j // nc][j % nc].axis("off")
    if O is not None:
        ax[0][0].legend(fontsize=6.5, loc="upper left")
    save(fig, "fig_ident_profiles" + ("_with_v2" if O is not None else ""))


# literature ranges (shaded): E_a,r 18-54 kJ/mol (Chen2020, ORegan2022, Mohtat2020, Ecker2015), graphite / cell entropy
# |dU/dT| <= 0.3 mV/K; D_s,p of lm3c (Mohtat cells) as a point
LIT = {"Ea_r": (17.8e3, 53.4e3), "dUdT_n": (-0.3e-3, 0.3e-3), "dUdT0": (-0.3e-3, 0.3e-3), "Ds_p": (9.0e-14, 1.03e-13)}


def xform(nm, pref):
    """x axis of a profile: relative to the reference for log groups, physical units otherwise."""
    import common as C
    if C.BOUNDS[nm][2] == "log":
        return (lambda v: np.asarray(v) / pref[nm]), r"%s / %s$^\star$" % (LABEL.get(nm, nm), LABEL.get(nm, nm))
    if nm.startswith("Ea_"):
        return (lambda v: np.asarray(v) * 1e-3), r"%s [kJ/mol]" % LABEL.get(nm, nm)
    if nm.startswith("dUdT"):
        return (lambda v: np.asarray(v) * 1e3), r"%s [mV/K]" % LABEL.get(nm, nm)
    return (lambda v: np.asarray(v)), LABEL.get(nm, nm)


SETLAB = {"fit_v2": "v2: 17 groups (not converged)", "fit_v2b": "v2b: 17 groups (reference)",
          "fit_v3a": r"v3a: 15 ($D_{s,p}$, $s_p$ lm3c)", "fit_v3b": r"v3b: 12 (+ lit. $E_{a,r}$, $\partial_T U$)"}
SETCOL = {"fit_v2": "0.8", "fit_v2b": "0.5", "fit_v3a": "C1", "fit_v3b": "C0", "fit_v4a": "C2", "fit_v4b": "C3", "fit_v5": "C4"}
SETLAB.update({"fit_v4a": "v4a: set B, 12 groups (FD)", "fit_v4b": "v4b: set B, 17 groups (FD)", "fit_v5": "v5: set B, 17 groups, exact IFT Jacobian, OCP by subtraction"})


def fig_fits(tags, order=None, split=3, name="fig_ident_fits"):
    rows = []
    for tag in tags:
        f = os.path.join(RES, tag, "validation.csv")
        if os.path.exists(f):
            d = pd.read_csv(f); d["set"] = tag; rows.append(d)
    if not rows:
        print("no validation.csv yet"); return
    d = pd.concat(rows)
    order = order or ["HPPC00", "HPPC20", "HPPC40", "CYC20", "CAP00", "CAP20", "CAP40"]
    sets = list(dict.fromkeys(d["set"]))
    fig, ax = plt.subplots(figsize=(170 * mm, 60 * mm), layout="constrained")
    x = np.arange(len(order)); wb = 0.8 / len(sets)
    for i, s in enumerate(sets):
        dd = d[d["set"] == s].set_index("test")
        v = [dd.loc[t, "rmse_mV"] if t in dd.index else np.nan for t in order]
        ax.bar(x + (i - (len(sets) - 1) / 2) * wb, v, wb, label=SETLAB.get(s, s), color=SETCOL.get(s, "C%d" % i))
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi + (i - (len(sets) - 1) / 2) * wb, vi + 0.4, "%.0f" % vi, ha="center", fontsize=5.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.axvline(split - 0.5, color="k", lw=0.6, ls=":")
    ax.text((split - 1) / 2, ax.get_ylim()[1] * 0.97, "fit", ha="center", va="top", fontsize=8)
    ax.text((split + len(order) - 1) / 2, ax.get_ylim()[1] * 0.97, "held out", ha="center", va="top", fontsize=8)
    ax.set_xticks(x, order); ax.set_ylabel("RMSE [mV]")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="fit_v2b")
    ap.add_argument("--old", default="fit_v2")
    ap.add_argument("--only", default="screening,residual,profiles,fits")
    ap.add_argument("--with-old", action="store_true", help="profiles: also the capped refits at --old (other reference)")
    ap.add_argument("--sets", default="fit_v2,fit_v2b,fit_v3a,fit_v3b", help="fits figure: the parameter sets compared")
    a = ap.parse_args()
    todo = a.only.split(",")
    if "screening" in todo:
        fig_screening(a.ref)
    if "residual" in todo:
        fig_residual(a.ref)
    if "profiles" in todo:
        fig_profiles(a.ref, a.old, a.with_old)
    if "fits" in todo:
        fig_fits(a.sets.split(","))
    if "fitsB" in todo:
        fig_fits(["fit_v3b", "fit_v4a", "fit_v4b", "fit_v5"], order=["HPPC00", "HPPC20", "HPPC40", "CAP00F", "CAP40F", "AGE22C2", "AGE22V", "CAP20"], split=6, name="fig_ident_fits_B")


if __name__ == "__main__":
    main()
