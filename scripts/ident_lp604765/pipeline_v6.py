"""Unattended pipeline from fit_v5 to the reduced final set (fit_v6a / fit_v6b), with the decisions made by stated rules.

    python pipeline_v6.py            (resumable: a step whose output exists is skipped; log in <results>/pipeline_v6.log)

Steps (all fits: exact IFT Jacobian, fit set B = HPPC 0/20/40 + full 1C cycles CAP00F, CAP40F, AGE22C2):
 1. Profiles at fit_v5 (ocp s1, as v5): each candidate group fixed at its literature / reference value, the other groups
    re-optimised (profile_lik.py). Threshold: ident_v5 (N_eff from the residual autocorrelation).
 2. Decisions.
    fixA (data rule): Ea_s = 0 (at its lower bound, zero sensitivity), plus every candidate whose profile at its
          literature value stays below the threshold (practically non-identifiable: the literature value costs nothing).
    fixB (admissibility rule): fixA plus every temperature law outside its literature range at fit_v5
          (Ea_r > 54 kJ/mol, |dUdT_n| or |dUdT0| > 0.3 mV/K, Ea_Dp at its 90 kJ/mol bound), fixed at the literature value
          whatever its cost on the fit data (the held-out tests arbitrate, step 5).
 3. Fits v6a (free = v5 groups - fixA) and v6b (free = v5 groups - fixB) with the second-pass cathode OCP (s2, window of
    v5); OCP/window alternation: after each fit the OCP is subtracted again with its window; if the table moves by
    more than 3 mV on the measured range, the fit is repeated with the new table (at most 2 extra rounds).
 4. Validation (validate.py) and identifiability (ident.py: gamma of the free set, Cramer-Rao with N_eff) of both.
 5. Recommendation: the smaller set (v6b) unless its held-out AGE22V RMSE is worse than v6a's by more than 0.5 mV.
    The recommended set is exported (results/fit_v6x/lp604765_cellset.json, with its OCP); data/ is NOT touched.
 6. Comparison for the paper: v6a refitted from the same start with the finite-difference Jacobian (cost, Jacobians,
    wall time).
 7. Summary: <results>/PIPELINE_v6.md, figures in <results>/figs.
"""
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(ROOT, "results", "exp_LP604765")
PY = sys.executable
OCP = os.path.join(RES, "ocp")
LOG = os.path.join(RES, "pipeline_v6.log")
V5 = os.path.join(RES, "fit_v5", "best.json")
# literature / reference values of the candidates (IDENTIFIABILITY.md)
LIT = {"Ds_p": 9.65e-14,          # lm3c (Mohtat cells), top-of-window cathode diffusivity
       "dUdT0": 0.008e-3,         # O'Regan 2022 entropic terms projected on the window
       "dUdT_n": 0.285e-3,
       "dq_40": 0.0,              # start offset of HPPC40: none
       "Ea_r": 39570.0,           # Mohtat2020, NMC charge transfer
       "Ea_Dp": 65700.0}          # lm3c (Mohtat cells, measured temperature), cathode diffusion
LIT_RANGE = {"Ea_r": (17.8e3, 54e3), "dUdT_n": (-0.3e-3, 0.3e-3), "dUdT0": (-0.3e-3, 0.3e-3), "Ea_Dp": (5e3, 89.9e3)}
TESTS_HELD = ["AGE22V", "CAP20"]


def log(s):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out):
    log("run: " + " ".join(os.path.basename(a) if a.endswith(".py") else a for a in args))
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        rc = subprocess.call([PY, "-u"] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log("   rc %d, %.0f s -> %s" % (rc, time.time() - t0, os.path.relpath(out, RES)))
    return rc


def fit(tag, free, fix, start, ocp, note, jac="ift", workers=6):
    best = os.path.join(RES, "fit_" + tag, "best.json")
    if os.path.exists(best):
        log("skip fit %s (exists)" % tag); return best
    run(["fit.py", "--tag", tag, "--tests", "B", "--jac", jac, "--ocp-p", ocp, "--starts", "0", "--max-nfev", "150",
         "--workers", str(workers), "--from-json", start, "--free", ",".join(free),
         "--fix", ",".join("%s=%r" % kv for kv in fix.items()), "--note", note], os.path.join(RES, "fit_%s.stdout" % tag))
    return best if os.path.exists(best) else None


def ocp_change(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    y = np.asarray(A["y"]); lo, hi = B["y_measured"]
    m = (y >= lo) & (y <= hi)
    return float(1e3 * np.max(np.abs(np.asarray(B["U"])[m] - np.asarray(A["U"])[m])))


def converge_ocp(tag, free, fix, start, ocp, note):
    """fit, subtract the OCP with its window, refit while the table moves by > 3 mV (at most 2 extra rounds)."""
    hist = []
    for k in range(3):
        t = tag if k == 0 else "%s_r%d" % (tag, k)
        best = fit(t, free, fix, start, ocp, note + " (OCP %s)" % os.path.basename(ocp))
        if best is None:
            return None, ocp, hist
        new = os.path.join(OCP, "ocp_p_%s.json" % t)
        if not os.path.exists(new):
            run(["ocp_sub.py", "--fit", best, "--tag", t], os.path.join(OCP, "ocp_p_%s.stdout" % t))
        dmax = ocp_change(ocp, new)
        hist.append(dict(fit=t, ocp_used=os.path.basename(ocp), ocp_next=os.path.basename(new), change_mV=dmax,
                         cost_mV=json.load(open(best))["cost_mV"]))
        log("   %s: cost %.2f mV; OCP change with its window %.1f mV" % (t, hist[-1]["cost_mV"], dmax))
        if dmax <= 3.0 or k == 2:
            return best, ocp, hist
        start, ocp = best, new
    return best, ocp, hist


def main():
    log("=== pipeline_v6 start")
    v5 = json.load(open(V5)); free5 = v5["free"]; p5 = v5["params"]
    s1, s2 = os.path.join(OCP, "ocp_p_s1.json"), os.path.join(OCP, "ocp_p_s2.json")
    # 1. profiles at v5
    prof_json = os.path.join(RES, "fit_v5", "profile_v5.json")
    spec = ";".join("%s=%r" % (k, v) for k, v in LIT.items() if k in free5)
    need = True
    if os.path.exists(prof_json):
        have = {q["param"] for q in json.load(open(prof_json))["points"]}
        need = not all(k in have for k in LIT if k in free5)
    if need:
        run(["profile_lik.py", "--fit", V5, "--ident", os.path.join(RES, "fit_v5", "ident_v5.json"), "--profiles", spec,
             "--max-nfev", "60", "--workers", "6", "--tag", "v5", "--jac", "ift", "--tests", "B", "--ocp-p", s1],
            os.path.join(RES, "fit_v5", "profile_v5.stdout"))
    P = json.load(open(prof_json)); thr = P["threshold_rel"]
    pts = {q["param"]: q for q in P["points"]}
    # 2. decisions
    fixA = {"Ea_s": 0.0}
    rows = []
    for k, L in LIT.items():
        if k not in free5:
            continue
        q = pts.get(k)
        inc = q["rel_increase"] if q else np.nan
        flat = bool(q is not None and inc < thr)
        lo, hi = LIT_RANGE.get(k, (-np.inf, np.inf))
        adm = lo <= p5[k] <= hi
        rows.append(dict(group=k, v5=p5[k], literature=L, profile_increase_pct=100 * inc, threshold_pct=100 * thr,
                         practically_non_identifiable=flat, admissible_at_v5=adm))
        if flat:
            fixA[k] = L
    fixB = dict(fixA)
    for r in rows:
        if not r["admissible_at_v5"] and r["group"] in LIT_RANGE:
            fixB[r["group"]] = LIT[r["group"]]
    freeA = [g for g in free5 if g not in fixA]
    freeB = [g for g in free5 if g not in fixB]
    json.dump(dict(threshold_rel=thr, candidates=rows, fixA=fixA, fixB=fixB, freeA=freeA, freeB=freeB),
              open(os.path.join(RES, "pipeline_v6_decisions.json"), "w"), indent=1)
    log("decisions: fixA %s; fixB %s" % (fixA, fixB))
    # warm start: the profile point of the group whose fixing moves the model most (Ds_p, if fixed), else v5
    start = V5
    if "Ds_p" in fixA and "Ds_p" in pts:
        start = os.path.join(RES, "pipeline_v6_start_A.json")
        json.dump(dict(params=pts["Ds_p"]["params"], cost_mV=pts["Ds_p"]["rmse_mV"]), open(start, "w"), indent=1)
    # 3. fits with OCP/window alternation
    bestA, ocpA, histA = converge_ocp("v6a", freeA, fixA, start, s2, "v6a: data rule (profiles at v5), IFT, set B")
    bestB, ocpB, histB = (None, None, [])
    if bestA and fixB != fixA:
        bestB, ocpB, histB = converge_ocp("v6b", freeB, fixB, bestA, ocpA,
                                          "v6b: admissibility rule (literature temperature laws), IFT, set B")
    # 4. validation and identifiability
    summ = {}
    for best, ocp, free in ((bestA, ocpA, freeA), (bestB, ocpB, freeB)):
        if not best:
            continue
        d = os.path.dirname(best); tag = os.path.basename(d).replace("fit_", "")
        if not os.path.exists(os.path.join(d, "validation.csv")):
            run(["validate.py", "--fit", best, "--ocp-p", ocp], os.path.join(d, "validate.stdout"))
        if not os.path.exists(os.path.join(d, "ident_%s.json" % tag)):
            run(["ident.py", "--fit", best, "--params", "all", "--jac", "ift", "--tests", "B", "--ocp-p", ocp, "--workers", "6",
                 "--tag", tag, "--sets", "free=" + ",".join(free)], os.path.join(d, "ident_%s.log" % tag))
        summ[tag] = dict(best=best, ocp=ocp, free=free, d=d)
    # 5. recommendation and export
    import pandas as pd
    val = {t: pd.read_csv(os.path.join(s["d"], "validation.csv")).set_index("test") for t, s in summ.items()
           if os.path.exists(os.path.join(s["d"], "validation.csv"))}
    rec = None
    if val:
        ta = [t for t in val if t.startswith("v6a")]; tb = [t for t in val if t.startswith("v6b")]
        ta, tb = (ta[-1] if ta else None), (tb[-1] if tb else None)
        if tb and ta:
            rec = tb if val[tb].loc["AGE22V", "rmse_mV"] <= val[ta].loc["AGE22V", "rmse_mV"] + 0.5 else ta
        else:
            rec = ta or tb
        s = summ[rec]
        cs = os.path.join(s["d"], "lp604765_cellset.json")
        if not os.path.exists(cs):
            run(["export_cellset.py", "--fit", s["best"], "--ocp-p", s["ocp"]], os.path.join(s["d"], "export.stdout"))
        log("recommended: %s" % rec)
    # 6. FD comparison (same problem and start as v6a)
    if bestA:
        fit("v6a_fd", freeA, fixA, start, s2, "v6a_fd: as the first round of v6a, finite-difference Jacobian", jac="fd", workers=12)
    # 7. summary and figures
    try:
        sys.path.insert(0, HERE)
        import fig_ident as FI
        sets = ["fit_v3b", "fit_v5"] + ["fit_" + t for t in summ]
        FI.SETLAB.update({"fit_" + t: "%s: %d groups, IFT" % (t, len(summ[t]["free"])) for t in summ})
        FI.fig_fits(sets, order=["HPPC00", "HPPC20", "HPPC40", "CAP00F", "CAP40F", "AGE22C2", "AGE22V", "CAP20"], split=6,
                    name="fig_ident_fits_v6")
    except Exception:  # noqa: BLE001
        log("figure failed: " + traceback.format_exc().splitlines()[-1])
    write_summary(rows, thr, fixA, fixB, freeA, freeB, summ, val, rec, histA, histB)
    log("=== pipeline_v6 done")
    open(os.path.join(RES, "pipeline_v6.done"), "w").write("done\n")


def write_summary(rows, thr, fixA, fixB, freeA, freeB, summ, val, rec, histA, histB):
    L = ["# LP604765: pipeline v6 (unattended), %s\n" % time.strftime("%Y-%m-%d %H:%M"),
         "Fit set B (HPPC 0/20/40 C + full 1C cycles CAP00F, CAP40F, AGE22C2); exact IFT Jacobian; cathode OCP by",
         "subtraction at the rested points (window/OCP alternation). Held out: AGE22V (cycles 5-7 of 20261022) and CAP20",
         "(Oct 9, before the temperature tests: the cell's 1C capacity was 1.894 Ah instead of 1.93 Ah afterwards).\n",
         "## 1-2. Profiles at fit_v5 and decisions (threshold +%.2f %% of the RMSE)\n" % (100 * thr),
         "| group | v5 | literature | profile increase | non-identifiable | admissible at v5 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %.4g | %.4g | %+.2f %% | %s | %s |" % (r["group"], r["v5"], r["literature"], r["profile_increase_pct"],
                                                           "yes" if r["practically_non_identifiable"] else "no",
                                                           "yes" if r["admissible_at_v5"] else "no"))
    L += ["", "- fixA (data rule): %s -> %d free groups: %s" % (fixA, len(freeA), ", ".join(freeA)),
          "- fixB (admissibility rule): %s -> %d free groups: %s\n" % (fixB, len(freeB), ", ".join(freeB)),
          "## 3. Fits and OCP alternation\n"]
    for h in histA + histB:
        L.append("- %s: cost %.2f mV with %s; OCP subtracted with its window moves by %.1f mV (%s)"
                 % (h["fit"], h["cost_mV"], h["ocp_used"], h["change_mV"], h["ocp_next"]))
    L += ["", "## 4. Validation (RMSE, mV)\n"]
    order = ["HPPC00", "HPPC20", "HPPC40", "CAP00F", "CAP40F", "AGE22C2", "AGE22V", "CAP20"]
    L.append("| set | " + " | ".join(order) + " | AGE22V above 3 V |"); L.append("|---|" + "---|" * (len(order) + 1))
    for t, v in val.items():
        L.append("| %s | " % t + " | ".join("%.1f" % v.loc[k, "rmse_mV"] if k in v.index else "-" for k in order)
                 + " | %.1f |" % v.loc["AGE22V", "rmse_above3V_mV"])
    for t, s in summ.items():
        f = os.path.join(s["d"], "ident_%s.json" % t)
        if os.path.exists(f):
            idn = json.load(open(f)); st = idn["largest_identifiable"]["sets"].get("free", {})
            L.append("\n- %s: gamma of the free set %.1f; Cramer-Rao rel. sd (N_eff %.0f): %s"
                     % (t, st.get("gamma", np.nan), idn["N_eff"],
                        ", ".join("%s %.1f %%" % (k, 100 * v) for k, v in st.get("rel_sd_crlb_Neff", {}).items())))
        b = json.load(open(s["best"]))
        L.append("- %s: at bound %s; %s" % (t, b.get("at_bound"), {k: float("%.4g" % b["params"][k]) for k in s["free"]}))
    L += ["", "## 5. Recommendation\n", "- %s (rule: the smaller set unless its AGE22V RMSE is worse by > 0.5 mV). Cell set: results/exp_LP604765/fit_%s/lp604765_cellset.json (data/ not changed)." % (rec, rec)]
    fd = os.path.join(RES, "fit_v6a_fd", "best.json")
    ia = os.path.join(RES, "fit_v6a", "start_0.json")
    if os.path.exists(fd) and os.path.exists(ia):
        a_, b_ = json.load(open(ia)), json.load(open(os.path.join(RES, "fit_v6a_fd", "start_0.json")))
        L += ["", "## 6. Finite differences vs exact (IFT) Jacobian, same problem and start (v6a, first round)\n",
              "| Jacobian | cost [mV] | Jacobians | time per Jacobian [s] | wall [s] | stop |", "|---|---|---|---|---|---|"]
        for nm, r in (("exact (IFT)", a_), ("finite differences", b_)):
            L.append("| %s | %.3f | %s | %.1f | %.0f | %s |" % (nm, r["cost_mV"], r.get("n_jac"), r.get("t_jac_s", 0) / max(r.get("n_jac") or 1, 1),
                                                               r["wall_s"], r["message"]))
    L += ["", "Figures: figs/fig_ident_fits_v6.png; per test: fit_v6*/val_*.png. Log: pipeline_v6.log."]
    open(os.path.join(RES, "PIPELINE_v6.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log("wrote PIPELINE_v6.md")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log("FAILED: " + traceback.format_exc())
        raise
