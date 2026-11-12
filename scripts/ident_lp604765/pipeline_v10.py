"""Unattended two-step identification of ID01 (v10). Resumable; log <results>/pipeline_v10.log, summary PIPELINE_v10.md.

Step 1 (done before, ocv_step.py): cathode U_p by subtraction on the C/20 discharge (ocp_p_c20); ID01's window
  (x100, y100, Q_n, Q_p), entropic dUdT0 and the start offsets dq_40, dq_00 from the 18 HPPC rests; dUdT_n at its
  literature value (not identifiable on the rests). All fixed here.
Step 2: dynamic fit on the HPPCs and the full 1C cycles (fit set B), exact IFT Jacobian, 12 groups free: the cathode
  diffusivity and its D(y) law (Ds_p, drop, shape), the anode one (Ds_n), both exchange currents (k0_p, k0_n), tau,
  R_contact and the four activation energies (Ea_Dp, Ea_Dn, Ea_r, Ea_s).
Step 3: identifiability at the optimum (ident.py); every group whose Cramer-Rao relative sd (N_eff) exceeds 50 % is
  profiled at its reference value (REF); a flat profile (below the N_eff threshold) fixes it there; refit.
Step 4: validation, identifiability and export of the final set; summary. Held out: AGE22V, CAP20 (and the EKF runs).
"""
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(ROOT, "results", "exp_LP604765")
PY = sys.executable
LOG = os.path.join(RES, "pipeline_v10.log")
OCP = os.path.join(RES, "ocp", "ocp_p_c20.json")
STEP1 = os.path.join(RES, "step1_ocv_c20.json")
START = os.path.join(RES, "fit_v9", "best.json")
FREE = ["Ds_p", "drop", "shape", "k0_p", "tau", "R_contact", "Ea_Dp", "Ea_r", "Ea_s", "Ds_n", "k0_n", "Ea_Dn"]
REF = {"Ds_p": 9.65e-14, "drop": 1.55, "shape": 1.37, "k0_p": 1.0, "tau": 1.0, "R_contact": 3.2e-3,     # lm3c (Mohtat cells)
       "Ds_n": 6.16e-13, "k0_n": 1.0, "Ea_Dn": 30800.0,                                                # lm3c
       "Ea_Dp": 65700.0, "Ea_r": 39570.0, "Ea_s": 0.0}                                                 # lm3c / Mohtat2020 / none
CRLB_MAX = 0.5


def log(s):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def run(args, out):
    log("run: " + " ".join(args))
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        rc = subprocess.call([PY, "-u"] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log("   rc %d, %.0f s" % (rc, time.time() - t0))
    return rc


def fit(tag, free, fix, start, note):
    best = os.path.join(RES, "fit_" + tag, "best.json")
    if not os.path.exists(best):
        run(["fit.py", "--tag", tag, "--tests", "B", "--jac", "ift", "--ocp-p", OCP, "--starts", "0", "--max-nfev", "150",
             "--workers", "6", "--from-json", start, "--free", ",".join(free),
             "--fix", ",".join("%s=%r" % kv for kv in fix.items()), "--note", note], os.path.join(RES, "fit_%s.stdout" % tag))
    return best if os.path.exists(best) else None


def post(tag, free):
    d = os.path.join(RES, "fit_" + tag); best = os.path.join(d, "best.json")
    if not os.path.exists(os.path.join(d, "validation.csv")):
        run(["validate.py", "--fit", best, "--ocp-p", OCP], os.path.join(d, "validate.stdout"))
    if not os.path.exists(os.path.join(d, "ident_%s.json" % tag)):
        run(["ident.py", "--fit", best, "--params", "all", "--jac", "ift", "--tests", "B", "--ocp-p", OCP, "--workers", "6",
             "--tag", tag, "--sets", "free=" + ",".join(free)], os.path.join(d, "ident_%s.log" % tag))
    return json.load(open(os.path.join(d, "ident_%s.json" % tag)))


def main():
    log("=== pipeline_v10 start")
    s1 = json.load(open(STEP1))
    fix1 = {k: s1["params"][k] for k in ("x100", "y100", "Q_n", "Q_p", "dUdT0", "dUdT_n", "dq_40", "dq_00")}
    fix1["dq_20"] = 0.0
    note = "v10: two-step. Step 1: C/20 U_p (ocp_p_c20) + window/entropic/offsets from the HPPC rests (step1_ocv_c20). Step 2: 12 dynamic groups, IFT, set B"
    b10 = fit("v10", FREE, fix1, START, note)
    if b10 is None:
        log("FAILED: fit v10"); return
    idn = post("v10", FREE)
    sd = idn["largest_identifiable"]["sets"]["free"]["rel_sd_crlb_Neff"]
    weak = [g for g in FREE if sd.get(g, 0.0) > CRLB_MAX]
    log("v10: cost %.2f mV; gamma(free) %.1f; groups with CRLB > %.0f %%: %s"
        % (json.load(open(b10))["cost_mV"], idn["largest_identifiable"]["sets"]["free"]["gamma"], 100 * CRLB_MAX, weak))
    fix2, prof = {}, {}
    if weak:
        pj = os.path.join(RES, "fit_v10", "profile_v10.json")
        spec = ";".join("%s=%r" % (g, REF[g]) for g in weak)
        have = {q["param"] for q in json.load(open(pj))["points"]} if os.path.exists(pj) else set()
        if not all(g in have for g in weak):
            run(["profile_lik.py", "--fit", b10, "--ident", os.path.join(RES, "fit_v10", "ident_v10.json"), "--profiles", spec,
                 "--max-nfev", "60", "--workers", "6", "--tag", "v10", "--jac", "ift", "--tests", "B", "--ocp-p", OCP],
                os.path.join(RES, "fit_v10", "profile_v10.stdout"))
        P = json.load(open(pj)); thr = P["threshold_rel"]
        for q in P["points"]:
            prof[q["param"]] = q["rel_increase"]
            if q["param"] in weak and q["rel_increase"] < thr:
                fix2[q["param"]] = REF[q["param"]]
        log("profiles (threshold %.2f %%): %s; fixed: %s" % (100 * thr, {k: round(100 * v, 2) for k, v in prof.items()}, fix2))
    final = "v10"
    if fix2:
        free2 = [g for g in FREE if g not in fix2]
        start = b10
        pts = {q["param"]: q for q in json.load(open(os.path.join(RES, "fit_v10", "profile_v10.json")))["points"]}
        if len(fix2) == 1:                                   # warm start at the converged profile point
            g = next(iter(fix2)); start = os.path.join(RES, "pipeline_v10_start.json")
            json.dump(dict(params=pts[g]["params"]), open(start, "w"), indent=1)
        b = fit("v10r", free2, {**fix1, **fix2}, start, note + "; profile-flat groups at reference values %s" % fix2)
        if b:
            post("v10r", free2); final = "v10r"
    d = os.path.join(RES, "fit_" + final)
    if not os.path.exists(os.path.join(d, "lp604765_cellset.json")):
        run(["export_cellset.py", "--fit", os.path.join(d, "best.json"), "--ocp-p", OCP], os.path.join(d, "export.stdout"))
    summary(s1, idn, prof, fix2, final)
    log("=== pipeline_v10 done (final %s)" % final)
    open(os.path.join(RES, "pipeline_v10.done"), "w").write(final + "\n")


def summary(s1, idn10, prof, fix2, final):
    import pandas as pd
    L = ["# LP604765 ID01: two-step identification (pipeline v10), %s\n" % time.strftime("%Y-%m-%d %H:%M"),
         "## Step 1: OCV (C/20 U_p by subtraction; window and entropic terms from the 18 HPPC rests)\n",
         "- rms %.1f mV (%s); window 3.0-4.2 V at 20 C: %.3f Ah" % (s1["rms_mV"], ", ".join("%s %.1f" % kv for kv in s1["per_test_rms_mV"].items()), s1["window_ends"]["Q_window_Ah"]),
         "- " + ", ".join("%s %.4g (sd %.2g)" % (k, v, s1["sd"].get(k, float("nan"))) for k, v in s1["params"].items()), "",
         "## Step 2-3: dynamic fit, identifiability, profiles\n"]
    for tag in ("v10", "v10r"):
        f = os.path.join(RES, "fit_%s" % tag, "best.json")
        if not os.path.exists(f):
            continue
        b = json.load(open(f)); fi = os.path.join(RES, "fit_%s" % tag, "ident_%s.json" % tag)
        i = json.load(open(fi)) if os.path.exists(fi) else None
        st = i["largest_identifiable"]["sets"]["free"] if i else {}
        L.append("- %s: cost %.2f mV (%s); at bound %s; gamma %.1f" % (tag, b["cost_mV"], ", ".join("%s %.1f" % kv for kv in b["per_test_mV"].items()),
                                                                     b["at_bound"], st.get("gamma", float("nan"))))
        L.append("  - values: " + ", ".join("%s %.4g" % (k, b["params"][k]) for k in b["free"]))
        L.append("  - CRLB rel. sd (N_eff %.0f): " % (i["N_eff"] if i else float("nan")) + ", ".join("%s %.1f %%" % (k, 100 * v) for k, v in st.get("rel_sd_crlb_Neff", {}).items()))
    if prof:
        L.append("- profiles at the reference values: " + ", ".join("%s %+.2f %%" % (k, 100 * v) for k, v in prof.items()) + "; fixed: %s" % fix2)
    L += ["", "## Validation (RMSE, mV)\n"]
    order = ["HPPC00", "HPPC20", "HPPC40", "CAP00F", "CAP40F", "AGE22C2", "AGE22V", "CAP20", "CYC20"]
    L.append("| set | " + " | ".join(order) + " | AGE22V above 3 V |"); L.append("|---|" + "---|" * (len(order) + 1))
    for tag in ("v7s1", "v9", "v10", "v10r"):
        f = os.path.join(RES, "fit_%s" % tag, "validation.csv")
        if os.path.exists(f):
            v = pd.read_csv(f).set_index("test")
            L.append("| %s | " % tag + " | ".join("%.1f" % v.loc[k, "rmse_mV"] if k in v.index else "-" for k in order) + " | %.1f |" % v.loc["AGE22V", "rmse_above3V_mV"])
    L += ["", "Final set: fit_%s (cell set exported there, with ocp_p_c20). data/ not changed." % final]
    open(os.path.join(RES, "PIPELINE_v10.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log("FAILED: " + traceback.format_exc()); raise
