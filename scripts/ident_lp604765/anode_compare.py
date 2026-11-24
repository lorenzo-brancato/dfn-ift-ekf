"""Comparison of literature anode OCPs through the whole identification (unattended, resumable).

    python anode_compare.py [--cands Chen2020,OKane2022,...] [--parallel 3]

Per candidate U_n (results/exp_LP604765/ocp_neg/ocp_neg_<name>.json, from ocp_neg_select.py; 'base' = the cell set's
Mohtat2020 table), with LP_OCP_N pointing at it:
  1. ocp_c20.py   cathode U_p by subtraction on the C/20 discharge (overpotential correction from fit_v9)
  2. ocv_step.py  window, entropic term and start offsets on the 18 HPPC rests (dUdT_n fixed at 0.285 mV/K)
  3. fit.py       dynamic fit as v12: 12 free groups, thermal state (C_th 40 J/K), IFT Jacobian, set B, start from v12
  4. validate.py  all tests (AGE22V held out)
Summary: <results>/anode_compare.csv and ANODE_COMPARE.md; log anode_compare.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(ROOT, "results", "exp_LP604765")
PY = sys.executable
FREE = "Ds_p,drop,shape,k0_p,tau,R_contact,Ea_Dp,Ea_r,Ea_s,Ds_n,Ea_Dn,th_hA"
LOG = os.path.join(RES, "anode_compare.log")


def log(s):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def run(args, out, env):
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        rc = subprocess.call([PY, "-u"] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
    return rc, time.time() - t0


def table_of(name):
    return os.path.join(RES, "ocp_neg", "ocp_neg_%s.json" % "".join(ch if ch.isalnum() else "_" for ch in name)[:40])


def one(name):
    tag = "an_" + name
    env = dict(os.environ)
    if name != "base":
        env["LP_OCP_N"] = table_of(name)
    else:
        env.pop("LP_OCP_N", None)
    ocp_p = os.path.join(RES, "ocp", "ocp_p_c20_%s.json" % tag)
    s1 = os.path.join(RES, "step1_ocv_c20_%s.json" % tag)
    fitd = os.path.join(RES, "fit_" + tag)
    if not os.path.exists(ocp_p):
        rc, dt = run(["ocp_c20.py", "--fit", os.path.join(RES, "fit_v9", "best.json"), "--ocp-model", os.path.join(RES, "ocp", "ocp_p_c20.json"),
                      "--tag", "c20_" + tag], os.path.join(RES, "ocp", "%s.stdout" % tag), env)
        log("%s: ocp_c20 rc %d (%.0f s)" % (name, rc, dt))
        if not os.path.exists(ocp_p):
            return dict(candidate=name, failed="ocp_c20")
    if not os.path.exists(s1):
        rc, dt = run(["ocv_step.py", "--ocp-p", ocp_p, "--tag", "c20_" + tag, "--fix", "dUdT_n=0.000285"],
                     os.path.join(RES, "step1_%s.stdout" % tag), env)
        log("%s: ocv_step rc %d (%.0f s)" % (name, rc, dt))
        if not os.path.exists(s1):
            return dict(candidate=name, failed="ocv_step")
    S = json.load(open(s1))
    fix = {k: S["params"][k] for k in ("x100", "y100", "Q_n", "Q_p", "dUdT0", "dUdT_n", "dq_40", "dq_00")}
    fix.update(k0_n=1.0, dq_20=0.0, th_C=40.0)
    env["LP_THERMAL"] = "1"
    if not os.path.exists(os.path.join(fitd, "best.json")):
        rc, dt = run(["fit.py", "--tag", tag, "--tests", "B", "--jac", "ift", "--thermal", "--ocp-p", ocp_p, "--starts", "0",
                      "--max-nfev", "150", "--workers", "4", "--from-json", os.path.join(RES, "fit_v12", "best.json"), "--free", FREE,
                      "--fix", ",".join("%s=%r" % kv for kv in fix.items()), "--note", "anode OCP %s; v12 configuration" % name],
                     os.path.join(RES, "fit_%s.stdout" % tag), env)
        log("%s: fit rc %d (%.0f s)" % (name, rc, dt))
        if not os.path.exists(os.path.join(fitd, "best.json")):
            return dict(candidate=name, failed="fit", rest_rms_mV=S["rms_mV"])
    if not os.path.exists(os.path.join(fitd, "validation.csv")):
        rc, dt = run(["validate.py", "--thermal", "--fit", os.path.join(fitd, "best.json"), "--ocp-p", ocp_p],
                     os.path.join(fitd, "validate.stdout"), env)
        log("%s: validate rc %d (%.0f s)" % (name, rc, dt))
    import pandas as pd
    b = json.load(open(os.path.join(fitd, "best.json")))
    row = dict(candidate=name, rest_rms_mV=S["rms_mV"], window_Ah=S["window_ends"]["Q_window_Ah"], x100=fix["x100"], Q_n=fix["Q_n"],
               fit_cost_mV=b["cost_mV"], at_bound=";".join(b.get("at_bound", [])))
    vf = os.path.join(fitd, "validation.csv")
    if os.path.exists(vf):
        v = pd.read_csv(vf).set_index("test")
        for t in ("HPPC00", "HPPC20", "HPPC40", "CAP00F", "CAP40F", "AGE22C2", "AGE22V", "CYC20"):
            if t in v.index:
                row[t] = float(v.loc[t, "rmse_mV"])
        if "AGE22V" in v.index and "rmse_above3V_mV" in v.columns:
            row["AGE22V_above3V"] = float(v.loc["AGE22V", "rmse_above3V_mV"])
    log("%s: done, rest %.1f mV, fit %.2f mV, AGE22V %s mV" % (name, S["rms_mV"], b["cost_mV"], row.get("AGE22V")))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="base,Mohtat2020,Chen2020,OKane2022,Marquis2019,NCA_Kim2011,Ecker2015,Ramadass2004,Ai2020,Chayambuka2022")
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    log("=== anode_compare start: %s" % a.cands)
    with ThreadPoolExecutor(a.parallel) as ex:
        rows = list(ex.map(one, a.cands.split(",")))
    import pandas as pd
    df = pd.DataFrame(rows)
    if "AGE22V" in df:
        df = df.sort_values("AGE22V")
    df.to_csv(os.path.join(RES, "anode_compare.csv"), index=False)
    cols = [c for c in ("candidate", "rest_rms_mV", "window_Ah", "fit_cost_mV", "AGE22V", "AGE22V_above3V", "CYC20", "HPPC20", "CAP00F", "at_bound", "failed") if c in df]
    md = ["# Anode OCP comparison through the identification (%s)\n" % time.strftime("%Y-%m-%d %H:%M"),
          "Each candidate: U_p by subtraction on the C/20, window on the HPPC rests, v12 dynamic fit, validation. RMSE in mV.\n",
          df[cols].round(2).to_markdown(index=False)]
    open(os.path.join(RES, "ANODE_COMPARE.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    log("=== anode_compare done")
    open(os.path.join(RES, "anode_compare.done"), "w").write("done\n")


if __name__ == "__main__":
    main()
