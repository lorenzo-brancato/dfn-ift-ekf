"""The Mohtat EKF on every cell, at three states of life, in parallel; one summary table.

    python scripts/mohtat_batch.py \
        --plant-root "<Mohtat NMC>/processed/Mohtat2020" --ref "<...>/per_checkup_window.csv" \
        --cells 1-21 --workers 7 --results results/mohtat/batch

States of life per cell: the first cycles (BOL), the cycle whose SOH (summary.csv, C/20 anchors) is closest to 90 %
and to 80 % (the last cycles when the cell stops above 80 %). From each start, consecutive cycles until at least
--min-hours of data (at least 3 cycles, at most 12): the first cycle is the filter's convergence, the statistics are on
the rest. A group with a logger fault (a gap > 60 s inside a file, a sample V <= 1 V, a voltage step > 0.15 V between
files beyond the ohmic jump of the current reversal, 0.12 V + 65 mOhm |dI|, as realdata.load in SOC SOH Coestimation)
is moved forward one cycle at a time (backward for the last state of life).

Filter options are passed to scripts/ekf_exp.py --dataset exp_Mohtat (defaults: Rf, LAM_n, LLI; sigma 14 mV; q 1e-6; lithium
constraint 1e-4; Ds,p and the D_p(y) law as identified). Statistics (from the second cycle): SOH, SOC error, LLI, LAM_n,
Rf, innovations; references: the file's SOH_ref (C/20) and the electrode-window fit of the nearest check-up (LLI in A.h
from that cell's first check-up, LAM_n, LAM_p).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mohtat_ekf_figs import window  # noqa: E402


def cells_of(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def fault(plant, cycles):
    prev = None
    for c in cycles:
        r = pd.read_csv(os.path.join(plant, "cycle_%04d.csv" % c))
        t, V, I = r["Time [s]"].values, r["Voltage [V]"].values, r["Current [A]"].values
        if np.any(np.diff(t) > 60) or np.any(V <= 1.0):
            return True
        if prev is not None and abs(V[0] - prev[0]) > 0.12 + 0.065 * abs(I[0] - prev[1]):
            return True
        prev = (V[-1], I[-1])
    return False


def plan(plant, min_hours, max_cycles=12):
    s = pd.read_csv(os.path.join(plant, "summary.csv"))
    have = set(int(c) for c in s.Cycle)
    dur = {}

    def hours(c):
        if c not in dur:
            r = pd.read_csv(os.path.join(plant, "cycle_%04d.csv" % c), usecols=["Time [s]"])
            dur[c] = float(r["Time [s]"].iloc[-1]) / 3600
        return dur[c]

    def group(c0):
        cyc, h = [], 0.0
        c = c0
        while c in have and (len(cyc) < 3 or h < min_hours) and len(cyc) < max_cycles:
            cyc.append(c); h += hours(c); c += 1
        return cyc

    starts = {"BOL": int(s.Cycle.iloc[0])}
    for lab, target in (("SOH90", 90.0), ("SOH80", 80.0)):
        i = int((s.SOH - target).abs().idxmin())
        starts[lab] = int(s.Cycle[i])
    out = {}
    for lab, c0 in starts.items():
        g = group(c0)
        while len(g) < 3 and c0 > int(s.Cycle.iloc[0]):        # near the end of the record: start earlier
            c0 -= 1; g = group(c0)
        tries = 0
        while fault(plant, g) and tries < 25:
            c0 += 1 if lab != "SOH80" else -1
            g = group(c0); tries += 1
        out[lab] = g
    return out


def stats(cs, f, ref):
    d = np.load(f, allow_pickle=True)
    cell = str(d["cell"]); c = cs["cells"][cell]
    co = [str(x) for x in d["coords"]]
    v = d["vals"]
    n1 = int(d["n_first"]) if "n_first" in d else 0
    if not np.all(np.isfinite(v[n1:])):
        return dict(finite=False)
    lam = v[:, co.index("LAM")] if "LAM" in co else np.zeros(len(v))
    lli = v[:, co.index("LLI")] if "LLI" in co else np.zeros(len(v))
    dxw = float(d["dx_window_n"])
    xe = d["xbar"] - lli * dxw; y = d["ybar"]
    k = np.arange(n1, len(y), 30)
    Qnom = window(cs, cell, c["x100"], c["y100"], 0.0)[0]
    W = np.array([window(cs, cell, xe[i], y[i], lam[i]) for i in k])
    soh = 100 * W[:, 0] / Qnom
    yt = np.interp(np.arange(len(y)), k, W[:, 1]); yb = np.interp(np.arange(len(y)), k, W[:, 2])
    soc = (yb - y) / (yb - yt)
    qc = np.cumsum(d["I"]) * float(d["dt"]) / 3600
    kt = int(np.argmax(d["y"][n1:])) + n1
    soc_ref = 1 - (qc - qc[kt]) / (float(d["SOH_ref"]) / 100 * c["q_ocv_Ah"])
    e = 100 * (soc - soc_ref)[n1:]
    out = dict(finite=True, SOH=np.nanmean(soh), SOH_min=np.nanmin(soh), SOH_max=np.nanmax(soh), SOH_ref=float(d["SOH_ref"]),
               SOC_rms=float(np.sqrt(np.mean(e ** 2))), SOC_max=float(np.max(np.abs(e))),
               LLI_Ah=float(np.mean(lli[n1:]) * dxw * c["Q_n"]), LAM_n=float(np.mean(lam[n1:])),
               Rf=float(np.median(v[n1:, co.index("Rf")])) if "Rf" in co else np.nan,
               innov_mV=float(1e3 * np.sqrt(np.mean(d["innov"][n1:] ** 2))))
    if ref is not None:
        rc = ref[ref.cell == int(cell[1:])]
        if len(rc):
            cyc = int(d["cycle"])
            r = rc.iloc[int(np.argmin(np.abs(rc.cycle.values - cyc)))]
            out.update(ref_cycle=int(r.cycle), ref_first_cycle=int(rc.cycle.min()),
                       LLI_ref_Ah=float(rc[rc.cycle == rc.cycle.min()].n_Li.iloc[0] - r.n_Li),
                       LAM_n_ref=float(r.LAM_n), LAM_p_ref=float(r.LAM_p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(ROOT, "data", "exp_Mohtat", "params", "mohtat_lm3c.json"))
    ap.add_argument("--plant-root", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--cells", default="1-21")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--min-hours", type=float, default=8.0)
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "mohtat", "batch"))
    ap.add_argument("--ekf-args", default="--coords Rf,LAM,LLI --sigma-mV 14 --q-th 1e-6 --li-sigma 1e-4 --seeds=-1")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    jobs = []
    for n in cells_of(a.cells):
        plant = os.path.join(a.plant_root, "real_cell_%02d" % n, "Plant")
        for lab, g in plan(plant, a.min_hours).items():
            jobs.append(dict(cell="C%02d" % n, state=lab, plant=plant, start=g[0], n=len(g)))
            print("C%02d %-5s cycles %d-%d (%d)" % (n, lab, g[0], g[-1], len(g)), flush=True)
    json.dump(jobs, open(os.path.join(a.results, "plan.json"), "w"), indent=1)
    if a.plan_only:
        return

    def run(j):
        log = os.path.join(a.results, "%s_%s.log" % (j["cell"], j["state"]))
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "ekf_exp.py"), "--dataset", "exp_Mohtat", "--start", "params", "--set", a.set,
               "--cell", j["cell"], "--plant", j["plant"], "--cycle", str(j["start"]), "--n-cycles", str(j["n"]),
               "--results", a.results] + a.ekf_args.split()
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print("done %s %s (rc %d)" % (j["cell"], j["state"], rc), flush=True)
        return rc
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))

    cs = json.load(open(a.set, encoding="utf-8"))
    ref = pd.read_csv(a.ref) if a.ref else None
    rows = []
    for j in jobs:
        tag = "cyc%04d" % j["start"] + ("x%d" % j["n"] if j["n"] > 1 else "")
        f = os.path.join(a.results, "ekf_%s_%s_s-1.npz" % (j["cell"], tag))
        row = dict(cell=j["cell"], state=j["state"], cycles="%d-%d" % (j["start"], j["start"] + j["n"] - 1))
        row.update(stats(cs, f, ref) if os.path.exists(f) else dict(finite=False))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.results, "summary.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
