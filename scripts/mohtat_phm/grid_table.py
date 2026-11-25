"""Table of an EKF tuning grid from the ekf_batch summaries: SOC rms error (second half) / +-2 sd coverage per run.

    .venv/Scripts/python.exe scripts/mohtat_phm/grid_table.py --names "rw_*,dc_*,soc_heldout"

One row per run folder (results/mohtat_phm/ekf/<name>/summary.jsonl), one column per cell_test (first run of each, or --err-ah);
cells: rms [%] / coverage [%]. Markdown on stdout.
"""
import argparse
import fnmatch
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EKF = os.path.join(ROOT, "results", "mohtat_phm", "ekf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="rw_*,dc_*")
    ap.add_argument("--err-ah", type=float, default=None, help="keep only runs with this start error (default: all; the first per cell)")
    a = ap.parse_args()
    pats = a.names.split(",")
    names = sorted(n for n in os.listdir(EKF) if any(fnmatch.fnmatch(n, p) for p in pats)
                   and os.path.isfile(os.path.join(EKF, n, "summary.jsonl")))
    rows, cols = {}, []
    for n in names:
        for line in open(os.path.join(EKF, n, "summary.jsonl")):
            r = json.loads(line)
            if a.err_ah is not None and abs((r.get("err_Ah") or 0.0) - a.err_ah) > 1e-9:
                continue
            c = "%s_%s" % (r["cell"], r.get("test", "VAL"))
            if c not in cols:
                cols.append(c)
            if c in rows.get(n, {}):
                continue
            rows.setdefault(n, {})[c] = "%.2f / %.0f" % (r["soc_err_rms_2nd_half_pct"], r["cover_2sd_pct"])
    cols.sort()
    print("| run | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for n in names:
        if n in rows:
            print("| %s | %s |" % (n, " | ".join(rows[n].get(c, "") for c in cols)))


if __name__ == "__main__":
    main()
