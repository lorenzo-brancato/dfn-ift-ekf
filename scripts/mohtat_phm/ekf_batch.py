"""EKF runs of the PHM configuration on Mohtat VAL tests with fit B (scripts/ekf_exp.py), one line per run.

    .venv/Scripts/python.exe scripts/mohtat_phm/ekf_batch.py --name soc --coords "" --cells C13,C08,C13_HPPC

Fixed: cell set mohtat_B_fin3 (--set), start 'soc', the measured temperature as input (--t-cell measured), 2-s model steps
(the cold 2C tests need them; the 10-s data are held over 5 steps) with one voltage update per 10-s sample.
Extra ekf_exp options after '--'. Summary: results/mohtat_phm/ekf/<name>/summary.jsonl.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KEYS = ("err_Ah", "innov_rmse_mV", "soc_err_rms_2nd_half_pct", "soc_err_max_2nd_half_pct", "soc_err_end_pct",
        "t_first_2pct_min", "cover_2sd_pct", "ms_per_step", "finite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--coords", default="")
    ap.add_argument("--cells", default="C13", help="cells, or cell_TEST entries (C13_HPPC); plain cells run --test")
    ap.add_argument("--test", default="VAL")
    ap.add_argument("--soc-err", default="0")
    ap.add_argument("--sigma-mV", type=float, default=15.0)
    ap.add_argument("--set", default=os.path.join(ROOT, "data", "exp_Mohtat", "params", "mohtat_B_fin3.json"))
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "mohtat_phm", "ekf", a.name)
    os.makedirs(out, exist_ok=True)
    extra = [x for x in a.extra if x != "--"]
    for entry in a.cells.split(","):
        cell, test = entry.split("_") if "_" in entry else (entry, a.test)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "ekf_exp.py"), "--dataset", "exp_Mohtat",
               "--test", "%s_%s" % (cell, test), "--set", a.set,
               "--start", "soc", "--coords", a.coords, "--sigma-mV", str(a.sigma_mV), "--soc-err", a.soc_err,
               "--t-cell", "measured", "--dt", "2", "--y-every-s", "10", "--no-openloop", "--results", out] + extra
        p = subprocess.run(cmd, capture_output=True, text=True)
        rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
        if not rows:
            print(cell, "FAILED", p.stderr[-800:], flush=True)
        with open(os.path.join(out, "summary.jsonl"), "a") as f:
            for r in rows:
                s = dict(cell=cell, test=test, **{k: r.get(k) for k in KEYS})
                s.update({k: r[k] for k in r if k.endswith("_end") and not k.startswith("soc")})
                f.write(json.dumps(s) + "\n")
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    main()
