"""How far from lithium plating: minimum LDP (anode potential vs Li/Li+ at the separator) while charging, per cell.

    .venv/Scripts/python.exe scripts/mohtat_phm/ldp_table.py --name s50_all_base [--ref soc_heldout,cal_q0_s15,dc_soc_s15,ref_rest] [--cells C08,C05]

For every ekf_<cell>_e*.npz of results/mohtat_phm/ekf/<name>/: the minimum of the LDP at the separator (plot_ekf.ldp_sep)
over the charging samples (I < 0), when it happens, the estimated and reference SOC there and the measured temperature
(read back from the dataset). With --ref (comma-separated folders), the same minimum from another run of the same cell (e.g.
the filter started at the reference charge) and the difference. Markdown on stdout.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from plot_ekf import ldp_sep  # noqa: E402
from dfnekf import datasets  # noqa: E402

EKF = os.path.join(ROOT, "results", "mohtat_phm", "ekf")


def runs(names):
    """names: one run folder, or several comma-separated (the first holding a cell wins)."""
    out = {}
    for name in names.split(","):
        for f in sorted(glob.glob(os.path.join(EKF, name, "ekf_*_e*.npz"))):
            out.setdefault(re.search(r"ekf_(C\d+_[A-Z0-9]+)_e", f).group(1), f)
    return out


def ldp_min(f):
    d = np.load(f)
    ldp = ldp_sep(d) * 1e3
    ch = d["I"] < -0.05
    if not ch.any():
        return None
    k = np.flatnonzero(ch)[np.argmin(ldp[ch])]
    return dict(ldp=ldp[k], t_h=d["t"][k] / 3600, soc=100 * d["soc"][k], soc_ref=100 * d["soc_ref"][k], I=d["I"][k], k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    rr = runs(a.name); ref = runs(a.ref) if a.ref else {}
    cells = [c if "_" in c else c + "_VAL" for c in a.cells.split(",")] if a.cells else sorted(rr)
    print("| test | T at min [degC] | I [A] | LDP min [mV] | t [h] | SOC est / ref [%]" + (" | LDP min, ref run [mV] | diff [mV]" if a.ref else "") + " |")
    print("|---" * (6 + 2 * bool(a.ref)) + "|")
    for c in cells:
        if c not in rr:
            continue
        r = ldp_min(rr[c])
        if r is None:
            continue
        dd = datasets.load("exp_Mohtat", c, dt=2.0)
        T = dd["T"][min(r["k"], len(dd["T"]) - 1)]
        row = "| %s | %.1f | %.2f | %.1f | %.2f | %.1f / %.1f" % (c, T, r["I"], r["ldp"], r["t_h"], r["soc"], r["soc_ref"])
        if a.ref:
            q = ldp_min(ref[c]) if c in ref else None
            row += " | %s | %s" % (("%.1f" % q["ldp"], "%+.1f" % (r["ldp"] - q["ldp"])) if q else ("", ""))
        print(row + " |")


if __name__ == "__main__":
    main()
