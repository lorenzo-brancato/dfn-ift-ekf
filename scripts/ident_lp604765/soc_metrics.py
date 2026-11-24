"""SOC metrics of SOC-only EKF runs against the re-anchored reference (fig_soh_nom.soc_reference: 100 % at the end of
every CV charge, full scale = the OCV-window capacity), after the first anchor.

    python soc_metrics.py --q-window 2.220 <runs dir> [<runs dir> ...]

Per run: SOC error RMS / max [%], coverage of the +-2 sd band [%], NIS (mean squared normalised innovation; 1 = consistent
voltage noise model), innovation RMS [mV]. Output: printed table and <first dir>/../soc_metrics.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig_soh_nom import soc_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--q-window", type=float, required=True)
    ap.add_argument("--scaled", action="store_true",
                    help="full scale = --q-window x the measured 1C capacity fade, interpolated across the block (soc_ref.reference)")
    ap.add_argument("--q-window-run", action="store_true", help="full scale = the run's own OCV window at each anchor (LLI runs: Q_window_Ah)")
    ap.add_argument("--pattern", default="ekf_*_e+0.00.npz", help="run files in each folder (e.g. ekf_*_e-*.npz: the SOC 50 %% starts)")
    a = ap.parse_args()
    rows = []
    for r in a.runs:
        for f in sorted(glob.glob(os.path.join(r, a.pattern))):
            d = np.load(f, allow_pickle=True)
            qw = d["Q_window_Ah"] if (a.q_window_run and "Q_window_Ah" in d) else a.q_window
            if a.scaled:
                from soc_ref import reference
                ref, anc, _ = reference(d["t"], d["I"], os.path.basename(f).split("_")[1], a.q_window)
            else:
                ref, anc, _ = soc_reference(d["t"], d["I"], q_window=qw)
            k = anc[0] if anc else 0
            es = 100 * (d["soc"] - ref)[k:]; sd = 100 * d["sd_soc"][k:]
            inn, Sy = d["innov"][k:], d["Sy"][k:]
            ok = np.isfinite(inn) & np.isfinite(Sy) & (Sy > 0)
            rows.append(dict(runs=os.path.basename(os.path.normpath(r)), test=os.path.basename(f).split("_")[1],
                             start=os.path.basename(f).rsplit("_", 1)[1][:-4],
                             soc_rms_pct=float(np.sqrt(np.mean(es ** 2))), soc_max_pct=float(np.max(np.abs(es))),
                             cover_2sd_pct=float(100 * np.mean(np.abs(es) <= 2 * sd)), sd_soc_mean_pct=float(np.mean(sd)),
                             nis=float(np.mean((inn[ok] / Sy[ok]) ** 2)), innov_rms_mV=float(1e3 * np.sqrt(np.mean(inn[ok] ** 2)))))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    out = os.path.join(os.path.dirname(os.path.normpath(a.runs[0])), "soc_metrics.csv")
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
