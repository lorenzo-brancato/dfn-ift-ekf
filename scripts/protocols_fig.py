"""Data of Fig. 9 (test protocols): voltage-current traces of the PyBaMM reference runs, subsampled for plotting.

Writes four plain CSV files (columns ``t_h,I_A,V_V``, no header comment, so that they can be read by pgfplots directly) into ``<results>/source_data/``:
    prot_F6_th2.csv    reference F6, state th2: every 4th sample (2 s), from ``ref_F6_th2_m40-20-40_dt0.5.npz``
    prot_F6Q_th2.csv   reference F6Q, state th2: every 4th sample (2 s), from ``ref_F6Q_th2_m40-20-40_dt0.5.npz``
    prot_RPT.csv       ageing reference run, cycle index 10 (the first reference performance test, RPT 1): every 20th sample (10 s)
    prot_P4.csv        ageing reference run, cycle index 11 (the first P4 cycle after RPT 1): every 10th sample (5 s)
Time is in hours from the start of the protocol / cycle; t_h is written with 5 decimals, current and voltage rounded to 4 decimals.

The ageing reference run is the file ``aging_dyn_rptslow_full.npz`` of the data folder (PyBaMM, 10 x [RPT + 9 P4 cycles] + final RPT, period 0.5 s). It is read here directly
with ``numpy.load``; the keys used are:
    t                         time (s), one sample per 0.5 s with repeated times at the boundaries of the experiment steps
    Current [A]               terminal current (positive = discharge)
    Terminal voltage [V]      terminal voltage
    cycle_start_t             (n_cycles,) start time (s) of each cycle; cycle c covers t in [cycle_start_t[c], cycle_start_t[c+1])
If the file is missing the script stops with an error: it is produced by the ageing reference script (see the README), not by this repository's quick tests.

Run:
    python scripts/protocols_fig.py --data data --results results
    python scripts/protocols_fig.py --quick --data data --results results      # only the two F6/F6Q traces (no ageing file needed)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from dfnekf import cli

AGEING_FILE = "aging_dyn_rptslow_full.npz"
RPT_CYCLE, P4_CYCLE = 10, 11


def write_trace(path, t_s, I, V):
    """Write the three columns in the format of the figure data files."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("t_h,I_A,V_V\n")
        for a, b, c in zip(t_s / 3600.0, np.round(I, 4), np.round(V, 4)): f.write("%.5f,%g,%g\n" % (a, b, c))


def main():
    p = cli.parser("Fig. 9", "test-protocol traces (reference voltage and current)")
    args = p.parse_args(); d = os.path.join(args.results, "source_data"); os.makedirs(d, exist_ok=True)
    for proto, step in (("F6", 4), ("F6Q", 4)):
        f = os.path.join(args.data, f"ref_{proto}_th2_m40-20-40_dt0.5.npz")
        if not os.path.exists(f): sys.exit(f"missing reference {f}")
        z = np.load(f); s = slice(None, None, step)
        write_trace(os.path.join(d, f"prot_{proto}_th2.csv"), z["t"][s], z["I"][s], z["V"][s])
    if args.quick: print("quick: wrote the F6 and F6Q traces only"); return
    fa = os.path.join(args.data, AGEING_FILE)
    if not os.path.exists(fa): sys.exit(f"missing ageing reference run {fa} (needed for prot_RPT.csv and prot_P4.csv; generate it with the ageing reference script, or pass --data pointing to it)")
    z = np.load(fa); t, I, V, cs = z["t"], z["Current [A]"], z["Terminal voltage [V]"], z["cycle_start_t"]
    for name, c, step in (("RPT", RPT_CYCLE, 20), ("P4", P4_CYCLE, 10)):
        a = np.searchsorted(t, cs[c], "left"); b = np.searchsorted(t, cs[c + 1], "left"); s = slice(a, b, step)
        write_trace(os.path.join(d, f"prot_{name}.csv"), t[s] - cs[c], I[s], V[s])
    print("written to", d)


if __name__ == "__main__":
    main()
