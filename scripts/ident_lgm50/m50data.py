"""LG M50 (Kirdaky dataset): reader of the tests used by the identification, and its test table.

    from m50data import TESTS, FIT, VAL, cut, rests
    d = cut("D_P05")                     # t [s], V [V], I [A, > 0 on discharge], T_meas [C], step, names, test, V_rest

Data: C:\\Users\\alexa\\Datasets Batteries\\Kirdaky DS LG50M (env M50_ROOT). Two groups of 8 cells, A-H in each:
  Expt 4 (cells "A".."H", drive-cycle ageing): the beginning-of-life check-up, EC-Lab .mpt exports (tab separated,
    I/mA < 0 on discharge). Every file starts with a CC-CV charge (1.5 A to 4.2 V) and a 2 h rest; a test starts at the
    first step after that rest (Ns 4). All check-ups were run in a 25 C chamber (the 10 / 25 / 40 C of the cell names is
    the ageing temperature): measured surface temperature 22-39 C.
      fast_01  C/10 discharge + 6 h rest + C/10 charge (1 s)                        -> C10
      fast_02  0.5C base with 10 s pulses 0.25-1.5C, discharge then charge (0.1 s) -> P05
      fast_03  1C base with 10 s pulses 0.5-1.5C (up to 7.5 A), discharge (0.1 s)  -> P10
      slow_02  0.5C discharge + 6 h rest + 0.5C charge                             -> CC05
      slow_03  GITT 25 x 288 s at 0.5C, 1 h rests                                  -> GITT
      slow_04  5 x 1440 s at 0.5C, 1 h rests                                       -> G20
  Expt 5 (cells "5A".."5H", standard cycling at 10 / 25 / 40 C): the first ageing cycle (Processed_Cycles/
    MPR_<T>degC_Cell<X>_cycling/cycle_0000.csv, 1 s, current > 0 on discharge): CC-CV charge at 0.3C to 50 mA, then at
    once a 1C (5 A) discharge to 2.5 V -> CYC = that discharge, started by OCV inversion of the voltage at the end of the
    CV (50 mA, near rest). Surface temperature 13-31 C (10 C cells), 40-49 C (40 C cells): the temperature range of the
    identification. The C/10 discharge of RPT0 (Processed Timeseries Data/0.1C Voltage Curves) -> C10 (the window).
The cell temperature of the model is the measured surface temperature, uniform in the cell (lpmodel LP_TMEAS=1).
WLTP drive cycles: not used (the processed files have a broken time base; the raw ageing records are binary .mpr).
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("M50_ROOT", r"C:\Users\alexa\Datasets Batteries\Kirdaky DS LG50M")
EXP4 = os.path.join(ROOT, "Expt 4 - Drive Cycle Aging (Control)")
EXP5 = os.path.join(ROOT, "Expt 5 - Standard Cycle Aging (Control)")
BOL = os.path.join(EXP4, "Raw Data", "Performance Checks", "BoL")
PROC = os.path.join(ROOT, "Processed_Cycles")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(REPO, "results", "exp_LGM50")
CACHE = os.path.join(RES, "cache")

AGE_T = dict(A=10.0, B=10.0, C=10.0, D=25.0, E=25.0, F=40.0, G=40.0, H=40.0)     # ageing chamber (both experiments)
CELL_T = {**{c: 25.0 for c in AGE_T}, **{"5" + c: t for c, t in AGE_T.items()}}  # chamber of the tests used here
Q_NOMINAL = 5.0                                   # [A.h] LG M50 nominal
FILES = dict(C10=("fast", 1), P05=("fast", 2), P10=("fast", 3), CC05=("slow", 2), GITT=("slow", 3), G20=("slow", 4))

FIT_CELLS = ("A", "D", "F", "5A", "5D", "5F")     # one per ageing group of each experiment
VAL_CELLS = ("B", "E", "G", "5B", "5E", "5G")
FIT_KINDS = ("P05", "P10", "GITT", "CYC")


def _key(cell, kind):
    return "%s_%s" % (cell, kind)


TESTS = {}
for c in AGE_T:
    for k in FILES:
        # GITT: the top of the cell's window (its rests fix the window); the others start by OCV inversion of the
        # rested voltage after their own CC-CV + 2 h rest (the CV ends differ: up to 7 mV at the top, tens of mAh)
        TESTS[_key(c, k)] = dict(cell=c, kind=k, T=25.0, dq=0.0 if k == "GITT" else "ocv", win=c)
    TESTS[_key("5" + c, "CYC")] = dict(cell="5" + c, kind="CYC", T=AGE_T[c], dq="ocv", win="5" + c)
FIT = {k: v for k, v in TESTS.items() if v["cell"] in FIT_CELLS and v["kind"] in FIT_KINDS}
VAL = {k: v for k, v in TESTS.items() if k not in FIT and v["cell"] in FIT_CELLS + VAL_CELLS}


def bol_file(cell, speed, n):
    pat = os.path.join(BOL, speed.capitalize(), "*%sdegC - cell %s - BoL (%s)_%02d_MB_*.mpt" % (int(AGE_T[cell]), cell, speed, n))
    f = glob.glob(pat)
    if len(f) != 1:
        raise FileNotFoundError(pat)
    return f[0]


def read_mpt(path):
    """t [s], V [V], I [A, > 0 on discharge], T_meas [C], ns (EC-Lab sequence number), cached as .npz."""
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, os.path.basename(path).replace(" ", "_") + ".npz")
    if os.path.exists(cf) and os.path.getmtime(cf) > os.path.getmtime(path):
        z = np.load(cf)
        return {k: z[k] for k in z.files}
    want = {"time/s": "t", "Ecell/V": "V", "I/mA": "I", "Ns": "ns"}
    d = pd.read_csv(path, sep="\t", encoding="latin1", low_memory=False)
    tcol = [c for c in d.columns if c.startswith("Temperature")]
    out = {v: d[k].to_numpy(dtype=float) for k, v in want.items()}
    out["I"] = -out["I"] / 1e3
    out["ns"] = out["ns"].astype(int)
    out["T_meas"] = d[tcol[0]].to_numpy(dtype=float) if tcol else np.full(len(d), np.nan)
    np.savez(cf, **out)
    return out


def clean_T(T, lo=-5.0, hi=70.0):
    """Measured surface temperature with logger glitches (out of range, isolated jumps > 3 C) replaced by neighbours."""
    T = np.asarray(T, dtype=float).copy()
    bad = ~np.isfinite(T) | (T < lo) | (T > hi)
    if bad.any():
        good = np.flatnonzero(~bad)
        T[bad] = np.interp(np.flatnonzero(bad), good, T[good]) if len(good) else 25.0
    if len(T) > 2:
        d1, d2 = T[1:-1] - T[:-2], T[1:-1] - T[2:]
        sp = np.flatnonzero((np.abs(d1) > 3) & (np.abs(d2) > 3) & (np.sign(d1) == np.sign(d2))) + 1
        T[sp] = 0.5 * (T[sp - 1] + T[sp + 1])
    return T


def read_cycle(cell, n=0):
    """Expt 5 ageing cycle n of cell '5X' (processed csv, 1 s): t, V, I (> 0 discharge), T_meas, ns = phase (0 charge, 1 discharge)."""
    c = cell[1:]
    f = os.path.join(PROC, "MPR_%ddegC_Cell%s_cycling" % (int(AGE_T[c]), c), "cycle_%04d.csv" % n)
    d = pd.read_csv(f)
    return dict(t=d["Time [s]"].to_numpy(float), V=d["Voltage [V]"].to_numpy(float), I=d["Current [A]"].to_numpy(float),
                T_meas=d["X-averaged cell temperature [C]"].to_numpy(float), ns=d["phase"].to_numpy(int), file=f)


def read_c10_rpt0(cell):
    """Expt 5 RPT0 C/10 discharge of cell '5X' (processed): t, V, I (> 0 discharge), T_meas."""
    c = cell[1:]
    f = os.path.join(EXP5, "Processed Timeseries Data", "0.1C Voltage Curves", "Cell %s" % c,
                     "Expt 5 - cell %s - RPT0 - 0.1C discharge data.csv" % c)
    d = pd.read_csv(f)
    return dict(t=d["Time (s)"].to_numpy(float), V=d["Voltage (V)"].to_numpy(float), I=-d["Current (mA)"].to_numpy(float) / 1e3,
                T_meas=d["Temperature (degC)"].to_numpy(float), ns=np.zeros(len(d), int), file=f)


def cut(key, ns_start=4):
    """The test in the format of lpmodel.prepare (step = Ns / phase), V_rest = the voltage it starts from.
    Expt 4: from the end of the 2 h rest after the initial CC-CV (Ns >= ns_start) to the end of the file.
    Expt 5 CYC: the 1C discharge of cycle 0, V_rest = the voltage at the end of the CV (50 mA).
    Expt 5 C10 (key '5X_C10'): the RPT0 C/10 discharge, V_rest unknown (nan)."""
    cell, kind = key.split("_", 1)
    if kind == "CYC":
        r = read_cycle(cell)
        i0 = int(np.flatnonzero(r["ns"] == 1)[0])
        V_rest = float(r["V"][i0 - 1])
        s = slice(i0, None); src = r["file"]
    elif cell.startswith("5"):
        r = read_c10_rpt0(cell)
        V_rest = float("nan"); s = slice(0, None); src = r["file"]
    else:
        sp, n = FILES[kind]
        src = bol_file(cell, sp, n)
        r = read_mpt(src)
        i0 = int(np.flatnonzero(r["ns"] >= ns_start)[0])
        V_rest = float(r["V"][i0 - 1])
        s = slice(i0, None)
    t = r["t"][s] - r["t"][s][0]
    return dict(t=t, V=r["V"][s], I=r["I"][s], T_meas=clean_T(r["T_meas"][s]), step=r["ns"][s], names=[os.path.basename(src)],
                test=key, V_rest=V_rest, file=src)


def rests(key, min_s=1800.0, i_rest=0.01):
    """Rested points of a test: (q [A.h] discharged since the start, V at the end of every rest >= min_s, duration, T [C])."""
    d = cut(key)
    t, I, V = d["t"], d["I"], d["V"]
    dt = np.diff(t, prepend=t[0])
    q = np.cumsum(I * dt) / 3600.0
    r = np.abs(I) < i_rest
    edges = np.flatnonzero(np.diff(r.astype(int)))
    starts = np.r_[0, edges + 1]; ends = np.r_[edges, len(I) - 1]
    out = []
    for a, b in zip(starts, ends):
        if r[a] and t[b] - t[a] >= min_s:
            out.append((float(q[b]), float(V[b]), float(t[b] - t[a]), float(d["T_meas"][b])))
    return out


if __name__ == "__main__":
    for k in ("D_C10", "D_P05", "D_GITT", "5A_CYC", "5D_CYC", "5F_CYC", "5A_C10"):
        d = cut(k)
        dt = np.diff(d["t"]); q = np.sum(np.clip(d["I"][1:], 0, None) * dt) / 3600
        print("%-7s %7d samples, %5.1f h, I %+.2f..%+.2f A, V %.3f..%.3f, discharged %.3f Ah, T %.1f..%.1f C, V_rest %.4f"
              % (k, len(d["t"]), d["t"][-1] / 3600, d["I"].min(), d["I"].max(), d["V"].min(), d["V"].max(), q,
                 np.nanmin(d["T_meas"]), np.nanmax(d["T_meas"]), d["V_rest"]))
    print("fit tests:", list(FIT))
