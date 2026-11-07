"""Experimental datasets: one loader interface for every measured cell under ``data/exp_<name>/``.

    from dfnekf import datasets
    datasets.names()                                   # ['exp_LP604765', 'exp_Mohtat']
    datasets.tests("exp_LP604765")                     # ['CAP00', ..., 'CYC20', 'HPPC00', ...]
    d = datasets.load("exp_LP604765", "CYC20", dt=1.0) # dict, see below
    cs = datasets.cellset("exp_Mohtat")                # the identified parameter set (dfnekf.cellset schema)
    q0 = datasets.initial_charge("exp_Mohtat", d, cs)  # A.h below the top of the window at t = 0

Layout (``DFNEKF_DATA`` overrides the root, default ``<repo>/data``)::

    data/sim/                   PyBaMM references of the paper (dfnekf.refio)
    data/exp_<name>/raw|tests/  measured files (git-ignored)
    data/exp_<name>/params/     identified cell set(s) (committed)
    data/exp_<name>/README.md   cell, tests, temperatures, fit / validation split (committed)

``load`` returns a dict with at least
    t [s] from 0, I [A] (> 0 on discharge), V [V], T [degC] (array; the nominal chamber value where none is measured),
    valid (bool array: samples usable as measurements), T_C (scalar: the temperature the model parameters are built at),
    dt (median sample period [s]), V_rest (rested voltage before t = 0, or the first sample), cell, test, dataset,
    meta (dict: dataset-specific extras).
With ``dt`` given, the record is binned to a uniform grid: I = bin mean, V = last sample of the bin (the implicit closure
reads the end-of-step state), as the identification's ``prepare`` with all step classes at ``dt``.

Only numpy (and pandas for the csv datasets); no jax.
"""
import json
import os
import re

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DATA = os.environ.get("DFNEKF_DATA", os.path.join(ROOT, "data"))
SIM = os.path.join(DATA, "sim")


def root(dataset, *parts):
    return os.path.join(DATA, dataset, *parts)


def names():
    return sorted(d for d in os.listdir(DATA) if d.startswith("exp_") and os.path.isdir(os.path.join(DATA, d)))


# ---------------------------------------------------------------- LP604765 (CNR STEMS, BATTAGE)
LP_RAW = root("exp_LP604765", "raw", "01. Electric Characterization", "01. ID01")
LP_CELL = "LP01"
LP_Q_REF = 1.95          # [A.h] window width of the identification (SOC = 1 - q / 1.95 A.h)
# FIT: the three HPPCs (from the first pulse, after the 1 h rest at the top of charge); dq = start below the top
# (dq_20 = 0 by definition of the window, dq_00 / dq_40 fitted). VAL: held out, placed by OCV inversion (dq = "ocv").
LP_FIT = {
    "HPPC20": dict(test="20261011_HPPC_20°C", start_step=2, T=20.0, dq="dq_20"),
    "HPPC40": dict(test="20261014_HPPC_40°C", start_step=2, T=40.0, dq="dq_40"),
    "HPPC00": dict(test="20261021_HPPC_00°C", start_step=2, T=0.0, dq="dq_00"),
}
LP_VAL = {
    "CYC20": dict(test="20261022_Ageing", files=["%03d_%s.txt" % (k, n) for k, n in
                                                 [(1, "Cycling_20°C"), (2, "Relaxation_20°C"),
                                                  (3, "Cycling_20°C_4500mbar"), (4, "Relaxation_20°C_4500mbar"),
                                                  (5, "Cycling_20°C"), (6, "Relaxation_20°C"),
                                                  (7, "Cycling_20°C"), (8, "Relaxation_20°C")]],
                  start_time=5345.0, T=20.0, dq="ocv"),
    "CAP20": dict(test="20261009_Capacity_20°C", start_step=2, end_step=4, T=20.0, dq="ocv"),
    "CAP40": dict(test="20261013_Capacity_40°C", start_step=2, end_step=4, T=40.0, dq="ocv"),
    "CAP00": dict(test="20261016_Capacity_00°C", start_step=2, end_step=4, T=0.0, dq="ocv"),
}
LP_TESTS = {**LP_FIT, **LP_VAL}


def lp_list_folders():
    """Test folders of the raw LP604765 data (``YYYYMMDD_<name>``)."""
    return sorted(d for d in os.listdir(LP_RAW) if re.match(r"\d{8}_", d) and os.path.isdir(os.path.join(LP_RAW, d)))


def lp_read_step(path):
    """One step file: CSStudio header (2 lines), columns E(V), i (A; discharge negative), T(s). Returns V, I (> 0 on
    discharge), t."""
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    for ln in lines[2:]:
        p = ln.strip().split("\t")
        if len(p) < 3:
            continue
        try:
            rows.append((float(p[0]), float(p[1]), float(p[2])))
        except ValueError:
            continue
    a = np.array(rows, dtype=float).reshape(-1, 3)
    return a[:, 0], -a[:, 1], a[:, 2]


def lp_step_files(folder):
    return sorted(f for f in os.listdir(os.path.join(LP_RAW, folder)) if re.match(r"\d{3}_.*\.txt$", f))


def lp_load_folder(folder, files=None):
    """A whole raw test folder joined into one record: t [s] (continuous), V, I, step (index of the file), names."""
    files = lp_step_files(folder) if files is None else files
    T, V, I, S = [], [], [], []
    t0 = 0.0
    for k, f in enumerate(files):
        v, i, t = lp_read_step(os.path.join(LP_RAW, folder, f))
        if len(t) == 0:
            continue
        T.append(t - t[0] + t0); V.append(v); I.append(i); S.append(np.full(len(t), k))
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
        t0 = T[-1][-1] + dt
    return dict(t=np.concatenate(T), V=np.concatenate(V), I=np.concatenate(I), step=np.concatenate(S),
                names=[os.path.splitext(f)[0] for f in files], test=folder)


def lp_temperature_of(folder):
    m = re.search(r"(\d{2})°C", folder)
    return float(m.group(1)) if m else None


def lp_cut(key):
    """The test ``key`` of LP_TESTS from its start (start_time or start_step) to end_step, with the rested voltage
    before the start. Time is NOT re-zeroed here (the identification's prepare() does that)."""
    ds = LP_TESTS[key]
    d = lp_load_folder(ds["test"], files=ds.get("files"))
    msk = d["t"] >= ds["start_time"] if "start_time" in ds else d["step"] >= ds.get("start_step", 0)
    k0 = int(np.argmax(msk))
    V_rest = float(d["V"][k0 - 1]) if k0 > 0 else float(d["V"][0])     # last sample before the start (rested)
    d = {**d, "t": d["t"][msk], "V": d["V"][msk], "I": d["I"][msk], "step": d["step"][msk]}
    keep = d["step"] >= ds.get("start_step", 0)
    if ds.get("end_step") is not None:
        keep &= d["step"] < ds["end_step"]
    d = {**d, **{k: d[k][keep] for k in ("t", "V", "I", "step")}}
    d["V_rest"] = V_rest
    return d


def _load_lp(test, dt=None):
    if test not in LP_TESTS:
        raise KeyError("unknown LP604765 test %r (known: %s)" % (test, ", ".join(sorted(LP_TESTS))))
    ds = LP_TESTS[test]
    d = lp_cut(test)
    t = d["t"] - d["t"][0]
    out = dict(t=t, I=d["I"], V=d["V"], step=d["step"])
    if dt is not None:
        out = bin_uniform(out, dt)
    n = len(out["t"])
    return dict(**out, T=np.full(n, ds["T"]), valid=np.ones(n, bool), T_C=float(ds["T"]),
                dt=float(dt) if dt is not None else float(np.median(np.diff(t))), V_rest=d["V_rest"], cell=LP_CELL,
                test=test, dataset="exp_LP604765",
                meta=dict(folder=ds["test"], names=d["names"], dq=ds["dq"], Q_ref=LP_Q_REF, split="fit" if test in LP_FIT else "val"))


# ---------------------------------------------------------------- Mohtat (real cells, processed in MIONet)
MOHTAT_TESTS = root("exp_Mohtat", "tests")


def _mohtat_split(test):
    m = re.fullmatch(r"(C\d{2})_(\w+?)(?:_BOL)?", test)
    if not m:
        raise KeyError("Mohtat test %r: use <cell>_<test>, e.g. C01_VAL" % test)
    return m.group(1), m.group(2)


def _load_mohtat(test, dt=None):
    import pandas as pd
    cell, kind = _mohtat_split(test)
    d = os.path.join(MOHTAT_TESTS, "DB_REAL_%s_%s_BOL" % (cell, kind))
    df = pd.read_csv(os.path.join(d, "cycles.csv"))
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    out = dict(t=df.t_s.values.astype(float), I=df.I_A.values.astype(float), V=df.V_V.values.astype(float),
               T=df.T_C.values.astype(float), valid=(df.eis.values == 0))
    if dt is not None and abs(dt - float(cfg["dt_s"])) > 1e-9:
        b = bin_uniform(dict(out, step=np.zeros(len(out["t"]), int)), dt, extra=("T",), flags=("valid",))
        out = {k: b[k] for k in ("t", "I", "V", "T", "valid")}
    return dict(**out, T_C=float(np.mean(out["T"])), dt=float(cfg["dt_s"]) if dt is None else float(dt),
                V_rest=float(out["V"][0]), cell=cell, test="%s_%s" % (cell, kind), dataset="exp_Mohtat",
                meta=dict(folder=d, config=cfg, eis=(~out["valid"]).astype(int)))


# ---------------------------------------------------------------- interface
_LOADERS = {"exp_LP604765": _load_lp, "exp_Mohtat": _load_mohtat}
_CELLSETS = {"exp_LP604765": "lp604765_cellset.json", "exp_Mohtat": "mohtat_lm3c.json"}


def tests(dataset):
    if dataset == "exp_LP604765":
        return sorted(LP_TESTS)
    if dataset == "exp_Mohtat":
        return sorted("%s_%s" % _mohtat_split(re.sub(r"^DB_REAL_", "", f)) for f in os.listdir(MOHTAT_TESTS)
                      if re.fullmatch(r"DB_REAL_C\d{2}_\w+_BOL", f))
    raise KeyError(dataset)


def load(dataset, test, dt=None):
    if dataset not in _LOADERS:
        raise KeyError("unknown dataset %r (known: %s)" % (dataset, ", ".join(_LOADERS)))
    return _LOADERS[dataset](test, dt=dt)


def cellset_path(dataset, name=None):
    return root(dataset, "params", name or _CELLSETS[dataset])


def cellset(dataset, name=None):
    from . import cellset as _cs
    return _cs.load(cellset_path(dataset, name))


def default_cell(dataset):
    return LP_CELL if dataset == "exp_LP604765" else "C01"


def initial_charge(dataset, d, cs):
    """Charge [A.h] below the top of the cell's window at t = 0, by the identification's rule of the dataset.
    LP604765: HPPC fit tests at the fitted offset (dq_20 = 0, dq_00 / dq_40 from the set's fit_extras); held-out tests by
    inverting the OCV (entropic term at T_C included) at the rested voltage. Mohtat: cellset.q_start (top of the
    window if the test starts under current, else OCV inversion at the first voltage)."""
    from . import cellset as _cs
    if dataset == "exp_LP604765":
        rule = d["meta"]["dq"]
        if rule == "ocv":
            return q_from_ocv(cs, d["cell"], d["V_rest"], d["T_C"] + 273.15)
        return 0.0 if rule == "dq_20" else float(cs["fit_extras"][rule])
    return _cs.q_start(cs, d["cell"], float(d["I"][0]), float(d["V"][0]))


def ocv(cs, cell, q, T_K):
    """Rested voltage [V] of `cell` with q [A.h] below the top of its window at T_K: U_p(y) - U_n(x) plus the entropic
    term (T - T_ref) (dUdT0 + dUdT_n (x - 0.5)) when the set has one (LP604765)."""
    c = cs["cells"][cell]
    q = np.asarray(q, dtype=float)
    x = c["x100"] - q / c["Q_n"]; y = c["y100"] + q / c["Q_p"]
    xn, Un = np.asarray(cs["ocp"]["n"]["x"]), np.asarray(cs["ocp"]["n"]["U"])
    yp, Up = np.asarray(cs["ocp"]["p"]["y"]), np.asarray(cs["ocp"]["p"]["U"])
    U = np.interp(np.clip(y, yp[0], yp[-1]), yp, Up) - np.interp(np.clip(x, xn[0], xn[-1]), xn, Un)
    tp = cs.get("temperature", {})
    if "dUdT0" in tp:
        U = U + (T_K - cs["T_ref_K"]) * (tp["dUdT0"] + tp.get("dUdT_n", 0.0) * (x - 0.5))
    return U


def q_from_ocv(cs, cell, V, T_K, lo=-0.4, hi=0.6, n=4001):
    """Inverse of ocv on [lo, hi] A.h by the nearest grid point (as the LP identification's Model.dq_from_ocv)."""
    q = np.linspace(lo, hi, n)
    return float(q[int(np.argmin(np.abs(ocv(cs, cell, q, T_K) - V)))])


def bin_uniform(d, dt=1.0, extra=(), flags=()):
    """Bins (t, I, V, step) to a uniform grid of width dt: I and `extra` = bin means, V and step = the last sample of
    the bin, `flags` = all() over the bin; t = end of the bin. Empty bins are dropped (their gaps stay in t)."""
    t, V, I, st = d["t"], d["V"], d["I"], d["step"]
    t = t - t[0]
    edges = np.arange(0.0, t[-1] + dt, dt)
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 1)
    nb = len(edges)
    cnt = np.bincount(b, minlength=nb)
    ok = cnt > 0
    last = np.zeros(nb, dtype=int); last[b] = np.arange(len(t))
    out = dict(t=edges[ok] + dt, I=np.bincount(b, weights=I, minlength=nb)[ok] / cnt[ok], V=V[last[ok]], step=st[last[ok]])
    for k in extra:
        out[k] = np.bincount(b, weights=d[k], minlength=nb)[ok] / cnt[ok]
    for k in flags:
        out[k] = (np.bincount(b, weights=(~np.asarray(d[k], bool)).astype(float), minlength=nb)[ok] == 0)
    return out
