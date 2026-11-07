"""Datasets, parameter transforms and a process pool for the LP604765 identification.

FIT: the three HPPCs (from the first pulse, after the 1 h rest at the top of charge).
VAL: 1C cycling at 20 C (first cycles of the first ageing block, BoL), the 1C capacity discharges at 0 and 40 C.
Initial state: uniform. Fit tests: at the top of the window shifted by the test's dq (dq_20 = 0 by definition of the
window, dq_00 / dq_40 free: the CV at 4.2 V ends at C/20 with a different overpotential). Held-out tests start at the
end of their first rest and are placed by inverting the model OCV at that rested voltage (dq = "ocv").
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(ROOT, "results", "exp_LP604765")          # git-ignored, like the data
CACHE = os.path.join(RES, "profiles_v2.pkl")         # v2: 1-s steps under current

from lpmodel import DFNEKF as _ROOT  # noqa: E402  (puts dfnekf on the path)
from dfnekf.datasets import LP_FIT as FIT, LP_VAL as VAL  # noqa: E402

ALL = {**FIT, **VAL}

# parameters of the model (lpmodel.NOMINAL) plus the per-test start offsets
from lpmodel import BOUNDS as _B, NOMINAL as _N  # noqa: E402

BOUNDS = {k: v for k, v in _B.items() if k != "dq"}
# search box of the fit (narrower window than lpmodel.BOUNDS: outside it the window cannot hold the 40 C capacity)
BOUNDS.update(x100=(0.5, 0.9, "lin"), y100=(0.3, 0.47, "lin"), Q_n=(3.0, 6.0, "lin"), Q_p=(3.0, 5.0, "lin"),
              Ds_p=(1e-15, 1e-12, "log"), dq_40=(-0.2, 0.1, "lin"), dq_00=(-0.1, 0.25, "lin"))
NOMINAL = {k: v for k, v in _N.items() if k != "dq"}
NOMINAL.update(dq_40=0.0, dq_00=0.0)
ALL_NAMES = list(BOUNDS)


def to_unit(name, v):
    lo, hi, kind = BOUNDS[name]
    return (np.log(v / lo) / np.log(hi / lo)) if kind == "log" else (v - lo) / (hi - lo)


def from_unit(name, u):
    lo, hi, kind = BOUNDS[name]
    return lo * (hi / lo) ** u if kind == "log" else lo + u * (hi - lo)


def model_params(p):
    """Full parameter dict (lpmodel keys) + dq for one dataset is added by the caller."""
    return {k: v for k, v in p.items() if not k.startswith("dq_")}


def dq_of(p, ds, prof=None, model=None):
    if ds["dq"] == "ocv":
        return model.dq_from_ocv(model_params(p), ds["T"], prof["V_rest"])
    return 0.0 if ds["dq"] == "dq_20" else float(p[ds["dq"]])


def profiles(keys=None, rebuild=False):
    from dfnekf.datasets import lp_cut
    from lpmodel import prepare
    cache = {}
    if os.path.exists(CACHE) and not rebuild:
        cache = pickle.load(open(CACHE, "rb"))
    keys = list(ALL) if keys is None else keys
    changed = False
    for k in keys:
        if k in cache:
            continue
        ds = ALL[k]
        d = lp_cut(k)                    # from the start, with the rested voltage before it
        V_rest = d["V_rest"]
        cache[k] = prepare(d, start_step=ds.get("start_step", 0), end_step=ds.get("end_step"))
        cache[k]["V_rest"] = V_rest
        changed = True
    if changed:
        os.makedirs(RES, exist_ok=True)
        pickle.dump(cache, open(CACHE, "wb"))
    return {k: cache[k] for k in keys}


def weights(prof):
    """sqrt(dt / total time): the weighted RMS of a residual is the time-averaged RMS of the test."""
    return np.sqrt(prof["dt"] / prof["dt"].sum())


# ---- process pool (one Model per worker, profiles loaded once)
_W = {}


def _init(keys):
    from lpmodel import Model
    _W["m"] = Model()
    _W["prof"] = profiles(keys)


def _sim(args):
    key, p, full = args
    ds = ALL[key]
    try:
        out = _W["m"].simulate(model_params(p), _W["prof"][key], ds["T"], dq=dq_of(p, ds, _W["prof"][key], _W["m"]),
                               full=full)
    except Exception as e:  # noqa: BLE001
        return None, repr(e)
    return out, None


def pool(keys, n=12):
    import multiprocessing as mp
    profiles(keys)                      # build the cache once in the parent
    return mp.get_context("spawn").Pool(n, initializer=_init, initargs=(keys,))
