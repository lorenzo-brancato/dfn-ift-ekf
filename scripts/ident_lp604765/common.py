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
CELL = os.environ.get("LP_CELL", "lp604765")         # lgm50: the LG M50 of scripts/ident_lgm50 (same model and fit)
if CELL == "lgm50":
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lgm50"))
    import m50data as M50  # noqa: E402
    RES = M50.RES
    for k_, v_ in dict(LP_BASE_SET=os.path.join(RES, "base_chen2020.json"), LP_Q_CELL="5.0", LP_BASE_Q="5.0",
                       LP_T_REF_C="25", LP_DQ_RANGE="-0.5,6.0", LP_TMEAS="1").items():
        os.environ.setdefault(k_, v_)                # read by lpmodel at import (and by the spawned workers)
else:
    RES = os.path.join(ROOT, "results", "exp_LP604765")          # git-ignored, like the data
CACHE = os.path.join(RES, "profiles_v2.pkl")         # v2: 1-s steps under current

from lpmodel import DFNEKF as _ROOT  # noqa: E402  (puts dfnekf on the path)
if CELL == "lgm50":
    FIT, VAL = M50.FIT, M50.VAL
    ALL = dict(M50.TESTS)
    TEST_SETS = {"fit": list(FIT), **{"fit_" + c: [k for k in FIT if FIT[k]["cell"] == c] for c in M50.FIT_CELLS}}
else:
    from dfnekf.datasets import LP_FIT as FIT, LP_VAL as VAL, LP_FIT_B as FIT_B, LP_VAL_B as VAL_B, LP_COLD as COLD  # noqa: E402
    ALL = {**FIT, **VAL, **FIT_B, **VAL_B, **COLD}
    TEST_SETS = {"hppc": list(FIT), "B": list(FIT) + list(FIT_B)}     # fit sets (fit.py --tests)

# parameters of the model (lpmodel.NOMINAL) plus the per-test start offsets
from lpmodel import BOUNDS as _B, NOMINAL as _N  # noqa: E402

BOUNDS = {k: v for k, v in _B.items() if k != "dq"}
# search box of the fit (narrower window than lpmodel.BOUNDS: outside it the window cannot hold the 40 C capacity)
BOUNDS.update(dq_20=(-0.2, 0.2, "lin"), x100=(0.5, 0.9, "lin"), y100=(0.3, 0.47, "lin"), Q_n=(3.0, 6.0, "lin"), Q_p=(3.0, 5.0, "lin"),
              Ds_p=(1e-15, 1e-12, "log"), dq_40=(-0.2, 0.1, "lin"), dq_00=(-0.1, 0.25, "lin"))
NOMINAL = {k: v for k, v in _N.items() if k != "dq"}
NOMINAL.update(dq_40=0.0, dq_00=0.0, dq_20=0.0)     # dq_20: fitted only when the window is fixed from another cell
WINDOWS = {}                                         # per-cell electrode windows (LG M50: one per cell, from its GITT rests)
if CELL == "lgm50":
    for k_ in ("dq_20", "dq_40", "dq_00"):
        BOUNDS.pop(k_, None); NOMINAL.pop(k_, None)
    # Chen2020 values (per-area kinetics through the k0 multipliers), Arrhenius starts, window boxes for a 5 Ah cell
    BOUNDS.update(x100=(0.6, 1.0, "lin"), y100=(0.15, 0.45, "lin"), Q_n=(4.5, 8.0, "lin"), Q_p=(4.5, 10.0, "lin"),
                  Ds_n=(1e-15, 1e-12, "log"), Ds_p=(1e-17, 1e-13, "log"), R_contact=(1e-4, 3e-2, "log"),
                  th_C=(20.0, 200.0, "log"))
    # literature boxes (LP_LIT_BOUNDS=1; values at 25 C, PyBaMM sets Chen2020, ORegan2022, OKane2022, Ecker2015,
    # Mohtat2020, Kim2011). The cell windows (x range 0.03-0.88, y 0.26-0.92) set the stoichiometries:
    #   Ds_n  ORegan2022 GITT 2.6e-16..2.7e-14 over the window, Chen2020 3.3e-14            -> 2e-16 .. 1e-13
    #   Ds_p  (value at y100) ORegan2022 2.8e-15..1.5e-14, Chen2020 4e-15                   -> 1e-15 .. 5e-14
    #   drop  ORegan2022 no drop up to y 0.95; NMC diffusivity up to ~3 decades over lithiation -> 0 .. 3
    #   k0    multipliers on the Chen2020 j0: ORegan2022 anode x3.9, cathode x0.74          -> n 0.3 .. 12, p 0.25 .. 3
    #   tau   on the Bruggeman (1.5) kappa, D_e: measured electrode tortuosities 1-5x Bruggeman -> 0.8 .. 5
    #   Ea_Dn 17.4 (ORegan), 30.3 (Chen/OKane, Ecker), 42.8 (Dualfoil, Mohtat), 48 (SiOx)  -> 10 .. 60 kJ/mol
    #   Ea_Dp 12.0 (ORegan), 18.6 (Mohtat), 20 (Kim), 25 (OKane), 80.6 (Ecker NCO)         -> 8 .. 85 kJ/mol
    #   Ea_r  (one for both electrodes) anode 35-53, cathode 17.8-43.6                     -> 15 .. 60 kJ/mol
    #   R_contact: not in the literature sets (Chen2020 has none): the wide box is kept; shape is a law parameter.
    if os.environ.get("LP_LIT_BOUNDS", "1") == "1":
        BOUNDS.update(Ds_n=(2e-16, 1e-13, "log"), Ds_p=(1e-15, 5e-14, "log"), drop=(0.0, 3.0, "lin"),
                      shape=(0.5, 8.0, "lin"), k0_n=(0.3, 12.0, "log"), k0_p=(0.25, 3.0, "log"), tau=(0.8, 5.0, "log"),
                      Ea_Dn=(10e3, 60e3, "lin"), Ea_Dp=(8e3, 85e3, "lin"), Ea_r=(15e3, 60e3, "lin"))
    # Arrhenius starts: ORegan2022 (LG M50): D_n 17.4, D_p 12.0 kJ/mol; j0 40 (anode) / 24 (cathode) -> one Ea_r of 32;
    # the series resistance without a law (Ea_s 0). The cell temperature is the measured surface temperature (LP_TMEAS).
    NOMINAL.update(Ds_n=3.3e-14, Ds_p=4e-15, drop=0.0, shape=1.0, k0_n=1.0, k0_p=1.0, R_contact=5e-3, tau=1.0,
                   Ea_Dn=17.4e3, Ea_Dp=12.0e3, Ea_r=32e3, Ea_s=0.0, dUdT0=0.0, dUdT_n=0.0,
                   x100=0.9014, y100=0.27, Q_n=5.828, Q_p=8.732)
    CACHE = os.path.join(RES, "profiles_tm.pkl")      # with the measured temperature per step
    _wf = os.path.join(RES, os.environ.get("LP_WINDOWS", "windows.json"))
    if os.path.exists(_wf):
        import json as _json
        _W_ = _json.load(open(_wf))
        WINDOWS = {c: {k_: float(w[k_]) for k_ in ("x100", "y100", "Q_n", "Q_p")} for c, w in _W_["cells"].items()}
        NOMINAL.update(dUdT0=float(_W_["dUdT0"]), dUdT_n=float(_W_["dUdT_n"]))     # shared entropic terms of the OCV step
        os.environ.setdefault("LP_OCP_P", _W_["ocp_p"])                           # the cathode table the windows were fitted with
        if _W_.get("ocp_n"):
            os.environ.setdefault("LP_OCP_N", _W_["ocp_n"])                       # and the anode table
GROUP_MODE = CELL != "lgm50" and os.environ.get("LP_GROUPS") == "1"
if GROUP_MODE:
    # LP604765 in parameter groups (ApEn draft, Section 2.3; lpmodel.Model.phys): the dynamics are identified as
    #   theta_n, theta_p = R^2 / D_s [s] (radius and diffusivity), I0_n, I0_p = A a L j0 [A] (area, specific surface,
    #   thickness, rate constant), R_e [Ohm] and tau_e [s] (conductivity / salt diffusivity with the stack: porosities,
    #   Bruggeman, thicknesses), R_c [Ohm]; temperature laws Ea_Dn, Ea_Dp, Ea_r, Ea_s; cathode law drop, shape.
    # Fixed: the stack shape, radii, t+, chi, c_e0 (Mohtat lm3c), the OCPs (anode Mohtat2020 table, cathode by
    # subtraction from the C/20), the window, entropic terms and HPPC offsets of step 1 (step1_ocv_c20_an_Mohtat2020).
    # No thermal model: every test at its chamber temperature.
    # Boxes, literature at 20-25 C (PyBaMM sets Chen2020, Ecker2015, Marquis2019, Mohtat2020, ORegan2022), mapped onto
    # this stack (a_n L_n A = 5.66 m2, a_p L_p A = 3.07 m2, R_n 2.5 um, R_p 3.5 um):
    #   theta_n: Mohtat lm3c 10 s, Chen 1.0e3 s, Ecker 4.8e3 s, Marquis 1e4 s              -> 10 .. 3e4 s
    #   theta_p: Mohtat2020 25 s, Marquis 1e3 s, Ecker 4e3 s, Chen 6.8e3 s                 -> 10 .. 2e4 s
    #   I0_n:    j0 at half stoichiometry 0.3 (Chen) .. 24 (lm3c) A/m2, box 0.1 .. 50 A/m2     -> 0.57 .. 283 A
    #   I0_p:    j0 0.48 (lm3c) .. 3.4 (Chen) A/m2, box 0.05 .. 20 A/m2                       -> 0.15 .. 61 A
    #   R_e, tau_e: measured tortuosities 1-5 x Bruggeman (b = 1.5)                          -> 0.6 .. 6 x the b = 1.5 value
    #   Ea_Dn 10-60, Ea_Dp 8-85, Ea_r 15-60 kJ/mol (ORegan, Chen/OKane, Ecker, Mohtat, Kim); Ea_s 0-60 kJ/mol
    import json as _json
    _R = os.path.join(ROOT, "results", "exp_LP604765")
    _AN = os.environ.get("LP_ANODE", "Mohtat2020")          # anode OCP table; step 1 and cathode OCP redone with it
    os.environ.setdefault("LP_OCP_N", os.path.join(_R, "ocp_neg", "ocp_neg_%s.json" % _AN))
    os.environ.setdefault("LP_OCP_P", os.path.join(_R, "ocp", "ocp_p_c20_an_%s.json" % _AN))
    os.environ.setdefault("LP_OCP_MONO", "1")           # cathode OCP made strictly decreasing (lpmodel._monotone): no closure stalls at rest
    # entropic term: the dOCV/dT measured on the HPPC rests (ocv_temperature.py), not the linear law of step 1
    # (LP_DUDT_TABLE= empty: the step-1 law dUdT0 + dUdT_n (x - 0.5))
    os.environ.setdefault("LP_DUDT_TABLE", os.path.join(_R, "ocv_T", "dUdT_table.json" if _AN == "Mohtat2020"
                                                         else "dUdT_table_an_%s.json" % _AN))
    if not os.environ["LP_DUDT_TABLE"]:
        os.environ.pop("LP_DUDT_TABLE")
    _S1 = _json.load(open(os.path.join(_R, "step1_ocv_c20_an_%s.json" % _AN)))["params"]
    from lpmodel import Model as _M
    _G0 = _M().G0
    for k_ in ("Ds_n", "Ds_p", "k0_n", "k0_p", "tau", "R_contact", "dq_20", "hys_H", "hys_g", "th_C", "th_hA"):
        if os.environ.get("LP_THERMAL") == "1" and k_ == "th_hA":
            continue                                  # lumped thermal state on (LP_THERMAL=1 before import): th_hA can be fitted
        if os.environ.get("LP_HYS") == "1" and k_ in ("hys_H", "hys_g"):
            continue                                  # OCV hysteresis on (LP_HYS=1): charge-branch offset and rate can be fitted
        BOUNDS.pop(k_, None)
    BOUNDS.update(theta_n=(10.0, 3e4, "log"), theta_p=(10.0, 2e4, "log"), I0_n=(0.57, 283.0, "log"), I0_p=(0.15, 61.0, "log"),
                  R_e=(0.6 * _G0["R_e"], 6 * _G0["R_e"], "log"), tau_e=(0.6 * _G0["tau_e"], 6 * _G0["tau_e"], "log"),
                  R_c=(1e-4, 4e-2, "log"), Ea_Dn=(10e3, 60e3, "lin"), Ea_Dp=(8e3, 85e3, "lin"), Ea_r=(15e3, 60e3, "lin"),
                  Ea_s=(0.0, 60e3, "lin"), drop=(0.0, 3.0, "lin"), shape=(0.5, 4.0, "lin"))
    for k_ in ("Ds_n", "Ds_p", "k0_n", "k0_p", "tau", "R_contact"):
        NOMINAL.pop(k_, None)
    # reference point: lm3c (Mohtat) groups on this stack, literature-mid activation energies
    NOMINAL.update(theta_n=10.15, theta_p=126.9, I0_n=_G0["I0_n"], I0_p=_G0["I0_p"], R_e=_G0["R_e"], tau_e=_G0["tau_e"],
                   R_c=3.2e-3, Ea_Dn=30e3, Ea_Dp=30e3, Ea_r=35e3, Ea_s=10e3, drop=1.55, shape=1.37,
                   hys_H=0.0, hys_g=50.0, th_C=40.0, th_hA=0.1, dq_20=0.0, **_S1)
    STEP1 = ("x100", "y100", "Q_n", "Q_p", "dUdT0", "dUdT_n", "dq_40", "dq_00")
    GROUPS = [k_ for k_ in BOUNDS if k_ not in STEP1]          # the 13 dynamic candidates of step 2
ALL_NAMES = list(BOUNDS)


def params_for(p, ds):
    """The parameters of one test: p with the window of the test's cell (WINDOWS[ds['win']]) when there is one."""
    w = WINDOWS.get(ds.get("win")) if WINDOWS else None
    return {**p, **w} if w else p


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
    if isinstance(ds["dq"], (int, float)):
        return float(ds["dq"])
    if ds["dq"] == "ocv":
        return model.dq_from_ocv(model_params(params_for(p, ds)), ds["T"], prof["V_rest"])
    return float(p.get(ds["dq"], 0.0))


def profiles(keys=None, rebuild=False):
    from lpmodel import prepare
    if CELL != "lgm50":
        from dfnekf.datasets import lp_cut
    cache = {}
    if os.path.exists(CACHE) and not rebuild:
        cache = pickle.load(open(CACHE, "rb"))
    keys = list(ALL) if keys is None else keys
    changed = False
    for k in keys:
        if k in cache:
            continue
        ds = ALL[k]
        d = M50.cut(k) if CELL == "lgm50" else lp_cut(k)     # from the start, with the rested voltage before it
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
    p = params_for(p, ds)
    try:
        out = _W["m"].simulate(model_params(p), _W["prof"][key], ds["T"], dq=dq_of(p, ds, _W["prof"][key], _W["m"]),
                               full=full)
    except Exception as e:  # noqa: BLE001
        return None, repr(e)
    return out, None


def _vjac(args):
    """Exact (IFT) sensitivities of one test: V and dV/dp for the groups `names` at the full parameter set p
    (physical values); the start charge by its rule (fixed, fitted dq_xx, or OCV inversion with IFT)."""
    key, names, p, _ = args
    import jax
    import jax.numpy as jnp
    if "m_ift" not in _W:
        from lpmodel import Model
        _W["m_ift"] = Model(ift=True); _W["m_ift"]._base(); _W["jac"] = {}
    m = _W["m_ift"]; ds = ALL[key]; pr = _W["prof"][key]
    jk = (key, tuple(names))
    if jk not in _W["jac"]:
        I, dt, idx = (jnp.asarray(pr[k_]) for k_ in ("I", "dt", "idx"))
        Tm = jnp.asarray(pr["Tm"]) if (m.tmeas and "Tm" in pr) else None    # measured temperature input (LP_TMEAS=1)

        def f(x, rest):
            pp = {**rest, **dict(zip(names, list(x)))}
            pm = {k_: v for k_, v in pp.items() if not k_.startswith("dq_")}
            if isinstance(ds["dq"], (int, float)):
                dq = jnp.asarray(float(ds["dq"]))
            elif ds["dq"] == "ocv":
                dq = m.dq_from_ocv_jax(pm, ds["T"], pr["V_rest"])
            else:
                dq = pp.get(ds["dq"], jnp.asarray(0.0))
            V = m._run(m.traced_jax(pm, ds["T"], dq), I, dt, idx, False, Tm)[0]
            return V, V
        _W["jac"][jk] = jax.jit(jax.jacfwd(f, has_aux=True))
    p = {**p, **{k_: v_ for k_, v_ in (WINDOWS.get(ds.get("win")) or {}).items() if k_ not in names}}   # the cell's window
    x = jnp.asarray([p[n_] for n_ in names], dtype=jnp.float64)
    rest = {k_: jnp.asarray(v, dtype=jnp.float64) for k_, v in p.items() if k_ not in names}
    try:
        J, V = _W["jac"][jk](x, rest)
        return (np.asarray(V), np.asarray(J)), None
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def pool(keys, n=12):
    import multiprocessing as mp
    profiles(keys)                      # build the cache once in the parent
    return mp.get_context("spawn").Pool(n, initializer=_init, initargs=(keys,))
