"""Cell parameter sets other than Chen2020 (the paper's cell), as exported by scripts/export_mohtat_lm3c.py.

    cs = cellset.load("data/exp_Mohtat/params/mohtat_lm3c.json")
    P = cellset.params_for(cs, "C01", T_K=296.0)        # SI parameters of the DFN at temperature T_K
    c0n, c0p = cellset.c0_top(cs, "C01")                # electrode concentrations at the top of charge (rested)

P carries, besides the usual keys of params.base(): "ocp_n", "ocp_p" (JAX interpolants of the set's OCP tables),
"T" [K], "R_contact" [Ohm], "entropic" (dT, dUdT0, dUdT_n, dUdT_p) and "dp_law" (the cathode D_s(y) law), which
closure.Alg and step.DFN read when present. The temperature is held constant over a run: the Arrhenius factors
of Ds_n, Ds_p (at the top of the window), k0 (both electrodes) and R_contact, and the electrolyte's kappa(c_e0, T),
D_e(c_e0, T), are applied here, once.
"""
import json

import numpy as np
import jax.numpy as jnp

R_GAS = 8.314462618


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _table(x, U):
    x, U = jnp.asarray(x, dtype=jnp.float64), jnp.asarray(U, dtype=jnp.float64)
    lo, hi = float(x[0]), float(x[-1])
    return lambda s: jnp.interp(jnp.clip(s, lo, hi), x, U)


def film_area(P):
    """A a_n L_n [m2]: a uniform film of resistance Rf [Ohm m2] acts in series as Rf / (A a_n L_n) [Ohm]."""
    return P["area"] * 3 * P["eps_sn"] / P["Rn"] * P["Ln"]


def params_for(cs, cell, T_K=None, film_share=0.0, ds_on="n", ea=None):
    """DFN parameters (dict) of `cell` at the constant temperature T_K (default: the set's reference).
    film_share: part of R_contact moved into an anode film Rf0 (series-equivalent at T_ref), so that the EKF's Rf multiplier
    has a non-zero baseline; the total series resistance at nominal is unchanged (the film acts in the kinetics, so the
    distribution changes slightly). ds_on: the electrode whose D_s the multiplier theta[2] scales ('p' for the Mohtat sets:
    theta_p is identified within 2 %, theta_n not at all)."""
    P = dict(cs["P"])
    c = cs["cells"][cell]
    P.update(cmax_n=c["cmax_n"], cmax_p=c["cmax_p"], k0_n=c["k0_n"], k0_p=c["k0_p"])
    if film_share > 0:
        P["Rf0"] = film_share * P["R_contact"] * film_area(P)
        P["R_contact"] = (1 - film_share) * P["R_contact"]
    P["ds_on"] = ds_on
    P["dx_window_n"] = c["x100"] - c["x0"]          # the anode window: LLI's unit in the EKF
    T_ref = cs["T_ref_K"]
    T = T_ref if T_K is None else float(T_K)
    t = cs["temperature"]
    if ea:                                        # override of the set's activation energies, e.g. {"Ea_Dp": 48e3} [J/mol]
        t = dict(t, **ea)
    ark = (1.0 / T - 1.0 / T_ref) / R_GAS
    P["Ds_n"] *= np.exp(-t["Ea_Dn"] * ark)
    P["Ds_p"] *= np.exp(-t["Ea_Dp"] * ark)
    P["k0_n"] *= np.exp(-t["Ea_r"] * ark)
    P["k0_p"] *= np.exp(-t["Ea_r"] * ark)
    P["R_contact"] *= np.exp(t["Ea_s"] * ark)
    e = t["electrolyte"]
    P["kappa"] = float(np.interp(T, e["T_K"], e["kappa"]))
    P["De"] = float(np.interp(T, e["T_K"], e["De"]))
    P["T"] = T
    P["soft_edges"] = True
    P["soft_eps"] = 1e-2      # edge width [c_max]: 1e-3 leaves the closure ill-conditioned (cond ~5e11) when a fast CV charge
                              # takes the separator-side anode surface past c_max, and the solve fails at random
    P["implicit_closure"] = True
    P["entropic"] = (T - T_ref, t["dUdT0"], t["dUdT_n"], t["dUdT_p"])
    P["dp_law"] = {k: float(cs["dp_law"][k]) for k in ("drop", "shape", "y0", "y1")}
    P["ocp_n"] = _table(cs["ocp"]["n"]["x"], cs["ocp"]["n"]["U"])
    P["ocp_p"] = _table(cs["ocp"]["p"]["y"], cs["ocp"]["p"]["U"])
    return P


def c0_top(cs, cell):
    """(c_n, c_p) at the top of charge: x100 c_max,n and y100 c_max,p (the cell's step-1 window)."""
    c = cs["cells"][cell]
    return c["x100"] * c["cmax_n"], c["y100"] * c["cmax_p"]


def c0_at(cs, cell, q):
    """(c_n, c_p) with q [A.h] discharged below the top of the window: x = x100 - q / Q_n, y = y100 + q / Q_p."""
    c = cs["cells"][cell]
    return (c["x100"] - q / c["Q_n"]) * c["cmax_n"], (c["y100"] + q / c["Q_p"]) * c["cmax_p"]


def q_start(cs, cell, I0, V0, i_rest=0.02, n=4000):
    """Charge [A.h] below the top of the window at t = 0, as ParamIdentification identification.initial_charges: a test
    that starts under current (|I0| > i_rest) starts AT the top (q = 0); one that starts at rest is placed by inverting
    the window's OCV at its first voltage V0."""
    if abs(I0) > i_rest:
        return 0.0
    c = cs["cells"][cell]
    xn, Un = np.asarray(cs["ocp"]["n"]["x"]), np.asarray(cs["ocp"]["n"]["U"])
    yp, Up = np.asarray(cs["ocp"]["p"]["y"]), np.asarray(cs["ocp"]["p"]["U"])
    q = np.linspace(0.0, 1.0, n) * 0.99 * min(c["Q_n"] * c["x100"], c["Q_p"] * (1 - c["y100"]))
    v = (np.interp(np.clip(c["y100"] + q / c["Q_p"], yp[0], yp[-1]), yp, Up)
         - np.interp(np.clip(c["x100"] - q / c["Q_n"], xn[0], xn[-1]), xn, Un))
    if V0 >= v[0]:
        return 0.0
    i = int(np.argmin(np.abs(v - V0)))
    j = int(np.clip(i + 1 if v[i] > V0 else i - 1, 0, n - 1))
    den = v[j] - v[i]
    frac = (V0 - v[i]) / den if abs(den) > 1e-12 else 0.0
    return float(max(q[i] + np.clip(frac, 0.0, 1.0) * (q[j] - q[i]), 0.0))
