"""Export of an identified Mohtat parameter set (ParamIdentification study, e.g. mohtat_2step_fv_lm3c) to the
physical (SI) parameters of the grey-box DFN, as one self-contained JSON read by dfnekf.cellset.

    python scripts/export_mohtat_lm3c.py --study "<ParamIdentification>/runs/studies/mohtat_2step_fv_lm3c" \
        --out data/exp_Mohtat/params/mohtat_lm3c.json

Inputs (all three are needed): dynamics.json (step 2: groups, fitted/fixed, cost), step1.json (per-cell
stoichiometry windows), ocp_correction.json (the step-1 cathode OCP table). Plus PyBaMM 'Mohtat2020' (particle
radii, active volume fractions, solid conductivities, c_e0, anode OCP) and 'ORegan2022' (electrolyte material).

Mapping of the groups (ParamIdentification srcs/ms_model.py, geometry='groups', electrolyte_groups='material'):
  L_n, L_s, L_p   = L_e (1 - l_s) r_np, L_e l_s, L_e (1 - l_s) (1 - r_np)
  eps_e           = eps_n, eps_s, eps_p ; Bruggeman brugg in all three regions
  A (area)        = L_e r_hat / (kappa R_e), r_hat the dimensionless ohmic response of a uniform split on the
                    identification's own grid (5 slices x 10 volumes, 12 in the separator), kappa of ORegan2022
                    at c_e0, 25 C. This and L_e also give tau_e = L_e^2 / (D_e |lam_1|), as in the fit.
  Ds_n            = R_n^2 / theta_n ; Ds_p = R_p^2 / theta_p is D_p at the top of the cathode window (y <= y100c)
  j0_k            = I0_k / (A a_k L_k), k0_k = j0_k / (0.5 c_max,k sqrt(c_e0))  (j0 = k0 sqrt(c_e c_s (c_max - c_s)))
  capacity        : the voltage sees only eps_s c_max (through Q = eps_s A L c_max F), so eps_s is kept from
                    Mohtat2020 (capped at 1 - eps_e) and c_max is the EFFECTIVE value that gives each cell's Q_n, Q_p.
  R_contact       : series resistance outside the electrolyte (contacts, tabs, solid); the solid conductivities of
                    Mohtat2020 are kept (their drop is < 0.01 mOhm here).
  film            : lm3c has no film resistance: Rf0 = 0 (set a baseline before estimating Rf with the EKF).
Temperature laws (dfnekf.cellset.params_for applies them): Ds_n (Ea_Dn), Ds_p (Ea_Dp), k0 (Ea_r), R_contact (Ea_s),
electrolyte D_e(c_e0, T), kappa(c_e0, T) tabulated from ORegan2022 (electrolyte_T = 'set'); t+ and TDF at c_e0, 25 C;
entropic term 'cell': (T - T_ref) (dUdT0 + dUdT_n (x_s - 0.5) + dUdT_p (y_s - 0.5)), x_s, y_s mean surface values.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

F, R = 96485.33212, 8.314462618
T_REF = 298.15


def ev(f, *args):
    import pybamm
    if not callable(f):
        return float(f)
    return float(f(*[pybamm.Scalar(float(a)) for a in args]).evaluate())


def r_hat(l_s, r_np, eps_n, eps_s, eps_p, brugg, N=5, npc=10, ns=12):
    """Ohmic response of a uniform split of 1 A on the dimensionless grid x / L_e (ParamIdentification
    BatchMS._group_maps, in numpy): phi_e averaged over the anode slices minus over the cathode slices,
    divided by kappa / L_e."""
    l_n, l_p = (1 - l_s) * r_np, (1 - l_s) * (1 - r_np)
    nn = N * npc
    dx = np.r_[np.full(nn, l_n / nn), np.full(ns, l_s / ns), np.full(nn, l_p / nn)]
    eps = np.r_[np.full(nn, eps_n), np.full(ns, eps_s), np.full(nn, eps_p)]
    deff = eps ** brugg
    n = dx.size
    slab = np.zeros((n, 2 * N))
    for i in range(N):
        slab[i * npc:(i + 1) * npc, i] = 1.0
        slab[nn + ns + i * npc:nn + ns + (i + 1) * npc, N + i] = 1.0
    width = dx @ slab
    u = slab / width[None, :]
    W = (slab * dx[:, None] / width[None, :]).T
    cumint = lambda v: np.cumsum(v * dx[:, None], 0) - 0.5 * v * dx[:, None]
    P = W @ cumint(-cumint(u) / deff[:, None])
    jv = np.r_[np.full(N, 1.0 / N), np.full(N, -1.0 / N)]
    pe = P @ jv
    return float(pe[:N].mean() - pe[N:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help="runs/studies/<study> of ParamIdentification")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pm", default="Mohtat2020")
    ap.add_argument("--electrolyte", default="ORegan2022")
    a = ap.parse_args()
    import pybamm

    ld = lambda f: json.load(open(os.path.join(a.study, f), encoding="utf-8"))
    dyn, s1, oc = ld("dynamics.json"), ld("step1.json"), ld("ocp_correction.json")
    g = dyn["groups"]
    pm, pe = pybamm.ParameterValues(a.pm), pybamm.ParameterValues(a.electrolyte)

    ce0 = float(pm["Initial concentration in electrolyte [mol.m-3]"])
    kappa_ref = ev(pe["Electrolyte conductivity [S.m-1]"], ce0, T_REF)
    De_ref = ev(pe["Electrolyte diffusivity [m2.s-1]"], ce0, T_REF)
    tplus = ev(pe["Cation transference number"], ce0, T_REF)
    tdf = ev(pe["Thermodynamic factor"], ce0, T_REF)

    L_e, l_s, r_np = g["L_e"], g["l_s"], g["r_np"]
    Ln, Ls, Lp = L_e * (1 - l_s) * r_np, L_e * l_s, L_e * (1 - l_s) * (1 - r_np)
    rh = r_hat(l_s, r_np, g["eps_n"], g["eps_s"], g["eps_p"], g["brugg"])
    A = L_e * rh / (kappa_ref * g["R_e"])

    Rn, Rp = float(pm["Negative particle radius [m]"]), float(pm["Positive particle radius [m]"])
    eps_sn = min(float(pm["Negative electrode active material volume fraction"]), 1.0 - g["eps_n"])
    eps_sp = min(float(pm["Positive electrode active material volume fraction"]), 1.0 - g["eps_p"])
    a_n, a_p = 3 * eps_sn / Rn, 3 * eps_sp / Rp
    j0n, j0p = g["I0_n"] / (A * a_n * Ln), g["I0_p"] / (A * a_p * Lp)

    cells = {}
    for c, w in sorted(s1["cells"].items()):
        cmax_n = w["Q_n"] * 3600 / (F * A * Ln * eps_sn)
        cmax_p = w["Q_p"] * 3600 / (F * A * Lp * eps_sp)
        cells[c] = dict(Q_n=w["Q_n"], Q_p=w["Q_p"], x100=w["x100"], y100=w["y100"], x0=w.get("x0"), y0=w.get("y0"),
                        cmax_n=cmax_n, cmax_p=cmax_p,
                        k0_n=j0n / (0.5 * cmax_n * np.sqrt(ce0)), k0_p=j0p / (0.5 * cmax_p * np.sqrt(ce0)),
                        ocv_fit_rmse_mV=w.get("rmse_mV"), C20_leg=w.get("C20_leg"), q_ocv_Ah=w.get("q_ocv_Ah"))

    # anode OCP: Mohtat2020 tabulated on the grid ParamIdentification uses (physics.ocps, 8001 points)
    xs = np.linspace(1e-5, 1 - 1e-5, 8001)
    Un = [ev(pm["Negative electrode OCP [V]"], x) for x in xs]
    # electrolyte temperature law at c_e0 (electrolyte_T = 'set')
    TK = np.arange(258.15, 333.15 + 1e-9, 2.5)
    e_T = dict(T_K=TK.tolist(), kappa=[ev(pe["Electrolyte conductivity [S.m-1]"], ce0, t) for t in TK],
               De=[ev(pe["Electrolyte diffusivity [m2.s-1]"], ce0, t) for t in TK])
    dl = s1["config"]
    out = {
        "schema": "dfnekf-cellset-1",
        "id": os.path.basename(os.path.normpath(a.study)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"study": os.path.abspath(a.study), "pm": a.pm, "electrolyte_set": a.electrolyte,
                   "pybamm": pybamm.__version__, "cost_mV": dyn["cost_mV"], "fitted": dyn["fitted"],
                   "rmse_per_test_mV": dyn.get("rmse_per_test"),
                   "rel_std_lower_bound": dyn.get("identifiability", {}).get("rel_std_lower_bound")},
        "groups": g,
        "T_ref_K": T_REF,
        "P": dict(Ln=Ln, Ls=Ls, Lp=Lp, Rn=Rn, Rp=Rp, eps_en=g["eps_n"], eps_es=g["eps_s"], eps_ep=g["eps_p"],
                  eps_sn=eps_sn, eps_sp=eps_sp, brug=g["brugg"],
                  sig_n=float(pm["Negative electrode conductivity [S.m-1]"]),
                  sig_p=float(pm["Positive electrode conductivity [S.m-1]"]),
                  alpha=0.5, Ds_n=Rn ** 2 / g["theta_n"], Ds_p=Rp ** 2 / g["theta_p"],
                  De=De_ref, kappa=kappa_ref, tplus=tplus, tdf=tdf, ce0=ce0, Rf0=0.0, area=A,
                  R_contact=g["R_contact"]),
        "derived": dict(r_hat=rh, j0_n=j0n, j0_p=j0p, L_e=L_e,
                        R_e_check=L_e * rh / (kappa_ref * A)),
        "temperature": dict(Ea_Dn=g["Ea_Dn"], Ea_Dp=g["Ea_Dp"], Ea_r=g["Ea_r"], Ea_s=g.get("Ea_s", 0.0),
                            dUdT0=g["dUdT0"], dUdT_n=g["dUdT_n"], dUdT_p=g.get("dUdT_p", 0.0), electrolyte=e_T),
        "dp_law": dict(drop=g["Dp_drop"], shape=g["Dp_shape"], y0=dl["y100"], y1=dl["y_end"],
                       note="D_p = Ds_p 10^(-drop u^shape), u = max(y - y0, 0) / (y1 - y0), y the particle's mean "
                            "stoichiometry at the start of the step (frozen over the step)"),
        "ocp": {"n": {"x": xs.tolist(), "U": Un, "source": "%s Negative electrode OCP [V]" % a.pm},
                "p": {"y": oc["table"]["y"], "U": oc["table"]["U"], "y_observed": oc.get("y_observed"),
                      "source": "ocp_correction.json table (step 1)"}},
        "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", a.out)
    print("A = %.4f m2, L = %.1f/%.1f/%.1f um, r_hat = %.4f, Ds_n = %.3e, Ds_p = %.3e, j0 = %.3g/%.3g A/m2"
          % (A, Ln * 1e6, Ls * 1e6, Lp * 1e6, rh, Rn ** 2 / g["theta_n"], Rp ** 2 / g["theta_p"], j0n, j0p))
    c = cells["C01"]
    print("C01: cmax_eff = %.0f / %.0f, k0 = %.3e / %.3e" % (c["cmax_n"], c["cmax_p"], c["k0_n"], c["k0_p"]))


if __name__ == "__main__":
    main()
