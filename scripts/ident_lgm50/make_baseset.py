"""Base cell set of the LG M50 identification from PyBaMM 'Chen2020' (the LG M50T parametrisation), in the schema
of data/exp_Mohtat/params/mohtat_lm3c.json that scripts/ident_lp604765/lpmodel.py reads (env LP_BASE_SET).

    python make_baseset.py            # -> results/exp_LGM50/base_chen2020.json

Geometry, porosities, radii, solid conductivities, c_max, kinetics (j0 = k0 sqrt(c_e c_s (c_max - c_s)), the PyBaMM
Chen2020 form) and the OCPs of Chen2020; area = electrode height x width. Electrolyte: ORegan2022 (LP57 of the LG M50
study), tabulated at c_e0 against T (kappa, D_e); t+ and the thermodynamic factor at c_e0, 25 C. Cathode diffusion law
D_p(y) off (drop 0; Chen2020 has a constant D_p). Cell 'C01' holds the Chen2020 window and q_ocv_Ah = 5.0, so the model's
area scale (Q_CELL / q_ocv_Ah) is 1. No contact resistance in Chen2020: R_contact is a fitted group.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

from m50data import RES

F = 96485.33212
T_REF = 298.15


def ev(f, *args):
    import pybamm
    if not callable(f):
        return float(f)
    return float(f(*[pybamm.Scalar(float(a)) for a in args]).evaluate())


def main():
    import pybamm
    pm, pe = pybamm.ParameterValues("Chen2020"), pybamm.ParameterValues("ORegan2022")
    g = lambda k: float(pm[k])
    ce0 = g("Initial concentration in electrolyte [mol.m-3]")
    A = g("Electrode height [m]") * g("Electrode width [m]")
    Ln, Ls, Lp = g("Negative electrode thickness [m]"), g("Separator thickness [m]"), g("Positive electrode thickness [m]")
    eps_sn, eps_sp = g("Negative electrode active material volume fraction"), g("Positive electrode active material volume fraction")
    cmax_n, cmax_p = g("Maximum concentration in negative electrode [mol.m-3]"), g("Maximum concentration in positive electrode [mol.m-3]")
    Qn = eps_sn * A * Ln * cmax_n * F / 3600; Qp = eps_sp * A * Lp * cmax_p * F / 3600
    x100 = g("Initial concentration in negative electrode [mol.m-3]") / cmax_n
    y100 = g("Initial concentration in positive electrode [mol.m-3]") / cmax_p
    # Chen2020 exchange current: m * c_e^0.5 * c_s^0.5 * (c_max - c_s)^0.5 -> k0 = m (evaluated as j0 / sqrt(...))
    cs_n, cs_p = 0.5 * cmax_n, 0.5 * cmax_p
    j0n = ev(pm["Negative electrode exchange-current density [A.m-2]"], ce0, cs_n, cmax_n, T_REF)
    j0p = ev(pm["Positive electrode exchange-current density [A.m-2]"], ce0, cs_p, cmax_p, T_REF)
    k0n, k0p = j0n / np.sqrt(ce0 * cs_n * (cmax_n - cs_n)), j0p / np.sqrt(ce0 * cs_p * (cmax_p - cs_p))
    TK = np.arange(258.15, 333.15 + 1e-9, 2.5)
    e_T = dict(T_K=TK.tolist(), kappa=[ev(pe["Electrolyte conductivity [S.m-1]"], ce0, t) for t in TK],
               De=[ev(pe["Electrolyte diffusivity [m2.s-1]"], ce0, t) for t in TK])
    xs = np.linspace(1e-4, 1 - 1e-4, 4001)
    Un = [ev(pm["Negative electrode OCP [V]"], x) for x in xs]
    Up = [ev(pm["Positive electrode OCP [V]"], y) for y in xs]
    P = dict(Ln=Ln, Ls=Ls, Lp=Lp, Rn=g("Negative particle radius [m]"), Rp=g("Positive particle radius [m]"),
             eps_en=g("Negative electrode porosity"), eps_es=g("Separator porosity"), eps_ep=g("Positive electrode porosity"),
             eps_sn=eps_sn, eps_sp=eps_sp, brug=1.5,
             sig_n=g("Negative electrode conductivity [S.m-1]"), sig_p=g("Positive electrode conductivity [S.m-1]"),
             alpha=0.5, Ds_n=ev(pm["Negative particle diffusivity [m2.s-1]"], 0.5, T_REF),
             Ds_p=ev(pm["Positive particle diffusivity [m2.s-1]"], 0.5, T_REF),
             De=float(np.interp(T_REF, TK, e_T["De"])), kappa=float(np.interp(T_REF, TK, e_T["kappa"])),
             tplus=ev(pe["Cation transference number"], ce0, T_REF), tdf=ev(pe["Thermodynamic factor"], ce0, T_REF),
             ce0=ce0, Rf0=0.0, area=A, R_contact=0.0)
    out = {
        "schema": "dfnekf-cellset-1", "id": "lgm50_chen2020_base", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": dict(pm="Chen2020", electrolyte_set="ORegan2022", pybamm=pybamm.__version__,
                       note="base set of the LG M50 identification (scripts/ident_lgm50/make_baseset.py)"),
        "T_ref_K": T_REF, "P": P,
        "temperature": dict(Ea_Dn=0.0, Ea_Dp=0.0, Ea_r=0.0, Ea_s=0.0, dUdT0=0.0, dUdT_n=0.0, dUdT_p=0.0, electrolyte=e_T),
        "dp_law": dict(drop=0.0, shape=1.0, y0=y100, y1=0.999, note="off (constant D_p)"),
        "ocp": {"n": dict(x=xs.tolist(), U=Un, source="Chen2020 Negative electrode OCP [V]"),
                "p": dict(y=xs.tolist(), U=Up, source="Chen2020 Positive electrode OCP [V]")},
        "cells": {"C01": dict(Q_n=Qn, Q_p=Qp, x100=x100, y100=y100, cmax_n=cmax_n, cmax_p=cmax_p, k0_n=k0n, k0_p=k0p,
                              q_ocv_Ah=5.0)},
    }
    os.makedirs(RES, exist_ok=True)
    f = os.path.join(RES, "base_chen2020.json")
    json.dump(out, open(f, "w", encoding="utf-8"))
    print("A %.4f m2, Q_n %.3f Ah, Q_p %.3f Ah, x100 %.4f, y100 %.4f, k0 %.3e / %.3e, kappa %.3f S/m, De %.3e, t+ %.3f, tdf %.3f"
          % (A, Qn, Qp, x100, y100, k0n, k0p, P["kappa"], P["De"], P["tplus"], P["tdf"]))
    print("wrote", f)


if __name__ == "__main__":
    main()
