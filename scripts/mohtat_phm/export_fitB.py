"""Export of the MPM-LE fit B (mpmle_identification study mohtat_B_fin) to the physical (SI) parameters of the
grey-box DFN, as one dfnekf cell-set JSON (schema dfnekf-cellset-1, read by dfnekf.cellset).

    python scripts/mohtat_phm/export_fitB.py                      # defaults: the study below -> data/exp_Mohtat/params/mohtat_B_fin.json

Same mapping as scripts/export_mohtat_lm3c.py (see its docstring), with the options of case B:
  theta_ref_C = 0  : theta_n, theta_p are the diffusion times at 0 degC; back to 25 degC through Ea_Dn, Ea_Dp
                     (mpmle srcs/identification.test_groups)
  slice_ratio = 4  : the area A comes from the ohmic response r_hat of a uniform reaction on the identification's own grid,
                     10 slices x 10 volumes per electrode (geometric widths, narrowest at the separator), 12 in the separator,
                     each slice carrying a share of the current equal to its width (mpmle srcs/ms_model._group_maps)
  De_f, dEa_De     : D_e(T) = De_f D_e,set(c_e0, T) exp(-dEa_De (1/T - 1/T_ref) / R)   (electrolyte_T = 'set')
  tdf_T            : chi(c_e0, T) of the electrolyte set, tabulated with kappa and D_e (cellset.params_for sets P["tdf"])
  Dp_wsurf         : the cathode D_p(y) law at y_mean + Dp_wsurf (y_surf - y_mean) (step.DFN.cp_law_of)
  anode OCP        : the Ai2020 table the study used (ocp_correction.json U_n_table)
  windows          : windows.json (common top/bottom of C07-C09, Q_n and Q_p from each cell's own C/20 capacity)
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

F, R = 96485.33212, 8.314462618
T_REF = 298.15
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MPMLE = os.environ.get("MPMLE_ROOT", r"C:\Users\alexa\OneDrive - Politecnico di Milano\Desktop\Batteries\Python codes"
                                     r"\MIONet\mpmle_identification")
FV_GROUPS = (10, 12)            # mpmle srcs/ms_model.BatchMS.FV_GROUPS: volumes per slice, volumes in the separator


def ev(f, *args):
    import pybamm
    if not callable(f):
        return float(f)
    return float(f(*[pybamm.Scalar(float(a)) for a in args]).evaluate())


def slice_widths(N, ratio):
    """Anode slice widths (collector -> separator, geometric, narrowest at the separator), summing to 1; the cathode's
    are the same reversed (mpmle ms_model._sw64)."""
    wn = ratio ** (-np.arange(N) / max(N - 1, 1))
    return wn / wn.sum()


def r_hat(l_s, r_np, eps_n, eps_s, eps_p, brugg, N, ratio, npc=FV_GROUPS[0], ns=FV_GROUPS[1]):
    """Ohmic response of a uniform reaction (each slice's share of 1 A = its width) on the dimensionless grid x / L_e,
    phi_e averaged (width-weighted) over the anode slices minus over the cathode slices, divided by kappa / L_e."""
    l_n, l_p = (1 - l_s) * r_np, (1 - l_s) * (1 - r_np)
    wn = slice_widths(N, ratio)
    wp = wn[::-1].copy()
    nn = N * npc
    dx = np.r_[l_n * np.repeat(wn, npc) / npc, np.full(ns, l_s / ns), l_p * np.repeat(wp, npc) / npc]
    eps = np.r_[np.full(nn, eps_n), np.full(ns, eps_s), np.full(nn, eps_p)]
    deff = eps ** brugg
    slab = np.zeros((dx.size, 2 * N))
    for i in range(N):
        slab[i * npc:(i + 1) * npc, i] = 1.0
        slab[nn + ns + i * npc:nn + ns + (i + 1) * npc, N + i] = 1.0
    width = dx @ slab
    u = slab / width[None, :]
    W = (slab * dx[:, None] / width[None, :]).T
    cumint = lambda v: np.cumsum(v * dx[:, None], 0) - 0.5 * v * dx[:, None]
    P = W @ cumint(-cumint(u) / deff[:, None])
    jv = np.r_[wn, -wp]
    pe = P @ jv
    return float(pe[:N] @ wn - pe[N:] @ wp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=os.path.join(MPMLE, "runs", "studies", "mohtat_B_fin"))
    ap.add_argument("--config", default=os.path.join(MPMLE, "configs", "mohtat_B_fin.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "exp_Mohtat", "params", "mohtat_B_fin.json"))
    a = ap.parse_args()
    import pybamm

    ld = lambda f: json.load(open(os.path.join(a.study, f), encoding="utf-8"))
    dyn, win, oc = ld("dynamics.json"), ld("windows.json"), ld("ocp_correction.json")
    cfg = json.load(open(a.config, encoding="utf-8"))
    mo, s1 = cfg["model"], cfg["step1"]
    assert mo.get("electrolyte_T") == "set" and mo.get("electrolyte_groups") == "material" and mo.get("geometry") == "groups"
    g = dict(dyn["groups"])
    pm, pe = pybamm.ParameterValues(cfg["pm"]), pybamm.ParameterValues(mo["electrolyte_set"])

    # theta at theta_ref_C -> 25 degC
    th_ref = mo.get("theta_ref_C")
    theta25 = {}
    for th, ea in (("theta_n", "Ea_Dn"), ("theta_p", "Ea_Dp")):
        theta25[th] = g[th] if th_ref is None else g[th] * np.exp(-g[ea] / R * (1.0 / (th_ref + 273.15) - 1.0 / T_REF))

    ce0 = float(pm["Initial concentration in electrolyte [mol.m-3]"])
    kappa_ref = ev(pe["Electrolyte conductivity [S.m-1]"], ce0, T_REF)
    De_set = ev(pe["Electrolyte diffusivity [m2.s-1]"], ce0, T_REF)
    tplus = ev(pe["Cation transference number"], ce0, T_REF)
    tdf_ref = ev(pe["Thermodynamic factor"], ce0, T_REF)
    De_f, dEa_De = g.get("De_f", 1.0), g.get("dEa_De", 0.0)

    L_e, l_s, r_np = g["L_e"], g["l_s"], g["r_np"]
    Ln, Ls, Lp = L_e * (1 - l_s) * r_np, L_e * l_s, L_e * (1 - l_s) * (1 - r_np)
    N, ratio = int(mo.get("n_slices", 5)), float(mo.get("slice_ratio", 1.0))
    rh = r_hat(l_s, r_np, g["eps_n"], g["eps_s"], g["eps_p"], g["brugg"], N, ratio)
    A = L_e * rh / (kappa_ref * g["R_e"])

    Rn, Rp = float(pm["Negative particle radius [m]"]), float(pm["Positive particle radius [m]"])
    eps_sn = min(float(pm["Negative electrode active material volume fraction"]), 1.0 - g["eps_n"])
    eps_sp = min(float(pm["Positive electrode active material volume fraction"]), 1.0 - g["eps_p"])
    a_n, a_p = 3 * eps_sn / Rn, 3 * eps_sp / Rp
    j0n, j0p = g["I0_n"] / (A * a_n * Ln), g["I0_p"] / (A * a_p * Lp)

    cells = {}
    for c, w in sorted(win.items()):
        cmax_n = w["Q_n"] * 3600 / (F * A * Ln * eps_sn)
        cmax_p = w["Q_p"] * 3600 / (F * A * Lp * eps_sp)
        cells[c] = dict(Q_n=w["Q_n"], Q_p=w["Q_p"], x100=w["x100"], y100=w["y100"], x0=w.get("x0"), y0=w.get("y0"),
                        cmax_n=cmax_n, cmax_p=cmax_p,
                        k0_n=j0n / (0.5 * cmax_n * np.sqrt(ce0)), k0_p=j0p / (0.5 * cmax_p * np.sqrt(ce0)),
                        ocv_fit_rmse_mV=w.get("rmse_mV"), C20_leg=w.get("C20_leg"), q_ocv_Ah=w.get("q_ocv_Ah"))

    # electrolyte temperature law at c_e0: the set's kappa(T); D_e with De_f and dEa_De; chi(T) (tdf_T)
    TK = np.arange(258.15, 333.15 + 1e-9, 2.5)
    ark = (1.0 / TK - 1.0 / T_REF) / R
    De_T = np.array([ev(pe["Electrolyte diffusivity [m2.s-1]"], ce0, t) for t in TK]) * De_f * np.exp(-dEa_De * ark)
    tdf_T = [ev(pe["Thermodynamic factor"], ce0, t) for t in TK] if mo.get("tdf_T") else [tdf_ref] * len(TK)
    e_T = dict(T_K=TK.tolist(), kappa=[ev(pe["Electrolyte conductivity [S.m-1]"], ce0, t) for t in TK],
               De=De_T.tolist(), tdf=list(map(float, tdf_T)))
    un = oc["U_n_table"]
    out = {
        "schema": "dfnekf-cellset-1",
        "id": os.path.basename(os.path.normpath(a.study)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"study": os.path.abspath(a.study), "config": os.path.abspath(a.config), "pm": cfg["pm"],
                   "electrolyte_set": mo["electrolyte_set"], "pybamm": pybamm.__version__, "cost_mV": dyn["cost_mV"],
                   "fitted": dyn["fitted"], "rmse_per_test_mV": dyn.get("rmse_per_test"), "model": mo,
                   "anode_ocp": un.get("source")},
        "groups": g,
        "T_ref_K": T_REF,
        "P": dict(Ln=Ln, Ls=Ls, Lp=Lp, Rn=Rn, Rp=Rp, eps_en=g["eps_n"], eps_es=g["eps_s"], eps_ep=g["eps_p"],
                  eps_sn=eps_sn, eps_sp=eps_sp, brug=g["brugg"],
                  sig_n=float(pm["Negative electrode conductivity [S.m-1]"]),
                  sig_p=float(pm["Positive electrode conductivity [S.m-1]"]),
                  alpha=0.5, Ds_n=Rn ** 2 / theta25["theta_n"], Ds_p=Rp ** 2 / theta25["theta_p"],
                  De=De_set * De_f, kappa=kappa_ref, tplus=tplus, tdf=tdf_ref, ce0=ce0, Rf0=0.0, area=A,
                  R_contact=g["R_contact"]),
        "derived": dict(r_hat=rh, j0_n=j0n, j0_p=j0p, L_e=L_e, theta25=theta25, n_slices=N, slice_ratio=ratio,
                        R_e_check=L_e * rh / (kappa_ref * A)),
        "temperature": dict(Ea_Dn=g["Ea_Dn"], Ea_Dp=g["Ea_Dp"], Ea_r=g["Ea_r"], Ea_s=g.get("Ea_s", 0.0),
                            dUdT0=g["dUdT0"], dUdT_n=g.get("dUdT_n", 0.0), dUdT_p=g.get("dUdT_p", 0.0),
                            electrolyte=e_T),
        "dp_law": dict(drop=g["Dp_drop"], shape=g["Dp_shape"], wsurf=g.get("Dp_wsurf", 0.0), y0=s1["y100"],
                       y1=s1["y_end"],
                       note="D_p = Ds_p 10^(-drop u^shape), u = max(y - y0, 0) / (y1 - y0), y = y_mean + wsurf "
                            "(y_surf - y_mean) at the start of the step (frozen over the step)"),
        "ocp": {"n": {"x": un["x"], "U": un["U"], "source": un.get("source")},
                "p": {"y": oc["table"]["y"], "U": oc["table"]["U"], "y_observed": oc.get("y_observed"),
                      "source": "ocp_correction.json table (step 1)"}},
        "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", a.out)
    print("A = %.4f m2, L = %.1f/%.1f/%.1f um, r_hat = %.4f, Ds_n = %.3e, Ds_p = %.3e (25 C), j0 = %.3g/%.3g A/m2, "
          "De = %.3e" % (A, Ln * 1e6, Ls * 1e6, Lp * 1e6, rh, Rn ** 2 / theta25["theta_n"], Rp ** 2 / theta25["theta_p"],
                         j0n, j0p, De_set * De_f))
    c = cells["C01"]
    print("C01: cmax_eff = %.0f / %.0f, k0 = %.3e / %.3e" % (c["cmax_n"], c["cmax_p"], c["k0_n"], c["k0_p"]))


if __name__ == "__main__":
    main()
