"""Export a fitted LP604765 set to the dfnekf cell-set JSON (schema dfnekf-cellset-1, read by dfnekf.cellset), so that the
EKF scripts of this repository (scripts/ekf_exp.py) can run on this cell.

    python export_cellset.py --fit results/fit_v2/best.json --out results/fit_v2/lp604765_cellset.json

Mapping: the physical P at T_ref = 20 C (lpmodel.Model.params); the electrolyte tables of lm3c divided by the fitted
tortuosity factor tau; one cell "LP01" with the fitted window (x100, y100, Q_n, Q_p) and k0 at T_ref. Temperature laws
as in dfnekf.cellset.params_for (same Arrhenius form and reference). The start offsets dq_00 / dq_40 are test-specific
and are not exported.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

import common as C
from lpmodel import Model, T_REF, Q_CELL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(C.RES, "fit_v2", "best.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    best = json.load(open(a.fit))
    p = C.model_params(best["params"])
    m = Model()
    P = m.params(p, T_REF - 273.15)
    lm = m.cs
    keep = ("Ln", "Ls", "Lp", "Rn", "Rp", "eps_en", "eps_es", "eps_ep", "eps_sn", "eps_sp", "brug", "sig_n", "sig_p", "alpha",
            "Ds_n", "Ds_p", "De", "kappa", "tplus", "tdf", "ce0", "Rf0", "area", "R_contact")
    e = lm["temperature"]["electrolyte"]
    y_end = min(p["y100"] + Q_CELL / p["Q_p"], 0.999)
    x0 = p["x100"] - Q_CELL / p["Q_n"]
    out = {
        "schema": "dfnekf-cellset-1",
        "id": "lp604765_" + os.path.basename(os.path.dirname(os.path.abspath(a.fit))),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"fit": os.path.abspath(a.fit), "cost_mV": best["cost_mV"], "per_test_mV": best["per_test_mV"],
                   "free": best.get("free"), "base_set": lm["id"],
                   "note": "LP604765 (CNR STEMS, BATTAGE), HPPC 0/20/40 C; OCPs of lm3c (NMC532 table, Mohtat2020 graphite)"},
        "T_ref_K": T_REF,
        "P": {k: float(P[k]) for k in keep},
        "temperature": dict(Ea_Dn=p["Ea_Dn"], Ea_Dp=p["Ea_Dp"], Ea_r=p["Ea_r"], Ea_s=p["Ea_s"], dUdT0=p["dUdT0"],
                            dUdT_n=p["dUdT_n"], dUdT_p=0.0,
                            electrolyte=dict(T_K=e["T_K"], kappa=[v / p["tau"] for v in e["kappa"]],
                                             De=[v / p["tau"] for v in e["De"]])),
        "dp_law": dict(drop=p["drop"], shape=p["shape"], y0=p["y100"], y1=y_end,
                       note="D_p = Ds_p 10^(-drop u^shape), u = max(y - y0, 0) / (y1 - y0), y the mean stoichiometry"),
        "ocp": lm["ocp"],
        "cells": {"LP01": dict(Q_n=p["Q_n"], Q_p=p["Q_p"], x100=p["x100"], y100=p["y100"], x0=x0, y0=y_end,
                               cmax_n=float(P["cmax_n"]), cmax_p=float(P["cmax_p"]), k0_n=float(P["k0_n"]),
                               k0_p=float(P["k0_p"]), q_ocv_Ah=Q_CELL)},
        "fit_extras": {k: best["params"][k] for k in best["params"] if k.startswith("dq_")} | {"tau": p["tau"]},
    }
    dst = a.out or os.path.join(os.path.dirname(os.path.abspath(a.fit)), "lp604765_cellset.json")
    json.dump(out, open(dst, "w", encoding="utf-8"))
    print("wrote", dst)
    # check: dfnekf.cellset.params_for gives the same P as lpmodel at 0, 20, 40 C
    from dfnekf import cellset
    cs = cellset.load(dst)
    for T in (0.0, 20.0, 40.0):
        Pa, Pb = cellset.params_for(cs, "LP01", T + 273.15), m.params(p, T)
        d = max(abs(Pa[k] / Pb[k] - 1) for k in ("Ds_n", "Ds_p", "k0_n", "k0_p", "R_contact", "kappa", "De", "cmax_n", "cmax_p"))
        print("T = %.0f C: max relative difference cellset vs lpmodel %.2e" % (T, d))


if __name__ == "__main__":
    main()
