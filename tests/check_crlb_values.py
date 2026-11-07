"""Check of the Cramer-Rao bounds against the stored values of the manuscript (Fig. 5 grey bars, Sec. III-B).

1. Linear algebra (no model run): from the stored 4-parameter table (tests/data/tab_crlb_m17.csv, P6Lx2, version stato_noto) rebuild the 4 x 4 covariance,
   take the Schur complement with respect to LLI and compare with the column `crlb` of tests/data/mc_budget.csv (tolerance: the 3 printed decimals).
2. Recomputation (JAX, ~20 s per state): sensitivities of the grey-box model on the P6Lx2 reference initial state -> conditional bound and 4-parameter marginal, compared
   with mc_budget.csv (crlb) and tab_crlb_m17.csv (sd_pct_range and correlations of the 4-parameter version).
   Uses the per-run files of `scripts/crlb.py` in --results if present (--from-files), otherwise recomputes the states in --states.
Usage: python tests/check_crlb_values.py [--states th1,th2] [--data data/sim] [--grid 4:1] [--no-recompute] [--results DIR --from-files]
Exit status 1 if a check fails.
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, pandas as pd
from dfnekf import crlb

p = argparse.ArgumentParser()
p.add_argument("--mc-budget", default=os.path.join(HERE, "data", "mc_budget.csv")); p.add_argument("--tab-crlb", default=os.path.join(HERE, "data", "tab_crlb_m17.csv"))
p.add_argument("--states", default="th1,th2,th3,th4,th5"); p.add_argument("--data", default="data/sim"); p.add_argument("--grid", default="4:1")
p.add_argument("--no-recompute", action="store_true"); p.add_argument("--results", default=None); p.add_argument("--from-files", action="store_true"); p.add_argument("--core", type=int, default=None)
a = p.parse_args()
mc = pd.read_csv(a.mc_budget); tab = pd.read_csv(a.tab_crlb, comment="#"); fails = []
PAR = {"Rf": "R_f", "LAM": "LAM_n", "Ds": "D_s"}


def check(name, got, ref, tol):
    ok = abs(got - ref) <= tol; print(f"{'ok  ' if ok else 'FAIL'} {name:42s} got {got:.4f} stored {ref:.4f}")
    if not ok: fails.append(name)


# ---- 1. Schur complement of the stored 4-parameter covariance
names = ["R_f", "LAM_n", "LLI", "D_s"]; keep = [0, 1, 3]
for tag in sorted(mc.state.unique()):
    g = tab[(tab.protocol == "P6Lx2") & (tab.theta_tag == tag) & (tab.version == "stato_noto")].set_index("parameter").loc[names]
    sd = g.sd.values; corr = g[[f"corr_{n}" for n in names]].values; C = corr * np.outer(sd, sd)
    S = crlb.schur_complement(C, keep); S2 = crlb.conditional(np.linalg.inv(C), keep); assert np.allclose(S, S2, rtol=1e-6)
    for i, par in zip(range(3), ["Rf", "LAM", "Ds"]):
        w = crlb.RANGE_WIDTH[par]; ref = float(mc[(mc.state == tag) & (mc.param == par)].crlb.iloc[0])
        check(f"schur(stored table) {tag} {par}", 100 * np.sqrt(S[i, i]) / w, ref, 6e-4)

# ---- 2. recomputation
if not a.no_recompute:
    states = [s for s in a.states.split(",") if s]
    if a.from_files:
        dVs = {t: np.load(os.path.join(a.results, "traj", f"crlb_dV_P6Lx2_{t}.npz"))["dV"] for t in states}
    else:
        from dfnekf import runtime; runtime.setup(a.core)
        from dfnekf import params as prm
        from dfnekf.experiment import TRUE
        from dfnekf.step import DFN
        m = DFN(prm.base(), 20, 10, 20, 12, 2, 0.5, a.grid)
        dVs = {t: crlb.sensitivities(m, TRUE[t], "P6Lx2", t, a.data) for t in states}
    for t, dV in dVs.items():
        rows = pd.DataFrame(crlb.table_rows(dV, "P6Lx2", t, [0, 0, 0, 0]))
        for par, nm in PAR.items():
            ref = float(mc[(mc.state == t) & (mc.param == par)].crlb.iloc[0])
            got = float(rows[(rows.version == "conditional_on_LLI") & (rows.parameter == nm)].sd_pct_range.iloc[0]); check(f"recomputed conditional {t} {par}", got, ref, 2e-3)
            old = float(tab[(tab.protocol == "P6Lx2") & (tab.theta_tag == t) & (tab.version == "stato_noto") & (tab.parameter == nm)].sd_pct_range.iloc[0])
            got = float(rows[(rows.version == "known_state") & (rows.parameter == nm)].sd_pct_range.iloc[0]); check(f"recomputed 4-par marginal {t} {par}", got, old, 2e-3 * max(1, old))
            if par != "Ds":
                pass
        g = rows[(rows.version == "known_state") & (rows.parameter == "LAM_n")].iloc[0]
        old = float(tab[(tab.protocol == "P6Lx2") & (tab.theta_tag == t) & (tab.version == "stato_noto") & (tab.parameter == "LAM_n")].corr_LLI.iloc[0]); check(f"corr(LAM, LLI) {t}", float(g.corr_LLI), old, 2e-3)
        # unknown SOC: LAM <= 0.5 %, change of the others <= 0.12 %
        eff = crlb.soc_effect(rows.to_dict("records")).set_index("parameter")
        print(f"     unknown SOC {t}: LAM {eff.loc['LAM_n','crlb_unknown_soc_pct']:.3f} %, change Rf {eff.loc['R_f','change_pct']:.3f}, Ds {eff.loc['D_s','change_pct']:.3f}")
        if eff.loc["LAM_n", "crlb_unknown_soc_pct"] > 0.5 + 5e-3 or max(eff.loc["R_f", "change_pct"], eff.loc["D_s", "change_pct"]) > 0.12 + 5e-3: fails.append(f"unknown SOC {t}")
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1 if fails else 0)
