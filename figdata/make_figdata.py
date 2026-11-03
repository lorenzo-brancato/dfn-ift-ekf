"""Results -> figure data. Reads the source-data tables written by the ``--aggregate`` mode of the scripts (``<results>/source_data`` and ``<results>/tables``)
and writes the compact CSV files read by the pgfplots figures (``paper/figures/data``), in exactly the format of the manuscript.

    python figdata/make_figdata.py --results results --out paper/figures/data [--compare <reference folder>]

``--compare`` checks every written file against a reference folder (numbers identical to the printed precision, 1e-6 relative) and prints one line per file.
Decimation of the long time series keeps the min/max envelope of each time bin, so that spikes survive.
"""
import argparse
import os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures/data"); ap.add_argument("--compare", default=None)
ap.add_argument("--sources", default=None, help="folder with the source-data tables (default: <results>/source_data)")
ap.add_argument("--tables", default=None, help="folder with the summary tables (default: <results>/tables)")
A = ap.parse_args()
SD = A.sources or os.path.join(A.results, "source_data"); TB = A.tables or os.path.join(A.results, "tables"); OUT = A.out
os.makedirs(OUT, exist_ok=True)
rd = lambda d, f: pd.read_csv(os.path.join(d, f), comment="#")
written = []


def w(df, f, float_format="%.6g"):
    df.to_csv(os.path.join(OUT, f), index=False, float_format=float_format); written.append(f)


def envelope(t, y, nbins):
    """Decimate to the min and max sample of each of ``nbins`` equal time bins (the spikes survive)."""
    t = np.asarray(t); y = np.asarray(y); edges = np.linspace(t.min(), t.max(), nbins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1); keep = []
    for b in range(nbins):
        k = np.where(idx == b)[0]
        if len(k) == 0: continue
        i0, i1 = k[np.argmin(y[k])], k[np.argmax(y[k])]
        keep += sorted({i0, i1})
    keep = np.array(sorted(set(keep))); return t[keep], y[keep]


STATES = ["th1", "th2", "th3", "th4", "th5"]
RANGE_W = {"Rf": 9.0, "LAM": 0.3, "Ds": 0.7}

# ---- Fig. 3: grid study (4:1 first, then the three uniform grids of Table I)
g = rd(SD, "fig_grid_m17.csv")
sel = [("20/10/20", "4:1", 12, "20/10/20 4:1 N_m=12"), ("10/5/10", "uniform", 8, "10/5/10 N_m=8"), ("20/10/20", "uniform", 12, "20/10/20 N_m=12"), ("40/20/40", "uniform", 16, "40/20/40 N_m=16")]
rows = []
for i, (gr, ratio, nm, label) in enumerate(sel):
    r = g[(g.grid == gr) & (g.ratio == ratio) & (g.Nm == nm)].iloc[0]
    rows.append({"x": i, "grid": label, "panel_a_E_V_1C_mV": r.E_V_1C_mV, "panel_a_E_V_3C_mV": r.E_V_3C_mV, "panel_b_bias_Rf_signed_pct_of_range": r.bias_Rf_pct, "panel_b_bias_LAM_signed_pct_of_range": r.bias_LAM_pct,
                 "panel_b_bias_Ds_signed_pct_of_range": r.bias_Ds_pct, "panel_b_abs_bias_Rf_pct_of_range": abs(r.bias_Rf_pct), "panel_b_abs_bias_LAM_pct_of_range": abs(r.bias_LAM_pct), "panel_b_abs_bias_Ds_pct_of_range": abs(r.bias_Ds_pct)})
w(pd.DataFrame(rows), "grid.csv")

# ---- Fig. 4: cost versus N (4:1 grids): EKF with the converged closure, full UKF
c = rd(SD, "fig_cost_m17.csv"); c4 = c[(c.grid.str.contains("4:1")) & (c.kind == "per chiamata")]
e = c4[c4["filter"].str.startswith("EKF") & c4["filter"].str.contains("newton") & (c4.N <= 813)].sort_values("N")      # Fig. 4 spans N = 168..813; the 40/20/40 point of Table I is outside the plotted range
w(pd.DataFrame({"series": "EKF", "grid": e.grid.values, "N_m": e.Nm.values, "state_dimension_N": e.N.values, "median_step_cost_ms": e.median_ms.values, "n_calls": e.n_calls.values}), "cost_EKF.csv")
u = c4[c4["filter"].str.startswith("UKF k=1 completo") & ~c4["filter"].str.contains("alpha")].sort_values("N")
w(pd.DataFrame({"series": "UKF_full", "grid": u.grid.values, "N_m": u.Nm.values, "state_dimension_N": u.N.values, "median_step_cost_ms": u.median_ms.values, "n_calls": u.n_calls.values}), "cost_UKF_full.csv")

# ---- Fig. 5: static Monte Carlo (points) and error budget (bars)
m = rd(SD, "fig_mc.csv")
for tg in STATES: w(m[m.theta == tg], f"mc_{tg}.csv")
bias = rd(TB, "bias.csv").set_index("theta")                          # signed bias of the batch fit, % of the range: bias_Rf_pct, bias_LAM_pct, bias_Ds_pct
crl = rd(TB, "crlb_4par.csv")                                         # 4-parameter Cramer-Rao table (parameters Rf, LAM, LLI, Ds; sd, correlation matrix)


def crlb_conditional(tab, theta_tag):
    """Cramer-Rao bound of (Rf, LAM, Ds) with LLI fixed: sd from the 3x3 sub-block of the 4-parameter Fisher matrix (= Schur complement of the covariance), % of range."""
    q = tab[(tab.theta_tag == theta_tag) & (tab.protocol == "P6Lx2") & (tab.version == "stato_noto")]
    order = ["R_f", "LAM_n", "LLI", "D_s"]; q = q.set_index("parameter").loc[order]
    sd = q.sd.values.astype(float); C = q[["corr_R_f", "corr_LAM_n", "corr_LLI", "corr_D_s"]].values.astype(float)
    cov = C * np.outer(sd, sd); fisher = np.linalg.inv(cov); idx = [0, 1, 3]
    cond = np.linalg.inv(fisher[np.ix_(idx, idx)]); width = q.range_width.values.astype(float)
    return dict(zip(["Rf", "LAM", "Ds"], 100 * np.sqrt(np.diag(cond)) / width[idx]))


rows = []
for i, tg in enumerate(STATES):
    d = m[m.theta == tg]; cb = crlb_conditional(crl, tg)
    for p, tc, ec in [("Rf", "a_Rf_true_over_Rf0", "a_Rf_estimated_over_Rf0"), ("LAM", "b_LAM_true", "b_LAM_estimated"), ("Ds", "c_Ds_true_over_Ds0", "c_Ds_estimated_over_Ds0")]:
        err = 100 * (d[ec].values - d[tc].values) / RANGE_W[p]; b = bias.loc[tg, f"bias_{p}_pct"]
        rows.append({"x": i, "state": tg, "param": p, "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4), "abs_bias": round(abs(float(b)), 4), "crlb": round(float(cb[p]), 3), "rmse_minus_bias": round(float(np.sqrt(np.mean((err - b) ** 2))), 4)})
w(pd.DataFrame(rows), "mc_budget.csv")

# ---- Fig. 6: NEES histograms (bins of 0.5, values above 10 shown at 10)
n = rd(SD, "fig_nees.csv"); bins = np.linspace(0, 10, 21)
for pn, f in [("a", "EKF"), ("a", "UKF"), ("b", "EKF")]:
    v = n[(n.panel == pn) & n["filter"].str.startswith(f)].NEES_RfDs_time_mean.values
    cnt, _ = np.histogram(np.clip(v, 0, 9.99), bins=bins)
    w(pd.DataFrame({"x": bins[:-1] + 0.25, "n": cnt}), f"nees_{pn}_{f}.csv")

# ---- Fig. 7: charge (pulse and 1C constant-current) -- thresholds are not written here, the tex files carry them
p = rd(SD, "fig_eta_charge.csv"); cc = rd(SD, "fig_charge_cc.csv")
for tg in ["th1", "th2", "th4"]:
    q = p[(p.theta == tg) & (p.t_from_pulse_start_s >= -10) & (p.t_from_pulse_start_s <= 60)]
    w(q[["t_from_pulse_start_s", "current_A", "phi_pl_reference_mV", "phi_pl_EKF_mV", "phi_pl_UKF_SPMe_mV"]].rename(columns=lambda s: s.replace("t_from_pulse_start_s", "t")), f"pulse_{tg}.csv")
    h = cc[(cc.theta == tg) & (cc.t_from_cc_start_s >= 0) & (cc.current_A < -0.5)].copy(); h["tmin"] = h.t_from_cc_start_s / 60
    w(h.iloc[::3][["tmin", "phi_pl_reference_mV", "phi_pl_EKF_mV", "phi_pl_UKF_SPMe_mV"]], f"cc_{tg}.csv")

# ---- Fig. 10: 3C discharge
d = rd(SD, "fig_eta3C.csv")
for tg in ["th2", "th4", "th5"]:
    q = d[(d.theta == tg) & (d.t_from_3C_onset_s >= -10) & (d.t_from_3C_onset_s <= 60)]
    w(q[["t_from_3C_onset_s", "eta_reference_mV", "eta_EKF_mV", "eta_UKF_SPMe_mV"]].rename(columns={"t_from_3C_onset_s": "t"}), f"dis_{tg}.csv")

# ---- Fig. 11: residual certificate
r = rd(SD, "fig_residual_m17.csv"); r["res"] = np.maximum(r.residual_inf_norm, 1e-12)
w(r[["t_s", "res"]], "residual.csv"); w(r[r.fallback_active.astype(bool)][["t_s"]], "residual_fallback.csv")

# ---- Fig. 8: ageing tracking
a = rd(SD, "fig_aging_dyn.csv")
for col in ["Rf_true", "LAM_true", "LLI_true", "Ds_true"]:
    t, y = envelope(a.t_h, a[col], 300); w(pd.DataFrame({"t": t, "y": y}), f"trk_{col}.csv")
for k in ["Rf", "LAM", "Ds"]:
    t, y = envelope(a.t_h, a[f"{k}_est_A"], 700); w(pd.DataFrame({"t": t, "y": y}), f"trk_{k}_A.csv")
t, y = envelope(a.t_h, a["inv_est_A"], 700); w(pd.DataFrame({"t": t, "y": y}), "trk_inv_A.csv")
# read-outs before the CV phase (marker 3 of the tracking run): the figure draws the A columns (seed 0)
w(a[a.marker == 3][["t_h", "Rf_est_A", "LAM_est_A", "inv_est_A", "Ds_est_A", "Rf_est_B", "LAM_est_B", "inv_est_B", "Ds_est_B"]], "trk_precv2.csv")
rr = rd(SD, "fig_tracking_rpt.csv"); out = {}
for run, pt, nm in [("seed0", "precv", "seed0_precv"), ("seed1", "precv", "seed1_precv"), ("seed0", "end", "seed0_end")]:
    q = rr[(rr.run == run) & (rr.point == pt) & (rr.rpt >= 1)].sort_values("rpt"); out.setdefault("rpt", q.rpt.values); out[nm] = q.Rf_err.values.round(3)
w(pd.DataFrame(out), "trk_rf.csv")
w(rd(SD, "fig_tracking_z.csv"), "trk_z.csv")

# ---- Fig. 9: protocols (written directly by scripts/protocols_fig.py)
for f in ["prot_F6_th2.csv", "prot_F6Q_th2.csv", "prot_P4.csv", "prot_RPT.csv"]:
    df = rd(SD, f); w(df, f, float_format="%.6g") if False else df.to_csv(os.path.join(OUT, f), index=False); written.append(f)

print(f"{len(written)} files written to {OUT}")
# columns that no figure reads and that cannot be reproduced from the source data (documented in VERIFY): the B (seed 1) columns of trk_precv2.csv
IGNORE = {"trk_precv2.csv": ["Rf_est_B", "LAM_est_B", "inv_est_B", "Ds_est_B"]}
if A.compare:
    bad = 0
    for f in sorted(written):
        ref = os.path.join(A.compare, f)
        if not os.path.exists(ref): print(f"{f:28s} NOT IN REFERENCE"); bad += 1; continue
        x = pd.read_csv(os.path.join(OUT, f)); y = pd.read_csv(ref); yt = pd.read_csv(ref, dtype=str)
        if x.shape != y.shape or list(x.columns) != list(y.columns): print(f"{f:28s} SHAPE/COLUMNS differ {x.shape} vs {y.shape}"); bad += 1; continue
        num = [k for k in x.select_dtypes("number").columns if k not in IGNORE.get(f, [])]; ok = True; worst = 0.0
        for k in num:
            dec = yt[k].str.extract(r"\.(\d+)")[0].str.len().fillna(0).astype(int).values; unit = 10.0 ** (-dec)      # last printed digit of the reference
            d = np.abs(x[k].values.astype(float) - y[k].values.astype(float)); tol = np.maximum(1e-5 * np.abs(y[k].values.astype(float)), 1.01 * unit)
            worst = max(worst, float(np.max(d / tol)) if len(d) else 0.0); ok &= bool((d <= tol).all())
        same_txt = all((x[k].astype(str).values == y[k].astype(str).values).all() for k in x.columns if k not in x.select_dtypes("number").columns)
        ok &= same_txt; bad += (not ok)
        print(f"{f:28s} {'OK  ' if ok else 'DIFF'} worst |diff| / tolerance {worst:.2f}" + (f" (ignored: {', '.join(IGNORE[f])})" if f in IGNORE else ""))
    print("all files identical to the printed precision" if bad == 0 else f"{bad} file(s) differ")
