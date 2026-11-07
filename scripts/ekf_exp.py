"""EKF of the grey-box DFN (dfnekf.ekf.EKF, the structured EKF with the implicit closure) on a measured test of an
experimental dataset (dfnekf.datasets), with the dataset's identified cell set; or on its simulated twin (--twin).

    python scripts/ekf_exp.py --dataset exp_LP604765 --test CYC20                    # SOC from wrong starts (defaults)
    python scripts/ekf_exp.py --dataset exp_Mohtat --test C01_VAL --seeds 0,1,2      # parameters from wrong starts
    python scripts/ekf_exp.py --dataset exp_Mohtat --test C01_VAL --twin             # model = plant, + white noise
    python scripts/ekf_exp.py --dataset exp_Mohtat --cell C05 --plant "<Mohtat NMC>/processed/Mohtat2020/real_cell_05/Plant" \
        --cycle 1 --n-cycles 5 --coords Rf,LAM,LLI --li-sigma 1e-4 --seeds=-1      # ageing cycles (scripts/mohtat_batch.py)

Two experiments (--start; the default depends on the dataset):
  soc     (exp_LP604765) Started with a wrong charge (--soc-err [A.h], + = filter starts emptier), does the filter find
          the true one, and does its +-2 sd band cover the error? Reference: Coulomb counting from the start placed by
          the dataset's rule (datasets.initial_charge), SOC = 1 - q / Q_ref. Initial covariance (full): a lithium-
          conserving charge offset of sd --p0-q [A.h] (all anode means down by dq / Q_n c_max,n, all cathode means up
          by dq / Q_p c_max,p, fully correlated) plus independent shell noise --p0-shell [c_max]. Also the open loop
          from the same wrong starts. Output: ekf_<test>.csv (rows merged by kind, err_Ah), ekf_<test>_e<err>.npz,
          open_<test>_e<err>.npy.
  params  (exp_Mohtat) Started away from the identified values theta = (Rf, LAM, Ds) = (1, 0, 1) by up to --offset of
          the ageing range (seed; -1 = --init), does the filter return to them, with error bars that cover the distance?
          Diagonal P0 (--p0-shell on the particle means). Output: ekf_<cell>_<test>[_twin].csv, ..._s<seed>.npz/.png.
Both: Rf = multiplier of a baseline film (--film-share of R_contact), Ds on the cathode (--ds-on p), D_p(y) law of the
set; no process noise on the DFN state, parameter random walk --q-th of the ageing range per sqrt(s);
R = (--sigma-mV)^2, the model error of the set (not the sensor's). The step is --dt (exp_LP604765: 1 s, binned from the
raw samples) or the test's own sampling period (exp_Mohtat: 10 s).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import runtime  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dfnekf import cellset, datasets  # noqa: E402
from dfnekf.ekf import EKF, RANGE  # noqa: E402
from dfnekf.rom import simulate  # noqa: E402
from dfnekf.step import DFN  # noqa: E402

TRUE = {"Rf": 1.0, "LAM": 0.0, "Ds": 1.0, "LLI": 0.0}   # the identified (beginning-of-life) values

# per-dataset defaults (any of them can be overridden on the command line)
DEFAULTS = {
    "exp_LP604765": dict(test="CYC20", start="soc", coords="Rf", sigma_mV=15.0, dims="10,6,10,12", dt=1.0, soft_eps=1e-3),
    "exp_Mohtat": dict(test="C01_VAL", start="params", coords="Rf,LAM,Ds", sigma_mV=10.0, dims="20,10,20,12", dt=None, soft_eps=None),
}


def load_cycles(plant_dir, cycles):
    """Consecutive ageing cycles of the processed Mohtat files (real_cell_NN/Plant/cycle_NNNN.csv: + discharge, ~10 s)
    joined into one record (the files are contiguous: each ends at 3.0 V on discharge and the next starts on charge at
    once); the SOH reference is the first cycle's (C/20 anchors)."""
    parts, t0, first = [], 0.0, None
    for c in cycles:
        f = os.path.join(plant_dir, "cycle_%04d.csv" % c)
        r = pd.read_csv(f)
        df = pd.DataFrame(dict(t_s=r["Time [s]"] - r["Time [s]"].iloc[0], I_A=r["Current [A]"], V_V=r["Voltage [V]"],
                               T_C=r["X-averaged cell temperature [C]"]))
        dt = float(np.median(np.diff(df.t_s.values)))
        df["t_s"] = df.t_s + t0
        t0 = float(df.t_s.iloc[-1]) + dt
        parts.append(df)
        first = first or dict(SOH_ref=float(r["SOH_ref [%]"].iloc[0]), source=f)
    out = pd.concat(parts, ignore_index=True)
    n = len(out)
    return dict(t=out.t_s.values, I=out.I_A.values, V=out.V_V.values, T=out.T_C.values, valid=np.ones(n, bool),
                T_C=float(out.T_C.mean()), dt=float(np.median(np.diff(out.t_s.values))), V_rest=float(out.V_V.values[0]),
                meta=dict(first, cycles=list(cycles), n_first=len(parts[0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--test", default=None, help="test of the dataset (datasets.tests); LP: CYC20, CAP20, HPPC20, ...; Mohtat: C01_VAL, ...")
    ap.add_argument("--set", default=None, help="cell set JSON (default: the dataset's params/)")
    ap.add_argument("--cell", default=None, help="cell of the set (default: the test's)")
    ap.add_argument("--start", default=None, choices=("soc", "params"))
    ap.add_argument("--twin", action="store_true", help="measured voltage replaced by the model's at the identified values + noise")
    ap.add_argument("--twin-noise-mV", type=float, default=2.0)
    ap.add_argument("--plant", default=None, help="exp_Mohtat: processed Plant folder (real_cell_NN/Plant); run --cycle from it")
    ap.add_argument("--cycle", type=int, default=None)
    ap.add_argument("--n-cycles", type=int, default=1, help="consecutive cycles from --cycle, joined into one run")
    ap.add_argument("--coords", default=None)
    ap.add_argument("--transform", default="free", choices=("free", "logit"),
                    help="parameter coordinates: free = unbounded (log for Rf, Ds; linear for LAM, LLI), logit = the paper's bounded ones")
    ap.add_argument("--sigma-mV", type=float, default=None)
    ap.add_argument("--film-share", type=float, default=0.5)
    ap.add_argument("--ds-on", default="p", choices=("n", "p"))
    ap.add_argument("--p0-shell", "--p0-soc", dest="p0_shell", type=float, default=0.005,
                    help="independent sd of each particle mean at t = 0 [fraction of c_max]")
    ap.add_argument("--p0-th", type=float, default=0.3, help="sd of the parameters at t = 0 [fraction of the ageing range]")
    ap.add_argument("--q-th", type=float, default=1e-4,
                    help="parameter random walk [fraction of the ageing range per sqrt(s)]; 1e-4 as experiment.Case")
    ap.add_argument("--li-sigma", type=float, default=None,
                    help="relative sd of the lithium pseudo-measurement (total lithium of the state = the window's); off by default")
    # soc experiment
    ap.add_argument("--soc-err", default="-0.4,0,0.4", help="[soc] initial charge errors [A.h] (+ = filter starts emptier)")
    ap.add_argument("--p0-q", type=float, default=0.5, help="[soc] sd of the initial charge [A.h]")
    ap.add_argument("--no-openloop", action="store_true")
    # params experiment
    ap.add_argument("--seeds", default="0,1,2", help="[params] seeds of the initial parameter offsets (-1: start at --init)")
    ap.add_argument("--offset", type=float, default=0.4, help="[params] initial parameter offset, fraction of the ageing range")
    ap.add_argument("--init", default="", help="[params] initial values overriding the identified ones, e.g. Ds=2 (seed -1 starts there)")
    ap.add_argument("--p0-coord", default="", help="[params] prior sd in filter coordinates (log for Rf, Ds), e.g. Ds=1.0: overrides --p0-th")
    # model
    ap.add_argument("--ea-dp", type=float, default=None, help="cathode diffusion activation energy [J/mol] overriding the set's")
    ap.add_argument("--soft-eps", type=float, default=None, help="width of the soft stoichiometry edges [fraction of c_max]")
    ap.add_argument("--t-input", action="store_true",
                    help="measured temperature as a known input on the cathode diffusivity: per step theta[2] *= the Arrhenius "
                         "factor (Ea_Dp) of the measured T (60 s moving average) against the run's mean T, at which P is built")
    ap.add_argument("--t-rise", type=float, default=1.0,
                    help="with --t-input: the temperature rise above the run's 5th percentile is multiplied by this (core vs surface)")
    ap.add_argument("--p-mult", default="", help="multipliers on P after the temperature laws, e.g. De=2,kappa=1.5 (k0 = both k0_n, k0_p)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--grid", default="4:1")
    ap.add_argument("--dims", default=None, help="Nn,Ns,Np,Nm")
    ap.add_argument("--dt", type=float, default=None, help="filter step [s] (default: the dataset's)")
    ap.add_argument("--results", default=None, help="output folder (default: results/<dataset>/ekf)")
    a = ap.parse_args()
    dflt = DEFAULTS[a.dataset]
    for k, v in dflt.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.results = a.results or os.path.join(ROOT, "results", a.dataset, "ekf")
    os.makedirs(a.results, exist_ok=True)

    # ---- data, cell set, initial charge
    cs = cellset.load(a.set or datasets.cellset_path(a.dataset))
    if a.cycle is not None:
        if a.dataset != "exp_Mohtat" or not a.plant:
            sys.exit("--cycle needs --dataset exp_Mohtat and --plant")
        d = load_cycles(a.plant, range(a.cycle, a.cycle + a.n_cycles))
        cell = a.cell or "C01"
        d.update(cell=cell, test="cyc%04d" % a.cycle + ("x%d" % a.n_cycles if a.n_cycles > 1 else ""))
        q0 = cellset.q_start(cs, cell, 0.0, d["V_rest"])     # an ageing cycle starts at the bottom, under charge
        tag = "%s_%s" % (cell, d["test"])
    else:
        d = datasets.load(a.dataset, a.test, dt=a.dt)
        cell = a.cell or d["cell"]
        q0 = datasets.initial_charge(a.dataset, d, cs)
        tag = d["test"]
    if a.max_hours:
        keep = d["t"] < a.max_hours * 3600 if a.start == "params" else np.arange(len(d["t"])) < int(a.max_hours * 3600 / d["dt"])
        d = {**d, **{k: d[k][keep] for k in ("t", "I", "V", "T", "valid")}}
        d["T_C"] = float(np.mean(d["T"]))       # parameters at the mean T of the run actually filtered
    t, I_np, dt = d["t"], d["I"], float(d["dt"])
    if a.dt is not None:
        gaps = np.diff(t)
        if np.any(np.abs(gaps - dt) > 1e-6):
            print("warning: %d gaps in the %g-s grid (max %.0f s), treated as %g s" % (np.sum(np.abs(gaps - dt) > 1e-6), dt, gaps.max(), dt))
    T_K = float(d["T_C"]) + 273.15
    coords = tuple(c for c in a.coords.split(",") if c)
    dims = tuple(int(v) for v in a.dims.split(","))
    P = cellset.params_for(cs, cell, T_K, film_share=a.film_share, ds_on=a.ds_on, ea={"Ea_Dp": a.ea_dp} if a.ea_dp else None)
    if a.soft_eps:
        P["soft_eps"] = a.soft_eps
    for k, v in (it.split("=") for it in a.p_mult.split(",") if it):
        for kk in (["k0_n", "k0_p"] if k == "k0" else [k]):
            P[kk] = P[kk] * float(v)
    m = DFN(P, *dims, 2, dt, a.grid)
    z_of = lambda q: np.asarray(m.init_state(*cellset.c0_at(cs, cell, q)))
    z0 = z_of(q0)
    I = jnp.asarray(I_np)
    th_true = jnp.array([TRUE["Rf"], 1.0 - TRUE["LAM"], TRUE["Ds"]])
    if a.twin:
        rng = np.random.default_rng(12345)
        V_model = np.asarray(jax.jit(lambda: simulate(m, jnp.asarray(z0), I, th_true, k=None)["V"])())
        y = V_model + rng.normal(0.0, a.twin_noise_mV * 1e-3, len(V_model))
        tag += "_twin"
    else:
        y = d["V"]
    ok = np.asarray(d["valid"], bool)
    tfac = None
    if a.t_input:
        Tc = pd.Series(d["T"]).rolling(max(1, int(round(60 / dt))), center=True, min_periods=1).mean().values
        Ta = np.percentile(Tc, 5); Tk = Ta + a.t_rise * (Tc - Ta) + 273.15
        ea_dp = a.ea_dp or cs["temperature"]["Ea_Dp"]
        tfac = jnp.asarray(np.exp(-ea_dp / 8.314462618 * (1 / Tk - 1 / T_K)))
        print("t-input: Ea_Dp %.0f J/mol, factor %.2f-%.2f" % (ea_dp, float(tfac.min()), float(tfac.max())), flush=True)

    # ---- filter
    li_ref = float(m.lithium(m.init_state(*cellset.c0_at(cs, cell, 0.0)), jnp.array([1.0, 1.0, 1.0])))
    # li_ref: the lithium of the identified beginning-of-life window (x100, y100; any q: x Q_n + y Q_p is constant on it)
    ek = EKF(m, coords=coords, fixed=dict(Rf=1.0, LAM=0.0, Ds=1.0), transform=a.transform, li_sigma=a.li_sigma, li_ref=li_ref)
    nel, nz = m.nel, m.nz
    rng_w = np.array([RANGE[c][1] - RANGE[c][0] for c in coords])
    lo = np.array([RANGE[c][0] for c in coords]); hi = np.array([RANGE[c][1] for c in coords])
    tv = np.array([TRUE[c] for c in coords])
    sd_shell = a.p0_shell * np.concatenate([np.full(m.Nn, P["cmax_n"]), np.full(m.Np, P["cmax_p"])])
    sd_rest = np.concatenate([np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0)])
    R = (a.sigma_mV * 1e-3) ** 2
    run = jax.jit(lambda x0, P0, Qd: ek.run(x0, P0, Qd, R, I, jnp.asarray(y), tfac=tfac))
    rows = []
    if a.start == "soc":
        rows = run_soc(a, d, cs, cell, m, ek, run, z_of, q0, y, t, I_np, coords, tv, rng_w, sd_shell, sd_rest, dt, tag, I, th_true)
        out = os.path.join(a.results, "ekf_%s.csv" % tag)
        old = pd.read_csv(out) if os.path.exists(out) else None
        df = pd.DataFrame(rows)
        if old is not None:
            df = pd.concat([old[~old.set_index(["kind", "err_Ah"]).index.isin(df.set_index(["kind", "err_Ah"]).index)], df])
        df.to_csv(out, index=False)
        print("wrote", out)
        return
    # params experiment
    kv = lambda spec: {k: float(v) for k, v in (it.split("=") for it in spec.split(",") if it)}
    init, p0c = kv(a.init), kv(a.p0_coord)
    dshift = P["dx_window_n"] * P["cmax_n"]
    for seed in [int(s) for s in a.seeds.split(",")]:
        rg = np.random.default_rng(200 + seed)
        th0 = np.clip(tv + rg.uniform(-a.offset, a.offset, len(coords)) * rng_w, lo, hi) if seed >= 0 else np.array([init.get(c, t_) for c, t_ in zip(coords, tv)])
        c0 = ek.coord_of(th0); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))
        z0s = z0.copy()
        if "LLI" in coords and not a.li_sigma:          # the closure sees z - LLI dx c_max,n: start consistent with the OCV
            z0s[:m.Nn] += th0[coords.index("LLI")] * dshift
        x0 = np.concatenate([z0s, c0])
        p0 = np.array([p0c.get(c, a.p0_th * w / d_) for c, w, d_ in zip(coords, rng_w, dc)])
        P0d = np.concatenate([sd_shell, sd_rest, p0])
        Qd = np.concatenate([np.zeros(nz), a.q_th * rng_w * np.sqrt(dt) / dc])
        t0 = time.time()
        H = {k: np.asarray(v) for k, v in run(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd)).items()}
        wall = time.time() - t0
        vals, sd = H["vals"], H["sd_vals"]
        err = vals - tv[None, :]
        nees = np.mean(np.sum((err / sd) ** 2, 1)[len(err) // 2:])
        innov = H["innov"][ok] * 1e3
        n3 = 2 * ok.sum() // 3
        row = dict(dataset=a.dataset, cell=cell, test=d["test"], twin=a.twin, seed=seed, n=len(t), dt_s=dt, T_C=T_K - 273.15,
                   wall_s=wall, ms_per_step=1e3 * wall / len(t), innov_rmse_mV=float(np.sqrt(np.mean(innov ** 2))),
                   nis=float(np.mean((H["innov"][ok] / H["Sy"][ok]) ** 2)),
                   nis_last_third=float(np.mean((H["innov"][ok][n3:] / H["Sy"][ok][n3:]) ** 2)),
                   sigma_mV=a.sigma_mV, q_th=a.q_th, p0_th=a.p0_th, li_sigma=a.li_sigma,
                   li_drift_pct=float(100 * H["li_rel"][-1]),
                   resF_max=float(H["resF"].max()), finite=bool(np.all(H["finite"])), nees_2nd_half=float(nees))
        for i, c in enumerate(coords):
            row["%s_0" % c] = float(th0[i]); row["%s_end" % c] = float(vals[-1, i]); row["%s_sd_end" % c] = float(sd[-1, i])
            row["%s_err_pct_range" % c] = float(100 * err[-1, i] / rng_w[i])
            tl = vals[2 * len(vals) // 3:, i]
            row["%s_last_third_spread" % c] = float(np.std(np.log10(tl)) if c in ("Rf", "Ds") else np.std(tl))
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        meta = d.get("meta", {})
        np.savez(os.path.join(a.results, "ekf_%s_s%d.npz" % (tag, seed)), t=t, I=I_np, y=y,
                 V=H["V"], vals=vals, sd=sd, innov=H["innov"], resF=H["resF"], coords=np.array(coords),
                 xbar=H["xbar"], ybar=H["ybar"], sd_xbar=H["sd_xbar"], sd_ybar=H["sd_ybar"], q0=q0, T_C=d["T"],
                 eis=(~ok).astype(int), dt=dt, cell=cell, test=d["test"], twin=a.twin, sigma_mV=a.sigma_mV, seed=seed,
                 cycle=-1 if a.cycle is None else a.cycle, SOH_ref=meta.get("SOH_ref", np.nan), dx_window_n=P["dx_window_n"],
                 n_first=meta.get("n_first", 0), n_cycles=a.n_cycles if a.cycle is not None else 0)
        plot_params(a, t, vals, sd, tv, coords, H, ok, cell, d["test"], seed, row, tag)
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "ekf_%s.csv" % tag), index=False)


def run_soc(a, d, cs, cell, m, ek, run, z_of, q0, y, t, I_np, coords, tv, rng_w, sd_shell, sd_rest, dt, tag, I, th_true):
    """The SOC experiment: open loop and EKF from the wrong starts q0 + err; returns the csv rows."""
    c = cs["cells"][cell]
    P = m.P
    Q_ref = d.get("meta", {}).get("Q_ref") or float(c.get("q_ocv_Ah", 1.0))
    q_ref = q0 + np.cumsum(I_np) * dt / 3600.0              # charge below the top after each step [A.h]
    soc_ref = 1.0 - q_ref / Q_ref
    errs = [float(e) for e in a.soc_err.split(",")]
    rows = []
    if not a.no_openloop:
        sim = jax.jit(lambda z0: simulate(m, z0, I, th_true, k=None)["V"])
        for e in errs:
            t0 = time.time()
            Vo = np.asarray(sim(jnp.asarray(z_of(q0 + e))))
            ok = np.isfinite(Vo)
            r = 1e3 * np.sqrt(np.mean((Vo[ok] - y[ok]) ** 2))
            print("open loop, start error %+.2f A.h: RMSE %.1f mV (%.1f s)" % (e, r, time.time() - t0))
            np.save(os.path.join(a.results, "open_%s_e%+.2f.npy" % (tag, e)), Vo)
            rows.append(dict(test=d["test"], kind="open", err_Ah=e, V_rmse_mV=r))
    nel, nz = m.nel, m.nz
    v = np.zeros(nz)
    v[:m.Nn] = -P["cmax_n"] / c["Q_n"]; v[m.Nn:nel] = P["cmax_p"] / c["Q_p"]    # d c_avg / d q [mol/m3 per A.h]
    c0 = ek.coord_of(tv); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))
    p0_th = a.p0_th * rng_w / dc
    Pz = np.diag(np.r_[sd_shell, sd_rest] ** 2)
    Pz += a.p0_q ** 2 * np.outer(v, v)
    P0 = np.zeros((ek.n, ek.n)); P0[:nz, :nz] = Pz; P0[nz:, nz:] = np.diag(p0_th ** 2)
    Qd = np.r_[np.zeros(nz), a.q_th * rng_w * np.sqrt(dt) / dc]
    for e in errs:
        x0 = np.r_[z_of(q0 + e), c0]
        t0 = time.time()
        H = {k: np.asarray(vv) for k, vv in run(jnp.asarray(x0), jnp.asarray(P0), jnp.asarray(Qd)).items()}
        wall = time.time() - t0
        q_p = (H["ybar"] - c["y100"]) * c["Q_p"]; q_n = (c["x100"] - H["xbar"]) * c["Q_n"]
        soc_p, soc_n = 1 - q_p / Q_ref, 1 - q_n / Q_ref
        sd_soc = H["sd_ybar"] * c["Q_p"] / Q_ref
        es = 100 * (soc_p - soc_ref)
        k10 = np.argmax(np.abs(es) < 2.0) if np.any(np.abs(es) < 2.0) else -1
        settle = np.where(np.abs(es) >= 2.0)[0]
        t_set = float(t[settle[-1]] if len(settle) and settle[-1] < len(t) - 1 else (np.nan if len(settle) else 0.0))
        innov = H["innov"] * 1e3
        half = len(t) // 2
        row = dict(test=d["test"], kind="ekf", err_Ah=e, wall_s=wall, ms_per_step=1e3 * wall / len(t),
                   innov_rmse_mV=float(np.sqrt(np.mean(innov ** 2))), finite=bool(np.all(H["finite"])),
                   soc_err_end_pct=float(es[-1]), soc_err_rms_2nd_half_pct=float(np.sqrt(np.mean(es[half:] ** 2))),
                   soc_err_max_2nd_half_pct=float(np.max(np.abs(es[half:]))), t_settle_2pct_h=t_set / 3600,
                   t_first_2pct_min=float(t[k10] / 60) if k10 >= 0 else np.nan,
                   cover_2sd_pct=float(100 * np.mean(np.abs(es) <= 200 * sd_soc)),
                   soc_n_minus_p_end_pct=float(100 * (soc_n[-1] - soc_p[-1])), li_drift_pct=float(100 * H["li_rel"][-1]),
                   resF_max=float(H["resF"].max()))
        for i, cc in enumerate(coords):
            row["%s_end" % cc] = float(H["vals"][-1, i]); row["%s_sd_end" % cc] = float(H["sd_vals"][-1, i])
        rows.append(row)
        print(json.dumps({k: (round(vv, 4) if isinstance(vv, float) else vv) for k, vv in row.items()}))
        np.savez(os.path.join(a.results, "ekf_%s_e%+.2f.npz" % (tag, e)), t=t, I=I_np, y=y, V=H["V"], innov=H["innov"],
                 Sy=H["Sy"], soc_p=soc_p, soc_n=soc_n, sd_soc=sd_soc, soc_ref=soc_ref, vals=H["vals"], sd_vals=H["sd_vals"],
                 coords=np.array(coords), resF=H["resF"], q0=q0, err=e, sigma_mV=a.sigma_mV)
    return rows


def plot_params(a, t, vals, sd, tv, coords, H, ok, cell, test, seed, row, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    th_h = t / 3600
    fig, ax = plt.subplots(len(coords) + 1, 1, figsize=(8, 2.1 * (len(coords) + 1)), sharex=True)
    for i, c in enumerate(coords):
        ax[i].fill_between(th_h, vals[:, i] - 2 * sd[:, i], vals[:, i] + 2 * sd[:, i], color="C0", alpha=0.2, lw=0)
        ax[i].plot(th_h, vals[:, i], "C0", lw=1)
        ax[i].axhline(tv[i], color="k", lw=0.8, ls="--")
        ax[i].set_ylabel(c)
    ax[-1].plot(th_h, np.where(ok, H["innov"] * 1e3, np.nan), "C3", lw=0.6)
    ax[-1].set_ylabel("innovation [mV]"); ax[-1].set_xlabel("t [h]")
    ax[0].set_title("%s %s%s  seed %d  |  innov. RMSE %.1f mV  |  sigma %.0f mV" % (cell, test, " (twin)" if a.twin else "",
                    seed, row["innov_rmse_mV"], a.sigma_mV), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "ekf_%s_s%d.png" % (tag, seed)), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
