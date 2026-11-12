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
from dfnekf.rom import simulate, simulate_vdt  # noqa: E402
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
    ap.add_argument("--q-shell", type=float, default=0.0,
                    help="[soc] process noise on the particle mean concentrations: random walk, fraction of c_max per sqrt(s), "
                         "independent per node (0: none, the model is trusted)")
    ap.add_argument("--q-q", type=float, default=0.0,
                    help="[soc] process noise on the cell's charge: random walk of sd q_q [A.h per sqrt(h)] along the "
                         "lithium-conserving direction (anode and cathode means together); 0: none")
    ap.add_argument("--soc-vlim", default=None,
                    help="[soc] V_max,V_min: SOC from the stoichiometry, on the window between the model OCVs V_max and V_min "
                         "(SOC = (y_0 - y) / (y_0 - y_100), anode (x - x_0) / (x_100 - x_0)); the Coulomb-counting reference is "
                         "normalised by the same window. Default: the set's window (q_ocv_Ah wide)")
    ap.add_argument("--soc0", type=float, default=None,
                    help="[soc] start every run at this SOC (fraction of the window, Q_ref) instead of truth + --soc-err")
    ap.add_argument("--li-sigma-Ah", type=float, default=None,
                    help="sd of the lithium pseudo-measurement in A.h of lithium (converted to --li-sigma with the electrode area)")
    ap.add_argument("--li-sigma", type=float, default=None,
                    help="relative sd of the lithium pseudo-measurement (total lithium of the state = the window's); off by default")
    # soc experiment
    ap.add_argument("--soc-err", default="0", help="[soc] initial charge errors [A.h] (+ = filter starts emptier); 0 = the placed start only")
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
    ap.add_argument("--adaptive", action="store_true",
                    help="variable step from the --dt grid: a step ends when the current leaves the band of its first sample "
                         "(|I - I_0| > --adapt-rel |I_0| + --adapt-abs) or after --adapt-max s; model current = the step's mean, "
                         "measured voltage = its last sample (as lpmodel.prepare)")
    ap.add_argument("--adapt-max", type=float, default=10.0, help="[s] longest step")
    ap.add_argument("--adapt-rel", type=float, default=0.05)
    ap.add_argument("--adapt-abs", type=float, default=0.02, help="[A] floor of the band (rests, end of CV)")
    ap.add_argument("--mask-vmax", type=float, default=None, help="no voltage update while charging at V >= this (late CC, CV) [V]")
    ap.add_argument("--mask-vmin", type=float, default=None, help="no voltage update at V <= this (end-of-discharge knee) [V]")
    ap.add_argument("--y-every-s", type=float, default=None,
                    help="voltage update only once per this many seconds (the others: prediction only) [s]")
    ap.add_argument("--y-burn-s", type=float, default=None,
                    help="with --y-every-s: voltage update at every sample for the first this many seconds [s]")
    ap.add_argument("--mask-hold-s", type=float, default=300.0, help="the mask-vmin mask is held this long after the voltage recovers [s]")
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
    dts = None
    if a.adaptive:
        d = adaptive_steps(d, dt, a.adapt_max, a.adapt_rel, a.adapt_abs)
        t, I_np, dts = d["t"], d["I"], d["dts"]
        print("adaptive: %d steps from %d (mean %.2f s; %d at the %g-s maximum)" % (len(dts), int(round(dts.sum() / dt)), dts.mean(),
              int(np.sum(dts >= a.adapt_max)), a.adapt_max), flush=True)
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
    mask = np.zeros(len(y), bool)
    if a.mask_vmax or a.mask_vmin:
        # no voltage update where the model is known to be poor: the top of charge (charging at V >= mask_vmax: late CC
        # and CV) and the end-of-discharge knee (V <= mask_vmin) plus the following mask_hold_s (relaxation from it)
        if a.mask_vmax:
            mask |= (I_np < -0.02) & (y >= a.mask_vmax)
        if a.mask_vmin:
            low = y <= a.mask_vmin
            hold = int(round(a.mask_hold_s / dt))
            last = -10 ** 9
            for i in range(len(y)):
                if low[i]:
                    last = i
                if i - last <= hold:
                    mask[i] = True
        print("masked: %.1f %% of the samples" % (100 * mask.mean()), flush=True)
    if a.y_every_s:
        # one voltage update per y_every_s seconds: the model error is correlated over minutes, so consecutive samples
        # are not independent information (the white-noise R would make the filter overconfident in the model)
        keep = np.r_[True, np.diff(np.floor(t / a.y_every_s)) > 0]
        if a.y_burn_s:                      # every sample during the burn-in: many small corrections from a wrong start
            keep |= t <= a.y_burn_s         # instead of one large linearised jump
        mask |= ~keep
        print("voltage update every %g s: %.1f %% of the samples used" % (a.y_every_s, 100 * (~mask).mean()), flush=True)
    y_f = np.where(mask, np.nan, y)
    ok = ok & ~mask
    tfac = None
    if a.t_input:
        Tc = pd.Series(d["T"]).rolling(max(1, int(round(60 / dt))), center=True, min_periods=1).mean().values
        Ta = np.percentile(Tc, 5); Tk = Ta + a.t_rise * (Tc - Ta) + 273.15
        ea_dp = a.ea_dp or cs["temperature"]["Ea_Dp"]
        tfac = jnp.asarray(np.exp(-ea_dp / 8.314462618 * (1 / Tk - 1 / T_K)))
        print("t-input: Ea_Dp %.0f J/mol, factor %.2f-%.2f" % (ea_dp, float(tfac.min()), float(tfac.max())), flush=True)

    # ---- filter
    li_ref = float(m.lithium(m.init_state(*cellset.c0_at(cs, cell, 0.0)), jnp.array([1.0, 1.0, 1.0])))
    if a.li_sigma_Ah:                      # mol/m2 of electrode -> A.h: x area x F / 3600
        a.li_sigma = a.li_sigma_Ah / (li_ref * P["area"] * 96485.33212 / 3600.0)
        print("lithium pseudo-measurement: sd %.2g A.h of %.3f A.h (relative %.3g)" % (a.li_sigma_Ah, li_ref * P["area"] * 96485.33212 / 3600.0, a.li_sigma), flush=True)
    # li_ref: the lithium of the identified beginning-of-life window (x100, y100; any q: x Q_n + y Q_p is constant on it)
    ek = EKF(m, coords=coords, fixed=dict(Rf=1.0, LAM=0.0, Ds=1.0), transform=a.transform, li_sigma=a.li_sigma, li_ref=li_ref)
    nel, nz = m.nel, m.nz
    rng_w = np.array([RANGE[c][1] - RANGE[c][0] for c in coords])
    lo = np.array([RANGE[c][0] for c in coords]); hi = np.array([RANGE[c][1] for c in coords])
    tv = np.array([TRUE[c] for c in coords])
    sd_shell = a.p0_shell * np.concatenate([np.full(m.Nn, P["cmax_n"]), np.full(m.Np, P["cmax_p"])])
    sd_rest = np.concatenate([np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0)])
    R = (a.sigma_mV * 1e-3) ** 2
    run = jax.jit(lambda x0, P0, Qd: ek.run(x0, P0, Qd, R, I, jnp.asarray(y_f), tfac=tfac, dts=dts))
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
                 n_first=meta.get("n_first", 0), n_cycles=a.n_cycles if a.cycle is not None else 0, masked=mask)
        plot_params(a, t, vals, sd, tv, coords, H, ok, cell, d["test"], seed, row, tag)
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "ekf_%s.csv" % tag), index=False)


def adaptive_steps(d, dt, max_s, rel, abs_):
    """Variable steps from a uniform record of step dt: a step starts at sample i and takes the following samples while
    |I_j - I_i| <= rel |I_i| + abs_ and the step is shorter than max_s; I = the step's mean, V = its last sample, t = the
    end of the step, T = the mean, valid = all. Returns d with t, I, V, T, valid replaced and dts [s] added."""
    I, n, L = d["I"], len(d["I"]), int(round(max_s / dt))
    starts, i = [], 0
    while i < n:
        j = i + 1
        while j < n and j - i < L and abs(I[j] - I[i]) <= rel * abs(I[i]) + abs_:
            j += 1
        starts.append((i, j)); i = j
    out = dict(d)
    out["t"] = np.array([d["t"][j - 1] for i, j in starts])
    out["I"] = np.array([I[i:j].mean() for i, j in starts])
    out["V"] = np.array([d["V"][j - 1] for i, j in starts])
    out["T"] = np.array([np.mean(d["T"][i:j]) for i, j in starts])
    out["valid"] = np.array([bool(np.all(d["valid"][i:j])) for i, j in starts])
    out["dts"] = np.array([(j - i) * dt for i, j in starts])
    return out


def window_lli(cs, cell, vmax, vmin, T_K, dxw, llis=np.linspace(-0.1, 0.6, 141), n=8001):
    """OCV window between vmax and vmin with the anode stoichiometry lowered by LLI dxw (lithium lost by the anode), on a
    grid of LLI [anode-window units]. Per LLI: x0, x100, y100, y0 at the window ends and its capacity Q [A.h]."""
    c = cs["cells"][cell]; tp = cs.get("temperature", {})
    xn, Un = np.asarray(cs["ocp"]["n"]["x"]), np.asarray(cs["ocp"]["n"]["U"])
    yp, Up = np.asarray(cs["ocp"]["p"]["y"]), np.asarray(cs["ocp"]["p"]["U"])
    q = np.linspace(-0.5, 3.0, n)
    out = {k: [] for k in ("x0", "x100", "y100", "y0", "Q")}
    for L in llis:
        x = c["x100"] - L * dxw - q / c["Q_n"]; y = c["y100"] + q / c["Q_p"]
        U = np.interp(np.clip(y, yp[0], yp[-1]), yp, Up) - np.interp(np.clip(x, xn[0], xn[-1]), xn, Un)
        if "dUdT0" in tp:
            U = U + (T_K - cs["T_ref_K"]) * (tp["dUdT0"] + tp.get("dUdT_n", 0.0) * (x - 0.5))
        ok = (y <= yp[-1]) & (x >= xn[0])
        qt = q[np.argmax(U <= vmax)]; qb = q[ok][np.where(U[ok] >= vmin)[0][-1]]
        out["x100"].append(c["x100"] - L * dxw - qt / c["Q_n"]); out["y100"].append(c["y100"] + qt / c["Q_p"])
        out["x0"].append(c["x100"] - L * dxw - qb / c["Q_n"]); out["y0"].append(c["y100"] + qb / c["Q_p"]); out["Q"].append(qb - qt)
    return dict(lli=llis, **{k: np.array(v) for k, v in out.items()})


def run_soc(a, d, cs, cell, m, ek, run, z_of, q0, y, t, I_np, coords, tv, rng_w, sd_shell, sd_rest, dt, tag, I, th_true):
    """The SOC experiment: open loop and EKF from the wrong starts q0 + err; returns the csv rows."""
    c = cs["cells"][cell]
    P = m.P
    # SOC window in charge below the top of the set's window, q in [q_top, q_bot]; electrode stoichiometries at its ends
    if a.soc_vlim:
        vmax, vmin = (float(v) for v in a.soc_vlim.split(","))
        T_K = float(d["T_C"]) + 273.15
        q_top, q_bot = datasets.q_from_ocv(cs, cell, vmax, T_K), datasets.q_from_ocv(cs, cell, vmin, T_K)
    else:
        q_top, q_bot = 0.0, d.get("meta", {}).get("Q_ref") or float(c.get("q_ocv_Ah", 1.0))
    Q_ref = q_bot - q_top
    x100w, y100w = c["x100"] - q_top / c["Q_n"], c["y100"] + q_top / c["Q_p"]
    x0w, y0w = c["x100"] - q_bot / c["Q_n"], c["y100"] + q_bot / c["Q_p"]
    print("SOC window: q %.3f..%.3f A.h (%.3f A.h); x %.4f..%.4f, y %.4f..%.4f" % (q_top, q_bot, Q_ref, x0w, x100w, y100w, y0w), flush=True)
    q_ref = q0 + np.cumsum(I_np * d.get("dts", dt)) / 3600.0   # charge below the top after each step [A.h]
    soc_ref = (q_bot - q_ref) / Q_ref                          # reference: Coulomb counting on the same window
    errs = [float(e) for e in a.soc_err.split(",")]
    if a.soc0 is not None:                  # absolute start at soc0 of the window
        errs = [q_bot - a.soc0 * Q_ref - q0]
        print("start at SOC %.0f %%: error %+.3f A.h against the reference start" % (100 * a.soc0, errs[0]), flush=True)
    rows = []
    if not a.no_openloop:
        sim = jax.jit(lambda z0: simulate(m, z0, I, th_true, k=None)["V"]) if "dts" not in d else             jax.jit(lambda z0: simulate_vdt(m, z0, I, d["dts"], th_true))
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
    qz = np.zeros(nz)
    qz[:m.Nn] = a.q_shell * P["cmax_n"] * np.sqrt(dt); qz[m.Nn:nel] = a.q_shell * P["cmax_p"] * np.sqrt(dt)
    Qd = np.r_[qz, a.q_th * rng_w * np.sqrt(dt) / dc]
    if a.q_q > 0:
        # random walk on the cell's charge along the lithium-conserving direction v (all anode means move together, all
        # cathode means opposite): SOC uncertainty without roughening the particle profiles. sd q_q A.h per sqrt(h).
        Qm = np.diag(Qd ** 2)
        Qm[:nz, :nz] += (a.q_q ** 2 * dt / 3600.0) * np.outer(v, v)
        Qd = Qm
    for e in errs:
        x0 = np.r_[z_of(q0 + e), c0]
        t0 = time.time()
        H = {k: np.asarray(vv) for k, vv in run(jnp.asarray(x0), jnp.asarray(P0), jnp.asarray(Qd)).items()}
        wall = time.time() - t0
        extra = {}
        if "LLI" in coords and a.soc_vlim:
            # LLI moves the anode stoichiometry the model sees (x_eff = x - LLI dx_window_n): the OCV window ends at V_max /
            # V_min and the window capacity move with it. SOC on the moved window; SOH = Q_window(LLI) / Q_window(0).
            il = list(coords).index("LLI"); lli = H["vals"][:, il]; sd_lli = H["sd_vals"][:, il]; dxw = P["dx_window_n"]
            W = window_lli(cs, cell, vmax, vmin, T_K, dxw)
            xw0, xw1, yw1, yw0, Qw = (np.interp(lli, W["lli"], W[k]) for k in ("x0", "x100", "y100", "y0", "Q"))
            x_eff = H["xbar"] - lli * dxw
            soc_p = (yw0 - H["ybar"]) / (yw0 - yw1); soc_n = (x_eff - xw0) / (xw1 - xw0)
            sd_soc = 0.5 * (H["sd_xbar"] / (xw1 - xw0) + H["sd_ybar"] / (yw0 - yw1))
            Q0w = float(np.interp(0.0, W["lli"], W["Q"])); dQ = np.gradient(W["Q"], W["lli"])
            soh = Qw / Q0w; sd_soh = np.abs(np.interp(lli, W["lli"], dQ)) * sd_lli / Q0w
            extra = dict(soh=soh, sd_soh=sd_soh, lli_Ah=lli * dxw * c["Q_n"], Q_window_Ah=Qw, Q_window0_Ah=Q0w)
            if a.dataset == "exp_LP604765":     # SOH on the nominal capacity (the only place it appears)
                extra.update(soh_nom=Qw / datasets.LP_Q_NOMINAL, sd_soh_nom=sd_soh * Q0w / datasets.LP_Q_NOMINAL)
            print("SOH (OCV window %g-%g V): %.2f %% of BOL at the end, sd %.2f %%; LLI %.0f mAh; window %.3f A.h" % (vmax, vmin, 100 * soh[-1], 100 * sd_soh[-1],
                  1e3 * extra["lli_Ah"][-1], Qw[-1]), flush=True)
        else:
            soc_p = (y0w - H["ybar"]) / (y0w - y100w)                  # SOC from the stoichiometry of each electrode
            soc_n = (H["xbar"] - x0w) / (x100w - x0w)
            sd_soc = 0.5 * (H["sd_xbar"] / (x100w - x0w) + H["sd_ybar"] / (y0w - y100w))   # bound: fully correlated electrodes
        soc = 0.5 * (soc_n + soc_p)                                 # the cell's SOC: mean of the two electrodes
        es = 100 * (soc - soc_ref)
        k10 = np.argmax(np.abs(es) < 2.0) if np.any(np.abs(es) < 2.0) else -1
        settle = np.where(np.abs(es) >= 2.0)[0]
        t_set = float(t[settle[-1]] if len(settle) and settle[-1] < len(t) - 1 else (np.nan if len(settle) else 0.0))
        innov = H["innov"] * 1e3
        half = len(t) // 2
        row = dict(test=d["test"], kind="ekf", err_Ah=e, wall_s=wall, ms_per_step=1e3 * wall / len(t),
                   innov_rmse_mV=float(np.sqrt(np.nanmean(innov ** 2))), finite=bool(np.all(H["finite"])),
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
                 Sy=H["Sy"], soc=soc, soc_p=soc_p, soc_n=soc_n, sd_soc=sd_soc, soc_ref=soc_ref, vals=H["vals"], sd_vals=H["sd_vals"],
                 coords=np.array(coords), resF=H["resF"], q0=q0, err=e, posdef=H["posdef"], minDiagP=H["minDiagP"],
                 ce_min=H["ce_min"], step_finite=H["finite"], li_rel=H["li_rel"], eta=H["eta"], sigma_mV=a.sigma_mV, soc_window=np.array([x0w, x100w, y100w, y0w]), cap_1C_t=d.get("meta", {}).get("cap_1C", (np.zeros(0), np.zeros(0)))[0], cap_1C=d.get("meta", {}).get("cap_1C", (np.zeros(0), np.zeros(0)))[1], **extra,
                 **anode_profile(m, H, coords))
    return rows


def anode_profile(m, H, coords):
    """Along the anode, from the closure solution g of each step (at the predicted state): x of the cell centres [m],
    phi_s - phi_e [V] and the plating potential phi_s - phi_e - R_f j_n [V vs Li/Li+] (step.DFN.interface, per node)."""
    A_ = m.A; Nn, Np, N = m.Nn, m.Np, m.N
    g = np.asarray(H["g"])
    jn = g[:, :Nn]; pe = g[:, Nn + Np:Nn + Np + N][:, :Nn]; sn = g[:, Nn + Np + N:2 * Nn + Np + N]
    th = np.asarray(H["vals"])[:, list(coords).index("Rf")] if "Rf" in coords else np.ones(len(g))   # R_f multiplier (fixed: 1)
    Rf = m.P["Rf0"] * th[:, None]
    dxn = np.asarray(A_.dxn)
    return dict(x_n=np.cumsum(dxn) - 0.5 * dxn, dphi_n=sn - pe, phi_pl_n=sn - pe - Rf * jn, j_n=jn)


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
