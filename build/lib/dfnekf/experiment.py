"""Experiment infrastructure: cases (PyBaMM reference or twin ROM as the truth), initial covariance and process noise, EKF / UKF / SPMe runs and run summaries."""
import os, json, time
import numpy as np, jax, jax.numpy as jnp
from . import params as prm, refio
from .step import DFN, interface_ref
from .rom import simulate
from .ekf import EKF, RANGE as TH_RANGE, LOGIT
from .ukf import UKF

TRUE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "theta_true.json")))
COORDS = ("Rf", "LAM", "Ds")
RANGEW = np.array([TH_RANGE[c][1] - TH_RANGE[c][0] for c in COORDS])
NAMES = {"Rf": "R_f", "LAM": "LAM_n", "Ds": "D_s"}


class Case:
    def __init__(self, tag, grid, twin=False, proto="P6Lx2", dims=(20, 10, 20, 12), dt=0.5, coords=COORDS, closure="newton", net=None, extra_ek=None, data_dir="data"):
        """tag: ageing state th1..th5; grid: 'uniform' or '4:1' (grey-box model); dims: (Nn, Ns, Np, Nm); data_dir: folder with the PyBaMM references."""
        self.tag, self.twin, self.dt, self.dims, self.grid_name, self.coords = tag, twin, dt, dims, grid, tuple(coords)
        self.grid = dims
        self.P0 = prm.base(); self.m = DFN(self.P0, *dims, 2, dt, grid)
        self.r = (refio.load_uniform(proto, tag, period=dt, data_dir=data_dir) if str(proto).startswith(("F6C", "F6Q")) else refio.load(proto, tag, period=dt, data_dir=data_dir)); self.N = self.r["N"]
        self.theta = np.array([self.r["Rf"], self.r["LAM"], self.r["LLI"], self.r["Ds"]])
        self.z0 = np.asarray(self.m.init_state(*[float(x) for x in self.r["c0_eff"]]))
        self.I = jnp.asarray(self.r["I"])
        th = jnp.array([self.r["Rf"], 1 - self.r["LAM"], self.r["Ds"]])
        if twin:
            self.Vtrue = np.asarray(jax.jit(lambda: simulate(self.m, jnp.asarray(self.z0), self.I, th, k=None)["V"])())
        else:
            self.Vtrue = self.r["V"]
        self.ek = EKF(self.m, net=net, closure=closure, coords=coords, fixed=dict(Rf=1.0, LAM=0.0, Ds=1.0), **(extra_ek or {}))
        self.nth = len(coords); self.nz = self.m.nz
        self.true_vals = np.array([{"Rf": self.r["Rf"], "LAM": self.r["LAM"], "Ds": self.r["Ds"]}[c] for c in coords])
        self.range = np.array([TH_RANGE[c][1] - TH_RANGE[c][0] for c in coords])
        # LLI nota: fissata per i filtri nello stato iniziale (z0 effettivo)

    def offsets(self, i, mag=0.4):
        """theta_0 = theta_vero + offset uniforme in +-40 % del range (seme 200 + i), ritagliato nel range di aging (non nel range logit: evita di partire in saturazione della sigmoide)."""
        rng = np.random.default_rng(200 + i); off = rng.uniform(-mag, mag, self.nth) * self.range
        lo = np.array([TH_RANGE[c][0] for c in self.coords]); hi = np.array([TH_RANGE[c][1] for c in self.coords])
        return np.clip(self.true_vals + off, lo, hi)

    def x0_P0_Q(self, th0, p0_scale=1.0, qth_scale=1.0):
        ek, m = self.ek, self.m; nel = m.nel
        c0 = ek.coord_of(th0); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))                     # dtheta/dc
        sig_th = 0.3 * self.range * p0_scale                                                      # 30 % del range in unita' theta
        q_th = 1e-4 * self.range * np.sqrt(self.dt) * qth_scale                                   # random walk 1e-4 del range al passo (dt = 1 s), scalato per sqrt(dt)
        x0 = np.concatenate([self.z0, c0])
        P0d = np.concatenate([np.full(nel, 10.0), np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0), sig_th / dc])
        Qd = np.concatenate([np.zeros(m.nz), q_th / dc])
        return x0, P0d, Qd

    def x_true(self):
        c = self.ek.coord_of(self.true_vals)
        return np.tile(np.concatenate([np.zeros(self.nz), c]), (self.N, 1))

    def eta_ref(self):
        return np.array([interface_ref(self.r, i, self.P0, self.r["Rf"])[0] for i in range(self.N)])


_JIT = {}
def run_ekf(case, i, p0_scale=1.0, qth_scale=1.0, y=None, seed=None, name_extra=""):
    ek = case.ek; key = ("ekf", id(ek))
    if key not in _JIT:
        _JIT[key] = jax.jit(lambda x0, P0d, Qd, y, xt, I: ek.run(x0, P0d, Qd, (2e-3) ** 2, I, y, xt))
    th0 = case.offsets(i); x0, P0d, Qd = case.x0_P0_Q(th0, p0_scale, qth_scale)
    seed = i if seed is None else seed
    y = case.Vtrue + 2e-3 * np.random.default_rng(seed).normal(size=case.N) if y is None else y
    t0 = time.time(); H = _JIT[key](jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), jnp.asarray(y), jnp.asarray(case.x_true()), case.I); jax.block_until_ready(H["vals"]); wall = time.time() - t0
    return summarize(case, "ekf", i, th0, H, wall, name_extra)


def run_ukf(case, i, closure_iters=1, p0_scale=1.0, qth_scale=1.0, y=None, seed=None, name_extra=""):
    """UKF on the same model and tuning as the EKF; ``closure_iters`` Newton iterations per sigma point (None: converged)."""
    ek = case.ek; key = ("ukf", id(ek), closure_iters)
    if key not in _JIT:
        uk = UKF(ek, closure_iters=closure_iters); _JIT[key] = jax.jit(lambda x0, P0d, Qd, y, xt, I: uk.run(x0, P0d, Qd, (2e-3) ** 2, I, y, xt))
    th0 = case.offsets(i); x0, P0d, Qd = case.x0_P0_Q(th0, p0_scale, qth_scale)
    seed = i if seed is None else seed
    y = case.Vtrue + 2e-3 * np.random.default_rng(seed).normal(size=case.N) if y is None else y
    t0 = time.time(); H = _JIT[key](jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), jnp.asarray(y), jnp.asarray(case.x_true()), case.I); jax.block_until_ready(H["vals"]); wall = time.time() - t0
    return summarize(case, "ukf", i, th0, H, wall, name_extra)


def summarize(case, filt, i, th0, H, wall, name_extra=""):
    v = np.asarray(H["vals"]); err = 100 * np.abs(v - case.true_vals) / case.range
    res = dict(filter=filt, closure=case.ek.closure, grid="-".join(map(str, case.dims[:3])) + f"_{case.grid_name}_Nm{case.dims[3]}", dt=case.dt, theta_tag=case.tag, twin=case.twin, run=i, seed=i,
               wall_s=wall, finite=bool(np.all(np.isfinite(v))), resF_median=float(np.median(H["resF"])), resF_p99=float(np.quantile(H["resF"], .99)), resF_max=float(np.max(H["resF"])))
    for j, c in enumerate(case.coords):
        res[f"{c}_true"] = case.true_vals[j]; res[f"{c}_0"] = th0[j]; res[f"{c}_end"] = v[-1, j]; res[f"{c}_err_pct"] = err[-1, j]
    nees = np.asarray(H["nees"]); res["nees_full_mean"] = float(np.nanmean(nees))
    if "nees2" in H: res["nees_RfDs_mean"] = float(np.nanmean(np.asarray(H["nees2"])))
    if filt == "ekf":
        res["extra_newton_frac"] = float(np.mean(np.asarray(H["need"]))); res["posdef_all"] = bool(np.all(H["posdef"])); res["minDiagP"] = float(np.min(H["minDiagP"]))
    # NEES marginale (R_f, D_s): 2 g.d.l.
    if filt == "ekf" and "sd_c" in H:
        pass
    # distanza dal bordo logit nei primi 60 s
    lo = np.array([LOGIT[c][0] for c in case.coords]); hi = np.array([LOGIT[c][1] for c in case.coords])
    k60 = int(60 / case.dt); frac = np.minimum(v[:k60] - lo, hi - v[:k60]) / (hi - lo)
    res["min_dist_logit_edge_first60s"] = float(frac.min()); res["touch_edge_first60s"] = bool(frac.min() < 0.01)
    if not case.twin:
        eta = np.asarray(H["eta"]); res["eta_rms_mV"] = float(1e3 * np.sqrt(np.mean((eta - case.eta_ref()) ** 2)))
    # serie salvate
    out = dict(vals=v, nees=nees)
    if "nees2" in H: out["nees2"] = np.asarray(H["nees2"])
    if "Pc" in H: out["Pc"] = np.asarray(H["Pc"])
    if "sd_vals" in H: out["sd_vals"] = np.asarray(H["sd_vals"])
    out["c"] = np.asarray(H["c"])
    for k in ["eta", "ppl", "ce_min", "innov"]:
        if k in H: out[k] = np.asarray(H[k])
    return res, out


def run_spme(case, i, p0_scale=1.0, qth_scale=1.0, name_extra=""):
    """UKF-SPMe (stesso theta, stesse P0/Q, stesso rumore) su un caso PyBaMM."""
    from .spme import SPMe, UKFSPMe
    ek = case.ek; sp = SPMe(case.m, ek); uk = UKFSPMe(sp); m = case.m
    th0 = case.offsets(i); c0 = ek.coord_of(th0); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))
    zs0 = np.asarray(sp.init(*[float(x) for x in case.r["c0_eff"]])); x0 = np.concatenate([zs0, c0])
    Nm, K = sp.Nm, sp.K
    sig_th = 0.3 * case.range * p0_scale; q_th = 1e-4 * case.range * np.sqrt(case.dt) * qth_scale
    P0d = np.concatenate([[10.0], np.full(Nm + K, 1.0), [10.0], np.full(Nm + K, 1.0), np.full(m.N, 3.0), sig_th / dc]); Qd = np.concatenate([np.zeros(sp.nzs), q_th / dc])
    y = case.Vtrue + 2e-3 * np.random.default_rng(i).normal(size=case.N)
    xt = np.tile(np.concatenate([np.zeros(sp.nzs), ek.coord_of(case.true_vals)]), (case.N, 1))
    fn = jax.jit(lambda x0, P0d, Qd, y, xt, I: uk.run(x0, P0d, Qd, (2e-3) ** 2, I, y, xt))
    t0 = time.time(); H = fn(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), jnp.asarray(y), jnp.asarray(xt), case.I); jax.block_until_ready(H["vals"]); wall = time.time() - t0
    res, out = summarize(case, "ukf_spme", i, th0, H, wall, name_extra); res["closure"] = "spme"
    return res, out
