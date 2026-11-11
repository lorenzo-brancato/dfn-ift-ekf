"""Exports the grey-box DFN + EKF of dfnekf to a C header for the embedded C++ port, plus golden files from Python.

    python embedded/export_model.py --dataset exp_LP604765 --dims 10,6,10,3,1 --dt 1,5,20
    python embedded/export_model.py --dataset exp_LP604765 --dims 10,3,10,3,1 --dt 5,20 --ekf-dt 5 --no-golden

Writes embedded/generated/<tag>/ (git-ignored):
  model.h                   every constant the C++ needs (static const real_t arrays: flash on the board)
  golden/openloop_<test>.csv  Python open loop: per step t, dt, I, V_meas, V, residual, SOC, mean stoichiometries, c_e ends
  golden/states_<test>.csv    the full state z every --state-every steps (debugging)
  golden/ekf_<test>_e<err>.csv  Python EKF (dfnekf.ekf.EKF): V prediction, innovation, SOC, sd, Rf, sd
  golden/jac.csv              dense step Jacobian (structjac.Struct.jac_full) at a few states of the EKF's model

Model: the dataset's cell set (dfnekf.cellset.params_for) at the test temperature, film share and soft-edge width as
scripts/ekf_exp.py (LP604765: --film-share 0.5, --soft-eps 1e-3, D_s multiplier on the cathode, implicit closure,
cathode D_p(y) law). Step lengths --dt: E and G (exact ZOH electrolyte maps) and the particle constants for each.
The cathode's particle constants (e, gain of the modes; rho, W of the tail bands) are tabulated against
s = (u + 1e-12)^shape, u = max(y - y0, 0) / (y1 - y0), on --ntab knots with their d/ds (cubic Hermite on the board):
log10 D_p is linear in s, so the tables are smooth. The anode's constants are scalars per step length.
Open-loop goldens: HPPC20 with the identification's mixed steps (1 s under current, 5 / 20 s on rests, lpmodel.prepare,
only when --dt is 1,5,20) and CYC20 on a uniform --ekf-dt grid. EKF golden: CYC20 at --ekf-dt from the start error
--ekf-err [A.h], filter set-up as scripts/ekf_exp.py --start soc.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
sys.path.insert(0, ROOT)
from dfnekf import runtime  # noqa: E402

runtime.single_thread()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from dfnekf import cellset, datasets  # noqa: E402
from dfnekf import particles as pt  # noqa: E402
from dfnekf.ekf import EKF, RANGE  # noqa: E402
from dfnekf.params import F, R as R_GAS  # noqa: E402
from dfnekf.step import DFN  # noqa: E402
from dfnekf.structjac import Struct  # noqa: E402

SCALE = dict(c=1000.0, a=100.0, s=100.0, e=100.0, th=0.5)       # EKF.make_step's default scaling


# ------------------------------------------------------------------ model
def build_model(a, dt):
    cs = cellset.load(datasets.cellset_path(a.dataset, getattr(a, "set", None)))
    cell = datasets.default_cell(a.dataset)
    T_K = a.T + 273.15
    P = cellset.params_for(cs, cell, T_K, film_share=a.film_share, ds_on="p")
    if a.soft_eps:
        P["soft_eps"] = a.soft_eps
    Nn, Ns, Np, Nm, K = a.dims
    m = DFN(P, Nn, Ns, Np, Nm, K, dt, a.grid)
    return cs, cell, P, m


def s_of_y(law, y):
    u = np.maximum((y - law["y0"]) / (law["y1"] - law["y0"]), 0.0)
    return (u + 1e-12) ** law["shape"]


def cathode_tables(m, dts, ntab, s_lo=0.0, s_extra=0.0):
    """Per step length: knots s_j on [s_lo, s_max + s_extra] (s_max at y = 1) and the constants f(s) = [e (Nm),
    gain (Nm), rho (K), W (K)] with df/ds, for D_p(s) = Ds_p 10^(-drop s). The range beyond [0, s_max] serves other
    temperatures: D_p(T) = Ds_p 10^(-drop (s + shift(T))), shift = -log10(Ds_p(T) / Ds_p) / drop."""
    P, law, Nm, K = m.P, m.dlaw, m.Nm, m.K
    smax = float(s_of_y(law, 1.0)) + s_extra
    sk = np.linspace(s_lo, smax, ntab)
    out = []
    for dt in dts:
        def f(s, dt=dt):
            C = pt.consts(P["Rp"], P["Ds_p"] * 10.0 ** (-law["drop"] * s), dt, Nm, K)
            return jnp.concatenate([C["e"], C["gain"], C["rho"], C["W"]])
        val = np.asarray(jax.vmap(f)(jnp.asarray(sk)))                      # ntab x nf
        der = np.asarray(jax.vmap(jax.jacfwd(f))(jnp.asarray(sk)))
        # accuracy of the cubic Hermite at the mid points
        h = sk[1] - sk[0]
        mid = sk[:-1] + 0.5 * h
        ex = np.asarray(jax.vmap(f)(jnp.asarray(mid)))
        herm = 0.5 * (val[:-1] + val[1:]) + h / 8.0 * (der[:-1] - der[1:])
        err = np.max(np.abs(herm - ex) / (np.abs(ex).max(0) + 1e-300), 0)
        out.append(dict(val=val, der=der, err=err))
    return sk, smax, out


def thermal_tables(a, cs, cell, P_ref, dts, tgrid):
    """The temperature-dependent constants at each grid temperature [degC], as cellset.params_for builds them:
    Arrhenius k0 (both), R_contact, D_s,n (anode particle constants), D_s,p (as a shift of the cathode tables), kappa
    (as a factor on the electrolyte conductances), D_e (E, G), F / 2RT and chi."""
    Nn, Ns, Np, Nm, K = a.dims
    rows = []
    for Tc in tgrid:
        P = cellset.params_for(cs, cell, Tc + 273.15, film_share=a.film_share, ds_on="p")
        if a.soft_eps:
            P["soft_eps"] = a.soft_eps
        m = DFN(P, Nn, Ns, Np, Nm, K, dts[0], a.grid)
        th1 = jnp.array([1.0, 1.0, 1.0])
        EG = [m.elec_mats(th1, d) for d in dts]
        an = anode_consts(m, dts)
        assert abs(P["Rf0"] - P_ref["Rf0"]) < 1e-15 * max(1.0, P_ref["Rf0"])
        rows.append(dict(
            k0_n=P["k0_n"], k0_p=P["k0_p"], R_contact=P["R_contact"], f_RT=F / (2 * R_GAS * m.A.T), chi=m.A.chi,
            kscale=P["kappa"] / P_ref["kappa"], sshift=-np.log10(P["Ds_p"] / P_ref["Ds_p"]) / P["dp_law"]["drop"],
            E=np.stack([np.asarray(e) for e, _ in EG]), G=np.stack([np.asarray(g) for _, g in EG]),
            an_e=np.stack([r["e"] for r in an]), an_gain=np.stack([r["gain"] for r in an]),
            an_rho=np.stack([r["rho"] for r in an]), an_W=np.stack([r["W"] for r in an])))
    return rows


def anode_consts(m, dts):
    P = m.P
    rows = []
    for dt in dts:
        C = pt.consts(P["Rn"], P["Ds_n"], dt, m.Nm, m.K)
        rows.append(dict(e=np.asarray(C["e"]), gain=np.asarray(C["gain"]), rho=np.asarray(C["rho"]),
                         W=np.asarray(C["W"])))
    return rows


# ------------------------------------------------------------------ python references
def kfac(C):
    return (3.0 * C["dt"] / C["R"] + jnp.sum(C["gain"], 0) + jnp.sum((1 - C["rho"]) * C["W"], 0)) / F


def openloop(m, z0, I, idx, dts, th, every):
    """Open loop with the implicit closure, converged Newton (solve_robust, tol 1e-10), step length dts[idx[k]]."""
    P, A_ = m.P, m.A
    Nn, Np = m.Nn, m.Np
    EG = [m.elec_mats(th, d) for d in dts]
    E = jnp.stack([e for e, _ in EG]); G = jnp.stack([g for _, g in EG])
    dtv = jnp.asarray(dts, dtype=float)
    wn = jnp.asarray(A_.dxn) / (P["Ln"] * P["cmax_n"]); wp = jnp.asarray(A_.dxp) / (P["Lp"] * P["cmax_p"])

    def body(carry, xs):
        z, g = carry
        Ik, k = xs
        dtk = dtv[k]
        c, a_, s_, ce = m.split(z)
        Cn = pt.consts(P["Rn"], P["Ds_n"], dtk, m.Nm, m.K)
        Cp = m.cathode_consts(c[Nn:], th, dtk)
        imp = (kfac(Cn), kfac(Cp), E[k], G[k])
        csn, csp, _ = m.surf(z)
        g, res, nit = A_.solve_robust(g, csn, csp, ce, Ik, th, tol=1e-10, imp=imp)
        res = jnp.linalg.norm(A_.residual(g, csn, csp, ce, Ik, th, imp), jnp.inf)
        V = A_.V_of(g, Ik, csn, csp)
        jn, jp = g[:Nn], g[Nn:Nn + Np]
        stn = pt.step((c[:Nn], a_[:, :Nn], s_[:, :Nn]), jn, Cn)
        stp = pt.step((c[Nn:], a_[:, Nn:], s_[:, Nn:]), jp, Cp)
        z2 = m.join(jnp.concatenate([stn[0], stp[0]]), jnp.concatenate([stn[1], stp[1]], axis=1),
                    jnp.concatenate([stn[2], stp[2]], axis=1), E[k] @ ce + G[k] @ jnp.concatenate([jn, jp]))
        out = dict(V=V, res=res, nit=nit, xbar=wn @ c[:Nn], ybar=wp @ c[Nn:], csn=jnp.mean(csn) / P["cmax_n"],
                   csp=jnp.mean(csp) / P["cmax_p"], ce0=ce[0], ceN=ce[-1], z=z)
        return (z2, g), out

    csn0, csp0, ce0 = m.surf(z0)
    c0 = m.split(z0)[0]
    Cn0 = pt.consts(P["Rn"], P["Ds_n"], dts[idx[0]], m.Nm, m.K)
    Cp0 = m.cathode_consts(c0[Nn:], th, dts[idx[0]])
    imp0 = (kfac(Cn0), kfac(Cp0), E[idx[0]], G[idx[0]])
    g0 = A_.newton(A_.guess(csn0, csp0, I[0], th), csn0, csp0, ce0, I[0], th, tol=1e-10, imp=imp0)[0]
    run = jax.jit(lambda z0_, g0_, I_, i_: jax.lax.scan(body, (z0_, g0_), (I_, i_))[1])
    t0 = time.time()
    H = run(jnp.asarray(z0), g0, jnp.asarray(I), jnp.asarray(idx))
    H = {k: np.asarray(v) for k, v in H.items()}
    H["wall"] = time.time() - t0
    return H


def write_csv(path, cols, fmt="%.12g"):
    names = list(cols)
    arr = np.column_stack([np.asarray(cols[k], dtype=float) for k in names])
    np.savetxt(path, arr, delimiter=",", header=",".join(names), comments="", fmt=fmt)


# ------------------------------------------------------------------ header writer
class H:
    def __init__(self):
        self.lines = []

    def raw(self, s):
        self.lines.append(s)

    def int_(self, name, v, doc=""):
        self.raw("constexpr int %s = %d;%s" % (name, int(v), ("  // " + doc) if doc else ""))

    def real(self, name, v, doc=""):
        self.raw("static const real_t %s = %s;%s" % (name, lit(v), ("  // " + doc) if doc else ""))

    def arr(self, name, a, doc=""):
        a = np.asarray(a, dtype=float)
        dims = "".join("[%d]" % d for d in a.shape)
        if doc:
            self.raw("// " + doc)
        self.raw("static const real_t %s%s = %s;" % (name, dims, nest(a)))


def lit(v):
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("non-finite constant")
    s = "%.17g" % v
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return "(real_t)" + s


def nest(a):
    if a.ndim == 1:
        out, line = [], []
        for i, v in enumerate(a):
            line.append(lit(v).replace("(real_t)", ""))
            if len(line) == 6:
                out.append(", ".join(line)); line = []
        if line:
            out.append(", ".join(line))
        return "{" + ",\n    ".join(out) + "}"
    return "{" + ",\n ".join(nest(x) for x in a) + "}"


# ------------------------------------------------------------------ filter (scripts/ekf_exp.py --start soc)
TRUE = {"Rf": 1.0, "LLI": 0.0}            # identified (beginning-of-life) values, as scripts/ekf_exp.py


def filter_setup(a, m, cs, cell, P, d):
    """EKF of the golden runs (scripts/ekf_exp.py): coordinates a.coords from Rf (log) and LLI (linear, in units of the
    anode window), LAM and Ds frozen, implicit closure, R = (a.sigma_mV)^2, parameter random walk a.q_th of the ageing
    range per sqrt(s). a.start 'soc': P0 = a lithium-conserving charge offset of sd a.p0_q [A.h] plus independent shell
    noise. 'params': diagonal P0 (shell noise, a.p0_th of the range on the coordinates), start at the identified values.
    a.li_sigma: relative sd of the lithium pseudo-measurement (None: off), referred to the beginning-of-life window."""
    import types
    Nn, Np, Nm, K, nel, N, nz = m.Nn, m.Np, m.Nm, m.K, m.nel, m.N, m.nz
    c = cs["cells"][cell]
    coords = tuple(getattr(a, "coords", ("Rf",)))
    start = getattr(a, "start", "soc")
    li_sigma = getattr(a, "li_sigma", None)
    q0 = datasets.initial_charge(a.dataset, d, cs)
    z_of = lambda q: np.asarray(m.init_state(*cellset.c0_at(cs, cell, q)))
    v = np.zeros(nz)
    v[:Nn] = -P["cmax_n"] / c["Q_n"]; v[Nn:nel] = P["cmax_p"] / c["Q_p"]
    th1 = jnp.array([1.0, 1.0, 1.0])
    li_ref = float(m.lithium(m.init_state(*cellset.c0_at(cs, cell, 0.0)), th1))
    ek = EKF(m, coords=coords, fixed=dict(Rf=1.0, LAM=0.0, Ds=1.0), transform="free", li_sigma=li_sigma, li_ref=li_ref)
    rng_w = np.array([RANGE[k][1] - RANGE[k][0] for k in coords])
    c0 = ek.coord_of(np.array([TRUE[k] for k in coords])); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))
    sd_shell = a.p0_shell * np.concatenate([np.full(Nn, P["cmax_n"]), np.full(Np, P["cmax_p"])])
    sd_rest = np.concatenate([np.full(Nm * nel, 1.0), np.full(K * nel, 1.0), np.full(N, 3.0)])
    p0_th = a.p0_th * rng_w / dc
    P0sd = np.r_[sd_shell, sd_rest, p0_th]
    Qsd = np.r_[np.zeros(nz), a.q_th * rng_w * np.sqrt(m.dt) / dc]
    scale = np.concatenate([np.full(nel, SCALE["c"]), np.full(Nm * nel, SCALE["a"]), np.full(K * nel, SCALE["s"]),
                            np.full(N, SCALE["e"]), np.full(len(coords), SCALE["th"])])
    p0_q = a.p0_q if start == "soc" else 0.0
    P0 = np.zeros((ek.n, ek.n))
    P0[:nz, :nz] = np.diag(P0sd[:nz] ** 2) + p0_q ** 2 * np.outer(v, v)
    P0[nz:, nz:] = np.diag(p0_th ** 2)
    # lithium (step.DFN.lithium at theta = 1): Li = li_w . z [mol/m2], linear in the particle means and c_e
    A_ = m.A
    li_w = np.zeros(nz)
    li_w[:Nn] = P["eps_sn"] * np.asarray(A_.dxn); li_w[Nn:nel] = P["eps_sp"] * np.asarray(A_.dxp)
    li_w[nz - N:] = np.asarray(A_.eps) * np.asarray(A_.dx)
    assert abs(li_w @ z_of(0.0) - li_ref) < 1e-9 * li_ref
    # SOC from the stoichiometry (scripts/ekf_exp.py --soc-vlim): the window between the model OCVs soc_vlim, by OCV
    # inversion at the test temperature; SOC = (y_bot - y) / (y_bot - y_top), Coulomb reference on the same window
    vmax, vmin = (float(s_) for s_ in getattr(a, "soc_vlim", "4.2,2.75").split(","))
    T_K = float(d["T_C"]) + 273.15
    q_top, q_bot = datasets.q_from_ocv(cs, cell, vmax, T_K), datasets.q_from_ocv(cs, cell, vmin, T_K)
    win = dict(vmax=vmax, vmin=vmin, q_top=q_top, q_bot=q_bot, y_top=c["y100"] + q_top / c["Q_p"],
               y_bot=c["y100"] + q_bot / c["Q_p"], x_top=c["x100"] - q_top / c["Q_n"], x_bot=c["x100"] - q_bot / c["Q_n"])
    return types.SimpleNamespace(q0=q0, z_of=z_of, v=v, ek=ek, c0=c0, P0sd=P0sd, Qsd=Qsd, R=(a.sigma_mV * 1e-3) ** 2, win=win,
                                 scale=scale, Q_ref=d["meta"]["Q_ref"], z0=z_of(q0), rng_w=rng_w, P0=P0, cell=c,
                                 coords=coords, start=start, li_sigma=li_sigma, li_ref=li_ref, li_w=li_w, p0_q=p0_q,
                                 dshift=P["dx_window_n"] * P["cmax_n"])


def ekf_soc(fs, d, err):
    """Python EKF on the record d from the start q0 + err [A.h]: per-step SOC (cathode read-out), its sd, the Coulomb-
    counting reference, and the filter's history H."""
    c = fs.cell
    run = jax.jit(lambda x0_: fs.ek.run(x0_, jnp.asarray(fs.P0), jnp.asarray(fs.Qsd), fs.R, jnp.asarray(d["I"]),
                                        jnp.asarray(d["V"])))
    t0 = time.time()
    H = {k: np.asarray(vv) for k, vv in run(jnp.asarray(np.r_[fs.z_of(fs.q0 + err), fs.c0])).items()}
    wall = time.time() - t0
    w = fs.win
    soc = (w["y_bot"] - H["ybar"]) / (w["y_bot"] - w["y_top"])           # SOC from the cathode stoichiometry
    sd = H["sd_ybar"] / (w["y_bot"] - w["y_top"])
    soc_ref = (w["q_bot"] - (fs.q0 + np.cumsum(d["I"]) * d["dt"] / 3600.0)) / (w["q_bot"] - w["q_top"])
    return soc, sd, soc_ref, H, wall


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="exp_LP604765")
    ap.add_argument("--set", default=None, help="cell set file in data/<dataset>/params (default: the dataset's current "
                    "one); its id goes into the header (SET) and the replay's '?' reply")
    ap.add_argument("--dims", default="10,6,10,3,1", help="Nn,Ns,Np,Nm,K")
    ap.add_argument("--dt", default="1,5,20", help="step lengths [s] the board can use")
    ap.add_argument("--ekf-dt", type=float, default=5.0, help="EKF step [s] of the golden runs (one of --dt)")
    ap.add_argument("--T", type=float, default=20.0, help="test temperature [degC]")
    ap.add_argument("--T-grid", default="-10:50:5", help="temperature tables lo:hi:step [degC] for a measured T on the "
                    "board ('' for none: the model then stays at --T)")
    ap.add_argument("--T-check", default="0,27.5,40", help="open-loop goldens at these temperatures [degC] (the "
                    "board's temperature tables against models built at T)")
    ap.add_argument("--film-share", type=float, default=0.5)
    ap.add_argument("--soft-eps", type=float, default=1e-3)
    ap.add_argument("--grid", default="4:1")
    ap.add_argument("--ntab", type=int, default=129, help="knots of the cathode tables against s")
    ap.add_argument("--sigma-mV", type=float, default=15.0)
    ap.add_argument("--p0-q", type=float, default=0.5)
    ap.add_argument("--p0-shell", type=float, default=0.005)
    ap.add_argument("--p0-th", type=float, default=0.3)
    ap.add_argument("--q-th", type=float, default=1e-4)
    ap.add_argument("--ekf-err", default="0.4", help="[soc] start errors [A.h] of the golden EKF runs")
    ap.add_argument("--coords", default="Rf", help="filter coordinates: Rf, or Rf,LLI (ageing)")
    ap.add_argument("--start", default="soc", choices=("soc", "params"),
                    help="soc: CYC20 from wrong charges (lithium-conserving P0); params: --test from the identified values")
    ap.add_argument("--test", default="CYC20", help="record of the EKF golden run (and of q0, z0)")
    ap.add_argument("--li-sigma", type=float, default=None, help="relative sd of the lithium pseudo-measurement")
    ap.add_argument("--soc-vlim", default="4.2,2.75", help="V_max,V_min: SOC from the stoichiometry on the window "
                    "between these model OCVs (scripts/ekf_exp.py --soc-vlim)")
    ap.add_argument("--ol-hours", type=float, default=2.0, help="CYC20 open-loop golden length [h]")
    ap.add_argument("--state-every", type=int, default=200)
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--no-ekf", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.coords = tuple(c_ for c_ in a.coords.split(",") if c_)
    if "Rf" not in a.coords or set(a.coords) - {"Rf", "LLI"}:
        sys.exit("--coords: Rf or Rf,LLI")
    a.dims = tuple(int(v) for v in a.dims.split(","))
    dts = [float(v) for v in a.dt.split(",")]
    if a.ekf_dt not in dts:
        sys.exit("--ekf-dt must be one of --dt")
    Nn, Ns, Np, Nm, K = a.dims
    tag = "%s_T%g_%d-%d-%d_m%dk%d" % (a.dataset.replace("exp_", "").lower(), a.T, Nn, Ns, Np, Nm, K)
    if a.coords != ("Rf",) or a.start != "soc" or a.test != "CYC20":
        tag += "_%s_%s" % ("".join(a.coords), a.test)
    out = a.out or os.path.join(HERE, "generated", tag)
    os.makedirs(os.path.join(out, "golden"), exist_ok=True)

    cs, cell, P, m = build_model(a, a.ekf_dt)
    A_ = m.A
    c = cs["cells"][cell]
    law = m.dlaw
    th1 = jnp.array([1.0, 1.0, 1.0])
    nel, N, nz = m.nel, m.N, m.nz
    print("model %s: cell set %s, states %d, closure unknowns %d" % (tag, cs.get("id", "?"), nz, A_.ng))

    # ---- constants
    tgrid = []
    if a.T_grid:
        lo_, hi_, st_ = (float(v) for v in a.T_grid.split(":"))
        tgrid = list(np.round(np.arange(lo_, hi_ + 0.5 * st_, st_), 6))
        if a.T not in tgrid:
            sys.exit("--T must be on --T-grid")
    thermal = thermal_tables(a, cs, cell, P, dts, tgrid) if tgrid else []
    sh = [r["sshift"] for r in thermal] or [0.0]
    s_lo = min(0.0, min(sh)) - 0.02
    s_extra = max(0.0, max(sh)) + 0.02
    sk, smax, ctab = cathode_tables(m, dts, a.ntab if not thermal else max(a.ntab, 161), s_lo, s_extra)
    for dt, tb in zip(dts, ctab):
        print("cathode table dt %g s: max rel. Hermite error e %.1e gain %.1e rho %.1e W %.1e" % (
            dt, tb["err"][:Nm].max(), tb["err"][Nm:2 * Nm].max(), tb["err"][2 * Nm:2 * Nm + K].max(), tb["err"][2 * Nm + K:].max()))
    an = anode_consts(m, dts)
    EG = [m.elec_mats(th1, d) for d in dts]
    a_n, a_p, _ = A_.coef(th1)
    ent = P.get("entropic") or (0.0, 0.0, 0.0, 0.0)

    # ---- EKF set-up (scripts/ekf_exp.py, --start soc)
    d_cyc = datasets.load(a.dataset, a.test, dt=a.ekf_dt)
    fs = filter_setup(a, m, cs, cell, P, d_cyc)
    nth = len(fs.coords)
    q0, z_of, v, ek, c0, P0sd, Qsd, R, scale, Q_ref, z0, rng_w = (
        fs.q0, fs.z_of, fs.v, fs.ek, fs.c0, fs.P0sd, fs.Qsd, fs.R, fs.scale, fs.Q_ref, fs.z0, fs.rng_w)

    # ---- header
    h = H()
    h.raw("// Generated by embedded/export_model.py -- do not edit.")
    h.raw("// %s" % " ".join(sys.argv))
    h.raw("#pragma once")
    h.raw('#include "dfn/config.h"')
    h.raw("namespace dfn_model {")
    h.raw('static const char TAG[] = "%s";' % tag)
    h.raw('#define DFN_MODEL_SET "%s"  // cell set id (%s, created %s)' % (
        cs.get("id", "?"), os.path.basename(datasets.cellset_path(a.dataset, a.set)), cs.get("created_utc", "?")))
    for nm_, v_ in (("NN", Nn), ("NS", Ns), ("NP", Np), ("NM", Nm), ("K", K), ("NEL", nel), ("N", N), ("NZ", nz),
                    ("NG", A_.ng), ("NDT", len(dts)), ("NTAB", len(sk)), ("NOCPN", len(cs["ocp"]["n"]["x"])),
                    ("NOCPP", len(cs["ocp"]["p"]["y"])), ("NTH", nth),
                    ("ITH_LLI", fs.coords.index("LLI") if "LLI" in fs.coords else -1)):
        h.int_(nm_, v_)
    h.arr("DT", dts, "step lengths [s]")
    h.int_("EKF_IDT", dts.index(a.ekf_dt), "index of the EKF step of the golden runs")
    h.raw("// ---- cell and kinetics (at the test temperature)")
    for k_ in ("cmax_n", "cmax_p", "k0_n", "k0_p", "alpha", "area", "sig_p", "R_contact", "Rf0", "Rn", "Rp", "Ln", "Lp", "T"):
        h.real(k_, P[k_])
    h.real("f_RT", F / (2 * R_GAS * A_.T), "F / (2 R T)")
    h.real("chi", A_.chi)
    h.real("a_n", a_n); h.real("a_p", a_p)
    h.real("soft_e_n", P["soft_eps"] * P["cmax_n"], "soft-edge width [mol/m3]")
    h.real("soft_e_p", P["soft_eps"] * P["cmax_p"])
    h.real("s_e", A_.s_e); h.real("s_sn", A_.s_sn); h.real("s_sp", A_.s_sp)
    h.real("ent_dT", ent[0]); h.real("ent_d0", ent[1]); h.real("ent_dn", ent[2]); h.real("ent_dp", ent[3])
    h.real("FARADAY", F)
    h.raw("// ---- grid and operators")
    h.arr("dxn", A_.dxn); h.arr("dxp", A_.dxp)
    h.arr("Kd", A_.Kd, "kappa tor_e / dx_e on the N-1 electrolyte edges")
    h.arr("wl_c", A_.wl_c, "weight of the left cell in the 1/c_e edge mean")
    # solid-phase operators in flux form: (B v)_i = sum over the edges c_k (v_i - v_neighbour) (+ the anode's Dirichlet
    # term c_D v_0), so that rounded coefficients keep the charge balance exact
    for nm_, B, dirich in (("Bn", A_.B_n, True), ("Bp", A_.B_p, False)):
        cedge = -np.diag(B, 1)
        n_ = B.shape[0]
        Bre = np.zeros_like(B)
        for k_ in range(n_ - 1):
            Bre[k_, k_] += cedge[k_]; Bre[k_ + 1, k_ + 1] += cedge[k_]; Bre[k_, k_ + 1] -= cedge[k_]; Bre[k_ + 1, k_] -= cedge[k_]
        cD = B[0, 0] - Bre[0, 0]
        Bre[0, 0] += cD
        assert np.allclose(B, Bre, rtol=1e-12, atol=0) and (dirich or abs(cD) < 1e-9 * B[0, 0])
        h.arr(nm_ + "_c", cedge, "edge conductances sigma / dx_edge")
        h.real(nm_ + "_dir", cD if dirich else 0.0, "Dirichlet term at the collector (anode)")
    h.raw("// ---- OCP tables (uniform knots, linear interpolation, clipped)")
    xn = np.asarray(cs["ocp"]["n"]["x"]); yp = np.asarray(cs["ocp"]["p"]["y"])
    h.real("ocpn_x0", xn[0]); h.real("ocpn_dx", (xn[-1] - xn[0]) / (len(xn) - 1)); h.arr("ocpn_U", cs["ocp"]["n"]["U"])
    h.real("ocpp_x0", yp[0]); h.real("ocpp_dx", (yp[-1] - yp[0]) / (len(yp) - 1)); h.arr("ocpp_U", cs["ocp"]["p"]["U"])
    h.raw("// ---- electrolyte: c_e+ = E c_e + G j_el, per step length")
    h.arr("E", np.stack([np.asarray(e) for e, _ in EG]))
    h.arr("G", np.stack([np.asarray(g) for _, g in EG]))
    h.raw("// ---- anode particle constants per step length")
    h.arr("an_e", np.stack([r["e"] for r in an])); h.arr("an_gain", np.stack([r["gain"] for r in an]))
    h.arr("an_rho", np.stack([r["rho"] for r in an])); h.arr("an_W", np.stack([r["W"] for r in an]))
    h.raw("// ---- cathode D_p(y) law and tables against s = (u + 1e-12)^shape: f = [e(NM), gain(NM), rho(K), W(K)]")
    h.real("law_y0", law["y0"]); h.real("law_y1", law["y1"]); h.real("law_shape", law["shape"]); h.real("law_drop", law["drop"])
    h.real("tab_s0", sk[0], "first knot of the cathode tables (below 0: higher temperatures)")
    h.real("tab_smax", smax)
    h.arr("cp_val", np.stack([t["val"] for t in ctab]), "[NDT][NTAB][2NM+2K]")
    if thermal:
        h.raw("// ---- temperature tables (thermal_tables): linear interpolation on TG; dfn::Thermal")
        h.raw("#define DFN_MODEL_THERMAL 1")
        h.int_("NTG", len(tgrid))
        h.arr("TG", tgrid, "grid temperatures [degC]")
        h.real("ent_Tref", cs["T_ref_K"], "reference temperature of the entropic term [K]")
        for k_ in ("k0_n", "k0_p", "R_contact", "f_RT", "chi", "kscale", "sshift"):
            h.arr("th_" + k_, [r[k_] for r in thermal])
        for k_ in ("E", "G", "an_e", "an_gain", "an_rho", "an_W"):
            h.arr("th_" + k_, np.stack([r[k_] for r in thermal]))
    h.arr("cp_der", np.stack([t["der"] for t in ctab]), "d/ds")
    h.raw("// ---- window, SOC read-out (cathode mean), EKF set-up (scripts/ekf_exp.py --start soc)")
    for k_ in ("Q_n", "Q_p", "x100", "y100"):
        h.real(k_, c[k_])
    h.real("Q_ref", Q_ref, "nominal capacity [A.h] (SOH = 1 - LLI / Q_ref; the SOC does not use it)")
    h.raw("// SOC from the stoichiometry: window between the model OCVs soc_vmax, soc_vmin at the export temperature;")
    h.raw("// SOC = (soc_y_bot - ybar) / (soc_y_bot - soc_y_top); the board recomputes the ends at a measured temperature")
    h.raw("#define DFN_MODEL_SOC_WINDOW 1")
    for k_ in ("vmax", "vmin", "y_top", "y_bot", "x_top", "x_bot", "q_top", "q_bot"):
        h.real("soc_" + k_, fs.win[k_])
    h.arr("w_p", np.asarray(A_.dxp) / (P["Lp"] * P["cmax_p"]), "ybar = w_p . c_avg,p")
    h.real("q0", q0, "start of %s [A.h below the top]" % a.test)
    h.arr("z0", z0, "state at q0 (rested)")
    h.arr("dz_dq", v, "d z / d q [per A.h]: the lithium-conserving charge direction")
    h.real("p0_q", fs.p0_q, "sd of the initial charge [A.h] (0: diagonal P0, start at the identified values)")
    h.arr("P0_sd", P0sd, "independent initial sd (physical units; theta in log coordinates)")
    h.arr("Q_sd", Qsd, "process noise sd per step")
    h.real("R_meas", R)
    h.arr("scale", scale, "EKF coordinate scaling")
    h.real("Rf_range", rng_w[0])
    h.raw("// filter coordinates: %s (Rf: log of the multiplier; LLI: linear, units of the anode window)" % ",".join(fs.coords))
    h.raw("#define DFN_MODEL_COORDS 1")
    h.arr("C0", c0, "initial coordinates")
    h.real("dshift", fs.dshift, "LLI = 1 lowers the anode means the closure sees by dx_window c_max,n [mol/m3]")
    h.arr("li_w", fs.li_w, "lithium = li_w . z [mol/m2]")
    h.real("li_ref", fs.li_ref, "lithium of the beginning-of-life window [mol/m2]")
    h.real("li_sigma", fs.li_sigma or 0.0, "relative sd of the lithium pseudo-measurement (0: off)")
    h.raw("}  // namespace dfn_model")
    with open(os.path.join(out, "model.h"), "w", encoding="utf-8") as f:
        f.write("\n".join(h.lines) + "\n")
    print("wrote", os.path.join(out, "model.h"), "(%.0f kB)" % (os.path.getsize(os.path.join(out, "model.h")) / 1e3))
    if a.no_golden:
        return

    gdir = os.path.join(out, "golden")
    if a.start == "params":                       # ageing golden only: the EKF from the identified values on --test
        soc, sd_soc, soc_ref, Hk, wall = ekf_soc(fs, d_cyc, 0.0)
        cols = dict(t=d_cyc["t"], I=d_cyc["I"], y=d_cyc["V"], V=Hk["V"], innov=Hk["innov"], Sy=Hk["Sy"], soc=soc,
                    sd_soc=sd_soc, res=Hk["resF"], li_rel=Hk["li_rel"])
        for i, c_ in enumerate(fs.coords):
            cols[c_] = Hk["vals"][:, i]; cols["sd_" + c_] = Hk["sd_vals"][:, i]
        write_csv(os.path.join(gdir, "ekf_%s.csv" % a.test), cols)
        print("EKF %s (%s): %d steps, %.2f ms/step, innov rmse %.1f mV, end %s, lithium drift %.2e" % (
            a.test, ",".join(fs.coords), len(soc), 1e3 * wall / len(soc), 1e3 * np.sqrt(np.mean(Hk["innov"] ** 2)),
            ", ".join("%s %.4f +- %.4f" % (c_, Hk["vals"][-1, i], Hk["sd_vals"][-1, i]) for i, c_ in enumerate(fs.coords)),
            Hk["li_rel"][-1]))
        return
    # ---- open loop: HPPC20 on mixed steps, CYC20 uniform
    runs = []
    if a.dataset == "exp_LP604765" and dts == [1.0, 5.0, 20.0]:
        sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
        from lpmodel import prepare
        raw = datasets.lp_cut("HPPC20")
        pr = prepare(raw, start_step=0)
        d_h = datasets.load(a.dataset, "HPPC20")
        q_h = datasets.initial_charge(a.dataset, d_h, cs)
        runs.append(("HPPC20", pr["t"], pr["I"], pr["idx"].astype(int), pr["V"], q_h))
    n_ol = int(a.ol_hours * 3600 / a.ekf_dt)
    runs.append(("CYC20", d_cyc["t"][:n_ol], d_cyc["I"][:n_ol], np.full(min(n_ol, len(d_cyc["t"])), dts.index(a.ekf_dt)),
                 d_cyc["V"][:n_ol], q0))
    for name, t, I, idx, Vm, qs in runs:
        Hh = openloop(m, z_of(qs), I, idx, dts, th1, a.state_every)
        soc = (fs.win["y_bot"] - Hh["ybar"]) / (fs.win["y_bot"] - fs.win["y_top"])
        rmse = 1e3 * np.sqrt(np.mean((Hh["V"] - Vm) ** 2))
        print("open loop %s: %d steps, RMSE vs measured %.1f mV, max ||F|| %.1e, %.2f ms/step" % (
            name, len(t), rmse, Hh["res"].max(), 1e3 * Hh["wall"] / len(t)))
        write_csv(os.path.join(gdir, "openloop_%s.csv" % name),
                  dict(t=t, idt=idx, I=I, V_meas=Vm, V=Hh["V"], res=Hh["res"], nit=Hh["nit"], soc=soc, xbar=Hh["xbar"],
                       ybar=Hh["ybar"], csn=Hh["csn"], csp=Hh["csp"], ce0=Hh["ce0"], ceN=Hh["ceN"]))
        ks = np.arange(0, len(t), a.state_every)
        np.savetxt(os.path.join(gdir, "states_%s.csv" % name), np.column_stack([ks, Hh["z"][ks]]), delimiter=",",
                   fmt="%.17g", header="k," + ",".join("z%d" % i for i in range(nz)), comments="")
        with open(os.path.join(gdir, "openloop_%s.start" % name), "w") as f:
            f.write("%.17g\n" % qs)

    # ---- temperature check: the CYC20 open loop with Python models built at other temperatures (same start)
    if thermal and a.T_check:
        import copy
        for Tc in [float(v) for v in a.T_check.split(",")]:
            aT = copy.copy(a); aT.T = Tc
            _, _, _, mT = build_model(aT, a.ekf_dt)
            HT = openloop(mT, z0, d_cyc["I"][:n_ol], np.full(n_ol, dts.index(a.ekf_dt)), dts, th1, a.state_every)
            write_csv(os.path.join(gdir, "openloop_CYC20_T%g.csv" % Tc),
                      dict(t=d_cyc["t"][:n_ol], idt=np.full(n_ol, dts.index(a.ekf_dt)), I=d_cyc["I"][:n_ol],
                           V_meas=d_cyc["V"][:n_ol], V=HT["V"], res=HT["res"],
                           soc=(fs.win["y_bot"] - HT["ybar"]) / (fs.win["y_bot"] - fs.win["y_top"])))
            with open(os.path.join(gdir, "openloop_CYC20_T%g.start" % Tc), "w") as f:
                f.write("%.17g\n" % q0)
            print("open loop CYC20 at %g degC: %d steps, max ||F|| %.1e" % (Tc, n_ol, HT["res"].max()))

    # ---- dense Jacobian at a few states of the CYC20 open loop (EKF model, dt = ekf_dt)
    S = Struct(m)
    Hc = openloop(m, z0, d_cyc["I"][:n_ol], np.full(n_ol, dts.index(a.ekf_dt)), dts, th1, a.state_every)
    picks = [k for k in (10, n_ol // 3, (2 * n_ol) // 3) if k < n_ol]
    rows = []
    for k in picks:
        z = jnp.asarray(Hc["z"][k]); Ik = float(d_cyc["I"][k])
        g = ek.init_g(z, Ik, th1)
        Jz, Jt = S.jac_full(z, Ik, th1, g)
        rows.append(np.r_[k, Ik, np.asarray(z), np.asarray(g), np.asarray(Jz).ravel(), np.asarray(Jt)[:, 0]])
    np.savetxt(os.path.join(gdir, "jac.csv"), np.array(rows), delimiter=",", fmt="%.17g",
               header="k,I,z[%d],g[%d],Jz[%dx%d],Jrf[%d]" % (nz, A_.ng, nz + 1, nz, nz + 1), comments="")
    print("wrote jac.csv at steps", picks)

    # ---- EKF golden
    if a.no_ekf:
        return
    for e in [float(s) for s in a.ekf_err.split(",")]:
        soc, sd_soc, soc_ref, Hk, wall = ekf_soc(fs, d_cyc, e)
        es = 100 * (soc - soc_ref)
        half = len(es) // 2
        print("EKF CYC20 e%+.2f: %d steps, %.2f ms/step, SOC err 2nd half rms %.2f %%, innov rmse %.1f mV, Rf end %.2f" % (
            e, len(es), 1e3 * wall / len(es), np.sqrt(np.mean(es[half:] ** 2)), 1e3 * np.sqrt(np.mean(Hk["innov"] ** 2)),
            Hk["vals"][-1, 0]))
        write_csv(os.path.join(gdir, "ekf_CYC20_e%+.2f.csv" % e),
                  dict(t=d_cyc["t"], I=d_cyc["I"], y=d_cyc["V"], V=Hk["V"], innov=Hk["innov"], Sy=Hk["Sy"], soc=soc,
                       sd_soc=sd_soc, soc_ref=soc_ref, Rf=Hk["vals"][:, 0], sd_Rf=Hk["sd_vals"][:, 0], res=Hk["resF"]))


if __name__ == "__main__":
    main()
