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
    cs = cellset.load(datasets.cellset_path(a.dataset))
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


def cathode_tables(m, dts, ntab):
    """Per step length: knots s_j on [0, s_max] (s_max at y = 1) and the constants f(s) = [e (Nm), gain (Nm), rho (K),
    W (K)] with df/ds, for D_p(s) = Ds_p 10^(-drop s)."""
    P, law, Nm, K = m.P, m.dlaw, m.Nm, m.K
    smax = float(s_of_y(law, 1.0))
    sk = np.linspace(0.0, smax, ntab)
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
def filter_setup(a, m, cs, cell, P, d):
    """EKF of the golden runs: Rf free (log coordinate), LAM and Ds frozen, implicit closure; P0 = a lithium-conserving
    charge offset of sd a.p0_q [A.h] plus independent shell noise; Rf random walk; R = (a.sigma_mV)^2."""
    import types
    Nn, Np, Nm, K, nel, N, nz = m.Nn, m.Np, m.Nm, m.K, m.nel, m.N, m.nz
    c = cs["cells"][cell]
    q0 = datasets.initial_charge(a.dataset, d, cs)
    z_of = lambda q: np.asarray(m.init_state(*cellset.c0_at(cs, cell, q)))
    v = np.zeros(nz)
    v[:Nn] = -P["cmax_n"] / c["Q_n"]; v[Nn:nel] = P["cmax_p"] / c["Q_p"]
    ek = EKF(m, coords=("Rf",), fixed=dict(Rf=1.0, LAM=0.0, Ds=1.0), transform="free")
    rng_w = np.array([RANGE["Rf"][1] - RANGE["Rf"][0]])
    c0 = ek.coord_of(np.array([1.0])); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))
    sd_shell = a.p0_shell * np.concatenate([np.full(Nn, P["cmax_n"]), np.full(Np, P["cmax_p"])])
    sd_rest = np.concatenate([np.full(Nm * nel, 1.0), np.full(K * nel, 1.0), np.full(N, 3.0)])
    p0_th = a.p0_th * rng_w / dc
    P0sd = np.r_[sd_shell, sd_rest, p0_th]
    Qsd = np.r_[np.zeros(nz), a.q_th * rng_w * np.sqrt(m.dt) / dc]
    scale = np.concatenate([np.full(nel, SCALE["c"]), np.full(Nm * nel, SCALE["a"]), np.full(K * nel, SCALE["s"]),
                            np.full(N, SCALE["e"]), np.full(1, SCALE["th"])])
    P0 = np.zeros((ek.n, ek.n))
    P0[:nz, :nz] = np.diag(P0sd[:nz] ** 2) + a.p0_q ** 2 * np.outer(v, v)
    P0[nz:, nz:] = np.diag(p0_th ** 2)
    return types.SimpleNamespace(q0=q0, z_of=z_of, v=v, ek=ek, c0=c0, P0sd=P0sd, Qsd=Qsd, R=(a.sigma_mV * 1e-3) ** 2,
                                 scale=scale, Q_ref=d["meta"]["Q_ref"], z0=z_of(q0), rng_w=rng_w, P0=P0, cell=c)


def ekf_soc(fs, d, err):
    """Python EKF on the record d from the start q0 + err [A.h]: per-step SOC (cathode read-out), its sd, the Coulomb-
    counting reference, and the filter's history H."""
    c = fs.cell
    run = jax.jit(lambda x0_: fs.ek.run(x0_, jnp.asarray(fs.P0), jnp.asarray(fs.Qsd), fs.R, jnp.asarray(d["I"]),
                                        jnp.asarray(d["V"])))
    t0 = time.time()
    H = {k: np.asarray(vv) for k, vv in run(jnp.asarray(np.r_[fs.z_of(fs.q0 + err), fs.c0])).items()}
    wall = time.time() - t0
    soc = 1 - (H["ybar"] - c["y100"]) * c["Q_p"] / fs.Q_ref
    sd = H["sd_ybar"] * c["Q_p"] / fs.Q_ref
    soc_ref = 1.0 - (fs.q0 + np.cumsum(d["I"]) * d["dt"] / 3600.0) / fs.Q_ref
    return soc, sd, soc_ref, H, wall


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="exp_LP604765")
    ap.add_argument("--dims", default="10,6,10,3,1", help="Nn,Ns,Np,Nm,K")
    ap.add_argument("--dt", default="1,5,20", help="step lengths [s] the board can use")
    ap.add_argument("--ekf-dt", type=float, default=5.0, help="EKF step [s] of the golden runs (one of --dt)")
    ap.add_argument("--T", type=float, default=20.0, help="test temperature [degC]")
    ap.add_argument("--film-share", type=float, default=0.5)
    ap.add_argument("--soft-eps", type=float, default=1e-3)
    ap.add_argument("--grid", default="4:1")
    ap.add_argument("--ntab", type=int, default=129, help="knots of the cathode tables against s")
    ap.add_argument("--sigma-mV", type=float, default=15.0)
    ap.add_argument("--p0-q", type=float, default=0.5)
    ap.add_argument("--p0-shell", type=float, default=0.005)
    ap.add_argument("--p0-th", type=float, default=0.3)
    ap.add_argument("--q-th", type=float, default=1e-4)
    ap.add_argument("--ekf-err", default="0.4", help="start errors [A.h] of the golden EKF runs")
    ap.add_argument("--ol-hours", type=float, default=2.0, help="CYC20 open-loop golden length [h]")
    ap.add_argument("--state-every", type=int, default=200)
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--no-ekf", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.dims = tuple(int(v) for v in a.dims.split(","))
    dts = [float(v) for v in a.dt.split(",")]
    if a.ekf_dt not in dts:
        sys.exit("--ekf-dt must be one of --dt")
    Nn, Ns, Np, Nm, K = a.dims
    tag = "%s_T%g_%d-%d-%d_m%dk%d" % (a.dataset.replace("exp_", "").lower(), a.T, Nn, Ns, Np, Nm, K)
    out = a.out or os.path.join(HERE, "generated", tag)
    os.makedirs(os.path.join(out, "golden"), exist_ok=True)

    cs, cell, P, m = build_model(a, a.ekf_dt)
    A_ = m.A
    c = cs["cells"][cell]
    law = m.dlaw
    th1 = jnp.array([1.0, 1.0, 1.0])
    nel, N, nz = m.nel, m.N, m.nz
    print("model %s: states %d, closure unknowns %d" % (tag, nz, A_.ng))

    # ---- constants
    sk, smax, ctab = cathode_tables(m, dts, a.ntab)
    for dt, tb in zip(dts, ctab):
        print("cathode table dt %g s: max rel. Hermite error e %.1e gain %.1e rho %.1e W %.1e" % (
            dt, tb["err"][:Nm].max(), tb["err"][Nm:2 * Nm].max(), tb["err"][2 * Nm:2 * Nm + K].max(), tb["err"][2 * Nm + K:].max()))
    an = anode_consts(m, dts)
    EG = [m.elec_mats(th1, d) for d in dts]
    a_n, a_p, _ = A_.coef(th1)
    ent = P.get("entropic") or (0.0, 0.0, 0.0, 0.0)

    # ---- EKF set-up (scripts/ekf_exp.py, --start soc)
    d_cyc = datasets.load(a.dataset, "CYC20", dt=a.ekf_dt)
    fs = filter_setup(a, m, cs, cell, P, d_cyc)
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
    for nm_, v_ in (("NN", Nn), ("NS", Ns), ("NP", Np), ("NM", Nm), ("K", K), ("NEL", nel), ("N", N), ("NZ", nz),
                    ("NG", A_.ng), ("NDT", len(dts)), ("NTAB", a.ntab), ("NOCPN", len(cs["ocp"]["n"]["x"])),
                    ("NOCPP", len(cs["ocp"]["p"]["y"])), ("NTH", 1)):
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
    h.real("tab_smax", smax)
    h.arr("cp_val", np.stack([t["val"] for t in ctab]), "[NDT][NTAB][2NM+2K]")
    h.arr("cp_der", np.stack([t["der"] for t in ctab]), "d/ds")
    h.raw("// ---- window, SOC read-out (cathode mean), EKF set-up (scripts/ekf_exp.py --start soc)")
    for k_ in ("Q_n", "Q_p", "x100", "y100"):
        h.real(k_, c[k_])
    h.real("Q_ref", Q_ref)
    h.arr("w_p", np.asarray(A_.dxp) / (P["Lp"] * P["cmax_p"]), "ybar = w_p . c_avg,p")
    h.real("q0", q0, "start of CYC20 [A.h below the top]")
    h.arr("z0", z0, "state at q0 (rested)")
    h.arr("dz_dq", v, "d z / d q [per A.h]: the lithium-conserving charge direction")
    h.real("p0_q", a.p0_q, "sd of the initial charge [A.h]")
    h.arr("P0_sd", P0sd, "independent initial sd (physical units; theta in log coordinates)")
    h.arr("Q_sd", Qsd, "process noise sd per step")
    h.real("R_meas", R)
    h.arr("scale", scale, "EKF coordinate scaling")
    h.real("Rf_range", rng_w[0])
    h.raw("}  // namespace dfn_model")
    with open(os.path.join(out, "model.h"), "w", encoding="utf-8") as f:
        f.write("\n".join(h.lines) + "\n")
    print("wrote", os.path.join(out, "model.h"), "(%.0f kB)" % (os.path.getsize(os.path.join(out, "model.h")) / 1e3))
    if a.no_golden:
        return

    gdir = os.path.join(out, "golden")
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
        soc = 1.0 - (Hh["ybar"] - c["y100"]) * c["Q_p"] / Q_ref
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
