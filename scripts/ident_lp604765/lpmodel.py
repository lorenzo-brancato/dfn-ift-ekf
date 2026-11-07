"""Grey-box DFN of the LP604765 on the dfnekf package (two folders up: scripts/ident_lp604765/), with a variable time step.

    from lpmodel import Model, prepare, NOMINAL
    m = Model()                                        # grid 4:1, dims 10,6,10,12 (default)
    prof = prepare(load_test("20261011_HPPC_20°C"), start_step=2)
    V = m.simulate(NOMINAL, prof, T_C=20.0)            # model V at the end of every model step

Parameters (dict, see NOMINAL / BOUNDS): physical values at T_REF = 20 C; the geometry, porosities, particle radii and
the electrolyte material (ORegan2022 law in T at c_e0) are those of the Mohtat lm3c export, the area scaled to this
cell's capacity. Electrode window x100, y100, Q_n, Q_p (step 1) and the start of each test (dq [A.h] below the top).
Cathode D_p(y) = D_p 10^(-drop u^shape), u = (y - y100)/(y_end - y100) (as lm3c). Electrolyte: one tortuosity factor
tau divides kappa and D_e. Arrhenius: D_n, D_p, k0 (both), R_contact; entropic dT (dUdT0 + dUdT_n (x_s - 0.5)).
"""
import copy
import json
import os
import sys

import numpy as np

DFNEKF = os.environ.get("DFNEKF_ROOT", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
sys.path.insert(0, DFNEKF)
from dfnekf import runtime  # noqa: E402

runtime.single_thread()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from dfnekf import particles as pt  # noqa: E402
from dfnekf.step import DFN  # noqa: E402

F, R_GAS = 96485.33212, 8.314462618
T_REF = 293.15
HERE = os.path.dirname(os.path.abspath(__file__))
LM3C = os.path.join(DFNEKF, "data", "exp_Mohtat", "params", "mohtat_lm3c.json")
Q_CELL = 1.95                                     # [A.h] window width used to place D_p's law end and the scaling

# nominal: lm3c per-area kinetics/diffusion, step-1 window (EIS 3 h rests, lm3c OCP tables)
NOMINAL = dict(
    Ds_n=6.16e-13, Ds_p=9.65e-14, drop=1.55, shape=1.37,
    k0_n=1.0, k0_p=1.0,                           # multipliers on the lm3c exchange-current densities (per area)
    R_contact=3.2e-3, tau=1.0,
    Ea_Dn=30.8e3, Ea_Dp=65.7e3, Ea_r=36.4e3, Ea_s=10e3,
    dUdT0=0.45e-3, dUdT_n=-0.97e-3,
    x100=0.656, y100=0.405, Q_n=4.437, Q_p=3.541,
    dq=0.0,
)
# search box (log-scaled where marked)
BOUNDS = dict(
    Ds_n=(1e-15, 1e-11, "log"), Ds_p=(1e-16, 1e-12, "log"), drop=(0.0, 3.0, "lin"), shape=(0.5, 4.0, "lin"),
    k0_n=(0.01, 100.0, "log"), k0_p=(0.01, 100.0, "log"), R_contact=(1e-4, 4e-2, "log"), tau=(0.3, 10.0, "log"),
    Ea_Dn=(5e3, 90e3, "lin"), Ea_Dp=(5e3, 90e3, "lin"), Ea_r=(5e3, 90e3, "lin"), Ea_s=(0.0, 60e3, "lin"),
    dUdT0=(-1.5e-3, 1.5e-3, "lin"), dUdT_n=(-3e-3, 3e-3, "lin"),
    x100=(0.5, 0.95, "lin"), y100=(0.3, 0.5, "lin"), Q_n=(2.0, 6.0, "lin"), Q_p=(2.2, 5.0, "lin"),
    dq=(-0.15, 0.15, "lin"),
)


def _table(x, U):
    x, U = jnp.asarray(x, dtype=jnp.float64), jnp.asarray(U, dtype=jnp.float64)
    lo, hi = float(x[0]), float(x[-1])
    return lambda s: jnp.interp(jnp.clip(s, lo, hi), x, U)


class Model:
    def __init__(self, dims=(10, 6, 10, 12), grid="4:1", K=2, dts=(1.0, 5.0, 20.0)):
        self.cs = json.load(open(LM3C, encoding="utf-8"))
        self.dims, self.grid, self.K, self.dts = dims, grid, K, tuple(float(d) for d in dts)
        self.ocp_n = _table(self.cs["ocp"]["n"]["x"], self.cs["ocp"]["n"]["U"])
        self.ocp_p = _table(self.cs["ocp"]["p"]["y"], self.cs["ocp"]["p"]["U"])
        c = self.cs["cells"]["C01"]
        P0 = self.cs["P"]
        self.scale = Q_CELL / 4.91                # area ratio to the Mohtat cell (C01 C/20 capacity 4.91 A.h)
        self.area = P0["area"] * self.scale
        self.j0n_ref = c["k0_n"] * 0.5 * c["cmax_n"] * np.sqrt(P0["ce0"])     # lm3c j0 at x = 0.5, c_e0 [A/m2]
        self.j0p_ref = c["k0_p"] * 0.5 * c["cmax_p"] * np.sqrt(P0["ce0"])
        self._jit = {}

    # ---- parameters
    def params(self, p, T_C):
        P0 = self.cs["P"]
        T = float(T_C) + 273.15
        P = dict(P0)
        P["area"] = self.area
        P["cmax_n"] = p["Q_n"] * 3600 / (F * self.area * P0["Ln"] * P0["eps_sn"])
        P["cmax_p"] = p["Q_p"] * 3600 / (F * self.area * P0["Lp"] * P0["eps_sp"])
        ark = (1.0 / T - 1.0 / T_REF) / R_GAS
        P["Ds_n"] = p["Ds_n"] * np.exp(-p["Ea_Dn"] * ark)
        P["Ds_p"] = p["Ds_p"] * np.exp(-p["Ea_Dp"] * ark)
        fr = np.exp(-p["Ea_r"] * ark)
        P["k0_n"] = p["k0_n"] * self.j0n_ref / (0.5 * P["cmax_n"] * np.sqrt(P0["ce0"])) * fr
        P["k0_p"] = p["k0_p"] * self.j0p_ref / (0.5 * P["cmax_p"] * np.sqrt(P0["ce0"])) * fr
        P["R_contact"] = p["R_contact"] * np.exp(p["Ea_s"] * ark)
        e = self.cs["temperature"]["electrolyte"]
        P["kappa"] = float(np.interp(T, e["T_K"], e["kappa"])) / p["tau"]
        P["De"] = float(np.interp(T, e["T_K"], e["De"])) / p["tau"]
        P["T"] = T
        P["Rf0"] = 0.0
        P["soft_edges"] = True
        P["implicit_closure"] = True
        P["entropic"] = (T - T_REF, p["dUdT0"], p["dUdT_n"], 0.0)
        y_end = min(p["y100"] + Q_CELL / p["Q_p"], 0.999)
        P["dp_law"] = dict(drop=float(p["drop"]), shape=float(p["shape"]), y0=float(p["y100"]), y1=float(y_end))
        P["ocp_n"], P["ocp_p"] = self.ocp_n, self.ocp_p
        P["ds_on"] = "n"
        return P

    def ocv(self, p, T_C, dq):
        """Rested cell voltage [V] with dq [A.h] below the top of the window at T_C (entropic term included)."""
        dq = np.asarray(dq, dtype=float)
        x = p["x100"] - dq / p["Q_n"]
        y = p["y100"] + dq / p["Q_p"]
        U = np.asarray(self.ocp_p(jnp.asarray(y))) - np.asarray(self.ocp_n(jnp.asarray(x)))
        return U + (T_C + 273.15 - T_REF) * (p["dUdT0"] + p["dUdT_n"] * (x - 0.5))

    def dq_from_ocv(self, p, T_C, V, lo=-0.4, hi=0.6, n=4001):
        """Inverse of ocv on [lo, hi] A.h (monotone there): the start of a held-out test from its rested voltage."""
        q = np.linspace(lo, hi, n)
        v = self.ocv(p, T_C, q)
        i = int(np.argmin(np.abs(v - V)))
        return float(q[i])

    def c0(self, p, P, dq):
        q = p["dq"] if dq is None else dq
        return (p["x100"] - q / p["Q_n"]) * P["cmax_n"], (p["y100"] + q / p["Q_p"]) * P["cmax_p"]

    # ---- simulation
    NUM = ("cmax_n", "cmax_p", "Ds_n", "Ds_p", "k0_n", "k0_p", "R_contact", "kappa", "De", "T")

    def _base(self):
        if not hasattr(self, "_m0"):
            P = self.params(NOMINAL, 20.0)
            Nn, Ns, Np, Nm = self.dims
            self._m0 = DFN(P, Nn, Ns, Np, Nm, self.K, self.dts[0], self.grid)
        return self._m0

    def traced(self, p, T_C, dq=None):
        """The numeric leaves that change between calls (a pytree for the jitted run)."""
        P = self.params(p, T_C)
        c0n, c0p = self.c0(p, P, dq)
        out = {k: jnp.asarray(float(P[k])) for k in self.NUM}
        dT, d0, dn, _ = P["entropic"]
        out.update(c0n=c0n, c0p=c0p, dT=dT, dUdT0=d0, dUdT_n=dn, **{"law_" + k: v for k, v in P["dp_law"].items()})
        return {k: jnp.asarray(float(v)) for k, v in out.items()}

    def _bind(self, q):
        """A shallow copy of the base DFN whose parameter-dependent pieces are rebuilt from the traced leaves q."""
        m0 = self._base()
        m = copy.copy(m0); A_ = copy.copy(m0.A); m.A = A_
        P = dict(m0.P); P.update({k: q[k] for k in self.NUM})
        P["entropic"] = (q["dT"], q["dUdT0"], q["dUdT_n"], 0.0)
        P["dp_law"] = {k: q["law_" + k] for k in ("drop", "shape", "y0", "y1")}
        m.P = P; A_.P = P; m.dlaw = P["dp_law"]
        A_.T = q["T"]
        A_.chi = 2 * R_GAS * q["T"] * (1 - P["tplus"]) * P["tdf"] / F
        Kd = q["kappa"] * jnp.asarray(A_.tor_e / A_.dxe)
        A_.j_Kd = Kd
        A_.j_A_e = -jnp.asarray(A_.Dm) @ (Kd[:, None] * jnp.asarray(A_.Gm))
        tor_h = 1.0 / (A_.wl / A_.tor[:-1] + (1 - A_.wl) / A_.tor[1:])
        Kde = q["De"] * jnp.asarray(tor_h / A_.dxe)
        m.jMdiff = (jnp.asarray(A_.Dm) @ (Kde[:, None] * jnp.asarray(A_.Gm))) / jnp.asarray(A_.eps * A_.dx)[:, None]
        return m

    def _run(self, q, I, dt, idx, full):
        m = self._bind(q)
        P, A_ = m.P, m.A
        Nn_, Nm, K = m.Nn, m.Nm, self.K
        th = jnp.array([1.0, 1.0, 1.0])
        EG = [m.elec_mats(th, d) for d in self.dts]
        E = jnp.stack([a for a, _ in EG]); G = jnp.stack([b for _, b in EG])
        z0 = m.init_state(q["c0n"], q["c0p"])
        kf = lambda C: (3.0 * C["dt"] / C["R"] + jnp.sum(C["gain"], 0) + jnp.sum((1 - C["rho"]) * C["W"], 0)) / F

        def body(carry, xs):
            z, g = carry
            Ik, dtk, k = xs
            Cn = pt.consts(P["Rn"], P["Ds_n"], dtk, Nm, K)
            c, a, s, ce = m.split(z)
            Cp = m.cathode_consts(c[Nn_:], th, dtk)
            Ek, Gk = E[k], G[k]
            imp = (kf(Cn), kf(Cp), Ek, Gk)
            csn, csp, _ = m.surf(z)
            g, res, nit = A_.solve_robust(g, csn, csp, ce, Ik, th, tol=1e-9, imp=imp)
            V = A_.V_of(g, Ik, csn, csp)
            jn, jp = g[:Nn_], g[Nn_:Nn_ + m.Np]
            stn = pt.step((c[:Nn_], a[:, :Nn_], s[:, :Nn_]), jn, Cn)
            stp = pt.step((c[Nn_:], a[:, Nn_:], s[:, Nn_:]), jp, Cp)
            z2 = m.join(jnp.concatenate([stn[0], stp[0]]), jnp.concatenate([stn[1], stp[1]], axis=1),
                        jnp.concatenate([stn[2], stp[2]], axis=1), Ek @ ce + Gk @ jnp.concatenate([jn, jp]))
            out = (V, res, jnp.mean(csn) / P["cmax_n"], jnp.mean(csp) / P["cmax_p"]) if full else (V, res)
            return (z2, g), out

        csn0, csp0, ce0 = m.surf(z0)
        g0 = A_.newton(A_.guess(csn0, csp0, I[0], th), csn0, csp0, ce0, I[0], th)[0]
        return jax.lax.scan(body, (z0, g0), (I, dt, idx))[1]

    def simulate(self, p, prof, T_C, dq=None, full=False):
        """prof: dict(I [A, dis. > 0], dt [s], idx [int into self.dts]) per model step. Returns V [n] (full: V, ||F||,
        mean x_s, mean y_s)."""
        key = (len(prof["I"]), bool(full))
        self._base()                                   # built outside any trace (its jnp constants must be concrete)
        if key not in self._jit:
            self._jit[key] = jax.jit(lambda q, I, dt, idx: self._run(q, I, dt, idx, full))
        out = self._jit[key](self.traced(p, T_C, dq), jnp.asarray(prof["I"]), jnp.asarray(prof["dt"]),
                             jnp.asarray(prof["idx"]))
        out = [np.asarray(o) for o in out]
        return out if full else out[0]


# ---- data -> model steps
def prepare(d, start_step=0, end_step=None, fast_window=30.0, rest_mid=300.0, i_rest=0.01, dts=(1.0, 5.0, 20.0)):
    """Bins the raw test into model steps: 1 s under current and within fast_window of a current change, 5 s on an
    early rest, 20 s on rests older than rest_mid (checked against all-1-s runs: < 4 mV max on rests). I = mean current over the bin, V_meas = the
    last sample of the bin (the implicit closure reads the end-of-step state)."""
    t, V, I, st = d["t"], d["V"], d["I"], d["step"]
    keep = st >= start_step
    if end_step is not None:
        keep &= st < end_step
    t, V, I, st = t[keep], V[keep], I[keep], st[keep]
    t = t - t[0]
    # 1-s grid of means (0.1-s pulse data averaged), V at the end of each second
    edges = np.arange(0.0, t[-1] + 1.0, 1.0)
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 1)
    nb = len(edges)
    cnt = np.bincount(b, minlength=nb)
    ok = cnt > 0
    Ib = np.bincount(b, weights=I, minlength=nb)[ok] / cnt[ok]
    last = np.zeros(nb, dtype=int); last[b] = np.arange(len(t))          # last sample index per bin
    Vb = V[last[ok]]; sb = st[last[ok]]; tb = edges[ok] + 1.0
    # time since the last current change, and the type of each 1-s bin
    chg = np.r_[True, np.abs(np.diff(Ib)) > 0.02] | np.r_[True, sb[1:] != sb[:-1]]
    tc = np.zeros(len(Ib)); last_c = tb[0]
    for i in range(len(Ib)):
        if chg[i]:
            last_c = tb[i] - 1.0
        tc[i] = tb[i] - last_c
    rest = np.abs(Ib) < i_rest
    # under current always 1 s (5-s steps under load: up to 48 mV off the 1-s run near the end of discharge)
    want = np.where(~rest | (tc <= fast_window), 0, np.where(tc > rest_mid, 2, 1))
    # merge consecutive bins of class 1/2 into 5/20-s steps (break at changes)
    outI, outdt, outidx, outV, outt, outs = [], [], [], [], [], []
    i, n = 0, len(Ib)
    while i < n:
        k = want[i]
        L = int(dts[k])
        j = i + 1
        while j < n and j - i < L and want[j] == k and not chg[j]:
            j += 1
        if j - i < L:            # incomplete block: emit 1-s steps
            for q in range(i, j):
                outI.append(Ib[q]); outdt.append(1.0); outidx.append(0); outV.append(Vb[q]); outt.append(tb[q]); outs.append(sb[q])
        else:
            outI.append(Ib[i:j].mean()); outdt.append(float(L)); outidx.append(k); outV.append(Vb[j - 1]); outt.append(tb[j - 1]); outs.append(sb[j - 1])
        i = j
    return dict(I=np.array(outI), dt=np.array(outdt), idx=np.array(outidx, dtype=int), V=np.array(outV),
                t=np.array(outt), step=np.array(outs), names=d["names"], test=d["test"])
