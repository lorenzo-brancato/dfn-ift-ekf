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
T_REF = 273.15 + float(os.environ.get("LP_T_REF_C", "20"))   # reference temperature of the parameters (LG M50: 25 C)
HERE = os.path.dirname(os.path.abspath(__file__))
LM3C = os.environ.get("LP_BASE_SET") or os.path.join(DFNEKF, "data", "exp_Mohtat", "params", "mohtat_lm3c.json")   # base cell set
Q_CELL = float(os.environ.get("LP_Q_CELL", "1.95"))   # [A.h] window width used to place D_p's law end and the scaling
BASE_Q = float(os.environ.get("LP_BASE_Q", "4.91"))   # [A.h] capacity of the base set's cell C01 (the area scale is Q_CELL / BASE_Q)
DQ_LO, DQ_HI = (float(v) for v in os.environ.get("LP_DQ_RANGE", "-0.4,2.4").split(","))   # [A.h] range of the OCV inversion

# nominal: lm3c per-area kinetics/diffusion, step-1 window (EIS 3 h rests, lm3c OCP tables)
NOMINAL = dict(
    Ds_n=6.16e-13, Ds_p=9.65e-14, drop=1.55, shape=1.37,
    k0_n=1.0, k0_p=1.0,                           # multipliers on the lm3c exchange-current densities (per area)
    R_contact=3.2e-3, tau=1.0,
    Ea_Dn=30.8e3, Ea_Dp=65.7e3, Ea_r=36.4e3, Ea_s=10e3,
    dUdT0=0.45e-3, dUdT_n=-0.97e-3,
    x100=0.656, y100=0.405, Q_n=4.437, Q_p=3.541,
    dq=0.0,
    hys_H=0.0, hys_g=50.0,                        # OCV hysteresis (cell level): charge-branch offset [V], rate [1/A.h]
    th_C=40.0, th_hA=0.1,                         # lumped thermal state (--thermal): heat capacity [J/K], heat transfer [W/K]
)
# search box (log-scaled where marked)
BOUNDS = dict(
    Ds_n=(1e-15, 1e-11, "log"), Ds_p=(1e-16, 1e-12, "log"), drop=(0.0, 3.0, "lin"), shape=(0.5, 4.0, "lin"),
    k0_n=(0.01, 100.0, "log"), k0_p=(0.01, 100.0, "log"), R_contact=(1e-4, 4e-2, "log"), tau=(0.3, 10.0, "log"),
    Ea_Dn=(5e3, 90e3, "lin"), Ea_Dp=(5e3, 90e3, "lin"), Ea_r=(5e3, 90e3, "lin"), Ea_s=(0.0, 60e3, "lin"),
    dUdT0=(-1.5e-3, 1.5e-3, "lin"), dUdT_n=(-3e-3, 3e-3, "lin"),
    x100=(0.5, 0.95, "lin"), y100=(0.3, 0.5, "lin"), Q_n=(2.0, 6.0, "lin"), Q_p=(2.2, 5.0, "lin"),
    dq=(-0.15, 0.15, "lin"),
    hys_H=(0.0, 0.08, "lin"), hys_g=(1.0, 1000.0, "log"),
    th_C=(5.0, 200.0, "log"), th_hA=(0.005, 2.0, "log"),
)


def _table(x, U):
    if os.environ.get("LP_OCP_SMOOTH", "0") == "1":
        return _table_pchip(x, U)
    x, U = jnp.asarray(x, dtype=jnp.float64), jnp.asarray(U, dtype=jnp.float64)
    lo, hi = float(x[0]), float(x[-1])
    return lambda s: jnp.interp(jnp.clip(s, lo, hi), x, U)


def _monotone(y, U, w=0.0025, slope_min=0.02):
    """A cathode OCP from subtraction carries the measurement noise: short segments where U rises with y. Where the
    curve is flat that noise makes dU/dc_s > 0, the rest equilibrium between particles unstable and the closure Newton
    stalls (LP_OCP_MONO=1). Moving average over w in y, then strictly decreasing (slope at least slope_min V per unit y)."""
    o = np.argsort(y); y, U = y[o], U[o]
    dy = np.median(np.diff(y)); n = max(1, int(round(w / dy)))
    k = np.ones(2 * n + 1) / (2 * n + 1)
    Us = np.convolve(np.pad(U, n, mode="edge"), k, mode="valid")
    for i in range(1, len(Us)):
        Us[i] = min(Us[i], Us[i - 1] - slope_min * (y[i] - y[i - 1]))
    return y, Us


def _table_pchip(x, U):
    """Monotone cubic Hermite (PCHIP) through the table points: C1, so the slope dU/ds has no jumps at the nodes.
    With fast kinetics the closure root sits on U(c_s) itself; a piecewise-linear table puts a slope jump every node
    and Newton can cycle between two segments (long rests near an empty anode, LP_OCP_SMOOTH=1)."""
    from scipy.interpolate import PchipInterpolator
    x, U = np.asarray(x, float), np.asarray(U, float)
    o = np.argsort(x); x, U = x[o], U[o]
    keep = np.r_[True, np.diff(x) > 0]; x, U = x[keep], U[keep]
    d = PchipInterpolator(x, U).derivative()(x)
    xj, Uj, dj = (jnp.asarray(a, dtype=jnp.float64) for a in (x, U, d))
    lo, hi = float(x[0]), float(x[-1])

    def f(s):
        s = jnp.clip(s, lo, hi)
        i = jnp.clip(jnp.searchsorted(xj, s, side="right") - 1, 0, len(x) - 2)
        h = xj[i + 1] - xj[i]; t = (s - xj[i]) / h
        h00 = (1 + 2 * t) * (1 - t) ** 2; h10 = t * (1 - t) ** 2; h01 = t ** 2 * (3 - 2 * t); h11 = t ** 2 * (t - 1)
        return h00 * Uj[i] + h10 * h * dj[i] + h01 * Uj[i + 1] + h11 * h * dj[i + 1]
    return f


class Model:
    def __init__(self, dims=(10, 6, 10, 12), grid="4:1", K=2, dts=(1.0, 5.0, 20.0), ift=False):
        """ift: closure solved by Closure.solve_ift (derivatives by the implicit function theorem, for gradient-based
        fits); the values are the same as with solve_robust."""
        self.cs = json.load(open(LM3C, encoding="utf-8"))
        self.dims, self.grid, self.K, self.dts = dims, grid, K, tuple(float(d) for d in dts)
        self.ift = ift
        # lumped thermal state (env LP_THERMAL=1, set by --thermal): the cell temperature is a state; every Arrhenius law,
        # the electrolyte conductivity, the Butler-Volmer temperature and the entropic term follow it
        self.thermal = os.environ.get("LP_THERMAL") == "1"
        # measured temperature (env LP_TMEAS=1): the cell temperature, uniform in the cell, is the measured surface
        # temperature of every step (prof["Tm"], C), an input; the same laws follow it as with the thermal state
        self.tmeas = os.environ.get("LP_TMEAS") == "1"
        if os.environ.get("LP_OCP_N"):              # anode OCP from a literature table (ocp_neg_select.py json: x, U)
            tab = json.load(open(os.environ["LP_OCP_N"], encoding="utf-8"))
            self.cs["ocp"]["n"] = dict(x=list(tab["x"]), U=list(tab["U"]), source=os.environ["LP_OCP_N"])
        self.ocp_n = _table(self.cs["ocp"]["n"]["x"], self.cs["ocp"]["n"]["U"])
        self.ocp_p = _table(self.cs["ocp"]["p"]["y"], self.cs["ocp"]["p"]["U"])
        self.ocp_p_source = "lm3c"
        # measured entropic coefficient (env LP_DUDT_TABLE, ocv_temperature.py dUdT_table.json): dOCV/dT(x) [V/K],
        # one table below T_REF and one above; replaces dUdT0 + dUdT_n (x - 0.5) in the voltage and in the OCV
        self.dudt = None
        if os.environ.get("LP_DUDT_TABLE"):
            tb = json.load(open(os.environ["LP_DUDT_TABLE"], encoding="utf-8"))
            lo_, hi_ = _table(tb["x_lo"], tb["dUdT_lo"]), _table(tb["x_hi"], tb["dUdT_hi"])
            self.dudt = lambda x, dT: jnp.where(dT < 0, lo_(x), hi_(x))
        if os.environ.get("LP_OCP_P"):              # cathode OCP of this cell (ocp_sub.py), e.g. by subtraction
            tab = json.load(open(os.environ["LP_OCP_P"], encoding="utf-8"))
            yy, UU = np.asarray(tab["y"], float), np.asarray(tab["U"], float)
            if os.environ.get("LP_OCP_MONO", "0") == "1":
                yy, UU = _monotone(yy, UU, float(os.environ.get("LP_OCP_MONO_W", "0.0025")))
            self.ocp_p = _table(yy, UU); self.ocp_p_source = os.environ["LP_OCP_P"]
        c = self.cs["cells"]["C01"]
        P0 = self.cs["P"]
        self.scale = Q_CELL / BASE_Q              # area ratio to the base cell (Mohtat C01: C/20 capacity 4.91 A.h)
        self.area = P0["area"] * self.scale
        self.j0n_ref = c["k0_n"] * 0.5 * c["cmax_n"] * np.sqrt(P0["ce0"])     # lm3c j0 at x = 0.5, c_e0 [A/m2]
        self.j0p_ref = c["k0_p"] * 0.5 * c["cmax_p"] * np.sqrt(P0["ce0"])
        # parameter groups (the model reaches the voltage only through them; ApEn draft, Section 2.3), at T_REF:
        #   theta_k = R_k^2 / D_s,k [s]; I0_k = A (3 eps_s,k / R_k) L_k j0_k(half stoichiometry, c_e0) [A];
        #   R_e = (1/A)(L_n / 3 kappa_n,eff + L_s / kappa_s,eff + L_p / 3 kappa_p,eff) [Ohm], kappa_eff = eps^b kappa / tau_k;
        #   tau_e = L_e^2 / D_e,eff in the separator [s], D_e,eff = eps_s^b D_e / tau_D. Stack shape (L, eps, b), radii,
        #   t+, chi, c_e0 fixed (Mohtat lm3c). The base values are those of the multipliers = 1.
        e = self.cs["temperature"]["electrolyte"]
        kap = float(np.interp(T_REF, e["T_K"], e["kappa"])); De = float(np.interp(T_REF, e["T_K"], e["De"]))
        b = P0["brug"]
        self.G0 = dict(
            I0_n=self.area * 3 * P0["eps_sn"] / P0["Rn"] * P0["Ln"] * self.j0n_ref,
            I0_p=self.area * 3 * P0["eps_sp"] / P0["Rp"] * P0["Lp"] * self.j0p_ref,
            R_e=(P0["Ln"] / (3 * kap * P0["eps_en"] ** b) + P0["Ls"] / (kap * P0["eps_es"] ** b)
                 + P0["Lp"] / (3 * kap * P0["eps_ep"] ** b)) / self.area,
            tau_e=(P0["Ln"] + P0["Ls"] + P0["Lp"]) ** 2 / (De * P0["eps_es"] ** b))
        self._jit = {}

    GROUPS = ("theta_n", "theta_p", "I0_n", "I0_p", "R_e", "tau_e", "R_c")

    def phys(self, p):
        """p with the groups (GROUPS, where present) converted to the model's physical keys: Ds_n, Ds_p, k0_n, k0_p
        (multipliers), tau_k, tau_D (factors on 1/kappa and 1/D_e), R_contact. Floats or jnp scalars."""
        if not any(k in p for k in self.GROUPS):
            return p
        P0 = self.cs["P"]
        q = dict(p)
        if "theta_n" in p: q["Ds_n"] = P0["Rn"] ** 2 / p["theta_n"]
        if "theta_p" in p: q["Ds_p"] = P0["Rp"] ** 2 / p["theta_p"]
        if "I0_n" in p: q["k0_n"] = p["I0_n"] / self.G0["I0_n"]
        if "I0_p" in p: q["k0_p"] = p["I0_p"] / self.G0["I0_p"]
        if "R_e" in p: q["tau_k"] = p["R_e"] / self.G0["R_e"]
        if "tau_e" in p: q["tau_D"] = p["tau_e"] / self.G0["tau_e"]
        if "R_c" in p: q["R_contact"] = p["R_c"]
        return q

    def groups(self, p):
        """The groups of a physical set p (inverse of phys, floats)."""
        P0 = self.cs["P"]
        tk, tD = p.get("tau_k", p.get("tau", 1.0)), p.get("tau_D", p.get("tau", 1.0))
        return dict(theta_n=P0["Rn"] ** 2 / p["Ds_n"], theta_p=P0["Rp"] ** 2 / p["Ds_p"], I0_n=p["k0_n"] * self.G0["I0_n"],
                    I0_p=p["k0_p"] * self.G0["I0_p"], R_e=tk * self.G0["R_e"], tau_e=tD * self.G0["tau_e"], R_c=p["R_contact"])

    # ---- parameters
    def params(self, p, T_C):
        p = self.phys(p)
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
        P["kappa"] = float(np.interp(T, e["T_K"], e["kappa"])) / p.get("tau_k", p.get("tau", 1.0))
        P["De"] = float(np.interp(T, e["T_K"], e["De"])) / p.get("tau_D", p.get("tau", 1.0))
        P["T"] = T
        P["Rf0"] = 0.0
        P["soft_edges"] = True
        # width of the soft c_s edges [fraction of c_max]: 1e-2 (closure.SAT_EPS 1e-3 is too sharp for a CV charge at 1C,
        # where the anode surface at the separator saturates and Newton cannot move the reaction off it; only x_s > 0.97
        # or < 0.03 are affected). LP_SOFT_EPS overrides (1e-3: the value of the fits up to v12 / m50_v4).
        P["soft_eps"] = float(os.environ.get("LP_SOFT_EPS", "1e-2"))
        P["implicit_closure"] = True
        P["entropic"] = (T - T_REF, p["dUdT0"], p["dUdT_n"], 0.0)
        P["entropic_fn"] = self.dudt
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
        dT = T_C + 273.15 - T_REF
        if self.dudt is not None:
            return U + dT * np.asarray(self.dudt(jnp.asarray(x), dT))
        return U + dT * (p["dUdT0"] + p["dUdT_n"] * (x - 0.5))

    def dq_from_ocv(self, p, T_C, V, lo=None, hi=None, n=None):     # 0.25 mAh grid, down to the bottom of the window
        """Inverse of ocv on [lo, hi] A.h (monotone there): the start of a held-out test from its rested voltage."""
        lo = DQ_LO if lo is None else lo; hi = DQ_HI if hi is None else hi
        n = int(round((hi - lo) / 2.5e-4)) + 1 if n is None else n
        q = np.linspace(lo, hi, n)
        v = self.ocv(p, T_C, q)
        V = V - p.get("hys_H", 0.0)      # every test starts rested after a charge: on the charge branch (OCV + H)
        i = int(np.argmin(np.abs(v - V)))
        # linear refinement between the bracketing grid points: dq is then continuous in p (a grid snap makes it a
        # staircase, which corrupts finite-difference Jacobians of the window groups)
        j = i if (i + 1 < n and (v[i] - V) * (v[i + 1] - V) <= 0) else max(i - 1, 0)
        den = v[j + 1] - v[j]
        return float(q[j] + (V - v[j]) * (q[j + 1] - q[j]) / den) if abs(den) > 1e-15 else float(q[i])

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
        out.update(c0n=c0n, c0p=c0p, dT=dT, dUdT0=d0, dUdT_n=dn, **{"law_" + k: v for k, v in P["dp_law"].items()},
                   hys_H=p.get("hys_H", 0.0), hys_g=p.get("hys_g", NOMINAL["hys_g"]),
                   **{k: p.get(k, NOMINAL[k]) for k in ("Ea_Dn", "Ea_Dp", "Ea_r", "Ea_s", "th_C", "th_hA")})
        return {k: jnp.asarray(float(v)) for k, v in out.items()}

    # ---- differentiable twins (exact-gradient fits): p holds jnp scalars, T_C is a constant of the test
    def traced_jax(self, p, T_C, dq):
        """traced() in jnp: the leaves as differentiable functions of the fit groups p (dict of jnp scalars) and dq."""
        p = self.phys(p)
        P0 = self.cs["P"]
        T = float(T_C) + 273.15
        ark = (1.0 / T - 1.0 / T_REF) / R_GAS
        cmax_n = p["Q_n"] * 3600 / (F * self.area * P0["Ln"] * P0["eps_sn"])
        cmax_p = p["Q_p"] * 3600 / (F * self.area * P0["Lp"] * P0["eps_sp"])
        fr = jnp.exp(-p["Ea_r"] * ark)
        e = self.cs["temperature"]["electrolyte"]
        out = dict(
            cmax_n=cmax_n, cmax_p=cmax_p,
            Ds_n=p["Ds_n"] * jnp.exp(-p["Ea_Dn"] * ark), Ds_p=p["Ds_p"] * jnp.exp(-p["Ea_Dp"] * ark),
            k0_n=p["k0_n"] * self.j0n_ref / (0.5 * cmax_n * np.sqrt(P0["ce0"])) * fr,
            k0_p=p["k0_p"] * self.j0p_ref / (0.5 * cmax_p * np.sqrt(P0["ce0"])) * fr,
            R_contact=p["R_contact"] * jnp.exp(p["Ea_s"] * ark),
            kappa=float(np.interp(T, e["T_K"], e["kappa"])) / p.get("tau_k", p.get("tau", 1.0)),
            De=float(np.interp(T, e["T_K"], e["De"])) / p.get("tau_D", p.get("tau", 1.0)),
            T=jnp.asarray(T), dT=jnp.asarray(T - T_REF), dUdT0=p["dUdT0"], dUdT_n=p["dUdT_n"],
            c0n=(p["x100"] - dq / p["Q_n"]) * cmax_n, c0p=(p["y100"] + dq / p["Q_p"]) * cmax_p,
            law_drop=p["drop"], law_shape=p["shape"], law_y0=p["y100"],
            law_y1=jnp.minimum(p["y100"] + Q_CELL / p["Q_p"], 0.999),
            hys_H=p.get("hys_H", 0.0), hys_g=p.get("hys_g", NOMINAL["hys_g"]),
            **{k: p.get(k, NOMINAL[k]) for k in ("Ea_Dn", "Ea_Dp", "Ea_r", "Ea_s", "th_C", "th_hA")})
        return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in out.items()}

    def ocv_jax(self, p, T_C, dq):
        x = p["x100"] - dq / p["Q_n"]; y = p["y100"] + dq / p["Q_p"]
        dT = T_C + 273.15 - T_REF
        if self.dudt is not None:
            return self.ocp_p(y) - self.ocp_n(x) + dT * self.dudt(x, dT)
        return self.ocp_p(y) - self.ocp_n(x) + dT * (p["dUdT0"] + p["dUdT_n"] * (x - 0.5))

    def dq_from_ocv_jax(self, p, T_C, V):
        """dq_from_ocv with IFT derivatives: the grid inversion (no gradient), then one Newton correction on
        ocv(p, dq) = V through which d dq / d p = -(d ocv / d p) / (d ocv / d dq) flows (slope held constant)."""
        sg = jax.lax.stop_gradient
        ps = {k: sg(v) for k, v in p.items()}
        Vd = V - p.get("hys_H", 0.0)                  # rested after a charge: charge branch (see dq_from_ocv)
        dq0 = self._dq_grid_jax(ps, T_C, V - ps.get("hys_H", 0.0))
        slope = sg(jax.grad(lambda d: self.ocv_jax(ps, T_C, d))(dq0))
        return dq0 - (self.ocv_jax(p, T_C, dq0) - Vd) / slope

    def _dq_grid_jax(self, p, T_C, V, lo=None, hi=None, n=None):
        lo = DQ_LO if lo is None else lo; hi = DQ_HI if hi is None else hi
        n = int(round((hi - lo) / 2.5e-4)) + 1 if n is None else n
        q = jnp.linspace(lo, hi, n)
        v = self.ocv_jax(p, T_C, q)
        return jax.lax.stop_gradient(q[jnp.argmin(jnp.abs(v - V))])

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

    def _run(self, q, I, dt, idx, full, Tm=None):
        m = self._bind(q)
        P, A_ = m.P, m.A
        Nn_, Nm, K = m.Nn, m.Nm, self.K
        th = jnp.array([1.0, 1.0, 1.0])
        EG = [m.elec_mats(th, d) for d in self.dts]
        E = jnp.stack([a for a, _ in EG]); G = jnp.stack([b for _, b in EG])
        z0 = m.init_state(q["c0n"], q["c0p"])
        kf = lambda C: (3.0 * C["dt"] / C["R"] + jnp.sum(C["gain"], 0) + jnp.sum((1 - C["rho"]) * C["W"], 0)) / F

        thermal = self.thermal
        tmeas = self.tmeas and Tm is not None
        T0 = q["T"]
        if thermal or tmeas:
            e_ = self.cs["temperature"]["electrolyte"]
            kT, kV = jnp.asarray(e_["T_K"]), jnp.asarray(e_["kappa"])
            kap0 = jnp.interp(T0, kT, kV)

        def body(carry, xs):
            z, g, h, Tc = carry
            Ik, dtk, k, Tmk = xs
            if tmeas:
                Tc = Tmk + 273.15                      # measured surface temperature, uniform in the cell
            c, a, s, ce = m.split(z)
            mm, AA = m, A_
            Ds_n = P["Ds_n"]
            if thermal or tmeas:                       # parameters at the cell temperature Tc (computed at T0 = chamber)
                ark = (1.0 / Tc - 1.0 / T0) / R_GAS
                fr = jnp.exp(-q["Ea_r"] * ark); rk = jnp.interp(Tc, kT, kV) / kap0
                Ds_n = P["Ds_n"] * jnp.exp(-q["Ea_Dn"] * ark)
                mm = copy.copy(m); mm.P = dict(P, Ds_p=P["Ds_p"] * jnp.exp(-q["Ea_Dp"] * ark))
                AA = copy.copy(A_)
                AA.P = dict(P, k0_n=P["k0_n"] * fr, k0_p=P["k0_p"] * fr, R_contact=P["R_contact"] * jnp.exp(q["Ea_s"] * ark),
                            entropic=(Tc - T_REF, q["dUdT0"], q["dUdT_n"], 0.0))
                AA.T = Tc; AA.chi = A_.chi * Tc / T0; AA.j_Kd = A_.j_Kd * rk; AA.j_A_e = A_.j_A_e * rk
            Cn = pt.consts(P["Rn"], Ds_n, dtk, Nm, K)
            Cp = mm.cathode_consts(c[Nn_:], th, dtk)
            Ek, Gk = E[k], G[k]
            imp = (kf(Cn), kf(Cp), Ek, Gk)
            csn, csp, _ = m.surf(z)
            g, res, nit = (AA.solve_ift if self.ift else AA.solve_robust)(g, csn, csp, ce, Ik, th, tol=1e-9, imp=imp)
            # OCV hysteresis: h relaxes to H on charge and to 0 on discharge, exponentially in the charge passed; held at rest
            dec = jnp.exp(-q["hys_g"] * jnp.abs(Ik) * dtk / 3600.0)
            h = dec * h + (1.0 - dec) * jnp.where(Ik < 0, q["hys_H"], 0.0)
            V = AA.V_of(g, Ik, csn, csp) + h
            if thermal:                                # heat: I (U - V) - I T dU/dT; exact step of C dT/dt = Q - hA (T - T0)
                xb = jnp.mean(c[:Nn_]) / P["cmax_n"]; yb = jnp.mean(c[Nn_:]) / P["cmax_p"]
                dUdT = self.dudt(xb, Tc - T_REF) if self.dudt is not None else q["dUdT0"] + q["dUdT_n"] * (xb - 0.5)
                U = self.ocp_p(yb) - self.ocp_n(xb) + (Tc - T_REF) * dUdT
                Qh = Ik * (U - V) - Ik * Tc * dUdT
                ed = jnp.exp(-dtk * q["th_hA"] / q["th_C"])
                Tc = T0 + (Tc - T0) * ed + Qh / q["th_hA"] * (1.0 - ed)
            jn, jp = g[:Nn_], g[Nn_:Nn_ + m.Np]
            stn = pt.step((c[:Nn_], a[:, :Nn_], s[:, :Nn_]), jn, Cn)
            stp = pt.step((c[Nn_:], a[:, Nn_:], s[:, Nn_:]), jp, Cp)
            z2 = m.join(jnp.concatenate([stn[0], stp[0]]), jnp.concatenate([stn[1], stp[1]], axis=1),
                        jnp.concatenate([stn[2], stp[2]], axis=1), Ek @ ce + Gk @ jnp.concatenate([jn, jp]))
            out = (V, res, jnp.mean(csn) / P["cmax_n"], jnp.mean(csp) / P["cmax_p"]) if full else (V, res)
            return (z2, g, h, Tc), out

        csn0, csp0, ce0 = m.surf(z0)
        if self.ift:                                   # only a warm start: its value matters, not its derivative
            sg = jax.lax.stop_gradient; As = A_._stopped()
            a0 = [sg(v) for v in (csn0, csp0, ce0, I[0], th)]
            g0 = sg(As.newton(As.guess(a0[0], a0[1], a0[3], a0[4]), *a0)[0])
        else:
            g0 = A_.newton(A_.guess(csn0, csp0, I[0], th), csn0, csp0, ce0, I[0], th)[0]
        Tm = jnp.zeros_like(I) if Tm is None else Tm
        return jax.lax.scan(body, (z0, g0, jnp.asarray(q["hys_H"], dtype=jnp.float64), jnp.asarray(T0, dtype=jnp.float64)), (I, dt, idx, Tm))[1]   # h0 = H: rested after a charge

    def simulate(self, p, prof, T_C, dq=None, full=False):
        """prof: dict(I [A, dis. > 0], dt [s], idx [int into self.dts]) per model step. Returns V [n] (full: V, ||F||,
        mean x_s, mean y_s)."""
        key = (len(prof["I"]), bool(full))
        self._base()                                   # built outside any trace (its jnp constants must be concrete)
        if key not in self._jit:
            self._jit[key] = jax.jit(lambda q, I, dt, idx, Tm: self._run(q, I, dt, idx, full, Tm))
        Tm = jnp.asarray(prof["Tm"]) if (self.tmeas and "Tm" in prof) else None
        out = self._jit[key](self.traced(p, T_C, dq), jnp.asarray(prof["I"]), jnp.asarray(prof["dt"]),
                             jnp.asarray(prof["idx"]), Tm)
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
    Tm_ = d.get("T_meas")
    Tb = np.bincount(b, weights=np.asarray(Tm_)[keep], minlength=nb)[ok] / cnt[ok] if Tm_ is not None else np.full(len(Ib), np.nan)
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
    outI, outdt, outidx, outV, outt, outs, outT = [], [], [], [], [], [], []
    i, n = 0, len(Ib)
    while i < n:
        k = want[i]
        L = int(dts[k])
        j = i + 1
        while j < n and j - i < L and want[j] == k and not chg[j]:
            j += 1
        if j - i < L:            # incomplete block: emit 1-s steps
            for q in range(i, j):
                outI.append(Ib[q]); outdt.append(1.0); outidx.append(0); outV.append(Vb[q]); outt.append(tb[q]); outs.append(sb[q]); outT.append(Tb[q])
        else:
            outI.append(Ib[i:j].mean()); outdt.append(float(L)); outidx.append(k); outV.append(Vb[j - 1]); outt.append(tb[j - 1]); outs.append(sb[j - 1]); outT.append(Tb[i:j].mean())
        i = j
    return dict(I=np.array(outI), dt=np.array(outdt), idx=np.array(outidx, dtype=int), V=np.array(outV),
                t=np.array(outt), step=np.array(outs), names=d["names"], test=d["test"], Tm=np.array(outT))
