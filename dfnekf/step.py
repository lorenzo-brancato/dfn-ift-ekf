"""Exact parametric ROM: z+ = Phi(z, I, theta). Pure JAX function (x64).
z = [c_avg (n_el), a (Nm x n_el), s (K x n_el), c_e (N)], n_el = Nn + Np (anode then cathode).
theta = [Rf, as, Ds] (multipliers). Convention: sample k: y_k = h(z_{k-1}, I_k), z_k = Phi(z_{k-1}, I_k)."""
import numpy as np
import jax, jax.numpy as jnp
import jax.scipy.linalg as jsl
jax.config.update("jax_enable_x64", True)
from . import particles as pt
from .closure import Alg, U_n, U_p
from .grid import ratio_of
from .params import F


class DFN:
    def __init__(self, P, Nn, Ns, Np, Nm, K, dt, grid):
        self.P, self.Nn, self.Ns, self.Np, self.Nm, self.K, self.dt = P, Nn, Ns, Np, Nm, K, dt
        self.grid = grid; self.A = Alg(P, Nn, Ns, Np, "linear", "inv_mean", ratio_of(grid))
        self.N, self.nel = self.A.N, Nn + Np
        self.dlaw = P.get("dp_law")                      # cathode D_s(y), frozen per step (None: constant D_p)
        self.implicit = bool(P.get("implicit_closure", False))   # closure at the end-of-step concentrations (Alg.state_imp)
        self.ds_on = P.get("ds_on", "n")                 # electrode whose D_s the multiplier theta[2] scales ('n': the paper)
        n = self.nel
        self.sl = dict(c=slice(0, n), a=slice(n, n + Nm * n), s=slice(n + Nm * n, n + Nm * n + K * n),
                       e=slice(n + Nm * n + K * n, n + Nm * n + K * n + self.N))
        self.nz = n + Nm * n + K * n + self.N
        # electrolyte: diffusion operator (weighted harmonic mean of tor on the edges: tor*D multiplies a Gradient)
        A_ = self.A
        wl = A_.wl
        tor = A_.tor
        tor_h = 1.0 / (wl / tor[:-1] + (1 - wl) / tor[1:])
        Kd = P["De"] * tor_h / A_.dxe
        self.Mdiff = (A_.Dm @ (Kd[:, None] * A_.Gm)) / (A_.eps * A_.dx)[:, None]     # N x N
        self.jMdiff = jnp.asarray(self.Mdiff)

    # ---- indexing
    def split(self, z):
        n, Nm, K = self.nel, self.Nm, self.K
        return z[self.sl["c"]], z[self.sl["a"]].reshape(Nm, n), z[self.sl["s"]].reshape(K, n), z[self.sl["e"]]

    def join(self, c, a, s, ce):
        return jnp.concatenate([c, a.reshape(-1), s.reshape(-1), ce])

    def init_state(self, c0n, c0p):
        n = self.nel
        c = jnp.concatenate([jnp.full(self.Nn, c0n), jnp.full(self.Np, c0p)])
        return self.join(c, jnp.zeros((self.Nm, n)), jnp.zeros((self.K, n)), jnp.full(self.N, self.P["ce0"]))

    def surf(self, z):
        c, a, s, ce = self.split(z)
        cs = c + a.sum(0) + s.sum(0)
        return cs[:self.Nn], cs[self.Nn:], ce

    # ---- step constants (depend on theta)
    def part_consts(self, th, dt=None):
        dt = self.dt if dt is None else dt
        P = self.P
        dn, dp = (th[2], 1.0) if self.ds_on == "n" else (1.0, th[2])
        Cn = pt.consts(P["Rn"], P["Ds_n"] * dn, dt, self.Nm, self.K)
        Cp = pt.consts(P["Rp"], P["Ds_p"] * dp, dt, self.Nm, self.K)
        return Cn, Cp

    def dp_factor(self, cavg_p):
        """D_p(y) / D_p of the cathode law P["dp_law"] = dict(drop, shape, y0, y1): 10^(-drop u^shape),
        u = max(y - y0, 0) / (y1 - y0), y = c_avg / c_max the particle's mean stoichiometry."""
        d = self.dlaw
        u = jnp.maximum((cavg_p / self.P["cmax_p"] - d["y0"]) / (d["y1"] - d["y0"]), 0.0)
        return 10.0 ** (-d["drop"] * (u + 1e-12) ** d["shape"])

    def cathode_consts(self, cavg_p, th, dt=None):
        """Per-node cathode constants with D_p frozen over the step at the law's value at the step's initial mean stoichiometry
        (times theta[2] when the multiplier acts on the cathode)."""
        dt = self.dt if dt is None else dt
        mult = th[2] if self.ds_on == "p" else 1.0
        return pt.consts(self.P["Rp"], self.P["Ds_p"] * mult * self.dp_factor(cavg_p), dt, self.Nm, self.K)

    def imp_coeffs(self, z, th, consts=None, em=None):
        """(k_n, k_p, E, G) for the implicit closure (closure.Alg.state_imp): k = d c_surf / d j over one step (ZOH flux)
        from the modal constants, (3 dt / R + sum gain + sum (1 - rho) W) / F, k_p per node with the D_p(y) law; E, G
        the electrolyte's exact step maps. None unless P["implicit_closure"]."""
        if not self.implicit:
            return None
        Cn, Cp = self.part_consts(th) if consts is None else consts
        if self.dlaw is not None:
            Cp = self.cathode_consts(z[self.sl["c"]][self.Nn:], th)
        E, G = self.elec_mats(th) if em is None else em
        k = lambda C: (3.0 * C["dt"] / C["R"] + jnp.sum(C["gain"], 0) + jnp.sum((1 - C["rho"]) * C["W"], 0)) / F
        return k(Cn), k(Cp), E, G

    def elec_mats(self, th, dt=None):
        """E (N x N), G (N x n_el): c_e+ = E c_e + G j_el (exact ZOH, expm of the augmented block)."""
        dt = self.dt if dt is None else dt
        P, A_ = self.P, self.A
        a_n, a_p, _ = A_.coef(th)
        n, N = self.nel, self.N
        B = jnp.zeros((N, n))
        src_n = (1 - P["tplus"]) * a_n / F / P["eps_en"]
        src_p = (1 - P["tplus"]) * a_p / F / P["eps_ep"]
        B = B.at[jnp.arange(self.Nn), jnp.arange(self.Nn)].set(src_n)
        B = B.at[self.Nn + self.Ns + jnp.arange(self.Np), self.Nn + jnp.arange(self.Np)].set(src_p)
        M = jnp.zeros((N + n, N + n))
        M = M.at[:N, :N].set(self.jMdiff).at[:N, N:].set(B)
        Ph = jsl.expm(M * dt)
        return Ph[:N, :N], Ph[:N, N:]

    # ---- algebraic closure
    def closure_newton(self, z, I, th, g0=None, tol=1e-11):
        csn, csp, ce = self.surf(z)
        A_ = self.A
        if g0 is None:
            g0 = A_.guess(csn, csp, I, th)
        imp = self.imp_coeffs(z, th)
        f = lambda g: A_.residual(g, csn, csp, ce, I, th, imp)

        def solve(f_, g_):
            return A_.solve_robust(g_, csn, csp, ce, I, th, tol=tol, imp=imp)[0]
        # custom_root: implicit differentiation, dg = -J^-1 dF
        g = jax.lax.custom_root(f, g0, solve, lambda g_, y: jnp.linalg.solve(jax.jacobian(g_)(y), y))
        return g

    def closure_k(self, z, I, th, g0, k=2):
        csn, csp, ce = self.surf(z)
        g, _ = self.A.newton_fixed(g0, csn, csp, ce, I, th, k=k)
        return g

    # ---- integration
    def advance(self, z, g, th, consts=None, em=None, cp_law=None):
        """One step with the flux g[:n_el] held (ZOH). cp_law: the cathode mean concentrations the D_p(y) law is evaluated at
        (default: those of z), so that its Jacobian can be taken apart (structjac.Struct.dlaw_cols)."""
        Cn, Cp = self.part_consts(th) if consts is None else consts
        E, G = self.elec_mats(th) if em is None else em
        c, a, s, ce = self.split(z)
        Nn = self.Nn
        jn, jp = g[:Nn], g[Nn:Nn + self.Np]
        if self.dlaw is not None:
            Cp = self.cathode_consts(c[Nn:] if cp_law is None else cp_law, th)
        stn = pt.step((c[:Nn], a[:, :Nn], s[:, :Nn]), jn, Cn)
        stp = pt.step((c[Nn:], a[:, Nn:], s[:, Nn:]), jp, Cp)
        c2 = jnp.concatenate([stn[0], stp[0]]); a2 = jnp.concatenate([stn[1], stp[1]], axis=1)
        s2 = jnp.concatenate([stn[2], stp[2]], axis=1)
        ce2 = E @ ce + G @ jnp.concatenate([jn, jp])
        return self.join(c2, a2, s2, ce2)

    def phi(self, z, I, th, g0=None):
        """Step with converged Newton closure (differentiable). Returns z+, g, V."""
        g = self.closure_newton(z, I, th, g0)
        csn, csp, _ = self.surf(z)
        return self.advance(z, g, th), g, self.A.V_of(g, I, csn, csp)

    def lithium(self, z, th):
        """Total lithium [mol/m2 of cell, per unit electrode area] = sum eps_s c_avg dx + eps_e c_e dx."""
        P = self.P
        c, a, s, ce = self.split(z)
        n_n = P["eps_sn"] * th[1] * jnp.sum(c[:self.Nn] * self.A.dxn)
        n_p = P["eps_sp"] * jnp.sum(c[self.Nn:] * self.A.dxp)
        n_e = jnp.sum(self.A.eps * ce * self.A.dx)
        return n_n + n_p + n_e

    # ---- anode/separator interface quantities (linear extrapolation from the last two anode cells)
    def extrap_right(self, v):
        """Linear extrapolation to the anode/separator interface from the last two anode cell centres (= 1.5 v[-1] - 0.5 v[-2] for equal cells)."""
        w = self.A.dxn; f = w[-1] / (w[-1] + w[-2])
        return v[-1] + f * (v[-1] - v[-2])

    def interface(self, g, csn, th):
        A_ = self.A
        jn, jp, pe, sn, sp = A_.unpack(g)
        Nn = self.Nn
        phis = self.extrap_right(sn); phie = self.extrap_right(pe[:Nn]); cs = self.extrap_right(csn); j = self.extrap_right(jn)
        Rf = self.P["Rf0"] * th[0]
        eta_lit = phis - phie - self.A.Un(cs / self.P["cmax_n"])
        phi_pl = phis - phie - Rf * j
        return eta_lit, phi_pl


def interface_ref(r, i, P, Rf_mult, Nn=40):
    """Same quantities from the fine PyBaMM data (Nn = 40 anode cells)."""
    ex = lambda v: 1.5 * v[-1] - 0.5 * v[-2]
    phis = ex(r["phi_s_n"][i]); phie = ex(r["phi_e"][i][:Nn]); cs = ex(r["cs_n"][i]); j = ex(r["j_n"][i])
    Un = float(U_n(cs / P["cmax_n"]))
    return phis - phie - Un, phis - phie - P["Rf0"] * Rf_mult * j


def _g_hat(self, j, csn, csp, ce, I, th):
    """Given j (20), rebuilds g = [j, phi_e, phi_s] with the linear equations + inverse BV (no iteration); V; residual ||F||."""
    A_ = self.A; P = self.P
    Nn, Np, N, Ns = self.Nn, self.Np, self.N, self.Ns
    jn, jp = j[:Nn], j[Nn:]
    a_n, a_p, Rf = A_.coef(th)
    sn = jnp.linalg.solve(A_.j_B_n, -a_n * jn * A_.dxn)                         # anode Ohm (Dirichlet)
    tiny = 1e-6
    csn_ = jnp.clip(csn, tiny, P["cmax_n"] - tiny); csp_ = jnp.clip(csp, tiny, P["cmax_p"] - tiny)
    ce_n, ce_p = ce[:Nn], ce[Nn + Ns:]
    j0n = P["k0_n"] * (jnp.maximum(ce_n, tiny) * csn_ * (P["cmax_n"] - csn_)) ** P["alpha"]
    j0p = P["k0_p"] * (jnp.maximum(ce_p, tiny) * csp_ * (P["cmax_p"] - csp_)) ** P["alpha"]
    f2 = 2 * 8.314462618 * A_.T / 96485.33212
    eta_n = f2 * jnp.arcsinh(jn / (2 * j0n)); eta_p = f2 * jnp.arcsinh(jp / (2 * j0p))
    pe0 = sn[0] - A_.Un(csn_[0] / P["cmax_n"]) - Rf * jn[0] - eta_n[0]
    cinv_e = A_.j_Kd * 0 + (lambda wl: 1 / (wl * ce[:-1] + (1 - wl) * ce[1:]))(jnp.asarray(A_.wl_c))
    ie_c = A_.j_Kd * A_.chi * cinv_e * (A_.j_Gm @ ce)
    aj = jnp.concatenate([a_n * jn * A_.dxn, jnp.zeros(Ns), a_p * jp * A_.dxp])
    rhs = aj - A_.j_Dm @ ie_c
    pe_rest = jnp.linalg.solve(A_.j_A_e[1:, 1:], rhs[1:] - A_.j_A_e[1:, 0] * pe0)
    pe = jnp.concatenate([pe0[None], pe_rest])
    sp = pe[Nn + Ns:] + A_.Up(csp_ / P["cmax_p"]) + eta_p
    g = A_.pack(jn, jp, pe, sn, sp)
    V = A_.V_of(g, I, csn, csp)
    res = jnp.linalg.norm(A_.residual(g, csn, csp, ce, I, th), jnp.inf)
    return g, V, res


DFN.g_hat = _g_hat
