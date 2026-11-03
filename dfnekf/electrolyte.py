"""Fase 8: trasporto variabile D_e(c_e), kappa(c_e) (Nyman2008, Chen2020; t+ e fattore termodinamico costanti). Linearizzazione per passo: l'operatore di diffusione e la conducibilita'
sono valutati sullo stato c_e a inizio passo (E, G = expm aumentato per passo); la Jacobiana strutturata include la dipendenza di E(c), G(c) dallo stato (blocco E_eff = d(E c + G j)/dc)."""
import numpy as np
import jax, jax.numpy as jnp
import jax.scipy.linalg as jsl
jax.config.update("jax_enable_x64", True)
from .closure import Alg
from .step import DFN
from .structjac import Struct
from . import params as prm
from .params import F


def kappa_fn(c):
    x = c / 1000.0
    return 0.1297 * x ** 3 - 2.51 * x ** 1.5 + 3.329 * x


def De_fn(c):
    x = c / 1000.0
    return 8.794e-11 * x ** 2 - 3.972e-10 * x + 4.862e-10


class AlgVT(Alg):
    def kd_edges(self, ce):
        """kappa(c) tor interpolato linearmente sugli spigoli (pesi di cella), diviso dxe"""
        kt = kappa_fn(jnp.maximum(ce, 1.0)) * jnp.asarray(self.tor)
        wl = jnp.asarray(self.wl)
        return (wl * kt[:-1] + (1 - wl) * kt[1:]) / jnp.asarray(self.dxe)

    def A_e_of(self, ce):
        Kd = self.kd_edges(ce)
        return -self.j_Dm @ (Kd[:, None] * self.j_Gm)

    def residual(self, g, csn, csp, ce, I, th):
        P = self.P
        i_cc = I / P["area"]
        jn, jp, pe, sn, sp = self.unpack(g)
        j0n, j0p, eta_n, eta_p, f, a_n, a_p, Rf = self.bv_parts(g, csn, csp, ce, th)
        r_bvn = jn - 2 * j0n * jnp.sinh(f * eta_n)
        r_bvp = jp - 2 * j0p * jnp.sinh(f * eta_p)
        wl = jnp.asarray(self.wl_c)                                   # cell-width weights on non-uniform grids (M3); = wl on uniform ones
        cinv_e = 1 / (wl * ce[:-1] + (1 - wl) * ce[1:])
        Kd = self.kd_edges(ce)
        ie_c = Kd * self.chi * cinv_e * (self.j_Gm @ ce)
        aj = jnp.concatenate([a_n * jn * self.dxn, jnp.zeros(self.Ns), a_p * jp * self.dxp])
        r_e = self.A_e_of(ce) @ pe + self.j_Dm @ ie_c - aj
        r_sn = self.j_B_n @ sn + a_n * jn * self.dxn
        r_sp = (self.j_B_p @ sp + a_p * jp * self.dxp).at[-1].add(i_cc)
        return jnp.concatenate([r_bvn, r_bvp, r_e * self.s_e, r_sn * self.s_sn, r_sp * self.s_sp])

    def jac(self, g, csn, csp, ce, I, th):
        J = Alg.jac(self, g, csn, csp, ce, I, th)
        N, Nel = self.N, self.Nel
        dA = (self.A_e_of(ce) - self.j_A_e) * self.s_e
        return J.at[Nel:Nel + N, Nel:Nel + N].add(dA)


class DFNVT(DFN):
    def __init__(self, P, Nn=20, Ns=10, Np=20, Nm=12, K=2, dt=0.5):
        super().__init__(P, Nn, Ns, Np, Nm, K, dt)
        self.A = AlgVT(P, Nn, Ns, Np, "linear", "inv_mean")
        A_ = self.A
        self.eps_dx = jnp.asarray(A_.eps * A_.dx)
        n = self.nel; N = self.N
        B = np.zeros((N, n))
        src_n = (1 - P["tplus"]) * 3 * P["eps_sn"] / P["Rn"] / F / P["eps_en"]
        src_p = (1 - P["tplus"]) * 3 * P["eps_sp"] / P["Rp"] / F / P["eps_ep"]
        B[np.arange(Nn), np.arange(Nn)] = src_n
        B[Nn + Ns + np.arange(Np), Nn + np.arange(Np)] = src_p
        self.B0 = jnp.asarray(B)

    def M_of(self, ce):
        A_ = self.A
        wl = jnp.asarray(A_.wl_c)                                     # harmonic mean of D tor with cell-width weights (M3: one-step c_e error 0.66 vs 4.7 mol/m3 with distance weights on the 4:1 grid)
        tor = jnp.asarray(A_.tor)
        dt_ = De_fn(jnp.maximum(ce, 1.0)) * tor
        harm = 1.0 / (wl / dt_[:-1] + (1 - wl) / dt_[1:])
        Kd = harm / jnp.asarray(A_.dxe)
        return (A_.j_Dm @ (Kd[:, None] * A_.j_Gm)) / self.eps_dx[:, None]

    def elec_mats_vt(self, ce, th, dt=None):
        dt = self.dt if dt is None else dt
        N, n = self.N, self.nel
        B = self.B0.at[:, :self.Nn].multiply(th[1])
        M = jnp.zeros((N + n, N + n)).at[:N, :N].set(self.M_of(ce)).at[:N, N:].set(B)
        Ph = jsl.expm(M * dt)
        return Ph[:N, :N], Ph[:N, N:]

    def advance_vt(self, z, g, th, consts=None):
        import particles as pt
        Cn, Cp = self.part_consts(th) if consts is None else consts
        c, a, s, ce = self.split(z)
        Nn = self.Nn
        E, G = self.elec_mats_vt(ce, th)
        jn, jp = g[:Nn], g[Nn:Nn + self.Np]
        stn = pt.step((c[:Nn], a[:, :Nn], s[:, :Nn]), jn, Cn)
        stp = pt.step((c[Nn:], a[:, Nn:], s[:, Nn:]), jp, Cp)
        c2 = jnp.concatenate([stn[0], stp[0]])
        a2 = jnp.concatenate([stn[1], stp[1]], 1)
        s2 = jnp.concatenate([stn[2], stp[2]], 1)
        return self.join(c2, a2, s2, E @ ce + G @ jnp.concatenate([jn, jp]))


def simulate_vt(m: DFNVT, z0, Iseq, th):
    A_ = m.A
    consts = m.part_consts(th)
    Iseq = jnp.asarray(Iseq)
    csn, csp, ce = m.surf(z0)
    g0 = A_.newton(A_.guess(csn, csp, Iseq[0], th), csn, csp, ce, Iseq[0], th)[0]

    def body(carry, I):
        z, g = carry
        csn, csp, ce = m.surf(z)
        g, res, _ = A_.solve_robust(g, csn, csp, ce, I, th, tol=1e-10)
        V = A_.V_of(g, I)
        eta, ppl = m.interface(g, csn, th)
        return (m.advance_vt(z, g, th, consts), g), dict(V=V, eta_lit=eta, phi_pl=ppl, ce=ce, resF=res)
    (zT, gT), out = jax.lax.scan(body, (z0, g0), Iseq)
    out["zT"] = zT
    return out


class StructVT(Struct):
    """Jacobiano strutturato con trasporto variabile: il blocco E diventa E_eff = d(E(c) c + G(c) j)/dc (j fisso), B su c_e = G(c)."""
    def lam_B_vt(self, th, z, g):
        m = self.m
        n = self.n
        dvec, E, Bm = Struct.lam_B(self, th)
        c, a, s, ce = m.split(z)
        j = g[:n]

        def f(cv):
            Ev, Gv = m.elec_mats_vt(cv, th)
            return Ev @ cv + Gv @ j
        E_eff = jax.jacfwd(f)(ce)
        Ev, Gv = m.elec_mats_vt(ce, th)
        return dvec, E_eff, Bm.at[self.ie, :].set(Gv)

    def jac_full_vt(self, z, I, th, g):
        m = self.m
        n = self.n
        dvec, E, Bm = self.lam_B_vt(th, z, g)
        lu = self.lu(g, z, I, th)
        Sw, Sth, _ = self.sens(lu, g, z, I, th)
        C = self.C_of(Sw)
        Azz = self.lam_dense(dvec, E) + Bm @ C[:n]
        dir_th = jax.jacfwd(lambda t: m.advance_vt(z, g, t))(th)
        Jt = jnp.concatenate([Bm @ Sth[:n] + dir_th, Sth[n:n + 1]], 0)
        Jz = jnp.concatenate([Azz, C[n:n + 1]], 0)
        return Jz, Jt
