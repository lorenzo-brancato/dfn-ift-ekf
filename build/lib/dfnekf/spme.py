"""Baseline SPMe 'nostro' (BRIEF v4 §5): stesse particelle modali, stesso elettrolita (E, G su 50 celle) e stessa cinetica del grey-box, ma con densita' di corrente
di reazione UNIFORME in ciascun elettrodo (j_n = i/(a_n L_n), j_p = -i/(a_p L_p)): nessun sistema algebrico spaziale; potenziale elettrolita dal solve lineare con sorgente uniforme;
cadute ohmiche degli elettrodi analitiche (i L/(3 sigma)). Stato compatto [c_n, a_n, s_n, c_p, a_p, s_p, c_e]. UKF square-root sullo stesso vettore logit theta."""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from . import particles as pt
from .closure import U_n, U_p
from .params import F, R, T
from .ekf import EKF


class SPMe:
    def __init__(self, m, ek: EKF):
        self.m, self.ek = m, ek; self.P = m.P
        self.Nm, self.K, self.N = m.Nm, m.K, m.N
        self.nzs = 2 * (1 + self.Nm + self.K) + self.N
        self.n = self.nzs + ek.nth; self.nth = ek.nth

    def split(self, zs):
        Nm, K = self.Nm, self.K; o = 1 + Nm + K
        def part(v): return v[0:1], v[1:1 + Nm].reshape(Nm, 1), v[1 + Nm:o].reshape(K, 1)
        return part(zs[:o]), part(zs[o:2 * o]), zs[2 * o:]

    def join(self, pn, pp, ce):
        return jnp.concatenate([pn[0], pn[1].reshape(-1), pn[2].reshape(-1), pp[0], pp[1].reshape(-1), pp[2].reshape(-1), ce])

    def init(self, c0n, c0p):
        z = lambda a, b: (jnp.full(1, a), jnp.zeros((self.Nm, 1)), jnp.zeros((self.K, 1)))
        return self.join(z(c0n, 0), z(c0p, 0), jnp.full(self.N, self.P["ce0"]))

    def surf(self, zs):
        pn, pp, ce = self.split(zs)
        return (pn[0] + pn[1].sum(0) + pn[2].sum(0))[0], (pp[0] + pp[1].sum(0) + pp[2].sum(0))[0], ce

    def outputs(self, zs, I, th):
        """V, eta_lit (nodo di separatore), phi_pl; j uniforme."""
        P, m = self.P, self.m; A_ = m.A; Nn, Ns, Np = m.Nn, m.Ns, m.Np
        csn, csp, ce = self.surf(zs)
        i_cc = I / P["area"]
        a_n = 3 * P["eps_sn"] * th[1] / P["Rn"]; a_p = 3 * P["eps_sp"] / P["Rp"]
        jn = i_cc / (a_n * P["Ln"]); jp = -i_cc / (a_p * P["Lp"]); Rf = P["Rf0"] * th[0]
        tiny = 1e-6; csn_ = jnp.clip(csn, tiny, P["cmax_n"] - tiny); csp_ = jnp.clip(csp, tiny, P["cmax_p"] - tiny)
        wn, wp = jnp.asarray(A_.dxn) / P["Ln"], jnp.asarray(A_.dxp) / P["Lp"]            # volume weights (equal cells: plain means)
        ce_n, ce_p = jnp.maximum(ce[:Nn] @ wn, tiny), jnp.maximum(ce[Nn + Ns:] @ wp, tiny)
        j0n = P["k0_n"] * (ce_n * csn_ * (P["cmax_n"] - csn_)) ** P["alpha"]; j0p = P["k0_p"] * (ce_p * csp_ * (P["cmax_p"] - csp_)) ** P["alpha"]
        f2 = 2 * R * T / F; eta_n = f2 * jnp.arcsinh(jn / (2 * j0n)); eta_p = f2 * jnp.arcsinh(jp / (2 * j0p))
        # elettrolita: potenziale dal solve lineare con sorgente uniforme
        cinv = 1 / (jnp.asarray(A_.wl_c) * ce[:-1] + (1 - jnp.asarray(A_.wl_c)) * ce[1:])
        ie_c = A_.j_Kd * A_.chi * cinv * (A_.j_Gm @ ce)
        aj = jnp.concatenate([a_n * jn * jnp.asarray(A_.dxn), jnp.zeros(Ns), a_p * jp * jnp.asarray(A_.dxp)])
        rhs = aj - A_.j_Dm @ ie_c
        pe_rest = jnp.linalg.solve(A_.j_A_e[1:, 1:], rhs[1:]); pe_lin = jnp.concatenate([jnp.zeros(1), pe_rest])
        phis_n = -i_cc * P["Ln"] / (3 * P["sig_n"])                     # media di phi_s,n (phi_s,n(0) = 0)
        pe_n_avg = phis_n - (U_n(csn_ / P["cmax_n"]) + eta_n + Rf * jn)
        pe = pe_lin + (pe_n_avg - pe_lin[:Nn] @ wn)
        phis_p_avg = pe[Nn + Ns:] @ wp + U_p(csp_ / P["cmax_p"]) + eta_p
        V = phis_p_avg - i_cc * P["Lp"] / (3 * P["sig_p"])
        phie_sep = m.extrap_right(pe[:Nn])
        cs_sep = csn_
        eta_lit = phis_n - phie_sep - U_n(cs_sep / P["cmax_n"])
        phi_pl = phis_n - phie_sep - Rf * jn
        return V, eta_lit, phi_pl, jn, jp

    def f(self, x, I):
        ek, m = self.ek, self.m; nz = self.nzs
        zs = x[:nz]; c = x[nz:]; th = ek.th_of(c)
        pn, pp, ce = self.split(zs)
        V, eta, ppl, jn, jp = self.outputs(self.join(pn, pp, jnp.maximum(ce, 20.0)), I, th)
        Cn, Cp = m.part_consts(th); E, G = ek.em_of(th)
        pn2 = pt.step(pn, jnp.full(1, jn), Cn); pp2 = pt.step(pp, jnp.full(1, jp), Cp)
        ce2 = E @ jnp.maximum(ce, 20.0) + G @ jnp.concatenate([jnp.full(m.Nn, jn), jnp.full(m.Np, jp)])
        return jnp.concatenate([self.join(pn2, pp2, ce2), c]), V, jnp.stack([eta, ppl, jnp.min(jnp.maximum(ce, 20.0))]), 0.0


class UKFSPMe:
    """UKF square-root (alpha = 1, kappa = 0, beta = 2) sull'SPMe: stessa convenzione di UKF4."""
    def __init__(self, sp: SPMe):
        self.sp = sp; n = sp.n; self.n = n; self.gamma = np.sqrt(n)
        self.Wm = np.full(2 * n + 1, 1 / (2 * n)); self.Wm[0] = 0.0; self.Wc = self.Wm.copy(); self.Wc[0] = 2.0

    def run(self, x0, P0d, Qd, R, Iseq, y, x_true=None):
        sp = self.sp; n, nz, nth = self.n, sp.nzs, sp.nth; Nm, K = sp.Nm, sp.K
        scale = jnp.asarray(np.concatenate([np.full(1, 1000.0), np.full(Nm + K, 100.0), np.full(1, 1000.0), np.full(Nm + K, 100.0), np.full(sp.N, 100.0), np.full(nth, 0.5)]))
        x0 = jnp.asarray(x0); Iseq = jnp.asarray(Iseq)
        S0 = jnp.diag(jnp.asarray(P0d) / scale); sq_Q = jnp.diag(jnp.asarray(Qd) / scale)
        Wm, Wc = jnp.asarray(self.Wm), jnp.asarray(self.Wc); sqWc = jnp.sqrt(Wc); gam = self.gamma
        fv = jax.vmap(sp.f, in_axes=(0, None))
        def sigma(u, S): return jnp.concatenate([u[None, :], u[None, :] + gam * S.T, u[None, :] - gam * S.T], 0)
        phys = lambda U: x0 + U * scale
        xt = None if x_true is None else (jnp.asarray(x_true) - x0) / scale
        def step(carry, inp):
            u, S = carry; I, yk = inp[0], inp[1]
            U = sigma(u, S); Xn, Vp, Et, _ = fv(phys(U), I)
            zhat = jnp.sum(Wm * Vp); dz = Vp - zhat; Sy2 = jnp.sum(Wc * dz ** 2) + R
            Pxy = ((Wc * dz)[:, None] * (U - u)).sum(0); K_ = Pxy / Sy2; innov = yk - zhat
            u_up = u + K_ * innov; P = S @ S.T - Sy2 * jnp.outer(K_, K_); P = 0.5 * (P + P.T)
            S_up = jnp.linalg.cholesky(P + 1e-14 * jnp.eye(n))
            U2 = sigma(u_up, S_up); Xn2, Vv, Ee, Pl = fv(phys(U2), I)
            Un2 = (Xn2 - x0) / scale; um2 = jnp.sum(Wm[:, None] * Un2, 0)
            R2 = jnp.linalg.qr(jnp.concatenate([sqWc[:, None] * (Un2 - um2), sq_Q.T], 0), mode="r")
            c_up = x0[nz:] + scale[nz:] * u_up[nz:]
            Pc = (S_up @ S_up.T)[nz:, nz:] * (scale[nz:, None] * scale[None, nz:])
            Em = jnp.sum(Wm[:, None] * Et, 0)
            out = dict(vals=sp.ek.val_of(c_up), c=c_up, Pc=Pc, innov=innov, Vpred=zhat, resF=jnp.zeros(()), eta=Em[0], ppl=Em[1], ce_min=Em[2])
            if xt is not None:
                e = u_up[nz:] - inp[2][nz:]; Ps = Pc / (scale[nz:, None] * scale[None, nz:]); out["nees"] = e @ jnp.linalg.solve(Ps, e)
                i2 = [i for i, c_ in enumerate(sp.ek.coords) if c_ in ("Rf", "Ds")]
                if len(i2) == 2:
                    ii = jnp.array(i2); out["nees2"] = e[ii] @ jnp.linalg.solve(Ps[jnp.ix_(ii, ii)], e[ii])
            return (um2, R2.T), out
        inp = (Iseq, jnp.asarray(y)) + (() if xt is None else (xt,))
        _, H = jax.lax.scan(step, (jnp.zeros(n), S0), inp)
        return H
