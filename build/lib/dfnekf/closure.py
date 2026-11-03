"""Sistema algebrico del DFN a stato congelato (CONVENTIONS.md): incognite g = [j_n, j_p, phi_e(N), phi_s_n, phi_s_p].
Dati: c_s,surf (n, p), c_e (N), I, theta = [Rf, as, Ds] (moltiplicatori). Jacobiano analitico, Newton con line search."""
import numpy as np
import jax
import jax.numpy as jnp
SAT_EPS = 1e-3
jax.config.update("jax_enable_x64", True)
from .params import F, R, T, A_CELL
from .grid import electrode_widths


def U_n(s):
    return U_n0(s)


def U_n0(s):
    return (1.9793 * jnp.exp(-39.3631 * s) + 0.2482 - 0.0909 * jnp.tanh(29.8538 * (s - 0.1234))
            - 0.04478 * jnp.tanh(14.9159 * (s - 0.2769)) - 0.0205 * jnp.tanh(30.4444 * (s - 0.6103)))


def U_p(s):
    return U_p0(s)


def U_p0(s):
    return (-0.8090 * s + 4.4875 - 0.0428 * jnp.tanh(18.5138 * (s - 0.5542))
            - 17.7326 * jnp.tanh(15.7890 * (s - 0.3117)) + 17.5842 * jnp.tanh(15.9308 * (s - 0.3120)))


class Alg:
    def __init__(self, P, Nn, Ns, Np, edge_avg="linear", cinv="inv_mean", grid_ratio=None, smooth_sat=False):
        self.smooth_sat = smooth_sat
        self.P, self.Nn, self.Ns, self.Np = P, Nn, Ns, Np
        self.cinv_mode = cinv
        N = Nn + Ns + Np
        self.N = N
        wn, wp = electrode_widths(Nn, P["Ln"], grid_ratio, True), electrode_widths(Np, P["Lp"], grid_ratio, False)
        dx = np.concatenate([wn, np.full(Ns, P["Ls"] / Ns), wp])
        self.dx = dx
        self.x = np.concatenate([[0.0], np.cumsum(dx)])
        self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        eps = np.concatenate([np.full(Nn, P["eps_en"]), np.full(Ns, P["eps_es"]), np.full(Np, P["eps_ep"])])
        self.eps = eps
        tor = eps ** P["brug"]
        self.tor = tor
        dxe = 0.5 * (dx[:-1] + dx[1:])
        wl = 0.5 * dx[1:] / dxe                      # peso della cella sinistra (interpolazione lineare alla posizione dello spigolo)
        self.wl = wl
        self.wl_c = (1.0 - wl) if grid_ratio is not None else wl     # M2 block G: PyBaMM weights the 1/c_e edge mean with the cell widths (checked against the reference residual, 4e-14); default (uniform) unchanged
        if edge_avg == "linear":                      # come FiniteVolume.node_to_edge di PyBaMM su mesh non uniforme
            tor_e = wl * tor[:-1] + (1 - wl) * tor[1:]
        elif edge_avg == "arith":
            tor_e = 0.5 * (tor[:-1] + tor[1:])
        else:  # armonica pesata sulle semicelle
            tor_e = dxe / (0.5 * dx[:-1] / tor[:-1] + 0.5 * dx[1:] / tor[1:])
        self.dxe, self.tor_e = dxe, tor_e
        Gm = np.zeros((N - 1, N)); Gm[np.arange(N - 1), np.arange(N - 1)] = -1; Gm[np.arange(N - 1), np.arange(1, N)] = 1
        Dm = np.zeros((N, N - 1)); Dm[np.arange(N - 1), np.arange(N - 1)] = 1; Dm[np.arange(1, N), np.arange(N - 1)] = -1
        self.Gm, self.Dm = Gm, Dm
        Kd = P["kappa"] * tor_e / dxe
        self.Kd = Kd
        self.A_e = -Dm @ (Kd[:, None] * Gm)                      # N x N (su phi_e)
        self.chi = 2 * R * T * (1 - P["tplus"]) * P["tdf"] / F
        # elettrodi
        def lap(n, w, sig, dirich):
            """Finite-volume operator sigma * (flux balance), face distance = distance between cell centres; Dirichlet at the half cell on the left (anode) or Neumann."""
            B = np.zeros((n, n)); de = 0.5 * (w[:-1] + w[1:])
            for k in range(n):
                if k > 0: B[k, k] += sig / de[k - 1]; B[k, k - 1] -= sig / de[k - 1]
                if k < n - 1: B[k, k] += sig / de[k]; B[k, k + 1] -= sig / de[k]
            if dirich: B[0, 0] += sig / (0.5 * w[0])
            return B
        self.B_n = lap(Nn, wn, P["sig_n"], True)
        self.B_p = lap(Np, wp, P["sig_p"], False)
        self.dxn, self.dxp = wn, wp
        self.Nel = Nn + Np
        # indici
        self.iel = np.concatenate([np.arange(Nn), np.arange(Nn + Ns, N)])    # celle di elettrodo nel vettore N
        self.ng = 2 * self.Nel + N
        # scalature di riga (Jacobiano lineare: costanti)
        self.s_e = 1.0 / np.abs(np.diag(self.A_e)).max()
        self.s_sn = 1.0 / np.abs(np.diag(self.B_n)).max()
        self.s_sp = 1.0 / np.abs(np.diag(self.B_p)).max()
        self.jax_const()

    def jax_const(self):
        for k in ["Gm", "Dm", "A_e", "Kd", "B_n", "B_p", "dx", "iel", "tor_e"]:
            setattr(self, "j_" + k, jnp.asarray(getattr(self, k)))

    # ---- helper
    def unpack(self, g):
        Nn, Np, N = self.Nn, self.Np, self.N
        jn, jp = g[:Nn], g[Nn:Nn + Np]
        pe = g[Nn + Np: Nn + Np + N]
        sn = g[Nn + Np + N: 2 * Nn + Np + N]
        sp = g[2 * Nn + Np + N:]
        return jn, jp, pe, sn, sp

    def pack(self, jn, jp, pe, sn, sp):
        return jnp.concatenate([jn, jp, pe, sn, sp])

    def coef(self, th):
        P = self.P
        a_n = 3 * P["eps_sn"] * th[1] / P["Rn"]
        a_p = 3 * P["eps_sp"] / P["Rp"]
        if jnp.shape(th)[0] == 3: return a_n, a_p, P["Rf0"] * th[0]        # uniform film resistance (default, unchanged)
        return a_n, a_p, P["Rf0"] * th[0] * th[3:3 + self.Nn]                 # M10 Z2: th = [Rf, a_s, D_s, w_1..w_Nn], film resistance per anode cell Rf0 * Rf * w_i

    def bv_parts(self, g, csn, csp, ce, th):
        P = self.P
        jn, jp, pe, sn, sp = self.unpack(g)
        a_n, a_p, Rf = self.coef(th)
        ce_n, ce_p = ce[:self.Nn], ce[self.Nn + self.Ns:]
        pe_n, pe_p = pe[:self.Nn], pe[self.Nn + self.Ns:]
        tiny = 1e-6
        csp_ = jnp.clip(csp, tiny, P["cmax_p"] - tiny)
        if self.smooth_sat:                                   # M10 M-sat: anode c_surf upper edge regularised with a softplus at 1e-3 c_max (derivative never zero); lower edge unchanged
            eps_s = SAT_EPS * P["cmax_n"]; csn_ = P["cmax_n"] - eps_s * jax.nn.softplus((P["cmax_n"] - jnp.maximum(csn, tiny)) / eps_s)
        else:
            csn_ = jnp.clip(csn, tiny, P["cmax_n"] - tiny)
        al = P["alpha"]
        j0n = P["k0_n"] * (jnp.maximum(ce_n, tiny) * csn_ * (P["cmax_n"] - csn_)) ** al
        j0p = P["k0_p"] * (jnp.maximum(ce_p, tiny) * csp_ * (P["cmax_p"] - csp_)) ** al
        eta_n = sn - pe_n - U_n(csn_ / P["cmax_n"]) - Rf * jn
        eta_p = sp - pe_p - U_p(csp_ / P["cmax_p"])
        f = F / (2 * R * T)
        return j0n, j0p, eta_n, eta_p, f, a_n, a_p, Rf

    def residual(self, g, csn, csp, ce, I, th):
        """F(g) scalato (righe BV: A/m2; righe potenziali: ~V)."""
        P = self.P
        i_cc = I / P["area"]
        jn, jp, pe, sn, sp = self.unpack(g)
        j0n, j0p, eta_n, eta_p, f, a_n, a_p, Rf = self.bv_parts(g, csn, csp, ce, th)
        r_bvn = jn - 2 * j0n * jnp.sinh(f * eta_n)
        r_bvp = jp - 2 * j0p * jnp.sinh(f * eta_p)
        # elettrolita
        wl = jnp.asarray(self.wl_c)
        cinv_e = wl / ce[:-1] + (1 - wl) / ce[1:] if self.cinv_mode == 'mean_inv' else 1 / (wl * ce[:-1] + (1 - wl) * ce[1:])
        ie_c = self.j_Kd * self.chi * cinv_e * (self.j_Gm @ ce)
        aj = jnp.concatenate([a_n * jn * self.dxn, jnp.zeros(self.Ns), a_p * jp * self.dxp])
        r_e = (self.j_A_e @ pe) + self.j_Dm @ ie_c - aj
        r_sn = self.j_B_n @ sn + a_n * jn * self.dxn
        r_sp = self.j_B_p @ sp + a_p * jp * self.dxp
        r_sp = r_sp.at[-1].add(i_cc)
        return jnp.concatenate([r_bvn, r_bvp, r_e * self.s_e, r_sn * self.s_sn, r_sp * self.s_sp])

    def jac(self, g, csn, csp, ce, I, th):
        """Jacobiano analitico (scalato come residual)."""
        Nn, Np, Ns, N, Nel = self.Nn, self.Np, self.Ns, self.N, self.Nel
        j0n, j0p, eta_n, eta_p, f, a_n, a_p, Rf = self.bv_parts(g, csn, csp, ce, th)
        Gn = 2 * f * j0n * jnp.cosh(f * eta_n)       # d(2 j0 sinh(f eta))/d eta
        Gp = 2 * f * j0p * jnp.cosh(f * eta_p)
        J = jnp.zeros((self.ng, self.ng))
        o_pe, o_sn, o_sp = Nel, Nel + N, Nel + N + Nn
        rn, rp = jnp.arange(Nn), Nn + jnp.arange(Np)
        # BV
        J = J.at[rn, rn].set(1 + Gn * Rf)
        J = J.at[rp, rp].set(1.0)
        J = J.at[rn, o_pe + jnp.arange(Nn)].set(Gn)
        J = J.at[rp, o_pe + Ns + Nn + jnp.arange(Np)].set(Gp)
        J = J.at[rn, o_sn + jnp.arange(Nn)].set(-Gn)
        J = J.at[rp, o_sp + jnp.arange(Np)].set(-Gp)
        # elettrolita
        J = J.at[Nel:Nel + N, o_pe:o_pe + N].set(self.j_A_e * self.s_e)
        ce_rows = Nel + jnp.arange(N)
        J = J.at[Nel + jnp.arange(Nn), rn].set(-a_n * self.dxn * self.s_e)
        J = J.at[Nel + Nn + Ns + jnp.arange(Np), rp].set(-a_p * self.dxp * self.s_e)
        # elettrodi
        J = J.at[o_sn:o_sn + Nn, o_sn:o_sn + Nn].set(self.j_B_n * self.s_sn)
        J = J.at[o_sn + jnp.arange(Nn), rn].set(a_n * self.dxn * self.s_sn)
        J = J.at[o_sp:o_sp + Np, o_sp:o_sp + Np].set(self.j_B_p * self.s_sp)
        J = J.at[o_sp + jnp.arange(Np), rp].set(a_p * self.dxp * self.s_sp)
        return J

    def V_of(self, g, I):
        """Tensione: phi_s,p al bordo destro (Neumann: phi_N - dx/2 * i_cc/sigma) - phi_s,n(0)=0."""
        P = self.P
        sp = self.unpack(g)[4]
        return sp[-1] - 0.5 * self.dxp[-1] * (I / P["area"]) / P["sig_p"]

    # ---- guess e Newton
    def guess(self, csn, csp, I, th):
        P = self.P
        a_n, a_p, Rf = self.coef(th)
        i_cc = I / P["area"]
        jn = jnp.full(self.Nn, 1.0) * i_cc / (a_n * P["Ln"])
        jp = jnp.full(self.Np, -1.0) * i_cc / (a_p * P["Lp"])
        Un = U_n(csn / P["cmax_n"]); Up = U_p(csp / P["cmax_p"])
        un = Un.mean()
        pe = jnp.full(self.N, -un)
        sn = jnp.zeros(self.Nn)
        sp = jnp.full(self.Np, (Up.mean() - un))
        return self.pack(jn, jp, pe, sn, sp)

    def newton(self, g0, csn, csp, ce, I, th, tol=1e-11, maxit=60):
        """Newton con backtracking (loop con arresto): ritorna g, ||F||, n_iter."""
        res = lambda g: self.residual(g, csn, csp, ce, I, th)

        def cond(s):
            g, r, k = s
            return (jnp.linalg.norm(r, jnp.inf) > tol) & (k < maxit)

        def body(s):
            g, r, k = s
            J = self.jac(g, csn, csp, ce, I, th)
            dg = jnp.linalg.solve(J, -r)
            n0 = jnp.linalg.norm(r)

            def ls(c):
                a, gn, rn_, ok = c
                gt = g + a * dg
                rt = res(gt)
                good = jnp.isfinite(jnp.linalg.norm(rt)) & (jnp.linalg.norm(rt) < n0 * (1 - 1e-4 * a) + 1e-300)
                return (jnp.where(good, a, a * 0.5), jnp.where(good, gt, gn), jnp.where(good, rt, rn_), good)

            def lsc(c):
                return (~c[3]) & (c[0] > 1e-6)
            a, gn, rn_, ok = jax.lax.while_loop(lsc, ls, (1.0, g, r, False))
            # se nessun passo di discesa: prendi il passo pieno
            gn = jnp.where(ok, gn, g + dg)
            rn_ = jnp.where(ok, rn_, res(g + dg))
            return gn, rn_, k + 1
        g, r, k = jax.lax.while_loop(cond, body, (g0, res(g0), 0))
        return g, jnp.linalg.norm(r, jnp.inf), k

    def newton_fixed(self, g0, csn, csp, ce, I, th, k=2):
        """k passi di Newton pieni (senza line search), per il ROM in ciclo chiuso."""
        g = g0
        for _ in range(k):
            r = self.residual(g, csn, csp, ce, I, th)
            J = self.jac(g, csn, csp, ce, I, th)
            g = g + jnp.linalg.solve(J, -r)
        return g, jnp.linalg.norm(self.residual(g, csn, csp, ce, I, th), jnp.inf)

    def solve_robust(self, gprev, csn, csp, ce, I, th, tol=1e-10):
        """Newton con warm start; se non converge, riparte dal guess a freddo; se ancora no, continuazione in I. Ritorna il migliore (residuo minimo)."""
        g, r, k = self.newton(gprev, csn, csp, ce, I, th, tol=tol)

        def retry(_):
            g1, r1, k1 = self.newton(self.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=tol)
            b1 = r1 < r
            gb, rb, kb = jnp.where(b1, g1, g), jnp.where(b1, r1, r), jnp.where(b1, k1, k)

            def cont(_):
                gg = self.guess(csn, csp, 0.0 * I, th)
                kk = 0
                for s_ in (0.05, 0.2, 0.5, 1.0):
                    gg, rr, k2 = self.newton(gg, csn, csp, ce, s_ * I, th, tol=tol)
                    kk = kk + k2
                b2 = rr < rb
                return jnp.where(b2, gg, gb), jnp.where(b2, rr, rb), jnp.where(b2, kk, kb)
            return jax.lax.cond(rb > tol, cont, lambda _: (gb, rb, kb), None)
        return jax.lax.cond(r > tol, retry, lambda _: (g, r, k), None)
