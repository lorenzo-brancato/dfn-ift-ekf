"""Algebraic system of the DFN at frozen state (CONVENTIONS.md): unknowns g = [j_n, j_p, phi_e(N), phi_s_n, phi_s_p].
Given: c_s,surf (n, p), c_e (N), I, theta = [Rf, as, Ds] (multipliers). Analytical Jacobian, Newton with line search."""
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
        # cell set: OCPs, temperature and series resistance from P when given (another cell than Chen2020), else the defaults
        self.T = float(P.get("T", T))
        self.Un, self.Up = P.get("ocp_n", U_n), P.get("ocp_p", U_p)
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
        wl = 0.5 * dx[1:] / dxe                      # weight of the left cell (linear interpolation at the edge position)
        self.wl = wl
        self.wl_c = (1.0 - wl) if grid_ratio is not None else wl     # M2 block G: PyBaMM weights the 1/c_e edge mean with the cell widths (checked against the reference residual, 4e-14); default (uniform) unchanged
        if edge_avg == "linear":                      # as PyBaMM's FiniteVolume.node_to_edge on a non-uniform mesh
            tor_e = wl * tor[:-1] + (1 - wl) * tor[1:]
        elif edge_avg == "arith":
            tor_e = 0.5 * (tor[:-1] + tor[1:])
        else:  # harmonic mean weighted on the half cells
            tor_e = dxe / (0.5 * dx[:-1] / tor[:-1] + 0.5 * dx[1:] / tor[1:])
        self.dxe, self.tor_e = dxe, tor_e
        Gm = np.zeros((N - 1, N)); Gm[np.arange(N - 1), np.arange(N - 1)] = -1; Gm[np.arange(N - 1), np.arange(1, N)] = 1
        Dm = np.zeros((N, N - 1)); Dm[np.arange(N - 1), np.arange(N - 1)] = 1; Dm[np.arange(1, N), np.arange(N - 1)] = -1
        self.Gm, self.Dm = Gm, Dm
        Kd = P["kappa"] * tor_e / dxe
        self.Kd = Kd
        self.A_e = -Dm @ (Kd[:, None] * Gm)                      # N x N (on phi_e)
        self.chi = 2 * R * self.T * (1 - P["tplus"]) * P["tdf"] / F
        # electrodes
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
        # indices
        self.iel = np.concatenate([np.arange(Nn), np.arange(Nn + Ns, N)])    # electrode cells in the N vector
        self.ng = 2 * self.Nel + N
        # row scalings (linear Jacobian: constants)
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
        if P.get("soft_edges"):                               # cell sets: both edges of both electrodes regularised with a softplus at
            def soft(c, cm):                                  # SAT_EPS c_max, so U and j0 keep depending on c_s (restoring) beyond 0 and c_max
                e = P.get("soft_eps", SAT_EPS) * cm
                lo = e * jax.nn.softplus(c / e)
                return cm - e * jax.nn.softplus((cm - lo) / e)
            csn_, csp_ = soft(csn, P["cmax_n"]), soft(csp, P["cmax_p"])
        elif self.smooth_sat:                                   # M10 M-sat: anode c_surf upper edge regularised with a softplus at 1e-3 c_max (derivative never zero); lower edge unchanged
            eps_s = SAT_EPS * P["cmax_n"]; csn_ = P["cmax_n"] - eps_s * jax.nn.softplus((P["cmax_n"] - jnp.maximum(csn, tiny)) / eps_s)
        else:
            csn_ = jnp.clip(csn, tiny, P["cmax_n"] - tiny)
        al = P["alpha"]
        j0n = P["k0_n"] * (jnp.maximum(ce_n, tiny) * csn_ * (P["cmax_n"] - csn_)) ** al
        j0p = P["k0_p"] * (jnp.maximum(ce_p, tiny) * csp_ * (P["cmax_p"] - csp_)) ** al
        eta_n = sn - pe_n - self.Un(csn_ / P["cmax_n"]) - Rf * jn
        eta_p = sp - pe_p - self.Up(csp_ / P["cmax_p"])
        f = F / (2 * R * self.T)
        return j0n, j0p, eta_n, eta_p, f, a_n, a_p, Rf

    def state_imp(self, g, csn, csp, ce, imp):
        """Concentrations the closure sees. imp = None: those of the state at the start of the step (explicit coupling,
        the paper's scheme). imp = (k_n, k_p, E, G) (step.DFN.imp_coeffs): those at the END of the step under the step's
        own flux j, from the same exact ZOH maps that advance the state: c_s - k j (k = d c_surf / d j over the step) and
        E c_e + G j. Implicit coupling: stable where the explicit one is not (steep dU/dc_s near an empty anode, fast
        kinetics with a strongly non-uniform reaction). The state update itself is unchanged."""
        if imp is None:
            return csn, csp, ce
        jn, jp = g[:self.Nn], g[self.Nn:self.Nn + self.Np]
        return csn - imp[0] * jn, csp - imp[1] * jp, imp[2] @ ce + imp[3] @ g[:self.Nel]

    def residual(self, g, csn, csp, ce, I, th, imp=None):
        """Scaled F(g) (BV rows: A/m2; potential rows: ~V)."""
        P = self.P
        csn, csp, ce = self.state_imp(g, csn, csp, ce, imp)
        i_cc = I / P["area"]
        jn, jp, pe, sn, sp = self.unpack(g)
        j0n, j0p, eta_n, eta_p, f, a_n, a_p, Rf = self.bv_parts(g, csn, csp, ce, th)
        r_bvn = jn - 2 * j0n * jnp.sinh(f * eta_n)
        r_bvp = jp - 2 * j0p * jnp.sinh(f * eta_p)
        # electrolyte
        wl = jnp.asarray(self.wl_c)
        cinv_e = wl / ce[:-1] + (1 - wl) / ce[1:] if self.cinv_mode == 'mean_inv' else 1 / (wl * ce[:-1] + (1 - wl) * ce[1:])
        ie_c = self.j_Kd * self.chi * cinv_e * (self.j_Gm @ ce)
        aj = jnp.concatenate([a_n * jn * self.dxn, jnp.zeros(self.Ns), a_p * jp * self.dxp])
        r_e = (self.j_A_e @ pe) + self.j_Dm @ ie_c - aj
        r_sn = self.j_B_n @ sn + a_n * jn * self.dxn
        r_sp = self.j_B_p @ sp + a_p * jp * self.dxp
        r_sp = r_sp.at[-1].add(i_cc)
        return jnp.concatenate([r_bvn, r_bvp, r_e * self.s_e, r_sn * self.s_sn, r_sp * self.s_sp])

    def jac(self, g, csn, csp, ce, I, th, imp=None):
        """Analytical Jacobian (scaled as the residual). With imp, plus the dependence on j through the end-of-step
        concentrations: c_s - k j (diagonal, BV rows) and E c_e + G j (d F / d c_e from JAX, times G)."""
        Nn, Np, Ns, N, Nel = self.Nn, self.Np, self.Ns, self.N, self.Nel
        if imp is not None:
            csn, csp, ce = self.state_imp(g, csn, csp, ce, imp)

            def r_bv(cn, cp):
                j0n, j0p, eta_n, eta_p, f, *_ = self.bv_parts(g, cn, cp, ce, th)
                return -2 * j0n * jnp.sinh(f * eta_n), -2 * j0p * jnp.sinh(f * eta_p)
            _, (dn, dp) = jax.jvp(r_bv, (csn, csp), (jnp.ones_like(csn), jnp.ones_like(csp)))   # diagonal: row i sees c_s,i only
            dF_dce = jax.jacfwd(lambda c: self.residual(g, csn, csp, c, I, th))(ce)               # (ng, N), explicit residual
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
        # electrolyte
        J = J.at[Nel:Nel + N, o_pe:o_pe + N].set(self.j_A_e * self.s_e)
        ce_rows = Nel + jnp.arange(N)
        J = J.at[Nel + jnp.arange(Nn), rn].set(-a_n * self.dxn * self.s_e)
        J = J.at[Nel + Nn + Ns + jnp.arange(Np), rp].set(-a_p * self.dxp * self.s_e)
        # electrodes
        J = J.at[o_sn:o_sn + Nn, o_sn:o_sn + Nn].set(self.j_B_n * self.s_sn)
        J = J.at[o_sn + jnp.arange(Nn), rn].set(a_n * self.dxn * self.s_sn)
        J = J.at[o_sp:o_sp + Np, o_sp:o_sp + Np].set(self.j_B_p * self.s_sp)
        J = J.at[o_sp + jnp.arange(Np), rp].set(a_p * self.dxp * self.s_sp)
        if imp is not None:
            J = J.at[rn, rn].add(-dn * imp[0]).at[rp, rp].add(-dp * imp[1])
            J = J.at[:, :Nel].add(dF_dce @ imp[3])
        return J

    def V_of(self, g, I, csn=None, csp=None):
        """Voltage: phi_s,p at the right boundary (Neumann: phi_N - dx/2 * i_cc/sigma) - phi_s,n(0)=0, minus I R_contact
        (P["R_contact"], default 0). With P["entropic"] = (dT, dUdT0, dUdT_n, dUdT_p) and the surface concentrations
        given, plus the cell's entropic term dT (dUdT0 + dUdT_n (x_s - 0.5) + dUdT_p (y_s - 0.5)), x_s, y_s their means."""
        P = self.P
        sp = self.unpack(g)[4]
        V = sp[-1] - 0.5 * self.dxp[-1] * (I / P["area"]) / P["sig_p"] - I * P.get("R_contact", 0.0)
        ent = P.get("entropic")
        if ent is not None and csn is not None:
            dT, d0, dn, dp = ent
            V = V + dT * (d0 + dn * (jnp.mean(csn) / P["cmax_n"] - 0.5) + dp * (jnp.mean(csp) / P["cmax_p"] - 0.5))
        return V

    # ---- guess and Newton
    def guess(self, csn, csp, I, th):
        P = self.P
        a_n, a_p, Rf = self.coef(th)
        i_cc = I / P["area"]
        jn = jnp.full(self.Nn, 1.0) * i_cc / (a_n * P["Ln"])
        jp = jnp.full(self.Np, -1.0) * i_cc / (a_p * P["Lp"])
        Un = self.Un(csn / P["cmax_n"]); Up = self.Up(csp / P["cmax_p"])
        un = Un.mean()
        pe = jnp.full(self.N, -un)
        sn = jnp.zeros(self.Nn)
        sp = jnp.full(self.Np, (Up.mean() - un))
        return self.pack(jn, jp, pe, sn, sp)

    def newton(self, g0, csn, csp, ce, I, th, tol=1e-11, maxit=60, imp=None):
        """Newton with backtracking (loop with stopping test): returns g, ||F||, n_iter."""
        res = lambda g: self.residual(g, csn, csp, ce, I, th, imp)

        def cond(s):
            g, r, k = s
            return (jnp.linalg.norm(r, jnp.inf) > tol) & (k < maxit)

        def body(s):
            g, r, k = s
            J = self.jac(g, csn, csp, ce, I, th, imp)
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
            # if no descent step is found: take the full step
            gn = jnp.where(ok, gn, g + dg)
            rn_ = jnp.where(ok, rn_, res(g + dg))
            return gn, rn_, k + 1
        g, r, k = jax.lax.while_loop(cond, body, (g0, res(g0), 0))
        return g, jnp.linalg.norm(r, jnp.inf), k

    def newton_fixed(self, g0, csn, csp, ce, I, th, k=2, imp=None):
        """k full Newton steps (no line search), for the closed-loop ROM."""
        g = g0
        for _ in range(k):
            r = self.residual(g, csn, csp, ce, I, th, imp)
            J = self.jac(g, csn, csp, ce, I, th, imp)
            g = g + jnp.linalg.solve(J, -r)
        return g, jnp.linalg.norm(self.residual(g, csn, csp, ce, I, th, imp), jnp.inf)

    def newton_plain(self, g0, csn, csp, ce, I, th, tol=1e-10, maxit=30, imp=None):
        """Undamped Newton with stopping test (stops on a non-finite residual): returns g, ||F||_inf, n_iter. A fallback for
        the backtracking Newton, whose monotone 2-norm test can stall in a curved valley where the full steps converge
        (steep U_n near a full anode at the end of a fast CV charge)."""
        res = lambda g: self.residual(g, csn, csp, ce, I, th, imp)

        def cond(s):
            g, r, k = s
            n = jnp.linalg.norm(r, jnp.inf)
            return (n > tol) & jnp.isfinite(n) & (k < maxit)

        def body(s):
            g, r, k = s
            g = g + jnp.linalg.solve(self.jac(g, csn, csp, ce, I, th, imp), -r)
            return g, res(g), k + 1
        g, r, k = jax.lax.while_loop(cond, body, (g0, res(g0), 0))
        return g, jnp.linalg.norm(r, jnp.inf), k

    def solve_robust(self, gprev, csn, csp, ce, I, th, tol=1e-10, imp=None):
        """Newton with warm start; if it does not converge, restarts from the cold guess; if it still does not, continuation in I. Returns the best (smallest residual).
        Cell sets (soft_edges): a non-finite residual counts as not converged, and the undamped Newton from the warm start
        is tried first among the fallbacks (the paper's sets never reach these branches with a non-finite residual)."""
        g, r, k = self.newton(gprev, csn, csp, ce, I, th, tol=tol, imp=imp)
        safe = bool(self.P.get("soft_edges"))
        better = (lambda a, b: jnp.isfinite(a) & ~(a >= b)) if safe else (lambda a, b: a < b)      # a better than b; NaN b loses to finite a
        bad = (lambda r_: ~(r_ <= tol)) if safe else (lambda r_: r_ > tol)

        def retry(_):
            if safe:
                g0, r0, k0 = self.newton_plain(gprev, csn, csp, ce, I, th, tol=tol, imp=imp)
                b0 = better(r0, r)
                g_, r_, k_ = jnp.where(b0, g0, g), jnp.where(b0, r0, r), jnp.where(b0, k0, k)
            else:
                g_, r_, k_ = g, r, k

            def cold(_):
                g1, r1, k1 = self.newton(self.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=tol, imp=imp)
                b1 = better(r1, r_)
                gb, rb, kb = jnp.where(b1, g1, g_), jnp.where(b1, r1, r_), jnp.where(b1, k1, k_)

                def cont(_):
                    gg = self.guess(csn, csp, 0.0 * I, th)
                    kk = 0
                    for s_ in (0.05, 0.2, 0.5, 1.0):
                        gg, rr, k2 = self.newton(gg, csn, csp, ce, s_ * I, th, tol=tol, imp=imp)
                        kk = kk + k2
                    b2 = better(rr, rb)
                    return jnp.where(b2, gg, gb), jnp.where(b2, rr, rb), jnp.where(b2, kk, kb)
                return jax.lax.cond(bad(rb), cont, lambda _: (gb, rb, kb), None)
            return jax.lax.cond(bad(r_), cold, lambda _: (g_, r_, k_), None)
        return jax.lax.cond(bad(r), retry, lambda _: (g, r, k), None)
