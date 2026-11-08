"""Extended Kalman filter of the grey-box DFN (state z augmented with the ageing parameters in logit coordinates).

``EKF``        structured EKF: step Jacobian A = Lambda + B C from the implicit function theorem on the converged closure, structured covariance
               prediction in O(N^2 n_j), rank-1 Joseph update. Closures: 'newton' (converged, warm start from the previous step, cold fallback;
               used for all results), 'cold' (converged from a cold guess) and, for the learned-closure study, 'net0', 'net1', 'net1c'.
``GuardedEKF`` adds per-step diagnostics and the optional state guard (rejects a correction that leaves the physical domain).
``LLIEKF``     guarded EKF with the lithium-inventory offset LLI as fourth parameter (ageing tracking).
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from .step import DFN
from .structjac import Struct
from .ageing import DX_N


LOGIT = {"Rf": (0.5, 12.0), "LAM": (-0.05, 0.4), "Ds": (0.15, 1.3), "LLI": (-0.05, 0.3)}       # logit range (widened), BRIEF v4 §2.3
RANGE = {"Rf": (1.0, 10.0), "LAM": (0.0, 0.3), "Ds": (0.3, 1.0), "LLI": (0.0, 0.2)}          # ageing range
IDX = {"Rf": 0, "LAM": 1, "Ds": 2, "LLI": 3}


def range_width(c): return RANGE[c][1] - RANGE[c][0]


class EKF:
    def __init__(self, m: DFN, net=None, closure="newton", coords=("Rf", "Ds"), fixed=None, jitter=1e-12, tol_lin=1e-4, qz_alpha=0.0, qz_mask=None, qz_unit=None, sigF=1.0, qz_cap=1e30, transform="logit", li_sigma=None, li_ref=None):
        """li_sigma: relative sd of a lithium pseudo-measurement (None: off). Each step, after the voltage update, the total
        lithium of the state (particles + electrolyte, m.lithium, before the LLI offset) is 'measured' equal to li_ref,
        the cell's lithium at the beginning of life [mol/m2] (default: the initial state's): Li(x) = Li_ref +- li_sigma Li_ref. The prediction conserves it exactly; the constraint keeps the Kalman
        corrections from moving lithium, so that a loss of inventory can only go into LLI
        (equivalently x_n,eff Q_n + x_p Q_p + LLI = const).
        transform: 'logit' (the paper's: each parameter bounded to its LOGIT range, saturates at the edges) or 'free'
        (unbounded: log coordinates for the multipliers Rf, Ds, which stay positive; identity for LAM and LLI)."""
        if transform not in ("logit", "free"):
            raise ValueError("transform must be 'logit' or 'free'")
        self.transform = transform
        self.li_sigma, self.li_ref = li_sigma, li_ref
        self.m, self.net, self.closure, self.coords = m, net, closure, tuple(coords)
        self.fixed = dict(Rf=1.0, LAM=0.0, Ds=1.0); self.fixed.update(fixed or {})
        self.nth = len(self.coords); self.nz = m.nz; self.n = m.nz + self.nth
        self.jitter, self.tol_lin = jitter, tol_lin
        self.S = Struct(m)
        E1, G1 = m.elec_mats(jnp.array([1.0, 1.0, 1.0])); self.E1, self.G1 = E1, G1
        self.qz_alpha, self.qz_mask, self.qz_unit, self.sigF, self.qz_cap = qz_alpha, qz_mask, qz_unit, sigF, qz_cap
        self.lo = jnp.array([LOGIT[c][0] for c in self.coords]); self.hi = jnp.array([LOGIT[c][1] for c in self.coords])
        self.logm = jnp.array([c in ("Rf", "Ds") for c in self.coords])              # 'free': log coordinates
        self.th_idx = [IDX[c] for c in self.coords if c in IDX and c != "LLI"]
        self.sign = jnp.array([-1.0 if c == "LAM" else 1.0 for c in self.coords])
        # LLI (cell sets): a constant offset of the anode means the closure sees, in units of the anode window
        # (P["dx_window_n"] c_max,n; the paper's LLIEKF uses DX_N of Chen2020): z_eff = z - LLI dx c_max,n on c_avg,n
        self.i_lli = self.coords.index("LLI") if "LLI" in self.coords else None
        if self.i_lli is not None:
            self.dshift = m.P.get("dx_window_n", DX_N) * m.P["cmax_n"]

    # --- theta maps
    def val_of(self, c):
        if self.transform == "free":
            return jnp.where(self.logm, jnp.exp(c), c)
        return self.lo + (self.hi - self.lo) * jax.nn.sigmoid(c)

    def dval_dc(self, c):
        if self.transform == "free":
            return jnp.where(self.logm, jnp.exp(c), jnp.ones_like(c))
        s = jax.nn.sigmoid(c); return (self.hi - self.lo) * s * (1 - s)

    def th_of(self, c):
        v = self.val_of(c); th = jnp.array([self.fixed["Rf"], 1.0 - self.fixed["LAM"], self.fixed["Ds"]])
        for i, name in enumerate(self.coords):
            if name == "LLI":
                continue
            th = th.at[IDX[name]].set(v[i] if name != "LAM" else 1.0 - v[i])
        return th

    def zeff_of(self, z, c):
        """The state the closure sees: z itself, or with the anode means lowered by LLI (when estimated)."""
        if self.i_lli is None:
            return z
        return z.at[:self.m.Nn].add(-self.val_of(c)[self.i_lli] * self.dshift)

    def coord_of(self, vals):
        vals = np.asarray(vals, float)
        if self.transform == "free":
            return np.where(np.asarray(self.logm), np.log(np.where(np.asarray(self.logm), vals, 1.0)), vals)
        p = (vals - np.asarray(self.lo)) / (np.asarray(self.hi) - np.asarray(self.lo)); return np.log(p / (1 - p))

    def em_of(self, th):
        return self.E1, self.G1.at[:, :self.m.Nn].multiply(th[1])

    # --- step and linearization
    def f_and_lin(self, x, I, g_prev, tf=None):
        """tf: optional known factor on theta[2] (the D_s multiplier) at this step, e.g. the Arrhenius factor of the
        measured temperature against the one P was built at; None = 1 (the paper's filter)."""
        m, S, nz, nj = self.m, self.S, self.nz, self.m.nel
        A_ = m.A
        z = x[:nz]; c = x[nz:]; th = self.th_of(c)
        if tf is not None:
            th = th.at[2].multiply(tf)
        cc, a, s, ce = m.split(z); z = m.join(cc, a, s, jnp.maximum(ce, 20.0))
        z_adv = z                                                           # advanced as is; the closure sees z_eff
        z = self.zeff_of(z, c)
        csn, csp, ce_ = m.surf(z)
        cl = self.closure
        consts = m.part_consts(th); em = self.em_of(th)
        imp = m.imp_coeffs(z, th, consts, em)                               # cell sets: implicit closure (None: the paper's)
        if imp is not None and cl not in ("newton", "cold"):
            raise ValueError("the implicit closure is implemented for the Newton closures only")
        if cl == "newton":
            g, res, _ = A_.solve_robust(g_prev, csn, csp, ce_, I, th, tol=1e-10, imp=imp); gl = g
            res = jnp.linalg.norm(A_.residual(g, csn, csp, ce_, I, th, imp), jnp.inf)
        elif cl == "cold":
            g, res, _ = A_.newton(A_.guess(csn, csp, I, th), csn, csp, ce_, I, th, tol=1e-10, imp=imp); gl = g
        elif cl == "net1":
            j = self.net(csn, csp, ce_, I, th)
            g, _, _ = m.g_hat(j, csn, csp, ce_, I, th)
            g, res = A_.newton_fixed(g, csn, csp, ce_, I, th, k=1)
            gl = jax.lax.cond(res > self.tol_lin, lambda g_: A_.newton(g_, csn, csp, ce_, I, th, tol=1e-10)[0], lambda g_: g_, g)
        elif cl == "net0":                                              # M17: network alone (k = 0): no Newton iteration, no completion; res = ||F||_inf of the network's g
            j = self.net(csn, csp, ce_, I, th)
            g, _, _ = m.g_hat(j, csn, csp, ce_, I, th); gl = g
            res = jnp.linalg.norm(A_.residual(g, csn, csp, ce_, I, th), jnp.inf)
        elif cl == "net1c":                                             # M17: network + 1 Newton + completion to convergence if ||F|| > tol_lin, also on V and on the state advance (net1 completes the linearization only); res = ||F|| after the single iteration
            j = self.net(csn, csp, ce_, I, th)
            g, _, _ = m.g_hat(j, csn, csp, ce_, I, th)
            g, res = A_.newton_fixed(g, csn, csp, ce_, I, th, k=1)
            g = jax.lax.cond(res > self.tol_lin, lambda g_: A_.newton(g_, csn, csp, ce_, I, th, tol=1e-10)[0], lambda g_: g_, g); gl = g
        V = A_.V_of(g, I, csn, csp)
        z2 = m.advance(z_adv, g, th, consts, em)
        eta, ppl = m.interface(g, csn, th)
        dvec, E, Bm = S.lam_B(th, em, z)
        lu = S.lu(gl, z, I, th, imp)
        Sw, Sth, X = S.sens(lu, gl, z, I, th, imp=imp, em_of=self.em_of)
        C = S.C_of(Sw)
        if m.dlaw is not None:                                              # D_p(y): Lambda, B (and k) depend on c_avg,p
            Ul = S.dlaw_cols(z_adv, gl, th, consts, em)                         # nz x Np, selected by c_avg,p
            if imp is not None:
                C = C.at[:, S.ic[m.Nn:]].add(S.sens_kp(X, gl, z, I, th, em))
        else:
            Ul = jnp.zeros((nz, 0))
        dv = self.dval_dc(c) * self.sign                                    # dtheta_full[idx]/dc
        if tf is not None and "Ds" in self.coords:                          # theta[2] = Ds * tf
            dv = dv.at[self.coords.index("Ds")].multiply(tf)
        ith = [i for i, nme in enumerate(self.coords) if nme != "LLI"]
        Cth = jnp.zeros((C.shape[0], self.nth)).at[:, jnp.array(ith, dtype=int)].set(Sth[:, self.th_idx] * dv[jnp.array(ith, dtype=int)][None, :])
        if self.i_lli is not None:                                          # d(j, V)/dLLI through the anode means the closure sees
            Cth = Cth.at[:, self.i_lli].set(-self.dshift * jnp.sum(C[:, :m.Nn], axis=1) * dv[self.i_lli])
        # direct dependence of Lambda, B (D_s) and G (a_s) on the coordinates: one tangent per coordinate with a direct effect
        dcols = []
        for i, name in enumerate(self.coords):
            if name in ("Ds", "LAM"):
                tang = jnp.zeros(3).at[IDX[name]].set(dv[i])
                dcols.append(jax.jvp(lambda t: m.advance(z_adv, gl, t, None, self.em_of(t)), (th,), (tang,))[1])
        D = jnp.stack(dcols, 1) if dcols else jnp.zeros((nz, 0))
        dl = [i for i, name in enumerate(self.coords) if name in ("Ds", "LAM")]
        return dict(f=jnp.concatenate([z2, c]), V=V, res=res, eta=eta, ppl=ppl, ce_min=jnp.min(ce_), li=m.lithium(z, th), g=gl, need=(res > self.tol_lin), dvec=dvec, E=E, Bm=Bm, C=C, Cth=Cth, D=D, dl=dl, Ul=Ul)

    def lowrank(self, L):
        """U (n x r) and V^T (r x n) of the step Jacobian's low-rank part, A = Lambda + U V^T: the flux columns (B, C), the
        direct theta columns (D_s, a_s), and with the D_p(y) law the c_avg,p columns (selected by unit rows)."""
        nz, n, nth, nj = self.nz, self.n, self.nth, self.m.nel
        nd = len(L["dl"]); nl = L["Ul"].shape[1]
        Bm_aug = jnp.concatenate([L["Bm"], jnp.zeros((nth, nj))], 0)
        Dcols = jnp.concatenate([L["D"], jnp.zeros((nth, nd))], 0)
        Lcols = jnp.concatenate([L["Ul"], jnp.zeros((nth, nl))], 0)
        U = jnp.concatenate([Bm_aug, Dcols, Lcols], 1)
        Cj = jnp.concatenate([L["C"][:nj], L["Cth"][:nj]], 1)
        Esel = jnp.zeros((nd, n)).at[jnp.arange(nd), nz + jnp.array(L["dl"], dtype=int)].set(1.0) if nd else jnp.zeros((0, n))
        Lsel = jnp.zeros((nl, n)).at[jnp.arange(nl), jnp.asarray(self.S.ic[self.m.Nn:])].set(1.0) if nl else jnp.zeros((0, n))
        Vt = jnp.concatenate([Cj, Esel, Lsel], 0)
        return U, Vt

    def make_step(self, x0, P0_diag, Q_diag, R, nees=False, scale=None, temp=False):
        """temp: the scan inputs carry a third entry, the factor tf on theta[2] (f_and_lin); x_true then comes fourth."""
        nz, n, nth, S, m = self.nz, self.n, self.nth, self.S, self.m
        if scale is None:
            nel = m.nel
            scale = jnp.asarray(np.concatenate([np.full(nel, 1000.0), np.full(m.Nm * nel, 100.0), np.full(m.K * nel, 100.0), np.full(m.N, 100.0), np.full(nth, 0.5)]))
        x0 = jnp.asarray(x0); nj = m.nel; ie = S.ie
        Q0 = (jnp.asarray(Q_diag) / scale) ** 2; P0 = jnp.diag((jnp.asarray(P0_diag) / scale) ** 2)
        # electrode-mean stoichiometries (volume-weighted particle means) and their sd, for SOC / capacity read-outs
        A_ = m.A
        w_n = jnp.zeros(n).at[jnp.asarray(S.ic[:m.Nn])].set(jnp.asarray(A_.dxn) / (m.P["Ln"] * m.P["cmax_n"])) * scale
        w_p = jnp.zeros(n).at[jnp.asarray(S.ic[m.Nn:])].set(jnp.asarray(A_.dxp) / (m.P["Lp"] * m.P["cmax_p"])) * scale
        qmask = None if self.qz_mask is None else jnp.asarray(self.qz_mask, float)
        qunit = None if self.qz_unit is None else jnp.asarray(self.qz_unit) / scale
        li_of = lambda u_: m.lithium((x0 + u_ * scale)[:nz], self.th_of((x0 + u_ * scale)[nz:]))   # mol / m2
        Li0 = li_of(jnp.zeros(n)) if self.li_ref is None else jnp.asarray(self.li_ref); R_li = (self.li_sigma * Li0) ** 2 if self.li_sigma else None

        def lam_left(M, dvec, E):
            return jnp.concatenate([dvec[:, None] * M[:S.nzp], E @ M[ie], M[S.nzp + m.N:]], 0)
        def lam_right(M, dvec, E):
            return jnp.concatenate([M[:, :S.nzp] * dvec[None, :], M[:, ie] @ E.T, M[:, S.nzp + m.N:]], 1)

        def step(carry, inp):
            u, P, g_prev = carry
            I, yk = inp[0], inp[1]
            x = x0 + u * scale
            L = self.f_and_lin(x, I, g_prev, inp[2] if temp else None)
            U, Vt = self.lowrank(L)                                                          # n x r, r x n
            h = jnp.concatenate([L["C"][nj], L["Cth"][nj]])
            f_u = (L["f"] - x0) / scale
            Us = U / scale[:, None]; Vts = Vt * scale[None, :]; hs = h * scale
            Ph = P @ hs; Sy = hs @ Ph + R; K = Ph / Sy
            innov = yk - L["V"]
            ok_y = jnp.isfinite(yk)                                                          # y = NaN: no voltage update (masked sample)
            K = jnp.where(ok_y, K, 0.0); du = K * jnp.where(ok_y, innov, 0.0)
            M_ = P - jnp.outer(K, Ph)
            Pp = M_ - jnp.outer(M_ @ hs, K) + R * jnp.outer(K, K)
            Pp = 0.5 * (Pp + Pp.T) + self.jitter * jnp.eye(n)
            li_innov = jnp.asarray(0.0)
            if self.li_sigma:                                                                # lithium pseudo-measurement
                h2 = jax.grad(li_of)(u + du)
                li_innov = Li0 - li_of(u + du)
                P2 = Pp @ h2; K2 = P2 / (h2 @ P2 + R_li)
                du = du + K2 * li_innov
                M2 = Pp - jnp.outer(K2, P2)
                Pp = M2 - jnp.outer(M2 @ h2, K2) + R_li * jnp.outer(K2, K2)
                Pp = 0.5 * (Pp + Pp.T) + self.jitter * jnp.eye(n)
            u_up = u + du
            Lam_du = jnp.concatenate([L["dvec"] * du[:S.nzp], L["E"] @ du[ie], du[S.nzp + m.N:]])
            u_next = f_u + Lam_du + Us @ (Vts @ du)
            PV = Pp @ Vts.T
            t2 = lam_left(PV, L["dvec"], L["E"]) @ Us.T
            t1 = lam_right(lam_left(Pp, L["dvec"], L["E"]), L["dvec"], L["E"])
            t4 = (Us @ (Vts @ PV)) @ Us.T
            Qd = Q0
            if self.qz_alpha > 0:
                Qd = Q0 + self.qz_alpha * jnp.minimum((L["res"] / self.sigF) ** 2, self.qz_cap) * (qmask * qunit ** 2)
            Pn = t1 + t2 + t2.T + t4 + jnp.diag(Qd)
            Pn = 0.5 * (Pn + Pn.T)
            c_up = x0[nz:] + scale[nz:] * u_up[nz:]
            vals = self.val_of(c_up)
            sd_c = jnp.sqrt(jnp.diag(Pp)[nz:]) * scale[nz:]
            xb = w_n @ (x0 / scale + u_up); yb = w_p @ (x0 / scale + u_up)
            out = dict(li_rel=li_of(u_up) / Li0 - 1.0, li_innov_rel=li_innov / Li0, xbar=xb, ybar=yb, sd_xbar=jnp.sqrt(w_n @ Pp @ w_n), sd_ybar=jnp.sqrt(w_p @ Pp @ w_p),
                       vals=vals, sd_vals=self.dval_dc(c_up) * sd_c, sd_c=sd_c, c=c_up, innov=innov, Sy=jnp.sqrt(Sy), V=L["V"], resF=L["res"], eta=L["eta"], ppl=L["ppl"], ce_min=L["ce_min"], li=L["li"], need=L["need"],
                       minDiagP=jnp.min(jnp.diag(Pp)), finite=jnp.all(jnp.isfinite(u_up)), posdef=jnp.all(jnp.isfinite(jnp.linalg.cholesky(Pp[nz:, nz:]))))
            if nees:
                e = u_up[nz:] - inp[3 if temp else 2][nz:]
                out["nees"] = e @ jnp.linalg.solve(Pp[nz:, nz:], e)
                i2 = [i for i, c_ in enumerate(self.coords) if c_ in ("Rf", "Ds")]
                if len(i2) == 2:
                    P2 = Pp[nz:, nz:][jnp.ix_(jnp.array(i2), jnp.array(i2))]; e2 = e[jnp.array(i2)]
                    out["nees2"] = e2 @ jnp.linalg.solve(P2, e2)
            return (u_next, Pn, L["g"]), out
        return step, scale

    def init_g(self, z0, I0, th, c0=None):
        m = self.m
        z0 = z0 if c0 is None else self.zeff_of(z0, c0)
        csn, csp, ce = m.surf(z0)
        imp = m.imp_coeffs(z0, th, None, self.em_of(th))
        return m.A.newton(m.A.guess(csn, csp, I0, th), csn, csp, ce, I0, th, tol=1e-10, imp=imp)[0]

    def run(self, x0, P0d, Qd, R, Iseq, y, x_true=None, tfac=None):
        """tfac: optional per-step factor on theta[2] (see f_and_lin), same length as Iseq.
        y: measured voltage; NaN entries are masked samples (prediction only; the lithium pseudo-measurement still applies).
        P0d: the initial standard deviations (1-D, physical units), or a full initial covariance (2-D, n x n, physical units)."""
        temp = tfac is not None
        full = np.ndim(P0d) == 2
        step, scale = self.make_step(x0, np.ones(self.n) if full else P0d, Qd, R, nees=x_true is not None, temp=temp)
        x0 = jnp.asarray(x0); Iseq = jnp.asarray(Iseq)
        th0 = self.th_of(x0[self.nz:])
        if temp:
            th0 = th0.at[2].multiply(jnp.asarray(tfac)[0])
        g0 = self.init_g(x0[:self.nz], Iseq[0], th0, x0[self.nz:])
        inp = (Iseq, jnp.asarray(y)) + ((jnp.asarray(tfac),) if temp else ()) + (() if x_true is None else ((jnp.asarray(x_true) - x0) / scale,))
        n = self.n
        P0 = jnp.asarray(P0d) / jnp.outer(scale, scale) if full else jnp.diag((jnp.asarray(P0d) / scale) ** 2)
        (uT, PT, gT), H = jax.lax.scan(step, (jnp.zeros(n), P0, g0), inp)
        return H


class GuardedEKF(EKF):
    def zeff(self, x):
        """Lithium-effective state used by the closure (identity here; EKF4L shifts the anode averages by LLI)."""
        return x[:self.nz]

    def make_step_g(self, x0, P0_diag, Q_diag, R, guard=False, scale=None, ce_lim=10.0, res_lim=1e-2):
        nz, n, nth, S, m = self.nz, self.n, self.nth, self.S, self.m
        A_ = m.A; nj = m.nel; ie = S.ie
        if scale is None:
            nel = m.nel
            scale = jnp.asarray(np.concatenate([np.full(nel, 1000.0), np.full(m.Nm * nel, 100.0), np.full(m.K * nel, 100.0), np.full(m.N, 100.0), np.full(nth, 0.5)]))
        x0 = jnp.asarray(x0)
        Q0 = (jnp.asarray(Q_diag) / scale) ** 2; P0 = jnp.diag((jnp.asarray(P0_diag) / scale) ** 2)
        cmx_n, cmx_p = m.P["cmax_n"], m.P["cmax_p"]

        def lam_left(M, dvec, E):
            return jnp.concatenate([dvec[:, None] * M[:S.nzp], E @ M[ie], M[S.nzp + m.N:]], 0)

        def lam_right(M, dvec, E):
            return jnp.concatenate([M[:, :S.nzp] * dvec[None, :], M[:, ie] @ E.T, M[:, S.nzp + m.N:]], 1)

        def ranges(z, th):
            cc, a, s_, ce = m.split(z)
            csn, csp, _ = m.surf(m.join(cc, a, s_, jnp.maximum(ce, 20.0)))
            return csn, csp, jnp.min(ce)

        def step(carry, inp):
            u, P, g_prev = carry
            I, yk, I_next = inp[0], inp[1], inp[2]
            Rk = R
            x = x0 + u * scale
            th = self.th_of(x[nz:])
            csn_p, csp_p, ce_raw_p = ranges(self.zeff(x), th)
            cc, a, s_, ce = m.split(self.zeff(x)); csn_c, csp_c, ce_c = m.surf(m.join(cc, a, s_, jnp.maximum(ce, 20.0)))
            res0 = jnp.linalg.norm(A_.residual(g_prev, csn_c, csp_c, ce_c, I, th), jnp.inf)
            _, rw, kw = A_.newton(g_prev, csn_c, csp_c, ce_c, I, th, tol=1e-10)
            L = self.f_and_lin(x, I, g_prev)
            U, Vt = self.lowrank(L)
            h = jnp.concatenate([L["C"][nj], L["Cth"][nj]])
            f_u = (L["f"] - x0) / scale
            Us = U / scale[:, None]; Vts = Vt * scale[None, :]; hs = h * scale
            Ph = P @ hs; Sy = hs @ Ph + Rk; K = Ph / Sy
            innov = yk - L["V"]; du = K * innov
            M_ = P - jnp.outer(K, Ph)
            Pp = M_ - jnp.outer(M_ @ hs, K) + Rk * jnp.outer(K, K)
            Pp = 0.5 * (Pp + Pp.T) + self.jitter * jnp.eye(n)
            u_up = u + du
            dv, E, dvec = None, L["E"], L["dvec"]

            def predict(Pin, dd):
                Lam_du = jnp.concatenate([dvec * dd[:S.nzp], E @ dd[ie], dd[S.nzp + m.N:]])
                un = f_u + Lam_du + Us @ (Vts @ dd)
                PV = Pin @ Vts.T
                t2 = lam_left(PV, dvec, E) @ Us.T
                t1 = lam_right(lam_left(Pin, dvec, E), dvec, E)
                t4 = (Us @ (Vts @ PV)) @ Us.T
                Pn = t1 + t2 + t2.T + t4 + jnp.diag(Q0)
                return un, 0.5 * (Pn + Pn.T)

            u_c, P_c = predict(Pp, du)
            # --- corrected-state checks
            x_up = x0 + u_up * scale
            th_up = self.th_of(x_up[nz:])
            csn_u, csp_u, ce_raw_u = ranges(self.zeff(x_up), th_up)
            cc2, a2, s2, ce2 = m.split(self.zeff(x_up)); csn_uc, csp_uc, ce_uc = m.surf(m.join(cc2, a2, s2, jnp.maximum(ce2, 20.0)))
            res_up = jnp.linalg.norm(A_.residual(L["g"], csn_uc, csp_uc, ce_uc, I, th_up), jnp.inf)
            # --- next step (after Newton with fallback) from the candidate next state
            x_n = x0 + u_c * scale; th_n = self.th_of(x_n[nz:])
            cc3, a3, s3, ce3 = m.split(self.zeff(x_n)); csn_n, csp_n, ce_n = m.surf(m.join(cc3, a3, s3, jnp.maximum(ce3, 20.0)))
            gN, rN, kN = A_.solve_robust(L["g"], csn_n, csp_n, ce_n, I_next, th_n, tol=1e-10)
            v_ce = ce_raw_u < ce_lim
            v_cs = (jnp.min(csn_u) < 0) | (jnp.max(csn_u) > cmx_n) | (jnp.min(csp_u) < 0) | (jnp.max(csp_u) > cmx_p)
            v_F = ~(rN <= res_lim)
            v_nf = ~jnp.all(jnp.isfinite(u_c))
            viol = v_ce | v_cs | v_F | v_nf
            reject = viol if guard else jnp.asarray(False)
            u_pred, P_pred = predict(P, jnp.zeros_like(du))
            u_next = jnp.where(reject, u_pred, u_c); Pn = jnp.where(reject, P_pred, P_c)
            u_rep = jnp.where(reject, u, u_up)
            c_up = x0[nz:] + scale[nz:] * u_rep[nz:]
            vals = self.val_of(c_up)
            sd_c = jnp.sqrt(jnp.diag(jnp.where(reject, P, Pp))[nz:]) * scale[nz:]
            Pth_ = jnp.where(reject, P, Pp)[nz:, nz:]; sdp_ = jnp.sqrt(jnp.diag(Pth_)); iu_ = np.triu_indices(nth, 1); corr_ut = (Pth_ / jnp.outer(sdp_, sdp_))[iu_]
            out = dict(vals=vals, sd_vals=self.dval_dc(c_up) * sd_c, innov=innov, V=L["V"], resF=L["res"], eta=L["eta"], ppl=L["ppl"], ce_min=L["ce_min"], li=L["li"], need=L["need"],
                       # diagnostics
                       csn_min=jnp.min(csn_p), csn_max=jnp.max(csn_p), csp_min=jnp.min(csp_p), csp_max=jnp.max(csp_p), ce_raw_min=ce_raw_p,
                       F_warm=res0, F_newton=L["res"], newton_it=kw, newton_fb=(rw > 1e-10) | (kw >= 60), K_norm=jnp.linalg.norm(K), K_th=jnp.linalg.norm(K[nz:]), du_norm=jnp.linalg.norm(du), du_th=jnp.linalg.norm(du[nz:]),
                       F_corr=res_up, F_next=rN, it_next=kN, ce_raw_min_up=ce_raw_u, csn_min_up=jnp.min(csn_u), csn_max_up=jnp.max(csn_u), csp_min_up=jnp.min(csp_u), csp_max_up=jnp.max(csp_u),
                       corr_ut=corr_ut, K_c=K[nz:], dvdc=self.dval_dc(c_up), viol_ce=v_ce, viol_cs=v_cs, viol_F=v_F, viol_nf=v_nf, rejected=reject, Sy=jnp.sqrt(Sy))
            return (u_next, Pn, L["g"]), out

        return step, scale

    def run_g(self, x0, P0d, Qd, R, Iseq, y, guard=False):
        step, scale = self.make_step_g(x0, P0d, Qd, R, guard=guard)
        x0 = jnp.asarray(x0); Iseq = jnp.asarray(Iseq)
        th0 = self.th_of(x0[self.nz:])
        g0 = self.init_g(x0[:self.nz], Iseq[0], th0)
        I_next = jnp.concatenate([Iseq[1:], Iseq[-1:]])
        P0 = jnp.diag((jnp.asarray(P0d) / scale) ** 2)
        (uT, PT, gT), H = jax.lax.scan(step, (jnp.zeros(self.n), P0, g0), (Iseq, jnp.asarray(y), I_next))
        return H


class LLIEKF(GuardedEKF):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        assert "LLI" in self.coords
        self.i_lli = self.coords.index("LLI")
        self.th_idx = [IDX[c] for c in self.coords if c != "LLI"]
        self.sign = jnp.array([-1.0 if c == "LAM" else 1.0 for c in self.coords])
        self.dshift = DX_N * self.m.P["cmax_n"]

    def th_of(self, c):
        v = self.val_of(c); th = jnp.array([self.fixed["Rf"], 1.0 - self.fixed["LAM"], self.fixed["Ds"]])
        for i, name in enumerate(self.coords):
            if name == "LLI": continue
            th = th.at[IDX[name]].set(v[i] if name != "LAM" else 1.0 - v[i])
        return th

    def zeff(self, x):
        lli = self.val_of(x[self.nz:])[self.i_lli]
        return x[:self.nz].at[:self.m.Nn].add(-lli * self.dshift)

    def f_and_lin(self, x, I, g_prev):
        m, S, nz, nj = self.m, self.S, self.nz, self.m.nel
        A_ = m.A; z = x[:nz]; c = x[nz:]; th = self.th_of(c)
        ze = self.zeff(x); cc, a, s, ce = m.split(ze); ze = m.join(cc, a, s, jnp.maximum(ce, 20.0))
        csn, csp, ce_ = m.surf(ze)
        g, res, _ = A_.solve_robust(g_prev, csn, csp, ce_, I, th, tol=1e-10); gl = g
        res = jnp.linalg.norm(A_.residual(g, csn, csp, ce_, I, th), jnp.inf)
        V = A_.V_of(g, I)
        consts = m.part_consts(th); em = self.em_of(th)
        zc = m.join(*m.split(z)[:3], jnp.maximum(m.split(z)[3], 20.0))
        z2 = m.advance(zc, g, th, consts, em)
        eta, ppl = m.interface(g, csn, th)
        dvec, E, Bm = S.lam_B(th, em)
        lu = S.lu(gl, ze, I, th)
        Sw, Sth, _ = S.sens(lu, gl, ze, I, th)
        C = S.C_of(Sw)
        dv = self.dval_dc(c) * self.sign
        ths = [i for i, nme in enumerate(self.coords) if nme != "LLI"]
        Cth_core = Sth[:, self.th_idx] * dv[jnp.array(ths)][None, :]
        col_lli = -self.dshift * jnp.sum(C[:, :m.Nn], axis=1) * dv[self.i_lli]
        Cth = jnp.zeros((C.shape[0], len(self.coords))).at[:, jnp.array(ths)].set(Cth_core).at[:, self.i_lli].set(col_lli)
        dcols = []
        for i, name in enumerate(self.coords):
            if name in ("Ds", "LAM"):
                tang = jnp.zeros(th.shape[0]).at[IDX[name]].set(dv[i])
                dcols.append(jax.jvp(lambda t_: m.advance(zc, gl, t_, None, self.em_of(t_)), (th,), (tang,))[1])
        D = jnp.stack(dcols, 1) if dcols else jnp.zeros((nz, 0))
        dl = [i for i, name in enumerate(self.coords) if name in ("Ds", "LAM")]
        return dict(f=jnp.concatenate([z2, c]), V=V, res=res, eta=eta, ppl=ppl, ce_min=jnp.min(ce_), li=m.lithium(ze, th), g=gl, need=(res > self.tol_lin), dvec=dvec, E=E, Bm=Bm, C=C, Cth=Cth, D=D, dl=dl, Ul=jnp.zeros((nz, 0)))
