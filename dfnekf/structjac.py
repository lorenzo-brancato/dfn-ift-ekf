"""Structured Jacobian of the step: z+ = Lambda z + B j(z, theta) (+ direct dependence of Lambda, B on D_s).
Structure in the code (step.py DFN.advance + particles.step + elec_mats):
  c_avg+  = c_avg - (3 dt / R F) j                      -> Lambda = 1,  B = -3 dt/(R F)
  a_m+    = exp(-kap_m dt) a_m - gain_m j / F           -> Lambda = e_m (D_s,n for the anode), B = -gain_m/F        (m = 1..N_m)
  s_k+    = rho_k s_k - (1-rho_k) W_k j / F             -> tail: K effective modes, Lambda = rho_k (depends on D_s), B = -(1-rho_k) W_k / F
  c_e+    = E c_e + G j_el                               -> fixed block E (depends on a_s only), G from the augmented expm
  j = j(c_surf(z), c_e, I, theta) from F(y; z, theta) = 0,  c_surf = c_avg + sum_m a_m + sum_k s_k  (y = [j, phi_e, phi_s]).
Hence A = dPhi/dz = Lambda + B C, C = dj/dz = (dj/dw)(dw/dz), w = (c_surf,n; c_surf,p; c_e); ZOH: j constant over the step (no dj/dt term).
Implicit function theorem: dy/dw = -J_F^-1 dF/dw. Reverse mode: pullback (J_F^T X = E_sel) with n_j + 1 = 21 outputs (j and V) against 45 inputs."""
import numpy as np
import jax, jax.numpy as jnp
import jax.scipy.linalg as jsl
jax.config.update("jax_enable_x64", True)
from .params import F as FARAD


class Struct:
    def __init__(self, m):
        self.m = m; n = m.nel; Nm, K, N = m.Nm, m.K, m.N
        self.n, self.Nm, self.K, self.N = n, Nm, K, N
        self.ic = np.arange(n)
        self.ia = n + np.arange(Nm * n).reshape(Nm, n)
        self.is_ = n + Nm * n + np.arange(K * n).reshape(K, n)
        self.ie = np.arange(n + Nm * n + K * n, m.nz)
        self.nzp = n + Nm * n + K * n
        self.ng = m.A.ng

    # --- Lambda (particle diagonal) and B (nz x nel) from the step constants
    def lam_B(self, th, em=None):
        m = self.m; n, Nn, Np = self.n, m.Nn, m.Np
        Cn, Cp = m.part_consts(th)
        E, G = m.elec_mats(th) if em is None else em
        def per_node(vn, vp):          # vn, vp [...]: anode/cathode -> [..., n]
            return jnp.concatenate([jnp.repeat(vn[..., None], Nn, -1), jnp.repeat(vp[..., None], Np, -1)], -1)
        dvec = jnp.concatenate([jnp.ones(n), per_node(Cn["e"], Cp["e"]).reshape(-1), per_node(Cn["rho"], Cp["rho"]).reshape(-1)])
        Bm = jnp.zeros((m.nz, n))
        cols = jnp.arange(n)
        bc = jnp.concatenate([jnp.full(Nn, -3 * Cn["dt"] / Cn["R"]), jnp.full(Np, -3 * Cp["dt"] / Cp["R"])]) / FARAD
        Bm = Bm.at[self.ic, cols].set(bc)
        Bm = Bm.at[self.ia, jnp.broadcast_to(cols, self.ia.shape)].set(per_node(-Cn["gain"], -Cp["gain"]) / FARAD)
        Bm = Bm.at[self.is_, jnp.broadcast_to(cols, self.is_.shape)].set(per_node(-(1 - Cn["rho"]) * Cn["W"], -(1 - Cp["rho"]) * Cp["W"]) / FARAD)
        Bm = Bm.at[self.ie, :].set(G)
        return dvec, E, Bm

    def lam_dense(self, dvec, E):
        L = jnp.zeros((self.m.nz, self.m.nz))
        L = L.at[jnp.arange(self.nzp), jnp.arange(self.nzp)].set(dvec)
        return L.at[self.ie[:, None], self.ie[None, :]].set(E)

    # --- sensitivities of y (j and V) with respect to w and theta at converged g
    def lu(self, g, z, I, th):
        m = self.m; csn, csp, ce = m.surf(z)
        return jsl.lu_factor(m.A.jac(g, csn, csp, ce, I, th))

    def sens(self, lu, g, z, I, th, with_theta=True):
        m = self.m; A_ = m.A; n, ng = self.n, self.ng
        csn, csp, ce = m.surf(z)
        w = jnp.concatenate([csn, csp, ce]); Nn = m.Nn
        res_w = lambda w_: A_.residual(g, w_[:Nn], w_[Nn:n], w_[n:], I, th)
        Fw = jax.jacfwd(res_w)(w)                                           # ng x (n + N)
        Esel = jnp.zeros((ng, n + 1)).at[jnp.arange(n), jnp.arange(n)].set(1.0).at[ng - 1, n].set(1.0)
        X = jsl.lu_solve(lu, Esel, trans=1)                                 # J^T X = Esel
        Sw = -X.T @ Fw                                                      # (n+1) x (n+N)
        if not with_theta:
            return Sw, None, X
        Ft = jax.jacfwd(lambda t: A_.residual(g, csn, csp, ce, I, t))(th)   # ng x 3
        return Sw, -X.T @ Ft, X

    def C_of(self, Sw):
        """(n+1) x nz: rows = j (n) and V; columns = z"""
        n, Nm, K = self.n, self.Nm, self.K
        Swc, Swe = Sw[:, :n], Sw[:, n:]
        return jnp.concatenate([Swc] + [Swc] * Nm + [Swc] * K + [Swe], axis=1)

    def jac_full(self, z, I, th, g):
        """Dense Jz [nz+1, nz] and Jtheta [nz+1, 3] (rows: z+, V) for the comparison with jacfwd; theta = [Rf, as, Ds] (3 columns)."""
        m = self.m; n = self.n
        dvec, E, Bm = self.lam_B(th)
        lu = self.lu(g, z, I, th)
        Sw, Sth, _ = self.sens(lu, g, z, I, th)
        C = self.C_of(Sw)
        Azz = self.lam_dense(dvec, E) + Bm @ C[:n]
        dir_th = jax.jacfwd(lambda t: m.advance(z, g, t))(th)               # direct dependence of Lambda, B, E, G on theta (g fixed)
        Jth_z = Bm @ Sth[:n] + dir_th
        Jz = jnp.concatenate([Azz, C[n:n + 1]], 0)
        Jt = jnp.concatenate([Jth_z, Sth[n:n + 1]], 0)
        return Jz, Jt
