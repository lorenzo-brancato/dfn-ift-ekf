"""Sensitivities of V along a ROM trajectory (converged Newton) with the state sensitivity equations and the structured Jacobians:
S_{k+1} = (Lambda + B C) S_k + dPhi/dpar_k,  dV_k/dpar = h_k S_k + dV/dpar_direct.  Parameters: Rf, LAM, Ds (via theta = [Rf, 1-LAM, Ds]), LLI (initial state), nuisance (d c_n0, d c_p0)."""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from .step import DFN
from .structjac import Struct
from . import params as prm
from .ekf import EKF

DX_N = 0.8735


def traj_sens(m: DFN, th_full, z0, Iseq, par=("Rf", "LAM", "Ds"), nuis=False, return_V=False):
    """par: ordered subset of {'Rf','LAM','LLI','Ds'}; nuis: adds dc_n0, dc_p0 (initial columns of S). Returns dV [N, npar+2*nuis]."""
    S = Struct(m); A_ = m.A; nj, nz, Nn = m.nel, m.nz, m.Nn
    ek = EKF(m, closure="newton", coords=("Rf", "Ds"))             # for em_of (G linear in a_s)
    ncol = len(par) + (2 if nuis else 0)
    S0 = jnp.zeros((nz, ncol))
    for j, name in enumerate(par):
        if name == "LLI": S0 = S0.at[S.ic[:Nn], j].set(-DX_N * prm.base()["cmax_n"])
    if nuis:
        S0 = S0.at[S.ic[:Nn], len(par)].set(1.0).at[S.ic[Nn:], len(par) + 1].set(1.0)
    pidx = {"Rf": (0, 1.0), "LAM": (1, -1.0), "Ds": (2, 1.0)}
    I = jnp.asarray(Iseq); th_full = jnp.asarray(th_full)
    csn, csp, ce = m.surf(z0)
    g0 = A_.newton(A_.guess(csn, csp, I[0], th_full), csn, csp, ce, I[0], th_full, tol=1e-10)[0]
    em = ek.em_of(th_full); consts = m.part_consts(th_full)

    def body(carry, Ik):
        z, g, Sm = carry
        csn, csp, ce = m.surf(z)
        g, res, _ = A_.solve_robust(g, csn, csp, ce, Ik, th_full, tol=1e-10)
        lu = S.lu(g, z, Ik, th_full); Sw, Sth, _ = S.sens(lu, g, z, Ik, th_full); C = S.C_of(Sw)
        dvec, E, Bm = S.lam_B(th_full, em)
        dirc = jax.jacfwd(lambda t: m.advance(z, g, t, None, ek.em_of(t)))(th_full)          # nz x 3
        V = A_.V_of(g, Ik)
        dV = C[nj] @ Sm
        # direct/indirect dependence on theta
        dz_th = jnp.zeros((nz, ncol)); dv_th = jnp.zeros(ncol)
        for j, name in enumerate(par):
            if name in pidx:
                ti, sg = pidx[name]
                dz_th = dz_th.at[:, j].set(sg * (Bm @ Sth[:nj, ti] + dirc[:, ti])); dv_th = dv_th.at[j].set(sg * Sth[nj, ti])
        dV = dV + dv_th
        LS = jnp.concatenate([dvec[:S.nzp, None] * Sm[:S.nzp], E @ Sm[S.ie]], 0)
        Sn = LS + Bm @ (C[:nj] @ Sm) + dz_th
        z2 = m.advance(z, g, th_full, consts, em)
        return (z2, g, Sn), (dV, V)
    _, (dV, V) = jax.lax.scan(body, (z0, g0, S0), I)
    return (dV, V) if return_V else dV


def fisher_crlb(dV, sigma=2e-3, ranges=None):
    F = dV.T @ dV / sigma ** 2
    C = np.linalg.pinv(F) if np.linalg.cond(F) > 1e15 else np.linalg.inv(F)
    sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
    return F, C, sd, corr
