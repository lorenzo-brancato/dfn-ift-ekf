"""Square-root unscented Kalman filter (QR) on the same grey-box model, parameterization and tuning as the EKF.

Scaled unscented transform with alpha = 1, kappa = 0, beta = 2 (weights >= 0). Order: measurement update with y_k = h(x_{k-1}, I_k), then prediction from
sigma points of the posterior. The kinetics closure is solved at every sigma point by ``closure_iters`` full Newton iterations started from the converged
solution at the mean point (``None``: Newton to convergence with line search, tol 1e-10, the criterion of the EKF).
"""
import time
import numpy as np
import jax, jax.numpy as jnp
import jax.scipy.linalg as jsl
jax.config.update("jax_enable_x64", True)
from .ekf import EKF


class UKF:
    def __init__(self, ek: EKF, closure_iters=1, alpha=1.0, kappa=0.0, beta=2.0):
        self.k_iter = closure_iters; self.ek = ek; self.m = ek.m; self.nz = ek.nz; self.n = ek.n; n = self.n
        self.alpha = alpha
        lam = alpha ** 2 * (n + kappa) - n
        self.gamma = np.sqrt(n + lam)
        self.Wm = np.full(2 * n + 1, 1 / (2 * (n + lam))); self.Wm[0] = lam / (n + lam)
        self.Wc = self.Wm.copy(); self.Wc[0] = lam / (n + lam) + 1 - alpha ** 2 + beta

    def pt(self, x, I, g0):
        """Physical state x -> (x+, V, residual, [eta, phi_pl, c_e,min]) with the closure solved from the warm start g0 (the solution at the mean point)."""
        ek, m = self.ek, self.m; A_ = m.A
        z = x[:self.nz]; c = x[self.nz:]; th = ek.th_of(c)
        cc, a, s, ce = m.split(z); z = m.join(cc, a, s, jnp.maximum(ce, 20.0))
        csn, csp, ce_ = m.surf(z)
        if self.k_iter is None:
            g, res, _ = A_.newton(g0, csn, csp, ce_, I, th, tol=1e-10)
        else:
            g, res = A_.newton_fixed(g0, csn, csp, ce_, I, th, k=self.k_iter)
        consts = m.part_consts(th); em = ek.em_of(th)
        z2 = m.advance(z, g, th, consts, em)
        eta, ppl = m.interface(g, csn, th)
        return jnp.concatenate([z2, c]), A_.V_of(g, I), res, jnp.stack([eta, ppl, jnp.min(ce_)])

    def mean_g(self, x, I, g_prev):
        ek, m = self.ek, self.m; z = x[:self.nz]; th = ek.th_of(x[self.nz:])
        cc, a, s, ce = m.split(z); z = m.join(cc, a, s, jnp.maximum(ce, 20.0)); csn, csp, ce_ = m.surf(z)
        return m.A.solve_robust(g_prev, csn, csp, ce_, I, th, tol=1e-10)[0]

    def run(self, x0, P0d, Qd, R, Iseq, y, x_true=None, return_step=False, chunk=None, progress=None):
        n, nz, nth = self.n, self.nz, self.ek.nth
        m = self.m; nel = m.nel
        scale = jnp.asarray(np.concatenate([np.full(nel, 1000.0), np.full(m.Nm * nel, 100.0), np.full(m.K * nel, 100.0), np.full(m.N, 100.0), np.full(nth, 0.5)]))
        x0 = jnp.asarray(x0); Iseq = jnp.asarray(Iseq)
        S0 = jnp.diag(jnp.asarray(P0d) / scale); sq_Q = jnp.diag(jnp.asarray(Qd) / scale)
        Wm, Wc = jnp.asarray(self.Wm), jnp.asarray(self.Wc); sqWc = jnp.sqrt(Wc); gam = self.gamma
        fv = jax.vmap(self.pt, in_axes=(0, None, None))
        def sigma(u, S): return jnp.concatenate([u[None, :], u[None, :] + gam * S.T, u[None, :] - gam * S.T], 0)
        def phys(U): return x0 + U * scale
        th0 = self.ek.th_of(x0[nz:]); g0 = self.ek.init_g(x0[:nz], Iseq[0], th0)
        xt = None if x_true is None else (jnp.asarray(x_true) - x0) / scale

        def step(carry, inp):
            u, S, g = carry
            I, yk = inp[0], inp[1]
            gm = self.mean_g(phys(u), I, g)
            U = sigma(u, S); Xn, Vp, Rs, Et = fv(phys(U), I, gm)
            zhat = jnp.sum(Wm * Vp); dz = Vp - zhat
            Sy2 = jnp.sum(Wc * dz ** 2) + R
            Pxy = ((Wc * dz)[:, None] * (U - u)).sum(0)
            K = Pxy / Sy2; innov = yk - zhat
            u_up = u + K * innov
            P = S @ S.T - Sy2 * jnp.outer(K, K); P = 0.5 * (P + P.T)
            S_up = jnp.linalg.cholesky(P + 1e-14 * jnp.eye(n))
            gm2 = self.mean_g(phys(u_up), I, gm)
            U2 = sigma(u_up, S_up); Xn2, _, _, _ = fv(phys(U2), I, gm2)
            Un2 = (Xn2 - x0) / scale; um2 = jnp.sum(Wm[:, None] * Un2, 0)
            R2 = jnp.linalg.qr(jnp.concatenate([sqWc[:, None] * (Un2 - um2), sq_Q.T], 0), mode="r"); S_next = R2.T
            c_up = x0[nz:] + scale[nz:] * u_up[nz:]
            Pc = (S_up @ S_up.T)[nz:, nz:] * (scale[nz:, None] * scale[None, nz:])
            Em = jnp.sum(Wm[:, None] * Et, 0)
            out = dict(vals=self.ek.val_of(c_up), c=c_up, Pc=Pc, innov=innov, Vpred=zhat, resF=Rs[0], resMax=jnp.max(Rs), eta=Em[0], ppl=Em[1], ce_min=Em[2])
            if xt is not None:
                e = u_up[nz:] - inp[2][nz:]; Ps = Pc / (scale[nz:, None] * scale[None, nz:]); out["nees"] = e @ jnp.linalg.solve(Ps, e)
                i2 = [i for i, c_ in enumerate(self.ek.coords) if c_ in ("Rf", "Ds")]
                if len(i2) == 2:
                    ii = jnp.array(i2); out["nees2"] = e[ii] @ jnp.linalg.solve(Ps[jnp.ix_(ii, ii)], e[ii])
            return (um2, S_next, gm2), out
        if return_step:
            return step, (jnp.zeros(n), S0, g0)
        inp = (Iseq, jnp.asarray(y)) + (() if xt is None else (xt,))
        if chunk is None:
            _, H = jax.lax.scan(step, (jnp.zeros(n), S0, g0), inp)
            return H
        # chunked scan (same step function, same carry): lets a long run report its progress; not to be called under jit
        scan_c = jax.jit(lambda c, xs: jax.lax.scan(step, c, xs)); carry = (jnp.zeros(n), S0, g0); Hs = []; N = len(inp[0]); t0 = time.time()
        for a in range(0, N, chunk):
            carry, Hc = scan_c(carry, tuple(v[a:a + chunk] for v in inp)); jax.block_until_ready(Hc["vals"]); Hs.append(Hc)
            if progress is not None: progress(min(a + chunk, N), N, Hc, time.time() - t0)
        return {k: jnp.concatenate([h[k] for h in Hs]) for k in Hs[0]}
