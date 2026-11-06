"""Closed-loop ROM: scan over the protocol with Newton at k iterations (warm start) or converged."""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from .step import DFN


def simulate(m: DFN, z0, Iseq, th, k=2, converged_first=True):
    """Returns a dict of arrays [N,...]: V (y_k = h(z_{k-1}, I_k)), cs_n, cs_p, ce (state at t_{k-1}), eta_lit, phi_pl, resF."""
    Iseq = jnp.asarray(Iseq)
    A_ = m.A
    consts = m.part_consts(th)
    em = m.elec_mats(th)
    csn0, csp0, _ = m.surf(z0)
    g0, _, _ = A_.newton(A_.guess(csn0, csp0, Iseq[0], th), csn0, csp0, z0[m.sl["e"]], Iseq[0], th)

    def body(carry, I):
        z, g = carry
        csn, csp, ce = m.surf(z)
        if k is None:                                   # converged Newton with line search (warm start)
            g, _, nit = A_.solve_robust(g, csn, csp, ce, I, th, tol=1e-10)
        else:
            g = A_.newton_fixed(g, csn, csp, ce, I, th, k=k)[0] if k > 0 else g
            nit = jnp.asarray(k)
        res = jnp.linalg.norm(A_.residual(g, csn, csp, ce, I, th), jnp.inf)
        V = A_.V_of(g, I)
        eta, ppl = m.interface(g, csn, th)
        z2 = m.advance(z, g, th, consts, em)
        return (z2, g), dict(V=V, cs_n=csn, cs_p=csp, ce=ce, eta_lit=eta, phi_pl=ppl, resF=res, nit=nit, li=m.lithium(z, th))
    (zT, gT), out = jax.lax.scan(body, (z0, g0), Iseq)
    out["zT"] = zT
    return out


def simulate_nn(m: DFN, net, z0, Iseq, th, kN=0):
    """Closed loop with the neural closure. kN = number of correcting Newton steps (0 = k=0). net(csn,csp,ce,I,th) -> j [20]."""
    Iseq = jnp.asarray(Iseq)
    consts = m.part_consts(th); em = m.elec_mats(th); A_ = m.A

    def body(z, I):
        csn, csp, ce = m.surf(z)
        j = net(csn, csp, ce, I, th)
        g, V, res = m.g_hat(j, csn, csp, ce, I, th)
        if kN > 0:
            g, res = A_.newton_fixed(g, csn, csp, ce, I, th, k=kN); V = A_.V_of(g, I)
        eta, ppl = m.interface(g, csn, th)
        z2 = m.advance(z, g, th, consts, em)
        return z2, dict(V=V, cs_n=csn, cs_p=csp, ce=ce, eta_lit=eta, phi_pl=ppl, resF=res, j=g[:20], li=m.lithium(z, th))
    zT, out = jax.lax.scan(body, z0, Iseq)
    out["zT"] = zT
    return out


def simulate_T(m: DFN, z0, Iseq, th3, Tc, Ea, chunk=20):
    """M12 K3: open loop with the measured surface temperature Tc (degC, same length as Iseq) as a known input. The Arrhenius factors (algebraic.arrhenius_th) are held constant over blocks of `chunk` steps (10 s at dt = 0.5 s) at the mean block temperature; the closure is the converged Newton of simulate(k=None). Returns V (same length as Iseq)."""
    from algebraic import arrhenius_th
    Iseq = jnp.asarray(Iseq); Tc = jnp.asarray(Tc); n = Iseq.shape[0]; nc = -(-n // chunk); pad = nc * chunk - n
    Ip = jnp.concatenate([Iseq, jnp.full(pad, Iseq[-1])]).reshape(nc, chunk); Tp = jnp.concatenate([Tc, jnp.full(pad, Tc[-1])]).reshape(nc, chunk).mean(1)
    A_ = m.A; csn0, csp0, _ = m.surf(z0); th0 = arrhenius_th(th3, Tp[0], Ea)
    g0, _, _ = A_.newton(A_.guess(csn0, csp0, Iseq[0], th0), csn0, csp0, z0[m.sl["e"]], Iseq[0], th0)

    def outer(carry, xs):
        Ic, Tm = xs; th = arrhenius_th(th3, Tm, Ea); consts = m.part_consts(th); em = m.elec_mats(th)

        def body(c, I):
            z, g = c; csn, csp, ce = m.surf(z); g, _, _ = A_.solve_robust(g, csn, csp, ce, I, th, tol=1e-10); V = A_.V_of(g, I); return (m.advance(z, g, th, consts, em), g), V
        return jax.lax.scan(body, carry, Ic)
    _, V = jax.lax.scan(outer, (z0, g0), (Ip, Tp))
    return V.reshape(-1)[:n]
