"""Learned closure (Appendix E): a reduced multiple-input operator network (MIONet) that maps the local fields, the current and the ageing
multipliers to the reaction current of every grid cell, with the charge balance enforced exactly.

j = raw + (target - sum_k w_k raw_k) per electrode, with w_k the normalized cell widths, so that sum_k w_k j_k = target for any network output
(the width weighting matters on the non-uniform 4:1 grid; with the plain mean the constraint is wrong there).
Architecture: field branch (MLP), scalar branch (MLP), trunk on the normalized cell centres; 56802 trainable parameters at p = 64.
"""
import pickle
from dataclasses import dataclass
import numpy as np
import jax, jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)
from . import params as prm
from .closure import Alg, U_n, U_p
from .grid import ratio_of

J_FLOOR = 0.01 * 1.49          # floor of the relative-error metric (1 % of the 1C scale, about 1.49 A/m2)


@dataclass(frozen=True)
class NetConfig:
    """Grid-dependent constants of the network: cell counts, trunk inputs (normalized cell centres per electrode) and the charge-constraint weights."""
    Nn: int
    Ns: int
    Np: int
    trunk_in: jnp.ndarray
    wn: jnp.ndarray
    wp: jnp.ndarray


def make_config(Nn, Ns, Np, grid):
    """Network constants for a grid ('uniform' or '4:1'), computed from the cell geometry used by the closure."""
    A = Alg(prm.base(), Nn, Ns, Np, grid_ratio=ratio_of(grid)); x = np.asarray(A.x); cen = 0.5 * (x[:-1] + x[1:]); dx = np.asarray(A.dx)
    xi_n = (cen[:Nn] - x[0]) / (x[Nn] - x[0]); xi_p = (cen[Nn + Ns:] - x[Nn + Ns]) / (x[Nn + Ns + Np] - x[Nn + Ns])
    xi = jnp.asarray(np.concatenate([xi_n, xi_p])); el = jnp.concatenate([jnp.zeros(Nn), jnp.ones(Np)])
    return NetConfig(Nn, Ns, Np, jnp.stack([xi, el], 1), jnp.asarray(dx[:Nn] / dx[:Nn].sum()), jnp.asarray(dx[Nn + Ns:] / dx[Nn + Ns:].sum()))


def features(cfg, csn, csp, ce, I, th):
    """csn [Nn], csp [Np], ce [N], I [A], th [3] -> (fields, scalars [6], charge targets [2])"""
    P = prm.base(); Nn, Ns = cfg.Nn, cfg.Ns
    ce_n, ce_p = ce[:Nn], ce[Nn + Ns:]
    j0n = P["k0_n"] * jnp.sqrt(ce_n * csn * (P["cmax_n"] - csn)); j0p = P["k0_p"] * jnp.sqrt(ce_p * csp * (P["cmax_p"] - csp))
    fields = jnp.concatenate([jnp.log(j0n), jnp.log(j0p), U_n(csn / P["cmax_n"]), U_p(csp / P["cmax_p"]), ce / 1000.0])
    i_cc = I / P["area"]
    a_n = 3 * P["eps_sn"] * th[1] / P["Rn"]; a_p = 3 * P["eps_sp"] / P["Rp"]
    t_n = i_cc / (a_n * P["Ln"]); t_p = -i_cc / (a_p * P["Lp"])
    Ih = I / prm.C_RATE_A
    scal = jnp.stack([jnp.arcsinh(Ih / 0.05) / 5.3, jnp.log(th[0]) / jnp.log(10.0), th[1], jnp.arcsinh(t_n / 0.05) / 5.3, jnp.arcsinh(t_p / 0.05) / 5.3,
                      jnp.arcsinh(t_n * th[0] * P["Rf0"] / 0.0257 * 10) / 3.0])
    return fields, scal, jnp.stack([t_n, t_p])


def init_mlp(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a) * (0.5 if i == len(sizes) - 2 else 1.0), jnp.zeros(b)))
    return ps


def mlp(ps, x, act=jnp.tanh):
    for W, b in ps[:-1]:
        x = act(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def init(key, cfg, p=64, hF=(128, 96), hS=64, hT=64):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    nf = 2 * (cfg.Nn + cfg.Np) + (cfg.Nn + cfg.Ns + cfg.Np)
    return dict(F=init_mlp(k1, [nf, *hF, p]), S=init_mlp(k2, [6, hS, hS, p]), T=init_mlp(k3, [2, hT, hT, hT, p]), out_b=jnp.zeros(1), skip=jnp.zeros(()))


def n_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def raw_out(params, cfg, fields_n, scal):
    bF = mlp(params["F"], fields_n); bS = mlp(params["S"], scal)
    t = mlp(params["T"], cfg.trunk_in, jnp.tanh)                      # [Nn + Np, p]
    return (t @ (bF * bS)) / jnp.sqrt(t.shape[1]) + params["out_b"][0]


def apply(params, norm, cfg, csn, csp, ce, I, th):
    """Reaction currents j [Nn + Np] (A/m2) with the width-weighted hard charge constraint."""
    Nn = cfg.Nn
    fields, scal, tg = features(cfg, csn, csp, ce, I, th)
    raw = raw_out(params, cfg, (fields - norm["mu"]) / norm["sd"], scal) * norm["js"]
    jn = raw[:Nn] + (tg[0] - jnp.dot(cfg.wn, raw[:Nn])); jp = raw[Nn:] + (tg[1] - jnp.dot(cfg.wp, raw[Nn:]))
    return jnp.concatenate([jn, jp])


def make_norm(cfg, D, idx):
    f = jax.vmap(lambda a, b, c, d, e: features(cfg, a, b, c, d, e))
    fields, _, _ = f(*(jnp.asarray(D[k][idx]) for k in ["csn", "csp", "ce", "I", "th"]))
    j = D["g"][idx, :cfg.Nn + cfg.Np]
    return dict(mu=fields.mean(0), sd=fields.std(0) + 1e-9, js=float(np.std(j)))


def rel_err(cfg, jp, jt):
    """Relative error per sample with a floor (vectorized)."""
    num = jnp.linalg.norm(jp - jt, axis=-1)
    den = jnp.maximum(jnp.linalg.norm(jt, axis=-1), J_FLOOR * np.sqrt(cfg.Nn + cfg.Np))
    return num / den


def train(cfg, D, tr, va, iters=20000, batch=512, lr0=1e-3, lr1=1e-5, seed=0, p=64, log_every=2000, verbose=True):
    """Adam with cosine decay on the relative squared error; returns (params, norm, history)."""
    key = jax.random.PRNGKey(seed)
    params = init(key, cfg, p=p)
    norm = make_norm(cfg, D, tr)
    X = {k: jnp.asarray(D[k]) for k in ["csn", "csp", "ce", "I", "th"]}; JT = jnp.asarray(D["g"][:, :cfg.Nn + cfg.Np])
    ap = jax.vmap(lambda prm_, a, b, c, d, e: apply(prm_, norm, cfg, a, b, c, d, e), in_axes=(None, 0, 0, 0, 0, 0))

    def loss_fn(params, ids):
        jp = ap(params, X["csn"][ids], X["csp"][ids], X["ce"][ids], X["I"][ids], X["th"][ids]); jt = JT[ids]
        scale2 = jnp.mean(jt ** 2, axis=1) + J_FLOOR ** 2
        return jnp.mean(jnp.mean((jp - jt) ** 2, axis=1) / scale2)

    sched = optax.cosine_decay_schedule(lr0, iters, alpha=lr1 / lr0)
    opt = optax.adam(sched)
    st = opt.init(params)
    rng = np.random.default_rng(seed)
    tr = np.asarray(tr)

    @jax.jit
    def upd(params, st, ids):
        l, g = jax.value_and_grad(loss_fn)(params, ids)
        u, st = opt.update(g, st, params); return optax.apply_updates(params, u), st, l

    @jax.jit
    def val_err(params, ids):
        jp = ap(params, X["csn"][ids], X["csp"][ids], X["ce"][ids], X["I"][ids], X["th"][ids])
        return rel_err(cfg, jp, JT[ids])
    hist = []
    for it in range(iters):
        ids = jnp.asarray(rng.choice(tr, batch, replace=False))
        params, st, l = upd(params, st, ids)
        if verbose and (it % log_every == 0 or it == iters - 1):
            e = np.asarray(val_err(params, jnp.asarray(va)))
            hist.append((it, float(l), float(np.median(e)), float(np.quantile(e, 0.99))))
            print(f"it {it} loss {float(l):.3e} val median {np.median(e):.4f} p99 {np.quantile(e, 0.99):.4f}", flush=True)
    return params, norm, hist


def load_net(path, grid="4:1", Nn=20, Ns=10, Np=20):
    """Load a trained network (pickle with 'params' and 'norm') and return (callable j(csn, csp, ce, I, th), config)."""
    cfg = make_config(Nn, Ns, Np, grid)
    S = pickle.load(open(path, "rb")); params = jax.tree_util.tree_map(jnp.asarray, S["params"]); norm = jax.tree_util.tree_map(jnp.asarray, S["norm"])
    return (lambda a, b, c, d, e: apply(params, norm, cfg, a, b, c, d, e)), cfg
