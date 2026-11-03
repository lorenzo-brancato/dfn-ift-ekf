"""M17: rete di chiusura sulla griglia 4:1. Stessa architettura di closure2 (nessun file modificato): cambia solo la posizione delle celle nel trunk (centri di cella normalizzati per elettrodo)."""
import os, pickle
os.environ.setdefault("DFN_GRID_RATIO", "4")
import numpy as np, jax, jax.numpy as jnp
import params as prm
import closure2 as C2
from algebraic import Alg


def set_grid_4to1():
    P0 = prm.base(); A = Alg(P0, 20, 10, 20); x = np.asarray(A.x); cen = 0.5 * (x[:-1] + x[1:])
    xi_n = (cen[:20] - x[0]) / (x[20] - x[0]); xi_p = (cen[30:] - x[30]) / (x[50] - x[30])
    C2.set_dims(20, 10, 20)
    C2.XI = jnp.asarray(np.concatenate([xi_n, xi_p])); C2.TRUNK_IN = jnp.stack([C2.XI, C2.EL], 1)
    return xi_n, xi_p


def weights_4to1():
    """Pesi di cella normalizzati per elettrodo (larghezze della griglia 4:1): il vincolo di carica esatto e' sum_k w_k j_k = target (verificato sulle etichette: 1e-8; con la media non pesata di closure2.apply lo scarto mediano e' 7 % / 2.7 %)."""
    dx = np.asarray(Alg(prm.base(), 20, 10, 20).dx)
    return jnp.asarray(dx[:20] / dx[:20].sum()), jnp.asarray(dx[30:] / dx[30:].sum())


def apply_w(params, norm, csn, csp, ce, I, th, wn=None, wp=None):
    """Come closure2.apply, con il vincolo di carica hard pesato con le larghezze di cella (unica differenza)."""
    fields, scal, tg = C2.features(csn, csp, ce, I, th)
    raw = C2.raw_out(params, (fields - norm["mu"]) / norm["sd"], scal) * norm["js"]
    jn = raw[:20] + (tg[0] - jnp.dot(wn, raw[:20])); jp = raw[20:] + (tg[1] - jnp.dot(wp, raw[20:]))
    return jnp.concatenate([jn, jp])


def load_net(pkl=None):
    pkl = pkl or os.environ.get("M17_NET", "data/c/closure_c20b_4to1w.pkl")
    set_grid_4to1()
    S = pickle.load(open(pkl, "rb")); params = jax.tree_util.tree_map(jnp.asarray, S["params"]); norm = jax.tree_util.tree_map(jnp.asarray, S["norm"])
    if S.get("weighted", False):
        wn, wp = weights_4to1(); return (lambda a, b, c, d, e: apply_w(params, norm, a, b, c, d, e, wn, wp)), S
    return (lambda a, b, c, d, e: C2.apply(params, norm, a, b, c, d, e)), S
