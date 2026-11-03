"""Training set of the learned closure (Appendix E): states, currents and ageing multipliers, labelled with the converged Newton solver.

The source states come from PyBaMM references (40/20/40 mesh) that are projected onto the cells of the grey-box grid (``grid`` = 'uniform' or '4:1') by the overlap of the cell
edges. The states are then perturbed (smooth random fields on the particle surfaces and the electrolyte), the ageing multipliers and the current are sampled, and every sample
is labelled with the reaction currents and potentials of the converged Newton solution (``Alg.newton``, tolerance 1e-12).

Used by ``scripts/train_closure.py``. The list of reference files is an explicit argument everywhere (``coarse_states(refs, ...)``): nothing is selected by file date.
"""
import numpy as np
import jax
import jax.numpy as jnp
from . import params as prm, refio
from .closure import Alg
from .grid import ratio_of

NN, NS, NP = 20, 10, 20                          # grey-box cells (anode, separator, cathode)
FINE = (40, 20, 40)                              # mesh of the PyBaMM references


def smooth_noise(rng, n, m, kmax=3):
    """Smooth random fields in [-1, 1] on m nodes: sum of low-frequency cosines. Returns an (n, m) array."""
    x = (np.arange(m) + 0.5) / m
    c = rng.normal(size=(n, kmax + 1)) / np.arange(1, kmax + 2)
    f = sum(c[:, [k]] * np.cos(np.pi * k * x)[None, :] for k in range(kmax + 1))
    return f / np.maximum(np.abs(f).max(1, keepdims=True), 1e-12)


def overlap_matrices(grid, Nn=NN, Ns=NS, Np=NP):
    """Projection of the fine (40/20/40) cell averages onto the cells of the grey-box grid: returns (W, Wn, Wp), with W [N, 100] for the electrolyte and
    Wn [Nn, 40], Wp [Np, 40] for the electrodes; each row is the normalized overlap length of a grey-box cell with the fine cells."""
    P = prm.base(); A = Alg(P, Nn, Ns, Np, grid_ratio=ratio_of(grid)); x = np.asarray(A.x)
    fine = np.concatenate([np.linspace(0, P["Ln"], FINE[0] + 1), P["Ln"] + np.linspace(0, P["Ls"], FINE[1] + 1)[1:], P["Ln"] + P["Ls"] + np.linspace(0, P["Lp"], FINE[2] + 1)[1:]])
    W = np.zeros((len(A.dx), len(fine) - 1))
    for k in range(len(A.dx)):
        for q in range(len(fine) - 1): W[k, q] = max(0.0, min(x[k + 1], fine[q + 1]) - max(x[k], fine[q]))
    W = W / W.sum(1, keepdims=True)
    return W, W[:Nn, :FINE[0]], W[Nn + Ns:, FINE[0] + FINE[1]:]


def split_ref_name(name):
    """'P6Lx2_th3' -> ('P6Lx2', 'th3') (the state tag is the part after the last underscore)."""
    proto, tag = name.rsplit("_", 1)
    return proto, tag


def coarse_states(refs, data_dir, grid, period=0.5, stride=5):
    """States (csn, csp, ce) of the references ``refs`` (list of '<protocol>_<state>' names) on the grey-box cells, one sample every ``stride`` steps
    (2.5 s at dt = 0.5 s: consecutive states are strongly correlated). Returns a dict of concatenated arrays."""
    W, Wn, Wp = overlap_matrices(grid); out = {k: [] for k in ("csn", "csp", "ce")}
    for name in refs:
        proto, tag = split_ref_name(name)
        r = refio.load(proto, tag, period=period, data_dir=data_dir); sub = np.arange(0, r["N"], stride)
        out["csn"].append(r["cs_n"][sub] @ Wn.T); out["csp"].append(r["cs_p"][sub] @ Wp.T); out["ce"].append(r["ce"][sub] @ W.T)
    return {k: np.concatenate(v) for k, v in out.items()}


def build(S, n_samples=80000, seed=1, n_var=5, hiI=0.5, ce_amp=0.4, Nn=NN, Ns=NS, Np=NP):
    """Training samples from the source states ``S`` (output of ``coarse_states``).

    ``n_samples / n_var`` base states are drawn from S; each is repeated ``n_var`` times, the first copy unperturbed and the others perturbed (surface concentrations by up to 8 %,
    electrolyte by up to ``ce_amp``). Ageing multipliers: R_f log-uniform in [1, 10], a_s uniform in [0.7, 1], D_s = 1. Current: log-uniform in [0.01, 5] C with random sign,
    a fraction ``hiI`` of the samples restricted to 1.5-5 C, 8 % at rest. ``base_id`` identifies the unperturbed parent (the validation split is made on it).
    """
    P0 = prm.base(); rng = np.random.default_rng(seed); n_base = n_samples // n_var
    idx = rng.choice(len(S["csn"]), n_base, replace=len(S["csn"]) < n_base)
    csn, csp, ce = [np.repeat(S[k][idx], n_var, 0) for k in ("csn", "csp", "ce")]; n = n_base * n_var; base_id = np.repeat(np.arange(n_base), n_var)
    pert = (np.arange(n) % n_var) != 0
    amp_s, amp_e = rng.uniform(0.0, 0.08, n) * pert, rng.uniform(0.0, ce_amp, n) * pert
    csn = csn * (1 + amp_s[:, None] * smooth_noise(rng, n, Nn)); csp = csp * (1 + amp_s[:, None] * smooth_noise(rng, n, Np)); ce = ce * (1 + amp_e[:, None] * smooth_noise(rng, n, Nn + Ns + Np))
    csn = np.clip(csn, 0.02 * P0["cmax_n"], 0.98 * P0["cmax_n"]); csp = np.clip(csp, 0.02 * P0["cmax_p"], 0.98 * P0["cmax_p"]); ce = np.clip(ce, 10.0, 3000.0)
    th = np.stack([np.exp(rng.uniform(0, np.log(10.0), n)), rng.uniform(0.7, 1.0, n), np.ones(n)], 1)
    Ihat = np.exp(rng.uniform(np.log(1e-2), np.log(5.0), n)) * rng.choice([-1.0, 1.0], n); hi = rng.random(n) < hiI
    Ihat[hi] = np.exp(rng.uniform(np.log(1.5), np.log(5.0), hi.sum())) * rng.choice([-1.0, 1.0], hi.sum())
    Ihat[rng.random(n) < 0.08] = 0.0
    return dict(csn=csn, csp=csp, ce=ce, th=th, I=Ihat * prm.C_RATE_A, base_id=base_id)


def label(D, grid, chunk=250, tol=1e-12, Nn=NN, Ns=NS, Np=NP):
    """Adds the converged Newton solution to the samples: ``g`` (unknowns [j_n, j_p, phi_e, phi_s_n, phi_s_p]), ``res`` (final residual) and ``nit`` (iterations)."""
    A = Alg(prm.base(), Nn, Ns, Np, grid_ratio=ratio_of(grid))
    solve = jax.jit(jax.vmap(lambda csn, csp, ce, I, th: A.newton(A.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=tol, maxit=80)))
    G, R, K = [], [], []
    for i in range(0, len(D["I"]), chunk):
        sl = slice(i, i + chunk); g, r, k = solve(*(jnp.asarray(D[k_][sl]) for k_ in ("csn", "csp", "ce", "I", "th")))
        G.append(np.asarray(g)); R.append(np.asarray(r)); K.append(np.asarray(k))
    D = dict(D); D["g"] = np.concatenate(G); D["res"] = np.concatenate(R); D["nit"] = np.concatenate(K)
    return D
