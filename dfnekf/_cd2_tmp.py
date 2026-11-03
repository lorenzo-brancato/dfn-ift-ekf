"""Dataset della chiusura a 20/10/20: stati dai riferimenti PyBaMM fini (coarse-grained a 20/10/20) + perturbati, theta e I campionati, etichette = Newton di riferimento (tol 1e-12)."""
import sys, os, glob, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import params as prm, cio
from algebraic import Alg
from closure_data import smooth_noise
P0 = prm.base(); Nn, Ns, Np = 20, 10, 20


def coarse_states(period=0.5):
    out = {k: [] for k in ["csn", "csp", "ce"]}
    for f in sorted(glob.glob(f"data/c/ref_*_m40-20-40_dt{period:g}.npz")):
        b = os.path.basename(f)[4:].replace(f"_m40-20-40_dt{period:g}.npz", ""); proto, tag = b.rsplit("_", 1)
        r = cio.load(proto, tag, period=period); n = r["N"]; sub = np.arange(0, n, 5)            # un campione ogni 2.5 s (stati molto correlati)
        out["csn"].append(r["cs_n"][sub].reshape(len(sub), 20, 2).mean(2)); out["csp"].append(r["cs_p"][sub].reshape(len(sub), 20, 2).mean(2))
        out["ce"].append(np.concatenate([r["ce"][sub, :40].reshape(len(sub), 20, 2).mean(2), r["ce"][sub, 40:60].reshape(len(sub), 10, 2).mean(2), r["ce"][sub, 60:].reshape(len(sub), 20, 2).mean(2)], 1))
    return {k: np.concatenate(v) for k, v in out.items()}


def build(n_samples=40000, seed=0, n_var=5, hiI=0.0, ce_amp=0.25):
    rng = np.random.default_rng(seed); S = coarse_states(); n_base = n_samples // n_var
    idx = rng.choice(len(S["csn"]), n_base, replace=len(S["csn"]) < n_base)
    csn, csp, ce = [np.repeat(S[k][idx], n_var, 0) for k in ["csn", "csp", "ce"]]; n = n_base * n_var; base_id = np.repeat(np.arange(n_base), n_var)
    pert = (np.arange(n) % n_var) != 0
    amp_s, amp_e = rng.uniform(0.0, 0.08, n) * pert, rng.uniform(0.0, ce_amp, n) * pert
    csn = csn * (1 + amp_s[:, None] * smooth_noise(rng, n, Nn)); csp = csp * (1 + amp_s[:, None] * smooth_noise(rng, n, Np)); ce = ce * (1 + amp_e[:, None] * smooth_noise(rng, n, Nn + Ns + Np))
    csn = np.clip(csn, 0.02 * P0["cmax_n"], 0.98 * P0["cmax_n"]); csp = np.clip(csp, 0.02 * P0["cmax_p"], 0.98 * P0["cmax_p"]); ce = np.clip(ce, 10.0, 3000.0)
    th = np.stack([np.exp(rng.uniform(0, np.log(10.0), n)), rng.uniform(0.7, 1.0, n), np.ones(n)], 1)
    Ihat = np.exp(rng.uniform(np.log(1e-2), np.log(5.0), n)) * rng.choice([-1.0, 1.0], n); hi = rng.random(n) < hiI
    Ihat[hi] = np.exp(rng.uniform(np.log(1.5), np.log(5.0), hi.sum())) * rng.choice([-1.0, 1.0], hi.sum())
    Ihat[rng.random(n) < 0.08] = 0.0
    return dict(csn=csn, csp=csp, ce=ce, th=th, I=Ihat * prm.C_RATE_A, base_id=base_id)


def label(D, chunk=250, tol=1e-12):
    A = Alg(P0, Nn, Ns, Np)
    solve = jax.jit(jax.vmap(lambda csn, csp, ce, I, th: A.newton(A.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=tol, maxit=80)))
    G, R, K = [], [], []
    for i in range(0, len(D["I"]), chunk):
        sl = slice(i, i + chunk); g, r, k = solve(*(jnp.asarray(D[k_][sl]) for k_ in ["csn", "csp", "ce", "I", "th"]))
        G.append(np.asarray(g)); R.append(np.asarray(r)); K.append(np.asarray(k))
    D = dict(D); D["g"] = np.concatenate(G); D["res"] = np.concatenate(R); D["nit"] = np.concatenate(K); return D


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000; out = sys.argv[2] if len(sys.argv) > 2 else "data/c/closure20.npz"
    hiI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; ceamp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    t0 = time.time(); D = label(build(n, seed=1 if hiI > 0 else 0, hiI=hiI, ce_amp=ceamp)); ok = D["res"] < 1e-9
    print("n", len(ok), "non convergenti", 1 - ok.mean(), "nit max", D["nit"].max(), "t", time.time() - t0)
    np.savez_compressed(out, **D)
