"""Closure residual at the UKF sigma points (supports the choice of 'one Newton iteration per sigma point' of the full UKF) and cost of converging Newton at every sigma point.

Twin experiment at th2, same configuration as ``twin_nees.py --filter ukf --ukf-iters 1``. The filter itself runs with ONE Newton iteration per sigma point, started from the
converged solution at the mean point. For every step, the algebraic residual of the closure at all 2n+1 sigma points is recorded in both passes (measurement update, prediction).
Every ``--chunk`` steps, the sigma points of the next step are re-solved with k = 1..6 Newton iterations and with Newton to convergence (tol 1e-10, line search): residual
statistics, number of iterations to converge, and the time per call of each variant.

Output (<results>/): res_steps_s<seed>.csv (per step: max, median, p99 and exceedance fractions of the residual, centre point, per pass), res_sweep_s<seed>.csv (k = 1..6 and k = -1
for converged Newton), res_cpu_s<seed>.csv.

  python scripts/ukf_closure_check.py --grid 4:1 --seed 0 --results results           # full protocol (8640 steps)
  python scripts/ukf_closure_check.py --quick                                         # 240 steps
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

p = cli.parser("Appendix B / Table IV (UKF closure iterations)", "closure residual at the UKF sigma points and cost of full convergence")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=None, help="number of steps (default: whole protocol)")
p.add_argument("--chunk", type=int, default=288, help="steps per scan chunk; the k-sweep is evaluated at the end of every chunk (a tail shorter than a chunk is dropped)")
args = p.parse_args()
runtime.setup(args.core)
import time
import numpy as np, pandas as pd, jax, jax.numpy as jnp
from dfnekf.experiment import Case
from dfnekf.ukf import UKF
from dfnekf.montecarlo import SIGMA_V, truncate

if args.quick: args.steps, args.chunk = 240, 120
CH, seed = args.chunk, args.seed
case = Case("th2", args.grid, twin=True, data_dir=args.data); ek = case.ek; m = ek.m; A_ = m.A
if args.steps: truncate(case, args.steps)
th0 = case.offsets(seed); x0, P0d, Qd = case.x0_P0_Q(th0); y = case.Vtrue + SIGMA_V * np.random.default_rng(seed).normal(size=case.N)
N = case.N; I_all = np.asarray(case.I)[:N]; y = y[:N]
uk = UKF(ek, closure_iters=1)


def build_step():
    """Copy of the scan step of ``UKF.run`` whose outputs are the residual statistics over all sigma points in both passes."""
    n, nz, nth = uk.n, uk.nz, ek.nth; nel = m.nel
    scale = jnp.asarray(np.concatenate([np.full(nel, 1000.0), np.full(m.Nm * nel, 100.0), np.full(m.K * nel, 100.0), np.full(m.N, 100.0), np.full(nth, 0.5)]))
    X0 = jnp.asarray(x0); S0 = jnp.diag(jnp.asarray(P0d) / scale); sq_Q = jnp.diag(jnp.asarray(Qd) / scale)
    Wm, Wc = jnp.asarray(uk.Wm), jnp.asarray(uk.Wc); sqWc = jnp.sqrt(Wc); gam = uk.gamma; R = SIGMA_V ** 2
    fv = jax.vmap(uk.pt, in_axes=(0, None, None))
    sigma = lambda u, S: jnp.concatenate([u[None, :], u[None, :] + gam * S.T, u[None, :] - gam * S.T], 0)
    phys = lambda U: X0 + U * scale
    g0 = ek.init_g(X0[:nz], jnp.asarray(I_all)[0], ek.th_of(X0[nz:]))
    st = lambda r: jnp.stack([jnp.max(r), jnp.median(r), jnp.quantile(r, .99), jnp.mean(r > 1e-10), jnp.mean(r > 1e-4), jnp.mean(r > 1e-2), r[0]])

    def step(carry, inp):
        u, S, g = carry; I, yk = inp
        gm = uk.mean_g(phys(u), I, g)
        U = sigma(u, S); Xn, Vp, Rs, Et = fv(phys(U), I, gm)
        zhat = jnp.sum(Wm * Vp); dz = Vp - zhat; Sy2 = jnp.sum(Wc * dz ** 2) + R
        Pxy = ((Wc * dz)[:, None] * (U - u)).sum(0); K = Pxy / Sy2; innov = yk - zhat
        u_up = u + K * innov
        P = S @ S.T - Sy2 * jnp.outer(K, K); P = 0.5 * (P + P.T); S_up = jnp.linalg.cholesky(P + 1e-14 * jnp.eye(n))
        gm2 = uk.mean_g(phys(u_up), I, gm)
        U2 = sigma(u_up, S_up); Xn2, _, Rs2, _ = fv(phys(U2), I, gm2)
        Un2 = (Xn2 - X0) / scale; um2 = jnp.sum(Wm[:, None] * Un2, 0)
        R2 = jnp.linalg.qr(jnp.concatenate([sqWc[:, None] * (Un2 - um2), sq_Q.T], 0), mode="r")
        return (um2, R2.T, gm2), dict(r1=st(Rs), r2=st(Rs2), innov=innov)
    return step, (jnp.zeros(n), S0, g0), (sigma, phys)


def conv_pt(x, I, g0):
    """One sigma point with Newton run to convergence (tol 1e-10, line search, warm start from the mean-point solution); returns (residual, iterations)."""
    z = x[:uk.nz]; th = ek.th_of(x[uk.nz:]); cc, a, s, ce = m.split(z); z = m.join(cc, a, s, jnp.maximum(ce, 20.0)); csn, csp, ce_ = m.surf(z)
    g, res, it = A_.newton(g0, csn, csp, ce_, I, th, tol=1e-10)
    return res, it


def timeit(f, *a, n=3):
    jax.block_until_ready(f(*a)); t = []
    for _ in range(n):
        s = time.perf_counter(); jax.block_until_ready(f(*a)); t.append(time.perf_counter() - s)
    return float(np.median(t) * 1e3)


if __name__ == "__main__":
    clock = cli.Clock(args, "ukf_closure_check.py")
    step, carry, (sigma, phys) = build_step()
    scan = jax.jit(lambda c, I, yy: jax.lax.scan(step, c, (I, yy)))
    fvk = {k: jax.jit(jax.vmap(UKF(ek, closure_iters=k).pt, in_axes=(0, None, None))) for k in range(1, 7)}
    fvc = jax.jit(jax.vmap(conv_pt, in_axes=(0, None, None)))
    rows_sw, rows_st = [], []; t0, c0 = time.time(), time.process_time()
    for a in range(0, N, CH):
        b = min(a + CH, N)
        if b - a != CH: break                                    # fixed chunk size (one compilation)
        carry, out = scan(carry, jnp.asarray(I_all[a:b]), jnp.asarray(y[a:b])); jax.block_until_ready(out["r1"])
        for key in ("r1", "r2"):
            v = np.asarray(out[key])
            rows_st += [dict(step=a + i, pass_=key, max=v[i, 0], median=v[i, 1], p99=v[i, 2], frac_gt1e10=v[i, 3], frac_gt1e4=v[i, 4], frac_gt1e2=v[i, 5], centre=v[i, 6]) for i in range(b - a)]
        if b < N:                                                # k-sweep at the sigma points of the next step
            u, S, g = carry; I = float(I_all[b]); gm = uk.mean_g(phys(u), I, g); X = phys(sigma(u, S))
            for k in range(1, 7):
                Rk = np.asarray(fvk[k](X, I, gm)[2])
                rows_sw.append(dict(step=b, k=k, max=Rk.max(), median=np.median(Rk), p99=np.quantile(Rk, .99), frac_gt1e10=float((Rk > 1e-10).mean()), frac_gt1e4=float((Rk > 1e-4).mean()), time_ms=timeit(fvk[k], X, I, gm)))
            rc, it = fvc(X, I, gm); rc, it = np.asarray(rc), np.asarray(it)
            rows_sw.append(dict(step=b, k=-1, max=float(rc.max()), median=float(np.median(rc)), p99=float(np.quantile(rc, .99)), frac_gt1e10=float((rc > 1e-10).mean()), frac_gt1e4=0.0, time_ms=timeit(fvc, X, I, gm),
                                it_mean=float(it.mean()), it_median=float(np.median(it)), it_p99=float(np.quantile(it, .99)), it_max=int(it.max())))
        print(f"seed {seed} step {b}/{N} wall {time.time() - t0:.0f}s  r1max {np.asarray(out['r1'])[:, 0].max():.2e}", flush=True)
        cli.write_csv(pd.DataFrame(rows_st), args, f"res_steps_s{seed}.csv", note="closure residual at the sigma points, one Newton iteration")
        cli.write_csv(pd.DataFrame(rows_sw), args, f"res_sweep_s{seed}.csv", note="k = 1..6 Newton iterations; k = -1: converged Newton (time_ms: per call over all sigma points)")
    cli.write_csv(pd.DataFrame([dict(seed=seed, steps=N, wall_s=time.time() - t0, cpu_s=time.process_time() - c0)]), args, f"res_cpu_s{seed}.csv")
    clock.log()
