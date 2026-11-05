"""Sec. II-C: structured IFT Jacobian against automatic differentiation, and its cost.

All comparisons use the SAME function Phi(z, theta, I) = [advance(z, closure_newton(z, I, theta, tol=1e-13)), V]: the converged closure, one step, the voltage.
  (i)   agreement (the converged closure is a lax.custom_root, i.e. itself implicit differentiation; therefore the agreement is also reported against jacfwd of EXPLICIT unrolled Newton maps without custom_root: k=1 started at the converged g, and k=3 started from g perturbed by 1e-3 relative; columns *_unrolled_k1, *_unrolled_k3_perturbed): relative error of dPhi/dz and dPhi/dtheta of the structured IFT Jacobian (`Struct.jac_full`, evaluated at the converged solution) with respect to
        `jax.jacfwd` through the converged Newton loop, on 7 states (P1_3C k=5,25; P2 k=100,640,1000; P6L k=20,600), 20/10/20, N_m = 12.   [paper: 6e-13]
  (ii)  cost on one state (N = 653, I = 5 A, theta = (4, 0.9, 0.55)): median and p99 of
          - the IFT pieces of the paper (LU + dj/dz + dj/dtheta) and the complete dense IFT Jacobian `jac_full` (dPhi/dz and dPhi/dtheta, all rows),
          - jacfwd of Phi with the converged closure (dPhi/dz, dPhi/dtheta), the same function as in (i). NOTE: `closure_newton` is a `lax.custom_root`, so this jacfwd
            differentiates implicitly (dense linear solve with the Jacobian of the residual), it does not unroll the Newton loop;
          - jacfwd through one explicit Newton iteration started at the converged solution (the cost of differentiating a solver iteration).
  (iii) for continuity with the manuscript (labelled 'legacy'): jacfwd of the step with Newton FIXED at one iteration from a cold guess, EKF coordinates (Rf, LAM, Ds)
        [paper: 0.5 ms vs 122 ms]; the share of the EKF step spent in the model and its derivatives (f_and_lin / full step; paper 8 %); worst-case / median step time
        (paper 2.95); amortized (lax.scan over 1000 steps) against the dispatch cost (paper: 17 % lower).
Timings must be taken on an idle machine with one thread (the script prints a warning and records the number of other busy processes if psutil is installed).

Output (one file per part, the timing parts can be run separately with --parts):
  <results>/jacobian_agreement.csv   one row per state  (columns as the earlier agreement table: proto, k, dPhi_dz_rel, dPhi_dtheta_rel, block errors)
  <results>/jacobian_cost.csv        timing rows (group, item, N, median_ms, p99_ms, p999_ms, worst_ms, worst_over_median, n_calls, note)
  <results>/jacobian_step.csv        step breakdown, shares, amortized cost, machine description
--aggregate writes <results>/tables/tab_jacobian.csv (the numbers quoted in Sec. II-C).

Commands:
  python scripts/jacobian_check.py --grid 4:1 --parts agreement --results results                      (minutes)
  python scripts/jacobian_check.py --grid 4:1 --parts cost,step --n-calls 3600 --n-jac 100 --results results      (idle machine; about 10 min)
  python scripts/jacobian_check.py --aggregate --results results
  python scripts/jacobian_check.py --quick --results /tmp/jac_quick
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime


def main():
    p = cli.parser("Sec. II-C (IFT Jacobian vs autodiff, cost)", "structured IFT Jacobian against jacfwd through the converged solver: agreement and timings on the same function")
    p.add_argument("--parts", default="agreement,cost,step", help="comma-separated: agreement, cost, step")
    p.add_argument("--n-calls", type=int, default=3600, help="calls for the timings of the IFT pieces and of the EKF step")
    p.add_argument("--n-jac", type=int, default=100, help="calls for the (slow) jacfwd timings, at least 100 for the paper")
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    runtime.setup(a.core)
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from dfnekf import params as prm, timing
    from dfnekf.step import DFN
    from dfnekf.rom import simulate
    from dfnekf.structjac import Struct
    from dfnekf.ekf import EKF
    from dfnekf.protocols import current_series, PROTOCOLS
    clock = cli.Clock(a, "jacobian_check.py"); parts = [s.strip() for s in a.parts.split(",")]
    n_calls, n_jac = (20, 3) if a.quick else (a.n_calls, a.n_jac)
    Nn, Ns, Np, Nm, dt = 20, 10, 20, 12, 0.5
    m = DFN(prm.base(), Nn, Ns, Np, Nm, 2, dt, a.grid); S = Struct(m); A_ = m.A; n = m.nel; nz = m.nz

    def Phi_V(z, th, I):                                              # converged closure + one step + voltage
        g = m.closure_newton(z, I, th, None, tol=1e-13)
        return jnp.concatenate([m.advance(z, g, th), A_.V_of(g, I)[None]])

    def Phi_unrolled(z, th, I, g_start, kit=1):                       # AD reference without custom_root: kit explicit Newton iterations from the constant start g_start
        csn, csp, ce = m.surf(z); g = A_.newton_fixed(g_start, csn, csp, ce, I, th, k=kit)[0]
        return jnp.concatenate([m.advance(z, g, th), A_.V_of(g, I)[None]])

    if "agreement" in parts:
        groups = {"c_avg": S.ic, "modes": S.ia.reshape(-1), "tail": S.is_.reshape(-1), "c_e": S.ie}
        cases = [("P1_3C", [1.0, 0, 1.0], [5, 25]), ("P2", [5.0, 0.3, 1.0], [100, 640, 1000]), ("P6L", [1.0, 0, 1.0], [20, 600])]
        if a.quick: cases = cases[:1]; cases = [(cases[0][0], cases[0][1], [5])]
        rows = []
        for proto, thv, ks in cases:
            I_all = current_series(proto, dt); soc = PROTOCOLS[proto]["soc"]
            c0 = (prm.C0_N_100, prm.C0_P_100) if soc == 1.0 else (prm.C0_N_50, prm.C0_P_50)
            th = jnp.array([thv[0], 1 - thv[1], thv[2]]); z0 = m.init_state(*c0)
            for k in ks:
                kk = int(k / dt)
                z = jax.jit(lambda z0, I, th: simulate(m, z0, I, th, k=None))(z0, jnp.asarray(I_all[:kk]), th)["zT"]; I = float(I_all[min(kk, len(I_all) - 1)])
                csn, csp, ce = m.surf(z)
                g = A_.solve_robust(A_.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=1e-12)[0]
                Jz, Jt = [np.asarray(x) for x in jax.jit(S.jac_full)(z, I, th, g)]
                Jz_ref = np.asarray(jax.jit(jax.jacfwd(Phi_V, 0))(z, th, I)); Jt_ref = np.asarray(jax.jit(jax.jacfwd(Phi_V, 1))(z, th, I))
                rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
                row = dict(grid=f"{Nn}-{Ns}-{Np} {a.grid}", Nm=Nm, dt=dt, proto=proto, k=k, dPhi_dz_rel=rel(Jz, Jz_ref), dPhi_dtheta_rel=rel(Jt, Jt_ref))
                gc = jax.lax.stop_gradient(g); gp = gc * (1.0 + 1e-3)                       # converged g (constant) and a 1e-3 relative perturbation of it
                for lab, (g_start, kit) in dict(unrolled_k1=(gc, 1), unrolled_k3_perturbed=(gp, 3)).items():
                    Rz = np.asarray(jax.jit(jax.jacfwd(Phi_unrolled, 0), static_argnames="kit")(z, th, I, g_start, kit=kit)); Rt = np.asarray(jax.jit(jax.jacfwd(Phi_unrolled, 1), static_argnames="kit")(z, th, I, g_start, kit=kit))
                    row[f"dPhi_dz_rel_{lab}"] = rel(Jz, Rz); row[f"dPhi_dtheta_rel_{lab}"] = rel(Jt, Rt)
                for rn, ri in dict(groups, V=np.array([nz])).items():
                    for cn, ci in groups.items():
                        nb = np.linalg.norm(Jz_ref[np.ix_(ri, ci)]); row[f"A[{rn},{cn}]"] = float(np.linalg.norm(Jz[np.ix_(ri, ci)] - Jz_ref[np.ix_(ri, ci)]) / max(nb, 1e-300))
                for j_, nm in enumerate(["Rf", "as", "Ds"]): row[f"dtheta[{nm}]"] = rel(Jt[:, j_], Jt_ref[:, j_])
                rows.append(row); print({q: row[q] for q in ["proto", "k", "dPhi_dz_rel", "dPhi_dtheta_rel"]}, flush=True)
        df = pd.DataFrame(rows); df["thr"] = 1e-6; df["pass"] = (df.dPhi_dz_rel <= 1e-6) & (df.dPhi_dtheta_rel <= 1e-6)
        cli.write_csv(df, a, "jacobian_agreement" + ("_quick" if a.quick else "") + ".csv", note="IFT Jacobian vs jacfwd through the converged Newton solve (tol 1e-13)")

    if "cost" in parts or "step" in parts:
        n_other = timing.idle_warning(); info = timing.machine_info()
        ek = EKF(m, closure="newton", coords=("Rf", "LAM", "Ds"))
        z0 = np.asarray(m.init_state(prm.C0_N_50, prm.C0_P_50)); x0 = np.concatenate([z0, ek.coord_of(np.array([4.0, 0.1, 0.55]))]); nth = 3
        P0d = np.concatenate([np.full(n, 10.0), np.full(Nm * n, 1.0), np.full(m.K * n, 1.0), np.full(m.N, 3.0), np.full(nth, 1.0)]); Qd = np.concatenate([np.zeros(nz), np.full(nth, 1e-3)])
        step, scale = ek.make_step(x0, P0d, Qd, 4e-6); th_full = ek.th_of(jnp.asarray(x0[nz:])); g0 = ek.init_g(jnp.asarray(x0[:nz]), 5.0, th_full)
        carry = (jnp.zeros(ek.n), jnp.diag((jnp.asarray(P0d) / scale) ** 2), g0); N = ek.n; I = 5.0
        note = f"{info['processor']}; 1 thread; JAX with Python dispatch; grid {a.grid}; other busy processes: {n_other}"
        rows = []

        def timed(group, item, f, args, nc):
            t = timing.time_calls(f, args, nc); r = dict(group=group, item=item, N=N, **timing.stats(t), note=note); rows.append(r)
            print(f"{item}: median {r['median_ms']:.3f} ms, p99 {r['p99_ms']:.3f} ms ({nc} calls)", flush=True); return r["median_ms"]

        if "step" in parts:
            x = jnp.asarray(x0)
            fstep = jax.jit(lambda c, I_, y: step(c, (I_, y))[0]); flin = jax.jit(lambda x_, I_, g: (lambda L: (L["f"], L["C"], L["Cth"], L["Bm"], L["D"], L["V"]))(ek.f_and_lin(x_, I_, g)))
            t_full = timed("step", "full EKF step (newton closure)", fstep, (carry, I, 3.7), n_calls)
            t_lin = timed("step", "model + derivatives (f_and_lin)", flin, (x, I, g0), n_calls)
            ns = 50 if a.quick else 1000
            t_scan = timing.scan_mean_ms(step, carry, (jnp.full(ns, 5.0), jnp.full(ns, 3.7)), ns)
            worst = rows[0]["worst_over_median"]
            sdf = pd.DataFrame([dict(item="share of the step in model + derivatives", value=t_lin / t_full), dict(item="worst / median step time", value=worst),
                                dict(item="step time, dispatch (median) [ms]", value=t_full), dict(item="step time, amortized in lax.scan (mean over steps) [ms]", value=t_scan),
                                dict(item="amortized cost relative to dispatch", value=t_scan / t_full), dict(item="n_other_busy_processes", value=np.nan if n_other is None else n_other)])
            for k, v in info.items(): sdf = pd.concat([sdf, pd.DataFrame([dict(item=f"machine: {k}", value=str(v))])])
            cli.write_csv(sdf, a, "jacobian_step" + ("_quick" if a.quick else "") + ".csv", note="EKF step breakdown, N = 653")
        if "cost" in parts:
            th = ek.th_of(jnp.asarray(x0[nz:])); z = jnp.asarray(x0[:nz]); csn, csp, ce = m.surf(z); g = A_.solve_robust(A_.guess(csn, csp, I, th), csn, csp, ce, I, th)[0]
            lu_f = jax.jit(S.lu); t_lu = timed("ift", "LU of J_F (assembly + lu_factor)", lu_f, (g, z, I, th), n_calls); lu = lu_f(g, z, I, th)
            djdz = jax.jit(lambda lu, g, z, I, th: S.C_of(S.sens(lu, g, z, I, th, with_theta=False)[0])); t_dz = timed("ift", "dj/dz (dF/dw + pullback + chain)", djdz, (lu, g, z, I, th), n_calls)
            X = S.sens(lu, g, z, I, th, with_theta=False)[2]
            djdth = jax.jit(lambda X, g, z, I, th: -X.T @ jax.jacfwd(lambda t: A_.residual(g, *m.surf(z), I, t))(th)); t_dth = timed("ift", "dj/dtheta (dF/dtheta + product with X)", djdth, (X, g, z, I, th), n_calls)
            t_ift = t_lu + t_dz + t_dth
            full = jax.jit(S.jac_full); t_dense = timed("ift", "complete dense IFT Jacobian (jac_full: dPhi/dz, dPhi/dtheta, V row)", full, (z, I, th, g), n_calls)
            # same function as in the agreement test: converged closure
            Jz_c = jax.jit(jax.jacfwd(Phi_V, 0)); Jt_c = jax.jit(jax.jacfwd(Phi_V, 1))
            t_cz = timed("jacfwd_converged", "jacfwd dPhi/dz, converged closure", Jz_c, (z, th, I), n_jac); t_ct = timed("jacfwd_converged", "jacfwd dPhi/dtheta, converged closure", Jt_c, (z, th, I), n_jac)
            # AD through one explicit Newton iteration started at the converged solution (g held constant): the cost of differentiating the solver iteration itself
            def Phi_w(z, th_, I_, gc):
                csn_, csp_, ce_ = m.surf(z); g_ = A_.newton_fixed(gc, csn_, csp_, ce_, I_, th_, k=1)[0]
                return jnp.concatenate([m.advance(z, g_, th_), A_.V_of(g_, I_)[None]])
            t_wz = timed("jacfwd_unrolled_k1", "jacfwd dPhi/dz, one Newton iteration unrolled from the converged solution", jax.jit(jax.jacfwd(Phi_w, 0)), (z, th, I, g), n_jac)
            t_wt = timed("jacfwd_unrolled_k1", "jacfwd dPhi/dtheta, one Newton iteration unrolled from the converged solution", jax.jit(jax.jacfwd(Phi_w, 1)), (z, th, I, g), n_jac)
            g3 = g * (1.0 + 1e-3); f3z = jax.jit(jax.jacfwd(lambda z_, th_, I_, g_: Phi_unrolled(z_, th_, I_, g_, 3), 0)); f3t = jax.jit(jax.jacfwd(lambda z_, th_, I_, g_: Phi_unrolled(z_, th_, I_, g_, 3), 1))
            t_3z = timed("jacfwd_unrolled_k3", "jacfwd dPhi/dz, three Newton iterations unrolled from a perturbed start", f3z, (z, th, I, g3), n_jac)
            t_3t = timed("jacfwd_unrolled_k3", "jacfwd dPhi/dtheta, three Newton iterations unrolled from a perturbed start", f3t, (z, th, I, g3), n_jac)
            # legacy: jacfwd of the step with Newton fixed at one iteration from a cold guess, EKF coordinates
            c_th = jnp.asarray(x0[nz:])

            def Phi1(z, c, I_):
                th_ = ek.th_of(c); csn_, csp_, ce_ = m.surf(z); g_ = A_.newton_fixed(A_.guess(csn_, csp_, I_, th_), csn_, csp_, ce_, I_, th_, k=1)[0]
                return jnp.concatenate([m.advance(z, g_, th_, m.part_consts(th_), ek.em_of(th_)), A_.V_of(g_, I_)[None]])
            t_1z = timed("legacy_1iteration", "legacy: jacfwd dPhi/dz, step with Newton fixed at 1 iteration from a cold guess", jax.jit(jax.jacfwd(Phi1, 0)), (z, c_th, I), n_jac)
            t_1t = timed("legacy_1iteration", "legacy: jacfwd dPhi/dtheta (3 coordinates), same function", jax.jit(jax.jacfwd(Phi1, 1)), (z, c_th, I), n_jac)
            nan = float("nan")
            rows += [dict(group="summary", item="IFT pieces (LU + dj/dz + dj/dtheta), sum of the medians", N=N, median_ms=t_ift), dict(group="summary", item="complete dense IFT Jacobian, median", N=N, median_ms=t_dense),
                     dict(group="summary", item="jacfwd converged closure: dPhi/dz + dPhi/dtheta, sum of the medians", N=N, median_ms=t_cz + t_ct),
                     dict(group="summary", item="ratio jacfwd converged / complete dense IFT Jacobian (same function)", N=N, median_ms=(t_cz + t_ct) / t_dense),
                     dict(group="summary", item="ratio jacfwd converged / IFT pieces", N=N, median_ms=(t_cz + t_ct) / t_ift),
                     dict(group="summary", item="jacfwd unrolled k=1 Newton from converged g: dPhi/dz + dPhi/dtheta, sum of the medians", N=N, median_ms=t_wz + t_wt),
                     dict(group="summary", item="jacfwd unrolled k=3 Newton from perturbed g: dPhi/dz + dPhi/dtheta, sum of the medians", N=N, median_ms=t_3z + t_3t),
                     dict(group="summary", item="legacy: jacfwd 1-iteration step: dPhi/dz + dPhi/dtheta, sum of the medians", N=N, median_ms=t_1z + t_1t),
                     dict(group="summary", item="legacy: ratio jacfwd 1-iteration / IFT pieces", N=N, median_ms=(t_1z + t_1t) / t_ift)]
        cli.write_csv(pd.DataFrame(rows), a, "jacobian_cost" + ("_quick" if a.quick else "") + ".csv", note=f"n_calls {n_calls} (IFT, step), {n_jac} (jacfwd)")
    clock.log()


def aggregate(a):
    import pandas as pd
    out = []
    f = os.path.join(a.results, "jacobian_agreement.csv")
    if os.path.exists(f):
        d = pd.read_csv(f, comment="#"); out += [dict(quantity="max relative error dPhi/dz, IFT vs jacfwd (converged closure)", value=d.dPhi_dz_rel.max(), unit="-", n=len(d)),
                                                 dict(quantity="max relative error dPhi/dtheta, IFT vs jacfwd (converged closure)", value=d.dPhi_dtheta_rel.max(), unit="-", n=len(d))]
    f = os.path.join(a.results, "jacobian_cost.csv")
    if os.path.exists(f):
        c = pd.read_csv(f, comment="#")
        for _, r in c.iterrows(): out.append(dict(quantity=r["item"], value=r.median_ms, unit="ratio" if ("ratio " in r["item"] and "iteration" not in r["item"].split("ratio")[0]) and r["group"] == "summary" and r["item"].replace("legacy: ", "").startswith("ratio") else "ms (median)", n=r.n_calls, p99_ms=r.p99_ms))
    f = os.path.join(a.results, "jacobian_step.csv")
    if os.path.exists(f):
        for _, r in pd.read_csv(f, comment="#").iterrows():
            if not str(r["item"]).startswith("machine"): out.append(dict(quantity=r["item"], value=r["value"], unit="", n=""))
    if not out: raise SystemExit(f"no jacobian_*.csv files in {a.results}")
    os.makedirs(os.path.join(a.results, "tables"), exist_ok=True)
    a2 = type("A", (), dict(item=a.item, grid=a.grid, results=os.path.join(a.results, "tables")))()
    print(cli.write_csv(pd.DataFrame(out), a2, "tab_jacobian.csv", note="numbers of Sec. II-C"))


if __name__ == "__main__":
    main()
