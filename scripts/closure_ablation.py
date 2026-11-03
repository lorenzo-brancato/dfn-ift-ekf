"""Appendix E, Table VIII: ablation of the learned closure inside the filter.

Question: does a trained network help the guarded EKF? Closures compared (state th2, estimation protocol P6Lx2 followed by the discharge pulses = protocol F6, 10620 steps, 4:1 grid,
R_f, LAM_n, D_s estimated with LLI fixed, guard on, ten noise sequences: measurement noise ``default_rng(seed)`` x 2 mV, initial offsets ``default_rng(200 + seed)``):
  net0   the network alone (no Newton iteration, no completion)
  net1   network + one Newton iteration; the converged solution is computed only for the linearization when the residual exceeds 1e-4 ('completion of the linearization')
  net1c  network + one Newton iteration + completion (voltage, state advance and linearization) when the residual exceeds 1e-4: the 'network + 1 Newton' row of Table VIII
  warm   converged Newton, warm start from the previous step with fallback to a cold start (the closure used for all results of the paper)
  cold   converged Newton from a fixed cold guess
The network is the one trained by ``scripts/train_closure.py`` (``--net``, default ``<results>/closure_4to1.pkl``).

Four kinds of jobs:
  filter runs   (default)   one run = one (closure, seed); outputs <results>/ablation_<closure>_s<seed>.csv and <results>/traj/ablation_<closure>_s<seed>.npz (all histories)
  --openloop                open-loop checks without filter, true parameters: voltage difference of net0/net1/net1c from warm on a 3C pulse of the nominal cell at 100 % SOC
                            (ablation_openloop3C.csv) and Newton iterations along F6 at th2 (ablation_iters.csv: iterations of cold/warm/fallback, completion fraction of the network)
  --cost                    cost of one filter step per closure (median over 3600 calls, one thread); MUST BE RUN ON AN IDLE MACHINE, one core (--core), nothing else running (ablation_cost.csv)
  --aggregate               Table VIII from the per-run files: tables/tab_ablation_m17.csv (one row per closure: means over the seeds), tables/ablation_runs.csv (one row per run, with the
                            voltage difference from warm), tables/ablation_text_numbers.csv (numbers quoted in Appendix E)

Commands (a filter run is about 575 s CPU on the machine of the paper, 10620 steps; the 50 runs are 8 h CPU; jobs are independent):
  python scripts/closure_ablation.py --grid 4:1 --data data --results results --closures warm --seeds 0-9
  python scripts/closure_ablation.py --grid 4:1 --data data --results results --closures net0,net1,net1c,cold --seeds 0-4 [--core 3]
  python scripts/closure_ablation.py --openloop --grid 4:1 --data data --results results                 (about 80 s)
  python scripts/closure_ablation.py --cost --grid 4:1 --data data --results results --core 2            (about 11 min, idle machine)
  python scripts/closure_ablation.py --aggregate --results results
  python scripts/closure_ablation.py --quick --grid 4:1 --data data --results results                    (needs the quick network of ``train_closure.py --quick``)
"""
import os
import sys
import re
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime                                             # no jax import here

CLOSURES = ["net0", "net1", "net1c", "warm", "cold"]
EKF_CLOSURE = dict(net0="net0", net1="net1", net1c="net1c", warm="newton", cold="cold")     # name in dfnekf.ekf.EKF
NEEDS_NET = ("net0", "net1", "net1c")
ORDER = {c: i for i, c in enumerate(CLOSURES)}
FIXED = dict(Rf=1.0, LAM=0.0, Ds=1.0)
QUICK_STEPS = 600
NET_TEXT = {"net0": "width-weighted charge constraint", "net1": "width-weighted charge constraint", "net1c": "width-weighted charge constraint", "warm": "none", "cold": "none"}
NOTE_TAIL = "state th2, protocol F6 (P6Lx2 + discharge pulses), grid 20/10/20, N_m = 12, dt = 0.5 s, sigma_V = 2 mV, guard on, LLI fixed"
THRESHOLD = 1e-4                                                            # completion threshold of the residual


def net_path(args):
    return args.net or os.path.join(args.results, "closure_4to1.pkl" if args.grid == "4:1" else "closure_uniform.pkl")


# ----------------------------------------------------------------------------------------------------------------------- filter runs
def run_jobs(args):
    runtime.setup(args.core)
    import time
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from dfnekf.experiment import Case
    from dfnekf.ekf import GuardedEKF
    from dfnekf import protocol_runs as pr
    if args.quick: args.results = os.path.join(args.results, "quick")
    clock = cli.Clock(args, "closure_ablation.py")
    closures = [c.strip() for c in args.closures.split(",") if c.strip()]; seeds = cli.parse_ids(args.seeds)
    if args.quick: closures, seeds = ["net1c", "warm"] if args.closures == ",".join(CLOSURES) else closures, seeds[:1]
    for c in closures:
        if c not in CLOSURES: raise SystemExit(f"unknown closure {c!r}; use {CLOSURES}")
    net = None
    if any(c in NEEDS_NET for c in closures):
        from dfnekf import closure_net as cn
        if not os.path.exists(net_path(args)): raise SystemExit(f"network file {net_path(args)} not found: run scripts/train_closure.py first (or give --net)")
        net = cn.load_net(net_path(args), args.grid)[0]
    case = Case("th2", args.grid, proto="F6", data_dir=args.data)
    if args.quick: pr.truncate(case, QUICK_STEPS)
    r = case.r; N = case.N; t = r["t"][:N]; I = np.asarray(r["I"])[:N]
    os.makedirs(os.path.join(args.results, "traj"), exist_ok=True)
    k4320 = min(int(np.searchsorted(t, 4320.0 - 1e-9, side="left")) - 1, N - 1); kend = N - 1
    for cl in closures:
        ek = GuardedEKF(case.m, net=net if cl in NEEDS_NET else None, closure=EKF_CLOSURE[cl], coords=case.coords, fixed=dict(FIXED))
        fn = jax.jit(lambda x0, P0d, Qd, y, I, ek=ek: ek.run_g(x0, P0d, Qd, (2e-3) ** 2, I, y, guard=True))
        for seed in seeds:
            traj = os.path.join(args.results, "traj", f"ablation_{cl}_s{seed}.npz")
            if args.resume and os.path.exists(traj): print(cl, seed, "exists, skipped", flush=True); continue
            th0 = case.offsets(seed); x0, P0d, Qd = case.x0_P0_Q(th0); y = case.Vtrue + 2e-3 * np.random.default_rng(seed).normal(size=N)
            c0 = time.process_time(); w0 = time.time()
            H = fn(jnp.asarray(x0), jnp.asarray(P0d), jnp.asarray(Qd), jnp.asarray(y), case.I); jax.block_until_ready(H["vals"]); H = {k: np.asarray(v) for k, v in H.items()}
            cpu = time.process_time() - c0; wall = time.time() - w0
            np.savez(traj, t=t, I=I, y=y, **H)
            vals = H["vals"]; err = 100 * (vals - case.true_vals) / case.range                           # signed, % of the range
            fin = np.isfinite(vals).all(1) & np.isfinite(H["V"]); n_fin = int(fin.sum()); last_fin = int(np.where(fin)[0].max()) if n_fin else -1
            first_bad = int(np.where(~fin)[0].min()) if (~fin).any() else -1
            ok = fin & ~H["rejected"].astype(bool) & np.isfinite(H["innov"])
            row = dict(closure=cl, seed=seed, steps=N, finite_all=bool(fin.all()), first_nonfinite_step=first_bad, last_finite_step=last_fin, t_last_finite_s=float(t[last_fin]) if last_fin >= 0 else np.nan,
                       innov_rms_mV=1e3 * float(np.sqrt(np.mean(H["innov"][ok] ** 2))) if ok.any() else np.nan, n_rejected=int(H["rejected"].sum()),
                       resF_median=float(np.nanmedian(H["resF"])), resF_p99=float(np.nanquantile(H["resF"], .99)), resF_max=float(np.nanmax(H["resF"])),
                       frac_res_gt_1em4=float(np.mean(H["resF"] > THRESHOLD)), frac_fallback_warm=float(np.mean(H["newton_fb"])), newton_it_median=float(np.median(H["newton_it"])), newton_it_p99=float(np.quantile(H["newton_it"], .99)),
                       t_at_4320_s=float(t[k4320]), cpu_s=cpu, wall_s=wall)
            for i, nm in enumerate(["Rf", "LAM", "Ds"]):
                row[f"{nm}_err_4320_pct"] = float(err[k4320, i]); row[f"{nm}_err_end_pct"] = float(err[kend, i])
            cli.write_csv(pd.DataFrame([row]), args, f"ablation_{cl}_s{seed}.csv", note=NOTE_TAIL + "; errors signed, % of the ageing range; cpu_s includes the compilation in the first run of a job", float_format=None)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    clock.log()


# ----------------------------------------------------------------------------------------------------------------------- open loop
def openloop(args):
    runtime.setup(args.core)
    import time
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from dfnekf import params as prm, refio, closure_net as cn
    from dfnekf.step import DFN
    from dfnekf.rom import simulate, simulate_nn
    if args.quick: args.results = os.path.join(args.results, "quick")
    clock = cli.Clock(args, "closure_ablation.py")
    if not os.path.exists(net_path(args)): raise SystemExit(f"network file {net_path(args)} not found: run scripts/train_closure.py first (or give --net)")
    net = cn.load_net(net_path(args), args.grid)[0]
    m = DFN(prm.base(), 20, 10, 20, 12, 2, 0.5, args.grid); A_ = m.A

    def sim_net1c(z0, I, th):
        """Network + 1 Newton + completion (converged Newton when the residual exceeds the threshold), open loop."""
        consts = m.part_consts(th); em = m.elec_mats(th)

        def body(z, Ik):
            csn, csp, ce = m.surf(z); j = net(csn, csp, ce, Ik, th); g, _, _ = m.g_hat(j, csn, csp, ce, Ik, th); g, res = A_.newton_fixed(g, csn, csp, ce, Ik, th, k=1)
            comp = res > THRESHOLD
            g2, r2, _ = jax.lax.cond(comp, lambda g_: A_.newton(g_, csn, csp, ce, Ik, th, tol=1e-10), lambda g_: (g_, res, jnp.asarray(0)), g)
            return m.advance(z, g2, th, consts, em), dict(V=A_.V_of(g2, Ik), resF=r2, comp=comp)
        return jax.lax.scan(body, z0, I)[1]

    # (1) 3C pulse of the nominal cell at 100 % SOC, true parameters, each closure with its own state
    r = refio.load("P1_3C", "nom", period=0.5, data_dir=args.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]]); I = jnp.asarray(r["I"])
    w = jax.jit(lambda z0, I, th: simulate(m, z0, I, th, k=None))(z0, I, th); Vw = np.asarray(w["V"]); rows = []; CASE = "P1_3C nominal cell, 100 % SOC, nominal theta"
    for nm, out in [("net0", jax.jit(lambda z0, I, th: simulate_nn(m, net, z0, I, th, 0))(z0, I, th)), ("net1", jax.jit(lambda z0, I, th: simulate_nn(m, net, z0, I, th, 1))(z0, I, th)), ("net1c", jax.jit(sim_net1c)(z0, I, th))]:
        V = np.asarray(out["V"]); d = 1e3 * (V - Vw); res = np.asarray(out["resF"])
        rows.append(dict(closure=nm, case=CASE, steps=len(V), dV_rms_vs_warm_mV=float(np.sqrt(np.mean(d ** 2))), dV_max_abs_mV=float(np.abs(d).max()), dV_end_mV=float(d[-1]),
                         resF_median=float(np.median(res)), resF_p99=float(np.quantile(res, .99)), resF_max=float(res.max()), V_rms_vs_PyBaMM_mV=1e3 * float(np.sqrt(np.mean((V - r["V"]) ** 2)))))
    rows.append(dict(closure="warm", case=CASE, steps=len(Vw), dV_rms_vs_warm_mV=0.0, dV_max_abs_mV=0.0, dV_end_mV=0.0, resF_median=float(np.median(w["resF"])), resF_p99=float(np.quantile(w["resF"], .99)),
                     resF_max=float(np.max(w["resF"])), V_rms_vs_PyBaMM_mV=1e3 * float(np.sqrt(np.mean((Vw - r["V"]) ** 2)))))
    D = pd.DataFrame(rows); cli.write_csv(D, args, "ablation_openloop3C.csv", note="open loop, no filter, true parameters; dV = closure minus converged Newton with warm start; grid 20/10/20, N_m = 12", float_format=None)
    print(D.round(4).to_string(), flush=True)

    # (2) Newton iterations along F6 at th2 (true theta, trajectory of the warm-started solver)
    r = refio.load("F6", "th2", period=0.5, data_dir=args.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]]); I = jnp.asarray(r["I"]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]])
    if args.quick: I = I[:QUICK_STEPS]
    consts = m.part_consts(th); em = m.elec_mats(th); csn, csp, ce = m.surf(z0); g0 = A_.newton(A_.guess(csn, csp, I[0], th), csn, csp, ce, I[0], th)[0]

    def body(carry, Ik):
        z, g = carry; csn, csp, ce = m.surf(z)
        gc, rc, kc = A_.newton(A_.guess(csn, csp, Ik, th), csn, csp, ce, Ik, th, tol=1e-10)                  # cold
        gw, rw, kw = A_.newton(g, csn, csp, ce, Ik, th, tol=1e-10)                                           # warm, one attempt
        gs, rs, ks = A_.solve_robust(g, csn, csp, ce, Ik, th, tol=1e-10)                                      # warm with fallback
        j = net(csn, csp, ce, Ik, th); g1, _, _ = m.g_hat(j, csn, csp, ce, Ik, th); g1, r1 = A_.newton_fixed(g1, csn, csp, ce, Ik, th, k=1)
        kcomp = jax.lax.cond(r1 > THRESHOLD, lambda g_: A_.newton(g_, csn, csp, ce, Ik, th, tol=1e-10)[2], lambda g_: jnp.asarray(0), g1)
        return (m.advance(z, gs, th, consts, em), gs), (kc, rc, kw, rw, ks, rs, r1, kcomp)
    _, out = jax.lax.scan(body, (z0, g0), I); kc, rc, kw, rw, ks, rs, r1, kcomp = map(np.asarray, out)
    it = []
    for nm, k_, extra in [("cold", kc, dict(frac_not_converged_1e10=float(np.mean(rc > 1e-10)))), ("warm (one attempt)", kw, dict(frac_not_converged_1e10=float(np.mean(rw > 1e-10)))),
                          ("warm with fallback", ks, dict(frac_not_converged_1e10=float(np.mean(rs > 1e-10)), frac_fallback=float(np.mean(rw > 1e-10)))),
                          ("network + 1 Newton + completion", 1 + kcomp, dict(frac_completed=float(np.mean(r1 > THRESHOLD)))), ("net0", np.zeros_like(kc), {})]:
        it.append(dict(closure=nm, steps=len(k_), it_median=float(np.median(k_)), it_p99=float(np.quantile(k_, .99)), it_max=int(np.max(k_)), **extra))
    Dit = pd.DataFrame(it); cli.write_csv(Dit, args, "ablation_iters.csv", note="Newton iterations per step along F6 at th2 with the true parameters (open loop); completion = converged Newton when the residual after one iteration exceeds 1e-4", float_format=None)
    print(Dit.round(4).to_string(), flush=True)
    clock.log()


# ----------------------------------------------------------------------------------------------------------------------- cost
COST_VARIANTS = [("net0", "net0", 1e-4, "network alone"), ("net1", "net1", 1e-4, "network + 1 Newton + completion of the linearization"),
                 ("net1c", "net1c", 1e-4, "network + 1 Newton + completion (voltage, state and linearization)"),
                 ("net1_always", "net1", 0.0, "net1 with the completion always active (tol_lin = 0)"), ("net1c_always", "net1c", 0.0, "net1c with the completion always active (tol_lin = 0)"),
                 ("warm", "newton", 1e-4, "converged Newton, warm start and fallback"), ("cold", "cold", 1e-4, "converged Newton from a fixed guess")]


def cost(args):
    runtime.setup(args.core)
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from dfnekf import params as prm, timing, closure_net as cn
    from dfnekf.step import DFN
    from dfnekf.ekf import EKF
    if args.quick: args.results = os.path.join(args.results, "quick")
    clock = cli.Clock(args, "closure_ablation.py")
    n_calls = 30 if args.quick else args.n_calls; variants = [v for v in COST_VARIANTS if v[0] in ("net0", "warm")] if args.quick else COST_VARIANTS
    if not os.path.exists(net_path(args)): raise SystemExit(f"network file {net_path(args)} not found: run scripts/train_closure.py first (or give --net)")
    net = cn.load_net(net_path(args), args.grid)[0]
    n_other = timing.idle_warning(); meta = timing.machine_info(); rows = []
    note = f"CPU, one thread, JAX with Python/XLA dispatch, grid 20/10/20 {args.grid}, N = 653, N_m = 12; not BMS hardware; busy other processes: {n_other}"
    for name, closure, tol, desc in variants:
        m = DFN(prm.base(), 20, 10, 20, 12, 2, 0.5, args.grid); ek = EKF(m, net=net, closure=closure, coords=("Rf", "LAM", "Ds"), tol_lin=tol); nel = m.nel
        z0 = np.asarray(m.init_state(prm.C0_N_50, prm.C0_P_50)); x0 = np.concatenate([z0, ek.coord_of(np.array([4.0, 0.1, 0.55]))])
        P0d = np.concatenate([np.full(nel, 10.0), np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0), np.full(3, 1.0)]); Qd = np.concatenate([np.zeros(m.nz), np.full(3, 1e-3)])
        step, scale = ek.make_step(x0, P0d, Qd, 4e-6); th_full = ek.th_of(jnp.asarray(x0[m.nz:])); g0 = ek.init_g(jnp.asarray(x0[:m.nz]), 5.0, th_full)
        carry = (jnp.zeros(ek.n), jnp.diag((jnp.asarray(P0d) / scale) ** 2), g0); f = jax.jit(lambda c, I, y: step(c, (I, y))[0])
        Ld = jax.jit(lambda: ek.f_and_lin(jnp.asarray(x0), 5.0, g0))(); res = float(Ld["res"]); need = bool(Ld["need"])
        t = timing.time_calls(f, (carry, 5.0, 3.7), n_calls)
        rows.append(dict(closure=name, description=desc, grid=f"20-10-20 {args.grid}", Nm=12, N=ek.n, kind="per call", **timing.stats(t), resF_at_test_point=res, completion_active_at_test_point=need, note=note)); print(rows[-1], flush=True)
        cli.write_csv(pd.DataFrame(rows), args, "ablation_cost.csv", note="median wall time of one jitted filter step (I = 5 A, y = 3.7 V), identical initial state for all closures", float_format=None)
    cli.write_csv(pd.DataFrame([dict(n_busy_other_processes=n_other, **meta)]), args, "ablation_cost_meta.csv", note="machine description of the cost measurement", float_format=None)
    clock.log()


# ----------------------------------------------------------------------------------------------------------------------- aggregation
def aggregate(args):
    import numpy as np, pandas as pd
    from dfnekf import params as prm
    res = args.results; tab = os.path.join(res, "tables"); os.makedirs(tab, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(res, "ablation_*_s*.csv"))):
        mt = re.match(r"ablation_(net0|net1|net1c|warm|cold)_s(\d+)\.csv$", os.path.basename(f))
        if not mt: continue
        r = pd.read_csv(f, comment="#").iloc[0].to_dict(); cl, seed = r["closure"], int(r["seed"])
        H = np.load(os.path.join(res, "traj", f"ablation_{cl}_s{seed}.npz")); Wp = os.path.join(res, "traj", f"ablation_warm_s{seed}.npz")
        if os.path.exists(Wp):
            W = np.load(Wp); ok = np.isfinite(H["V"]) & np.isfinite(W["V"]); d = 1e3 * (H["V"][ok] - W["V"][ok])
            r["dV_rms_vs_warm_mV"] = float(np.sqrt(np.mean(d ** 2))); r["dV_max_abs_vs_warm_mV"] = float(np.abs(d).max())
        else: r["dV_rms_vs_warm_mV"] = r["dV_max_abs_vs_warm_mV"] = np.nan
        r["csn_max_frac_of_cmax"] = float(np.nanmax(H["csn_max"]) / prm.base()["cmax_n"]); rows.append(r)
    if not rows: raise SystemExit(f"no ablation_<closure>_s<seed>.csv in {res}")
    D = pd.DataFrame(rows); D["net"] = D.closure.map(NET_TEXT); D["o"] = D.closure.map(ORDER); D = D.sort_values(["o", "seed"]).drop(columns="o")
    cli.write_csv(D, args, "tables/ablation_runs.csv", note="one row per run; dV = closed-loop voltage difference from the warm-start run of the same seed. " + NOTE_TAIL, float_format="%.10g")
    K = pd.read_csv(os.path.join(res, "ablation_cost.csv"), comment="#") if os.path.exists(os.path.join(res, "ablation_cost.csv")) else None
    IT = pd.read_csv(os.path.join(res, "ablation_iters.csv"), comment="#") if os.path.exists(os.path.join(res, "ablation_iters.csv")) else None
    itmap = {"net0": "net0", "net1": "network + 1 Newton + completion", "net1c": "network + 1 Newton + completion", "warm": "warm with fallback", "cold": "cold"}
    T = []
    for cl in [c for c in CLOSURES if c in set(D.closure)]:
        g = D[D.closure == cl]; e = {}
        for p in ["Rf", "LAM", "Ds"]:
            for w in ["4320", "end"]:
                x = g[f"{p}_err_{w}_pct"].values; e[f"{p}_err_{w}_pct_mean"] = x.mean(); e[f"{p}_err_{w}_pct_sd"] = x.std(ddof=1) if len(x) > 1 else np.nan; e[f"{p}_err_{w}_pct_maxabs"] = np.abs(x).max()
        it = IT[IT.closure == itmap[cl]] if IT is not None else None; kk = K[K.closure == cl] if K is not None else None
        T.append(dict(closure=cl, net=NET_TEXT[cl], n_runs=len(g), seeds=",".join(str(int(s)) for s in g.seed), dV_rms_vs_warm_mV_mean=g.dV_rms_vs_warm_mV.mean(), dV_rms_vs_warm_mV_max=g.dV_rms_vs_warm_mV.max(),
                      innov_rms_mV_mean=g.innov_rms_mV.mean(), resF_median=g.resF_median.median(), resF_p99=g.resF_p99.mean(), resF_max=g.resF_max.max(), frac_steps_res_gt_1em4=g.frac_res_gt_1em4.mean(),
                      frac_fallback_warm=g.frac_fallback_warm.mean() if cl == "warm" else np.nan,
                      newton_it_median_openloop=it.it_median.iloc[0] if it is not None and len(it) else np.nan, newton_it_p99_openloop=it.it_p99.iloc[0] if it is not None and len(it) else np.nan,
                      ms_per_step_median=kk.median_ms.iloc[0] if kk is not None and len(kk) else np.nan, ms_per_step_p99=kk.p99_ms.iloc[0] if kk is not None and len(kk) else np.nan,
                      n_rejected_total=int(g.n_rejected.sum()), n_finite=int(g.finite_all.sum()), **e))
    T = pd.DataFrame(T)
    cli.write_csv(T, args, "tables/tab_ablation_m17.csv", note="Table VIII: one row per closure, mean over the seeds (errors: estimate minus truth, % of the ageing range, signed, at t = 4320 s and at the end). " + NOTE_TAIL, float_format="%.6g")
    print(T.round(3).T.to_string())
    # numbers quoted in Appendix E
    def mean(cl, col): g = D[D.closure == cl]; return float(g[col].mean()) if len(g) else np.nan
    def mx(cl, col): g = D[D.closure == cl]; return float(g[col].max()) if len(g) else np.nan
    tn = []
    def add(name, value, text): tn.append(dict(quantity=name, value=value, quoted_as=text))
    add("dV_rms_net0_vs_warm_mV (mean over seeds)", mean("net0", "dV_rms_vs_warm_mV"), "0.53 mV (Step 1, Table VIII)")
    add("dV_rms_net1c_vs_warm_mV", mean("net1c", "dV_rms_vs_warm_mV"), "0.004 mV (Step 2, Table VIII)")
    add("dV_rms_net1_vs_warm_mV", mean("net1", "dV_rms_vs_warm_mV"), "row 'net1' (completion of the linearization only)")
    add("completion fraction net1c (steps with residual > 1e-4 after one iteration)", mean("net1c", "frac_res_gt_1em4"), "11.3 %")
    add("completion fraction net1", mean("net1", "frac_res_gt_1em4"), "11.3 % (same decision as net1c)")
    add("fallback fraction warm (Newton from the warm start failed, cold restart)", mean("warm", "frac_fallback_warm"), "0.11 % (Table VIII, dagger)")
    add("resF_max warm, max over seeds", mx("warm", "resF_max"), "residual below 1e-10 with the converged solver")
    add("resF_max cold, max over seeds", mx("cold", "resF_max"), "residual below 1e-10 with the converged solver")
    add("resF_p99 after one iteration from the network (net1, mean over seeds)", mean("net1", "resF_p99"), "99th percentile 3e-3")
    add("resF_p99 network alone (net0, mean over seeds)", mean("net0", "resF_p99"), "0.40")
    for p in ["Rf", "LAM", "Ds"]:
        for cl in ["net0", "net1", "net1c"]:
            add(f"final error {p}, {cl} minus warm (% of range, mean over seeds)", mean(cl, f"{p}_err_end_pct") - mean("warm", f"{p}_err_end_pct"), "network alone: shift of about 1 % each; net1c: less than 0.02 %" if cl != "net1" else "")
        add(f"final error {p}, warm (% of range, mean over seeds)", mean("warm", f"{p}_err_end_pct"), "-1.4, -0.1, -0.4 with the converged solver (Table VIII)")
    if K is not None and len(K):
        ms = {c: float(K[K.closure == c].median_ms.iloc[0]) for c in K.closure}
        if "net0" in ms and "warm" in ms: add("step cost net0 relative to warm (%)", 100 * (ms["net0"] / ms["warm"] - 1), "the network alone saves 5 %")
        if "net1c" in ms and "warm" in ms: add("step cost net1c relative to warm (%)", 100 * (ms["net1c"] / ms["warm"] - 1), "+3 % (7.61 vs 7.42 ms)")
        for c, v in ms.items(): add(f"step cost {c} (ms, median)", v, "Table VIII, column Step")
    add("guard rejections, all runs and closures", int(D.n_rejected.sum()), "the guard never rejected an update on the protocols of the paper")
    add("max anode surface concentration / c_max,n of the warm runs (F6 at th2)", mx("warm", "csn_max_frac_of_cmax"), "anode surface below 77 % of the maximum comes from the F6Q static tests (scripts/charge.py, tables/charge_validity.csv)")
    ol = os.path.join(res, "ablation_openloop3C.csv")
    if os.path.exists(ol):
        O = pd.read_csv(ol, comment="#"); add("open loop 3C pulse, dV_rms net0 vs warm (mV)", float(O[O.closure == "net0"].dV_rms_vs_warm_mV.iloc[0]), "1.5 mV on a 3C pulse in open loop (Table VIII, footnote)")
    if IT is not None and "frac_completed" in IT: add("open loop F6, completion fraction", float(IT.frac_completed.dropna().iloc[0]), "11 % of the steps along the true trajectory")
    cli.write_csv(pd.DataFrame(tn), args, "tables/ablation_text_numbers.csv", note="numbers quoted in Appendix E (Table VIII text). " + NOTE_TAIL, float_format="%.6g")


def main():
    p = cli.parser("Appendix E, Table VIII (learned-closure ablation)", "filter runs with five closures on F6 at th2, open-loop checks, step cost and the table", seeds="0-9", states="th2")
    p.add_argument("--closures", default=",".join(CLOSURES), help="comma-separated subset of " + ", ".join(CLOSURES))
    p.add_argument("--net", default=None, help="trained network (default <results>/closure_4to1.pkl)")
    p.add_argument("--resume", action="store_true", help="skip runs whose trajectory file exists")
    p.add_argument("--openloop", action="store_true", help="open-loop checks and Newton iteration statistics instead of filter runs")
    p.add_argument("--cost", action="store_true", help="step-cost measurement (idle machine, one core)")
    p.add_argument("--n-calls", type=int, default=3600, help="calls per closure in --cost")
    p.add_argument("--aggregate", action="store_true", help="build Table VIII from the per-run files (no simulation)")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    if a.openloop: return openloop(a)
    if a.cost: return cost(a)
    run_jobs(a)


if __name__ == "__main__":
    main()
