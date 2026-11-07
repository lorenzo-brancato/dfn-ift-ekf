"""Appendix A: checks of the grey-box model against PyBaMM and against analytic or convergence arguments.

Parts (--parts, default all):
  algebra    (a) closure against PyBaMM's algebraic solution on the same 4:1 mesh (20/10/20, 300 radial nodes, period 1 s): 50 sampled states per protocol
             (P1_1C, P1_3C [30 states], P2, P6L) and theta in {nominal, th4}, 360 states in all (rng seed 0): relative error of the reaction currents (L2),
             of phi_e, and of the voltage (eV_rel, RELATIVE); analytic residual Jacobian against central differences.   [paper: 1.2e-5 in j_n, 8e-5 in V]
             Also the mapping check at the corners of the ageing range (one parameter at a time at its limit, uniform 20/10/20 mesh, 3 protocols x 17 states);
             the largest relative difference is reported as it is (the paper rounds it to 1e-3).
  vt         (b) variable electrolyte transport (dfnekf.electrolyte): algebra against PyBaMM with variable transport, 3C 30 s closed-loop voltage error at th2 with variable and
             with constant transport (0.74 vs 1.45 mV), anode overpotential error, minimum electrolyte concentration error (0.25 % vs 36 %), structured Jacobian against jacfwd.
  lithium    (c) lithium conservation drift over the whole P6Lx2 run with the converged closure and with ONE Newton iteration per step, for the states of --states.
  statements (d) four statements without a dedicated script before: '12 modes give < 0.3 mV' (open loop, N_m = 4, 8, 12, 16 against a 600-mode particle),
             'about 250 modes at t* = 1e-4, 3 at t* = 1' (analytic, lambda_M^2 t* >= 60), 'coarser radial grids err by up to 20 mV in the first second of a 3C pulse'
             (needs PyBaMM: --with-pybamm; 30, 100 and 300 radial nodes), 'with constant transport a 3C pulse cannot last beyond about 55 s' (open loop until failure).
The grid of the grey-box model is --grid (the paper's checks use 4:1 for algebra, vt, lithium and statements; the corner check always uses the uniform mesh of its references).
Outputs in <results>: model_checks_gateA_states.csv, model_checks_gateA.csv, model_checks_corners.csv, model_checks_vt.csv, model_checks_lithium.csv, model_checks_statements.csv.
--aggregate writes <results>/tables/tab_model_checks.csv (one row per number quoted in Appendix A).

Commands (algebra about 3 min, corners 1 min, vt 5 min, lithium 6 min for 5 states, statements 3 min + PyBaMM 3 min):
  python scripts/model_checks.py --grid 4:1 --parts algebra,vt,lithium,statements --data data/sim --results results
  python scripts/model_checks.py --grid 4:1 --parts statements --with-pybamm --data data/sim --results results
  python scripts/model_checks.py --aggregate --results results
  python scripts/model_checks.py --quick --results /tmp/checks_quick
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime


def relj(a, b):
    import numpy as np
    b = np.asarray(b); return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 0.01 * 1.49 * np.sqrt(len(b))))


def algebra_state_errors(A, r, i, th, newton, jacf, resf, with_jac=True):
    """Errors of the closure at sample i of the reference r (state, current from the reference)."""
    import numpy as np
    import jax.numpy as jnp
    csn, csp, ce, I = (jnp.asarray(r[k][i]) for k in ("cs_n", "cs_p", "ce", "I"))
    g, nr, k = newton(A.guess(csn, csp, I, th), csn, csp, ce, I, th)
    jn, jp, pe, sn, sp = A.unpack(g); V = float(A.V_of(g, I)); pe_ref = r["phi_e"][i]
    out = dict(ej_n=relj(jn, r["j_n"][i]), ej_p=relj(jp, r["j_p"][i]), ephi_e=float(np.linalg.norm(np.asarray(pe) - pe_ref) / np.linalg.norm(pe_ref)), dphi_e_mV=1e3 * float(np.abs(np.asarray(pe) - pe_ref).max()),
               eV_rel=abs(V - float(r["V"][i])) / abs(float(r["V"][i])), dV_mV=1e3 * abs(V - float(r["V"][i])), newton_res=float(nr))
    if with_jac:                                                       # analytic Jacobian (at the reference solution) against central differences
        gref = A.pack(*(jnp.asarray(r[k_][i]) for k_ in ["j_n", "j_p", "phi_e", "phi_s_n", "phi_s_p"]))
        J = np.asarray(jacf(gref, csn, csp, ce, I, th)); Jfd = np.zeros_like(J)
        for c in range(len(gref)):
            h = 1e-6 * max(abs(float(gref[c])), 1e-3); e = jnp.zeros_like(gref).at[c].set(h)
            Jfd[:, c] = np.asarray(resf(gref + e, csn, csp, ce, I, th) - resf(gref - e, csn, csp, ce, I, th)) / (2 * h)
        out["jac_err"] = float(np.linalg.norm(J - Jfd) / np.linalg.norm(J))
    return out


def main():
    p = cli.parser("Appendix A (model checks)", "closure against PyBaMM, variable electrolyte transport, lithium conservation, four supporting statements", states="th1,th2,th3,th4,th5")
    p.add_argument("--parts", default="algebra,vt,lithium,statements", help="comma-separated: algebra, vt, lithium, statements")
    p.add_argument("--with-pybamm", action="store_true", help="statements: also run PyBaMM for the radial-node statement (needs pybamm)")
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.aggregate: return aggregate(a)
    runtime.setup(a.core)
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from dfnekf import params as prm, refio, reference
    from dfnekf.closure import Alg
    from dfnekf.grid import ratio_of
    from dfnekf.step import DFN, interface_ref
    from dfnekf.rom import simulate
    clock = cli.Clock(a, "model_checks.py"); parts = [s.strip() for s in a.parts.split(",")]
    P0 = prm.base(); dt = 0.5; Q = a.quick
    suffix = "_quick" if Q else ""
    W = lambda df, name, note="": cli.write_csv(df, a, name.replace(".csv", suffix + ".csv"), note=note)

    # ------------------------------------------------------------------ (a) closure against the PyBaMM algebraic solution
    if "algebra" in parts:
        if a.grid != "4:1": print("algebra gate: the references on the mesh of the filter exist for the 4:1 grid only; running the corner check only", flush=True)
        else:
            A = Alg(P0, 20, 10, 20, "linear", "inv_mean", ratio_of(a.grid)); assert A.dxn[0] / A.dxn[-1] > 3.9
            newton = jax.jit(A.newton); jacf = jax.jit(A.jac); resf = jax.jit(A.residual)
            rng = np.random.default_rng(0); rows = []
            for tag in ["nom", "th4"]:
                for proto in ["P1_1C", "P1_3C", "P2", "P6L"]:
                    r = refio.load(proto, tag, mesh=(20, 10, 20), period=1.0, data_dir=a.data, grid_ratio=4.0); n = r["N"]            # mesh tag 20-10-20nu4: PyBaMM on the same 4:1 mesh
                    th = jnp.array([r["Rf"], 1 - r["LAM"], r["Ds"]])
                    sel = np.concatenate([[0], rng.choice(np.arange(1, n), min(49, n - 1), replace=False)])
                    if Q: sel = sel[:3] if (tag == "nom" and proto == "P1_3C") else sel[:0]
                    for i in sel: rows.append(dict(theta=tag, proto=proto, i=int(i), **algebra_state_errors(A, r, int(i), th, newton, jacf, resf)))
                    print("algebra", tag, proto, "done", flush=True)
            df = pd.DataFrame(rows); W(df, "model_checks_gateA_states.csv", "closure against PyBaMM, 4:1 mesh"); out = []
            for tg, g in df.groupby("theta"):
                for nm, col, thr in [("j_n rel L2", "ej_n", 1e-3), ("j_p rel L2", "ej_p", 1e-3), ("phi_e rel L2", "ephi_e", 1e-3), ("V rel", "eV_rel", 1e-3), ("Jacobian vs FD", "jac_err", 1e-6)]:
                    out.append(dict(gate="closure vs PyBaMM algebra (4:1)", theta=tg, quantity=nm, value_max=float(g[col].max()), threshold=thr, passed=bool(g[col].max() < thr), n=len(g)))
                out.append(dict(gate="closure vs PyBaMM algebra (4:1)", theta=tg, quantity="info: V max abs [mV]", value_max=float(g.dV_mV.max()), threshold=np.nan, passed=True, n=len(g)))
                out.append(dict(gate="closure vs PyBaMM algebra (4:1)", theta=tg, quantity="info: phi_e max abs [mV]", value_max=float(g.dphi_e_mV.max()), threshold=np.nan, passed=True, n=len(g)))
            o = pd.DataFrame(out); W(o, "model_checks_gateA.csv", f"{len(df)} sampled states"); print(o.to_string(), flush=True)
        # mapping at the corners of the ageing range, uniform mesh
        A = Alg(P0, 20, 10, 20, "linear", "inv_mean", ratio_of("uniform")); newton = jax.jit(A.newton); jacf = jax.jit(A.jac); resf = jax.jit(A.residual)
        rng = np.random.default_rng(1); rowsA = []
        for tag in (["bRf"] if Q else ["bRf", "bLAM", "bLLI", "bDs"]):
            agg = []
            for proto in (["P1_3C"] if Q else ["P1_3C", "P2", "P6L"]):
                r = refio.load(proto, tag, mesh=(20, 10, 20), period=1.0, data_dir=a.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]])
                sel = np.concatenate([[0], rng.choice(r["N"], 16, replace=False)]); sel = sel[:3] if Q else sel
                for i in sel:
                    e = algebra_state_errors(A, r, int(i), th, newton, jacf, resf, with_jac=False); agg.append(dict(ej_n=e["ej_n"], ej_p=e["ej_p"], ephi_e=e["ephi_e"], eV=e["eV_rel"]))
            d = pd.DataFrame(agg)
            for c, nm in [("ej_n", "j_n"), ("ej_p", "j_p"), ("ephi_e", "phi_e"), ("eV", "V")]:
                rowsA.append(dict(gate="static algebra at the corners (uniform mesh)", theta=tag, quantity=nm, value_max=float(d[c].max()), threshold=1e-3, passed=bool(d[c].max() <= 1e-3), n=len(d)))
        o = pd.DataFrame(rowsA); W(o, "model_checks_corners.csv", "relative errors; corner theta = one parameter at its limit"); print(o.to_string(), flush=True)

    # ------------------------------------------------------------------ (b) variable electrolyte transport
    if "vt" in parts:
        from dfnekf.electrolyte import DFNVT, simulate_vt, StructVT
        m = DFNVT(P0, 20, 10, 20, 12, 2, dt, a.grid); A = m.A; rows = []
        r = refio.load("P1_3C", "nomVT", mesh=(20, 10, 20), period=1.0, data_dir=a.data, grid_ratio=4.0); n = r["N"]; newton = jax.jit(A.newton); th = jnp.array([1.0, 1.0, 1.0]); aggs = []
        for i in range(3 if Q else n):
            csn, csp, ce, I = (jnp.asarray(r[k][i]) for k in ("cs_n", "cs_p", "ce", "I"))
            g, nr, k = newton(A.guess(csn, csp, I, th), csn, csp, ce, I, th); jn, jp, pe, sn, sp = A.unpack(g); V = float(A.V_of(g, I))
            aggs.append(dict(ej_n=relj(jn, r["j_n"][i]), ej_p=relj(jp, r["j_p"][i]), ephi=float(np.linalg.norm(np.asarray(pe) - r["phi_e"][i]) / np.linalg.norm(r["phi_e"][i])), eV=abs(V - float(r["V"][i])) / abs(float(r["V"][i]))))
        a_ = pd.DataFrame(aggs); rows.append(dict(gate="algebra, variable transport (4:1)", **{c: float(a_[c].max()) for c in a_.columns}, threshold=1e-3)); print(rows[-1], flush=True)
        r = refio.load("P1_3C", "th2VT", period=dt, data_dir=a.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]])
        out = jax.jit(lambda z0, I, th: simulate_vt(m, z0, I, th))(z0, jnp.asarray(r["I"]), th)
        mc = DFN(P0, 20, 10, 20, 12, 2, dt, a.grid); oc = jax.jit(lambda z0, I, th: simulate(mc, z0, I, th, k=None))(z0, jnp.asarray(r["I"]), th)
        eta_ref = np.array([interface_ref(r, i, P0, r["Rf"]) for i in range(r["N"])]); ce_ref = r["ce"].min(1)
        for lab, o_ in [("variable transport (per-step linearization)", out), ("constant transport", oc)]:
            Vx = np.asarray(o_["V"]); e_lit = np.asarray(o_["eta_lit"]) - eta_ref[:, 0]; cem = np.asarray(o_["ce"]).min(1)
            rows.append(dict(gate="closed loop th2 3C 30 s (4:1)", model=lab, E_V_rms_mV=1e3 * np.sqrt(np.mean((Vx - r["V"]) ** 2)), E_V_max_mV=1e3 * np.abs(Vx - r["V"]).max(), E_eta_rms_mV=1e3 * np.sqrt(np.mean(e_lit ** 2)),
                             ce_min_relerr_max_pct=float(100 * np.max(np.abs(cem - ce_ref) / ce_ref)), ce_min_ref=float(ce_ref.min()), thr_mV=3.0)); print(rows[-1], flush=True)
        S = StructVT(m)

        def Phi_V(z, th, I):
            csn, csp, ce = m.surf(z); g = A.solve_robust(A.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=1e-13)[0]
            return jnp.concatenate([m.advance_vt(z, g, th), A.V_of(g, I)[None]])
        for k in ([10] if Q else [10, 30, 55]):
            z = jax.jit(lambda z0, I, th: simulate_vt(m, z0, I, th))(z0, jnp.asarray(r["I"][:k]), th)["zT"]; I = float(r["I"][k]); csn, csp, ce = m.surf(z)
            g = A.solve_robust(A.guess(csn, csp, I, th), csn, csp, ce, I, th, tol=1e-12)[0]
            Jz, Jt = [np.asarray(x) for x in jax.jit(S.jac_full_vt)(z, I, th, g)]
            Jz_ref = np.asarray(jax.jit(jax.jacfwd(Phi_V, 0))(z, th, I)); Jt_ref = np.asarray(jax.jit(jax.jacfwd(Phi_V, 1))(z, th, I)); rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
            rows.append(dict(gate="structured Jacobian vs jacfwd, variable transport (4:1)", k=k, dPhi_dz_rel=rel(Jz, Jz_ref), dPhi_dtheta_rel=rel(Jt, Jt_ref), threshold=1e-6)); print(rows[-1], flush=True)
        W(pd.DataFrame(rows), "model_checks_vt.csv", "variable electrolyte transport")

    # ------------------------------------------------------------------ (c) lithium conservation over P6Lx2
    if "lithium" in parts:
        m = DFN(P0, 20, 10, 20, 12, 2, dt, a.grid); rows = []
        for tag in (cli.states(a)[:1] if Q else cli.states(a)):
            r = refio.load("P6Lx2", tag, period=dt, data_dir=a.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]]); z0 = m.init_state(*[float(x) for x in r["c0_eff"]])
            ns = 600 if Q else r["N"]; I = jnp.asarray(r["I"][:ns])
            for k, lab in [(None, "converged"), (1, "one Newton iteration")]:
                out = jax.jit(lambda z0, I, th: simulate(m, z0, I, th, k=k))(z0, I, th); li = np.asarray(out["li"]); V = np.asarray(out["V"])
                rows.append(dict(theta=tag, proto="P6Lx2", closure=lab, steps=ns, lithium_drift_max=float(np.nanmax(np.abs(li / li[0] - 1))), E_V_rms_mV=1e3 * float(np.sqrt(np.mean((V - r["V"][:ns]) ** 2))),
                                 resF_max=float(np.nanmax(out["resF"])), finite=bool(np.all(np.isfinite(V)))))
                print(rows[-1], flush=True)
        W(pd.DataFrame(rows), "model_checks_lithium.csv", "drift = max |Li(t)/Li(0) - 1|")

    # ------------------------------------------------------------------ (d) the four supporting statements
    if "statements" in parts:
        rows = []
        add = lambda statement, quantity, value, unit, paper, note="": rows.append(dict(statement=statement, quantity=quantity, value=value, unit=unit, paper_value=paper, note=note))
        # d1: truncation of the particle modes, open loop against a 600-mode particle (the tail beyond 600 modes enters through its exact DC gain)
        tag = "th2"; ref_models = {}
        for proto in (["P1_3C"] if Q else ["P1_3C", "P6Lx2"]):
            r = refio.load(proto, tag, period=dt, data_dir=a.data); th = jnp.array([r["Rf"], r["as_"], r["Ds"]]); I = jnp.asarray(r["I"]); c0 = [float(x) for x in r["c0_eff"]]; Vs = {}
            for Nm, K in [(600, 1)] + [(nm, 2) for nm in ((12,) if Q else (4, 8, 12, 16))]:
                mm = DFN(P0, 20, 10, 20, Nm, K, dt, a.grid); z0 = mm.init_state(*c0)
                Vs[Nm] = np.asarray(jax.jit(lambda z0, I, th: simulate(mm, z0, I, th, k=None)["V"])(z0, I, th)); print("modes", proto, Nm, flush=True)
            for Nm in [k for k in Vs if k != 600]:
                e = Vs[Nm] - Vs[600]
                add("12 modes give < 0.3 mV", f"open-loop voltage difference, N_m={Nm} + 2 tail bands vs 600 modes, {proto} th2: max abs", 1e3 * float(np.abs(e).max()), "mV", "< 0.3 (N_m=12)")
                add("12 modes give < 0.3 mV", f"open-loop voltage difference, N_m={Nm} + 2 tail bands vs 600 modes, {proto} th2: rms", 1e3 * float(np.sqrt(np.mean(e ** 2))), "mV", "")
        # d2: number of modes for lambda_M^2 t* >= 60 (analytic)
        from scipy.optimize import brentq
        lam = lambda n: brentq(lambda x: np.tan(x) - x, n * np.pi + 1e-9, n * np.pi + np.pi / 2 - 1e-9)
        for ts, paper in [(1e-4, "about 250"), (1.0, "3")]:
            M = next(n for n in range(1, 5000) if lam(n) ** 2 * ts >= 60.0)
            add("modes needed: lambda_M^2 t* >= 60", f"number of modes M at t* = {ts:g}", M, "modes", paper, f"lambda_M = {lam(M):.2f}; exp(-60) = {np.exp(-60):.1e}; t* = {ts:g} is {ts * P0['Rn'] ** 2 / P0['Ds_n']:.3g} s for the anode (R^2/D = {P0['Rn'] ** 2 / P0['Ds_n']:.0f} s)")
        # d3: radial nodes of the PyBaMM reference
        if a.with_pybamm:
            res = {}
            for rn in ((30, 300) if Q else (30, 100, 300)):
                o = reference.run("P1_3C", theta=(1.0, 0.0, 0.0, 1.0), mesh=(20, 10, 20), period=0.5, rnodes=rn); res[rn] = (np.asarray(o["t"]), np.asarray(o["V"])); print("pybamm radial nodes", rn, f"{o['wall_s']:.0f} s", flush=True)
            t300, V300 = res[300]
            for rn in [k for k in res if k != 300]:
                t, V = res[rn]; n = min(len(V), len(V300)); d = np.abs(V[:n] - V300[:n]); tt = t300[:n]
                add("coarser radial grids err by up to 20 mV in the first second of a 3C pulse", f"{rn} vs 300 radial nodes, P1_3C nominal: error at t = 1 s", 1e3 * float(d[np.argmin(np.abs(tt - 1.0))]), "mV", "19.6 (30 nodes); 3.5 (100 nodes)")
                add("coarser radial grids err by up to 20 mV in the first second of a 3C pulse", f"{rn} vs 300 radial nodes: max abs error for t <= 1 s", 1e3 * float(d[tt <= 1.0 + 1e-9].max()), "mV", "up to 20")
                add("coarser radial grids err by up to 20 mV in the first second of a 3C pulse", f"{rn} vs 300 radial nodes: max abs error for 10 s <= t <= 30 s", 1e3 * float(d[(tt >= 10) & (tt <= 30 + 1e-9)].max()), "mV", "2.5-3.7 (30 nodes); 0.2-0.3 (100 nodes)")
        else:
            add("coarser radial grids err by up to 20 mV in the first second of a 3C pulse", "not run (needs PyBaMM: --with-pybamm)", float("nan"), "mV", "19.6 (30 nodes)")
        # d4: constant transport, 3C from the fully charged state, open loop until the electrolyte is depleted
        mm = DFN(P0, 20, 10, 20, 12, 2, dt, a.grid); z0 = mm.init_state(prm.C0_N_100, prm.C0_P_100); T = 30 if Q else 90; I = jnp.full(int(T / dt), 3 * prm.C_RATE_A); th = jnp.array([1.0, 1.0, 1.0])
        out = jax.jit(lambda z0, I, th: simulate(mm, z0, I, th, k=None))(z0, I, th); ce_min = np.asarray(out["ce"]).min(1); V = np.asarray(out["V"]); resF = np.asarray(out["resF"]); t = np.arange(len(I)) * dt
        first = lambda mask: float(t[np.argmax(mask)]) if mask.any() else float("nan")
        add("constant transport: a 3C pulse cannot last beyond about 55 s", "first time with c_e,min < 100 mol/m3 (3C from SOC 100 %, nominal theta, open loop)", first(ce_min < 100), "s", "about 55")
        add("constant transport: a 3C pulse cannot last beyond about 55 s", "first time with c_e,min < 20 mol/m3 (the filter's clip value)", first(ce_min < 20), "s", "about 55")
        add("constant transport: a 3C pulse cannot last beyond about 55 s", "first time with c_e,min < 1 mol/m3", first(ce_min < 1), "s", "about 55")
        add("constant transport: a 3C pulse cannot last beyond about 55 s", "first time the closure fails (residual > 1e-6 or non-finite voltage)", first((resF > 1e-6) | ~np.isfinite(V)), "s", "about 55", f"horizon {T} s; c_e,min at the end: {ce_min[-1]:.2f} mol/m3")
        W(pd.DataFrame(rows), "model_checks_statements.csv", "supporting statements of Appendix A")
        print(pd.DataFrame(rows)[["quantity", "value", "unit", "paper_value"]].to_string(), flush=True)
    clock.log()


def aggregate(a):
    import numpy as np
    import pandas as pd
    R = lambda n: (lambda f: pd.read_csv(f, comment="#") if os.path.exists(f) else None)(os.path.join(a.results, n))
    rows = []
    add = lambda item, value, paper, unit="", src="": rows.append(dict(item=item, value=value, paper_value=paper, unit=unit, source=src))
    g = R("model_checks_gateA.csv")
    if g is not None:
        for q, nm, paper in [("j_n rel L2", "relative error of the reaction current j_n (L2), max over states, nominal theta", "1.2e-5"), ("V rel", "relative voltage error eV_rel, max over states and theta", "8e-5")]:
            s = g[g.quantity == q]; s = s[s.theta == "nom"] if q.startswith("j_n") else s; add(nm, float(s.value_max.max()), paper, "", "model_checks_gateA.csv")
    gs = R("model_checks_gateA_states.csv")
    if gs is not None: add("number of sampled states (closure vs PyBaMM)", len(gs), "360")
    c = R("model_checks_corners.csv")
    if c is not None:
        i = c.value_max.idxmax(); add("largest relative difference at the corners of the ageing range", float(c.value_max.max()), "1e-3 (rounded)", "", f"{c.theta[i]}, {c.quantity[i]}; passed 1e-3: {bool(c.passed.all())}")
    v = R("model_checks_vt.csv")
    if v is not None:
        cl = v[v.gate.str.startswith("closed loop")]
        for _, r in cl.iterrows():
            tag = "variable" if r.model.startswith("variable") else "constant"
            add(f"3C 30 s closed-loop voltage error, {tag} transport", float(r.E_V_rms_mV), "0.74" if tag == "variable" else "1.45", "mV"); add(f"anode overpotential error, {tag} transport", float(r.E_eta_rms_mV), "0.06" if tag == "variable" else "", "mV")
            add(f"c_e,min relative error (max), {tag} transport", float(r.ce_min_relerr_max_pct), "0.25" if tag == "variable" else "36", "%")
        s1 = v[v.gate.str.startswith("structured")]
        if len(s1): add("structured Jacobian vs jacfwd, variable transport: max dPhi/dz", float(s1.dPhi_dz_rel.max()), "6e-13")
    l = R("model_checks_lithium.csv")
    if l is not None:
        for cl_, s in l.groupby("closure"): add(f"lithium drift max over P6Lx2 ({int(s.steps.max())} steps), {cl_}", float(s.lithium_drift_max.max()), "3e-13 (converged)", "", f"states {','.join(s.theta)}; finite: {bool(s.finite.all())}")
    st = R("model_checks_statements.csv")
    if st is not None:
        for _, r in st.iterrows(): add(f"{r.statement}: {r.quantity}", r.value, r.paper_value, r.unit, r.note if isinstance(r.note, str) else "")
    if not rows: raise SystemExit(f"no model_checks_*.csv in {a.results}")
    a2 = type("A", (), dict(item=a.item, grid=a.grid, results=os.path.join(a.results, "tables")))(); print(cli.write_csv(pd.DataFrame(rows), a2, "tab_model_checks.csv", note="numbers of Appendix A", float_format="%.6g"))


if __name__ == "__main__":
    main()
