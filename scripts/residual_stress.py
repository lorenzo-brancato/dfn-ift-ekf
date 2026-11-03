"""Fig. 11 and the residual-certificate numbers of Appendix E: where the learned closure should fail.

Stress case outside the ageing range: three 5C pulses of 10 s followed by 120 s of rest (protocol P5, 1000 steps of 0.5 s, initial SOC 90 % plus rest), film resistance 15 times nominal,
initial electrolyte concentration halved (500 mol/m^3), LAM = 0.3 (a_s = 0.7), D_s nominal. The true trajectory is the converged Newton solver with fallback; at every step the trained
network gives the reaction currents, one Newton iteration follows, and the residual ||F||_inf of the closure equations after that single iteration is recorded (the quantity of Fig. 11).
Where it exceeds 1e-4 the step is completed with Newton run to convergence ('completion', marks in the figure). For comparison the residual of the network alone and of the converged
solver along the same trajectory are recorded.

Outputs:
  <results>/residual_stress_series.csv        one row per step: t_s, I_A, residual_inf_norm (network + 1 Newton), fallback_active (completion), completion_newton_it, ce_min,
                                              threshold, residual_network_alone, residual_converged
  <results>/source_data/fig_residual_m17.csv  Fig. 11 data: t_s, residual_inf_norm, fallback_active, threshold (header comments as in the original file)
  <results>/tables/residual_stress_numbers.csv numbers quoted in the text: completed fraction (31 %), maximum residual, decades above the threshold ('four orders of magnitude'),
                                              longest time above the threshold ('up to a minute'), network alone, converged solver
The other numbers of the residual certificate of Appendix E come from the closure ablation (``scripts/closure_ablation.py --aggregate``, tables/ablation_text_numbers.csv): residual below
1e-10 with the converged solver on the protocols of the paper (warm/cold runs), 99th percentile 3e-3 after one iteration from the network, 0.40 with the network alone.

Commands (about 20 s):
  python scripts/residual_stress.py --grid 4:1 --results results [--net results/closure_4to1.pkl]
  python scripts/residual_stress.py --quick --grid 4:1 --results results          (first 300 steps: the first pulse; needs the quick network)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime                                             # no jax import here

THRESHOLD = 1e-4
N_STEPS, QUICK_STEPS, DT = 1000, 300, 0.5
STRESS = dict(Rf=15.0, LAM=0.3, Ds=1.0, ce0=500.0)
HEADER = ["# fig_residual_m17 (single panel). Infinity norm of the kinetics residual after the network (trained on the 4:1 grid) + one Newton iteration, over three 5C 10 s pulses (P5), "
          "case R_f x15, c_e,0/2, LAM 0.3; 4:1 grid 20/10/20, N_m = 12, dt = 0.5 s; 1000 steps.",
          "# fallback_active = 1 where the residual exceeds the 1e-4 threshold and Newton is completed. origin: scripts/residual_stress.py"]


def runs_above(flag, dt):
    """Lengths (s) and start indices of the contiguous runs of True in ``flag``."""
    import numpy as np
    f = np.r_[False, np.asarray(flag, bool), False]; d = np.diff(f.astype(int)); s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    return (e - s) * dt, s


def main():
    p = cli.parser("Fig. 11, Appendix E (residual certificate)", "residual of the learned closure after one Newton iteration in a stress case outside the ageing range")
    p.add_argument("--net", default=None, help="trained network (default <results>/closure_4to1.pkl)")
    a = p.parse_args()
    runtime.setup(a.core)
    import numpy as np, pandas as pd, jax, jax.numpy as jnp
    from dfnekf import params as prm, closure_net as cn
    from dfnekf.step import DFN
    from dfnekf.protocols import current_series
    if a.quick: a.results = os.path.join(a.results, "quick")
    net_file = a.net or os.path.join(a.results, "closure_4to1.pkl" if a.grid == "4:1" else "closure_uniform.pkl")
    if not os.path.exists(net_file): raise SystemExit(f"network file {net_file} not found: run scripts/train_closure.py first (or give --net)")
    clock = cli.Clock(a, "residual_stress.py")
    net = cn.load_net(net_file, a.grid)[0]
    m = DFN(prm.base(), 20, 10, 20, 12, 2, DT, a.grid); A_ = m.A
    n = QUICK_STEPS if a.quick else N_STEPS
    I_all = np.concatenate([current_series("P5", DT), np.zeros(300)])[:n]
    th = jnp.array([STRESS["Rf"], 1.0 - STRESS["LAM"], STRESS["Ds"]])
    z0 = m.init_state(prm.C0_N_100, prm.C0_P_100); c, a_, s, _ = m.split(z0); z0 = m.join(c, a_, s, jnp.full(m.N, STRESS["ce0"]))
    consts = m.part_consts(th); em = m.elec_mats(th); csn, csp, ce_ = m.surf(z0); g0 = A_.newton(A_.guess(csn, csp, I_all[0], th), csn, csp, ce_, I_all[0], th)[0]

    def body(carry, I):
        z, g = carry; csn, csp, ce = m.surf(z); ce = jnp.maximum(ce, 20.0); j = net(csn, csp, ce, I, th)
        g_net, _, res_net = m.g_hat(j, csn, csp, ce, I, th)                                             # network alone
        g1, res1 = A_.newton_fixed(g_net, csn, csp, ce, I, th, k=1)                                     # network + one Newton iteration (the quantity of Fig. 11)
        gc, resc, _ = A_.solve_robust(g, csn, csp, ce, I, th, tol=1e-10)                                # converged solver with fallback: the true trajectory
        need = res1 > THRESHOLD
        kc = jax.lax.cond(need, lambda g_: A_.newton(g_, csn, csp, ce, I, th, tol=1e-10)[2], lambda g_: jnp.asarray(0), g1)
        return (m.advance(z, gc, th, consts, em), gc), (res1, need, jnp.min(ce), kc, res_net, jnp.linalg.norm(A_.residual(gc, csn, csp, ce, I, th), jnp.inf))
    _, out = jax.lax.scan(body, (z0, g0), jnp.asarray(I_all)); res1, need, cemin, kc, res_net, res_conv = map(np.asarray, out)
    t = np.arange(len(I_all)) * DT
    S = pd.DataFrame(dict(t_s=t, I_A=I_all, residual_inf_norm=res1, fallback_active=need.astype(int), completion_newton_it=kc, ce_min=cemin, threshold=THRESHOLD, residual_network_alone=res_net, residual_converged=res_conv))
    cli.write_csv(S, a, "residual_stress_series.csv", note="stress case P5, R_f x15, c_e,0/2, LAM 0.3; residual_inf_norm = after the network and one Newton iteration, before completion", float_format=None)
    os.makedirs(os.path.join(a.results, "source_data"), exist_ok=True); sd = os.path.join(a.results, "source_data", "fig_residual_m17.csv")
    with open(sd, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(HEADER) + "\nt_s,residual_inf_norm,fallback_active,threshold\n")
        for tt, rr, nn in zip(t, res1, need): f.write(f"{tt:g},{rr:.9g},{int(nn)},0.0001\n")
    lens, starts = runs_above(res1 > THRESHOLD, DT)
    jump = np.log10(np.maximum(res1[1:], 1e-16) / np.maximum(res1[:-1], 1e-16))
    nums = [("steps", len(res1), ""), ("completed steps (residual > 1e-4)", int(need.sum()), ""),
            ("fraction of completed steps", float(need.mean()), "31 % of the total (Fig. 11 marks)"),
            ("residual after one iteration: median", float(np.median(res1)), ""), ("residual after one iteration: 99th percentile", float(np.quantile(res1, .99)), ""),
            ("residual after one iteration: maximum", float(res1.max()), ""),
            ("maximum residual in decades above the threshold", float(np.log10(res1.max() / THRESHOLD)), "jumps above the threshold by up to four orders of magnitude"),
            ("largest step-to-step jump (decades)", float(jump.max()), "idem (jump from one step to the next)"),
            ("longest time above the threshold (s)", float(lens.max()) if len(lens) else 0.0, "stays above it for up to a minute"),
            ("number of excursions above the threshold", int(len(lens)), "one or more per 5C pulse"), ("total time above the threshold (s)", float(lens.sum()) if len(lens) else 0.0, ""),
            ("median Newton iterations of the completion (completed steps)", float(np.median(kc[need])) if need.any() else 0.0, ""),
            ("residual of the network alone: maximum", float(res_net.max()), "the network alone returns a wrong current without warning"),
            ("residual of the network alone: median", float(np.median(res_net)), ""),
            ("residual of the converged solver: maximum", float(res_conv.max()), "below 1e-10 with the converged solver"), ("minimum c_e (mol/m3)", float(cemin.min()), "")]
    N = pd.DataFrame(nums, columns=["quantity", "value", "first_version_text"]); os.makedirs(os.path.join(a.results, "tables"), exist_ok=True)
    cli.write_csv(N, a, "tables/residual_stress_numbers.csv", note="numbers of the residual certificate in the stress case (Fig. 11)", float_format="%.6g")
    print(N.to_string(index=False)); clock.log()


if __name__ == "__main__":
    main()
