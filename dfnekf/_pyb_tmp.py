"""Riferimenti PyBaMM della campagna. theta = (Rf, LAM, LLI, Ds): Rf = R_f/R_f0 (rho_sei x Rf), LAM = perdita di materiale attivo anodo (eps_s,n x (1-LAM)),
LLI = perdita di litio ciclabile (frazione della finestra stechiometrica anodica a stechiometria catodica fissata: c_n0 -= LLI * dx_n * cmax_n), Ds = D_s,n/D_s,n0.
Griglia x: 'mesh' (n,s,p); nodi radiali 300; rtol 1e-9; trasporto costante come nel pilot."""
import sys, os, time, argparse, json
import numpy as np
import pybamm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as prm
from protocols import PROTOCOLS, d1_current
from pybamm_ref import VARS, make_model, step_str

DX_N = 0.8735                      # finestra stechiometrica anodica (x_100 - x_0) del Chen2020 (LLI = 1 -> intera finestra)
SOC_C0 = {1.0: (prm.C0_N_100, prm.C0_P_100), 0.5: (prm.C0_N_50, prm.C0_P_50)}


def c0_for_soc(soc):
    if soc in SOC_C0: return SOC_C0[soc]
    p = pybamm.ParameterValues("Chen2020")
    sim = pybamm.Simulation(pybamm.lithium_ion.DFN(), parameter_values=p)
    s = sim.solve([0, 1], initial_soc=soc)
    return (float(s["R-averaged negative particle concentration [mol.m-3]"].entries[..., 0].mean()), float(s["R-averaged positive particle concentration [mol.m-3]"].entries[..., 0].mean()))


def make_params(Rf=1.0, LAM=0.0, LLI=0.0, Ds=1.0, c0=None, Lsei=None, var_transport=False):
    p = pybamm.ParameterValues("Chen2020"); P = prm.base()
    if not var_transport:
        p.update({"Electrolyte diffusivity [m2.s-1]": P["De"], "Electrolyte conductivity [S.m-1]": P["kappa"]})
    p.update({
              "SEI resistivity [Ohm.m]": 2e5 * Rf,
              "Negative electrode active material volume fraction": P["eps_sn"] * (1 - LAM),
              "Negative particle diffusivity [m2.s-1]": P["Ds_n"] * Ds})
    if Lsei is not None:
        p.update({"Initial SEI thickness [m]": Lsei})
    if c0 is not None:
        p.update({"Initial concentration in negative electrode [mol.m-3]": c0[0] - LLI * DX_N * P["cmax_n"],
                  "Initial concentration in positive electrode [mol.m-3]": c0[1]})
    return p


def experiment_steps(segs, cutoff=None):
    return tuple(step_str(I, d, cutoff) for I, d in segs)


def run(proto, theta=(1.0, 0.0, 0.0, 1.0), mesh=(40, 20, 40), period=1.0, soc=None, rnodes=300, rtol=1e-9, atol=1e-10, I_custom=None, var_transport=False, grid_ratio=None):
    Rf, LAM, LLI, Ds = theta
    var_pts = {"x_n": mesh[0], "x_s": mesh[1], "x_p": mesh[2], "r_n": rnodes, "r_p": rnodes}
    if I_custom is not None:
        I = np.asarray(I_custom, float); segs = []; i = 0
        while i < len(I):
            j = i
            while j + 1 < len(I) and I[j + 1] == I[i]: j += 1
            segs.append((float(I[i]), float(j - i + 1))); i = j + 1
        pr = dict(segs=segs, soc=soc if soc is not None else 0.8, cutoff=None)
    else:
        pr = PROTOCOLS[proto]
    sc = pr["soc"] if soc is None else soc
    c0 = c0_for_soc(sc); p = make_params(Rf, LAM, LLI, Ds, c0, var_transport=var_transport)
    if pr.get("cutoff_idx") is not None:        # M2: voltage event only on one step
        ci = pr["cutoff_idx"]; ci = set(ci) if isinstance(ci, (list, tuple, set)) else {ci}         # M5: one index or a list of indices
        steps = tuple(step_str(I_, d_, pr["cutoff_V"] if k in ci else None) for k, (I_, d_) in enumerate(pr["segs"]))
    else:
        steps = experiment_steps(pr["segs"], pr["cutoff"])
    exp = pybamm.Experiment([steps], period=f"{period:g} seconds")
    kw_sub = {}
    if grid_ratio is not None:                  # M2 block G: electrode cells geometric (collector/separator width = grid_ratio), separator uniform
        P_ = prm.base(); def_ = make_model().default_submesh_types
        def wid(n, L, left):
            q = grid_ratio ** (-1.0 / (n - 1)); w = q ** np.arange(n); w = L * w / w.sum(); return w if left else w[::-1]
        en = np.concatenate([[0.0], np.cumsum(wid(mesh[0], P_["Ln"], True))]); es = en[-1] + np.linspace(0.0, P_["Ls"], mesh[1] + 1); ep = es[-1] + np.concatenate([[0.0], np.cumsum(wid(mesh[2], P_["Lp"], False))])
        en[-1] = P_["Ln"]; es[0] = P_["Ln"]; es[-1] = P_["Ln"] + P_["Ls"]; ep[0] = es[-1]; ep[-1] = P_["Ln"] + P_["Ls"] + P_["Lp"]      # exact domain limits
        gen = lambda e: pybamm.MeshGenerator(pybamm.UserSupplied1DSubMesh, submesh_params={"edges": e})
        st = dict(def_); st["negative electrode"] = gen(en); st["separator"] = gen(es); st["positive electrode"] = gen(ep); kw_sub["submesh_types"] = st
    sim = pybamm.Simulation(make_model(), parameter_values=p, experiment=exp, var_pts=var_pts, solver=pybamm.IDAKLUSolver(rtol=rtol, atol=atol), **kw_sub)
    t0 = time.time(); sol = sim.solve()
    out = {"t": sol["Time [s]"].entries}
    for k, v in VARS.items():
        out[k] = np.asarray(sol[v].entries)
        if out[k].ndim == 2: out[k] = out[k].T
    out["wall_s"] = time.time() - t0
    out["theta"] = np.array(theta); out["c0"] = np.array(c0); out["mesh"] = np.array(mesh); out["period"] = period
    return out


def fname(proto, tag, mesh, period=1.0, grid_ratio=None):
    ms = "-".join(str(m) for m in mesh) + (f"nu{grid_ratio:g}" if grid_ratio else "")
    return f"data/c/ref_{proto}_{tag}_m{ms}" + ("" if period == 1.0 else f"_dt{period:g}") + ".npz"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    from concurrent.futures import ProcessPoolExecutor
    def go(j): return _go(j)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_go if False else __import__("pyb_worker").go, jobs): print(r, flush=True)
