"""PyBaMM reference simulations (the truth of every test): the DFN on a fine grid (40/20/40 cells, 300 radial nodes per particle, rtol 1e-9) parameterized by the
ageing state theta = (Rf, LAM, LLI, Ds): Rf = R_f / R_f0 (SEI resistivity x Rf), LAM = loss of anode active material (eps_s,n x (1 - LAM)), LLI = loss of cyclable lithium
(fraction of the anode stoichiometric window, c_n0 -= LLI * dx_n * cmax_n), Ds = D_s,n / D_s,n0. Constant electrolyte transport as in the grey-box model, unless ``var_transport``.
Model options: SEI 'constant' with a distributed film resistance and the interfacial current as a state, so that eta = dphi - U - j R_f.

Command line (needs pybamm, which is imported lazily only here):  python -m dfnekf.reference <jobs.json> --workers 4 --data data/sim
A job is a dict {proto, tag, theta, mesh, period[, soc, var_transport, grid_ratio, force]}; ``standard_jobs()`` lists every reference the paper needs.
"""
import argparse
import json
import os
import time
import numpy as np
from . import params as prm
from .protocols import PROTOCOLS, d1_current

VARS = {
    "cs_n": "Negative particle surface concentration [mol.m-3]",
    "cs_p": "Positive particle surface concentration [mol.m-3]",
    "csav_n": "R-averaged negative particle concentration [mol.m-3]",
    "csav_p": "R-averaged positive particle concentration [mol.m-3]",
    "ce": "Electrolyte concentration [mol.m-3]",
    "j_n": "Negative electrode interfacial current density [A.m-2]",
    "j_p": "Positive electrode interfacial current density [A.m-2]",
    "phi_e": "Electrolyte potential [V]",
    "phi_s_n": "Negative electrode potential [V]",
    "phi_s_p": "Positive electrode potential [V]",
    "V": "Terminal voltage [V]",
    "I": "Current [A]",
}




def _pybamm():
    import pybamm
    return pybamm


def make_model():
    return _pybamm().lithium_ion.DFN(options={
        "SEI": "constant", "SEI film resistance": "distributed",
        "total interfacial current density as a state": "true",
        "thermal": "isothermal"})


def step_str(I, d, cutoff=None):
    a = abs(I)
    if a == 0:
        return f"Rest for {int(d)} seconds"
    verb = "Discharge" if I > 0 else "Charge"
    s = f"{verb} at {a} A for {int(d)} seconds"
    if cutoff is not None:
        s += f" or until {cutoff} V"
    return s




DX_N = 0.8735                      # stoichiometric window of the anode (x_100 - x_0) of Chen2020 (LLI = 1: the whole window)
SOC_C0 = {1.0: (prm.C0_N_100, prm.C0_P_100), 0.5: (prm.C0_N_50, prm.C0_P_50)}


def c0_for_soc(soc):
    if soc in SOC_C0: return SOC_C0[soc]
    p = _pybamm().ParameterValues("Chen2020")
    sim = _pybamm().Simulation(_pybamm().lithium_ion.DFN(), parameter_values=p)
    s = sim.solve([0, 1], initial_soc=soc)
    return (float(s["R-averaged negative particle concentration [mol.m-3]"].entries[..., 0].mean()), float(s["R-averaged positive particle concentration [mol.m-3]"].entries[..., 0].mean()))


def make_params(Rf=1.0, LAM=0.0, LLI=0.0, Ds=1.0, c0=None, Lsei=None, var_transport=False):
    p = _pybamm().ParameterValues("Chen2020"); P = prm.base()
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
    exp = _pybamm().Experiment([steps], period=f"{period:g} seconds")
    kw_sub = {}
    if grid_ratio is not None:                  # M2 block G: electrode cells geometric (collector/separator width = grid_ratio), separator uniform
        P_ = prm.base(); def_ = make_model().default_submesh_types
        def wid(n, L, left):
            q = grid_ratio ** (-1.0 / (n - 1)); w = q ** np.arange(n); w = L * w / w.sum(); return w if left else w[::-1]
        en = np.concatenate([[0.0], np.cumsum(wid(mesh[0], P_["Ln"], True))]); es = en[-1] + np.linspace(0.0, P_["Ls"], mesh[1] + 1); ep = es[-1] + np.concatenate([[0.0], np.cumsum(wid(mesh[2], P_["Lp"], False))])
        en[-1] = P_["Ln"]; es[0] = P_["Ln"]; es[-1] = P_["Ln"] + P_["Ls"]; ep[0] = es[-1]; ep[-1] = P_["Ln"] + P_["Ls"] + P_["Lp"]      # exact domain limits
        gen = lambda e: _pybamm().MeshGenerator(_pybamm().UserSupplied1DSubMesh, submesh_params={"edges": e})
        st = dict(def_); st["negative electrode"] = gen(en); st["separator"] = gen(es); st["positive electrode"] = gen(ep); kw_sub["submesh_types"] = st
    sim = _pybamm().Simulation(make_model(), parameter_values=p, experiment=exp, var_pts=var_pts, solver=_pybamm().IDAKLUSolver(rtol=rtol, atol=atol), **kw_sub)
    t0 = time.time(); sol = sim.solve()
    out = {"t": sol["Time [s]"].entries}
    for k, v in VARS.items():
        out[k] = np.asarray(sol[v].entries)
        if out[k].ndim == 2: out[k] = out[k].T
    out["wall_s"] = time.time() - t0
    out["theta"] = np.array(theta); out["c0"] = np.array(c0); out["mesh"] = np.array(mesh); out["period"] = period
    return out


def fname(proto, tag, mesh, period=1.0, grid_ratio=None, data_dir="data/sim"):
    ms = "-".join(str(m) for m in mesh) + (f"nu{grid_ratio:g}" if grid_ratio else "")
    return f"{data_dir}/ref_{proto}_{tag}_m{ms}" + ("" if period == 1.0 else f"_dt{period:g}") + ".npz"




def standard_jobs():
    """Every reference simulation needed by the paper (list of job dicts; see ``assets/reference_jobs.json``)."""
    return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "reference_jobs.json")))


def go(j):
    """Run one job (module-level so that it can be pickled by the process pool); skip if the file exists unless ``force``."""
    try:
        proto, tag, theta, mesh, period = j["proto"], j["tag"], tuple(j["theta"]), tuple(j["mesh"]), j.get("period", 1.0)
        gr = j.get("grid_ratio"); fn = fname(proto, tag, mesh, period, gr, j.get("data_dir", "data/sim"))
        if os.path.exists(fn) and not j.get("force"): return ("skip", fn)
        kw = dict(theta=theta, mesh=mesh, period=period, soc=j.get("soc"), var_transport=bool(j.get("var_transport", False)), grid_ratio=gr)
        if proto == "D1": kw["I_custom"] = d1_current(); kw["soc"] = 0.8
        o = run(proto, **kw)
        os.makedirs(os.path.dirname(fn) or ".", exist_ok=True)
        np.savez_compressed(fn, **o)
        return ("ok", fn, round(o["wall_s"], 1), len(o["t"]), float(o["V"][-1]), float(o["ce"].min()))
    except Exception as e:
        return ("FAIL", j, str(e)[:200])


def main():
    ap = argparse.ArgumentParser(description="generate PyBaMM reference simulations")
    ap.add_argument("jobs", help="JSON file with a list of jobs, or 'standard' for the references of the paper")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--data", default="data/sim")
    a = ap.parse_args()
    jobs = standard_jobs() if a.jobs == "standard" else json.load(open(a.jobs))
    for j in jobs: j["data_dir"] = a.data
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(go, jobs): print(r, flush=True)


if __name__ == "__main__":
    main()
