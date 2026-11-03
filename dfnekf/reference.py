"""Riferimento PyBaMM: DFN fine 40/20/40, 300 nodi radiali, tol 1e-9, parametrizzato in theta.
Opzioni (CONVENTIONS.md 0): SEI 'constant' + 'SEI film resistance' distributed + j come stato -> eta = dphi - U - j*R_f."""
import sys, time, argparse
import numpy as np
import pybamm
sys.path.insert(0, __file__.rsplit("src", 1)[0] + "src")
import params as prm
from protocols import PROTOCOLS

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


def make_params(Rf=1.0, as_=1.0, Ds=1.0, c0n=None, c0p=None):
    p = pybamm.ParameterValues("Chen2020")
    P = prm.base()
    p.update({
        "Electrolyte diffusivity [m2.s-1]": P["De"],
        "Electrolyte conductivity [S.m-1]": P["kappa"],
        "SEI resistivity [Ohm.m]": 2e5 * Rf,
        "Negative electrode active material volume fraction": P["eps_sn"] * as_,
        "Negative particle diffusivity [m2.s-1]": P["Ds_n"] * Ds,
    })
    if c0n is not None:
        p.update({"Initial concentration in negative electrode [mol.m-3]": c0n,
                  "Initial concentration in positive electrode [mol.m-3]": c0p})
    return p


def make_model():
    return pybamm.lithium_ion.DFN(options={
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


def run(proto, Rf=1.0, as_=1.0, Ds=1.0, rtol=1e-9, atol=1e-10, c0=None):
    pr = PROTOCOLS[proto]
    soc = pr["soc"]
    if c0 is None:
        c0 = (prm.C0_N_100, prm.C0_P_100) if soc == 1.0 else (prm.C0_N_50, prm.C0_P_50)
    p = make_params(Rf, as_, Ds, *c0)
    exp = pybamm.Experiment([tuple(step_str(I, d, pr["cutoff"]) for I, d in pr["segs"])], period="1 second")
    m = make_model()
    var_pts = {"x_n": 40, "x_s": 20, "x_p": 40, "r_n": 300, "r_p": 300}
    sim = pybamm.Simulation(m, parameter_values=p, experiment=exp, var_pts=var_pts,
                            solver=pybamm.IDAKLUSolver(rtol=rtol, atol=atol))
    t0 = time.time()
    sol = sim.solve()
    out = {"t": sol["Time [s]"].entries}
    for k, v in VARS.items():
        out[k] = np.asarray(sol[v].entries)
        if out[k].ndim == 2:
            out[k] = out[k].T
    out["wall_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("proto"); ap.add_argument("--Rf", type=float, default=1.0)
    ap.add_argument("--as_", type=float, default=1.0); ap.add_argument("--Ds", type=float, default=1.0)
    ap.add_argument("--tag", default="nom"); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    o = run(a.proto, a.Rf, a.as_, a.Ds)
    fn = f"{a.out}/ref_{a.proto}_{a.tag}.npz"
    np.savez_compressed(fn, Rf=a.Rf, as_=a.as_, Ds=a.Ds, **o)
    print(fn, "wall", o["wall_s"], "n", len(o["t"]), "Vend", o["V"][-1])
