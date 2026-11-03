"""PyBaMM reference of the dynamic ageing run (Fig. 8): accelerated SEI growth, current-driven loss of active material, film resistance varying along the electrode.

Cell: PyBaMM ``Chen2020`` with constant electrolyte transport (as everywhere in the study), full DFN (40/20/40 cells, 300 radial nodes, IDAKLU rtol 1e-9, atol 1e-10),
SEI "reaction limited" with the kinetic constant multiplied by 450, "SEI film resistance: distributed" (the film grows unevenly along the electrode), anode LAM
"current-driven" with rate -1e-8 |i|, interfacial current as a state, isothermal, no particle mechanics.

Sequence (``cycles``): ``nblocks`` x [RPT + 9 ageing cycles] + a final RPT, i.e. 101 cycles for ``nblocks = 10``.
  ageing cycle : discharge 2.5 A to 3.65 V, rest 10 min, charge 5 A to 4.2 V, hold 4.2 V until 0.25 A, rest 10 min.
  enriched RPT : discharge 2.5 A to 3.65 V, rest 30 min, pulse protocol P6L (2160 s), rest 30 min, discharge 2.5 A (C/2) to 2.5 V, rest 1 h,
                 charge 2.5 A (C/2) to 4.2 V, hold 4.2 V until 0.1 A (C/50), rest 30 min (about 7.5 h).
The solution is sampled every 0.5 s. Besides the terminal voltage and current the file stores the total lithium in each electrode, the x-averaged
SEI thickness and active-material fraction (the truth of the three ageing parameters, read by ``dfnekf.ageing.load``), the end time of every experiment step
(``step_end_t``: the reading points of the RPTs) and the x-profiles of the SEI thickness, active-material fraction and interfacial current every 10 min.
PyBaMM is imported lazily, only by ``run``.
"""
import time
import numpy as np

from . import params as prm
from .protocols import PROTOCOLS

SEI_ACCELERATION = 450.0                 # SEI kinetic rate constant = 1e-12 m/s x 450
LAM_RATE = 1e-8                          # d eps_s / dt = -LAM_RATE |i|  [m2/(A s)]
NAMES = ["Terminal voltage [V]", "Current [A]", "X-averaged negative SEI thickness [m]", "X-averaged negative electrode active material volume fraction",
         "Total lithium in negative electrode [mol]", "Total lithium in positive electrode [mol]", "Total lithium [mol]"]
PROFILES = ["Negative SEI thickness [m]", "Negative electrode active material volume fraction", "Negative electrode interfacial current density [A.m-2]"]
VAR_PTS = {"x_n": 40, "x_s": 20, "x_p": 40, "r_n": 300, "r_p": 300}


def step_str(I, d):
    """PyBaMM experiment step for a current I [A] (positive = discharge) held for d seconds."""
    a = abs(I)
    if a == 0: return f"Rest for {int(d)} seconds"
    return f"{'Discharge' if I > 0 else 'Charge'} at {a} A for {int(d)} seconds"


def make_params():
    """PyBaMM parameter set of the fresh cell (nominal ageing parameters, 100 % SOC) with the ageing mechanisms switched on."""
    import pybamm
    P = prm.base()
    p = pybamm.ParameterValues("Chen2020")
    p.update({"Electrolyte diffusivity [m2.s-1]": P["De"], "Electrolyte conductivity [S.m-1]": P["kappa"],
              "SEI resistivity [Ohm.m]": 2e5, "Negative electrode active material volume fraction": P["eps_sn"],
              "Negative particle diffusivity [m2.s-1]": P["Ds_n"],
              "Initial concentration in negative electrode [mol.m-3]": prm.C0_N_100, "Initial concentration in positive electrode [mol.m-3]": prm.C0_P_100})
    p["SEI kinetic rate constant [m.s-1]"] = 1e-12 * SEI_ACCELERATION
    p["Negative electrode current-driven LAM rate"] = (lambda i, T: -LAM_RATE * abs(i))
    p["Lower voltage cut-off [V]"] = 1.5
    return p


def cycle_definitions():
    """(ageing cycle, enriched RPT) as tuples of PyBaMM experiment steps."""
    cyc = ("Discharge at 2.5 A until 3.65 V", "Rest for 10 minutes", "Charge at 5 A until 4.2 V", "Hold at 4.2 V until 0.25 A", "Rest for 10 minutes")
    rpt = (("Discharge at 2.5 A until 3.65 V", "Rest for 30 minutes") + tuple(step_str(I, d) for I, d in PROTOCOLS["P6L"]["segs"])
           + ("Rest for 30 minutes", "Discharge at 2.5 A until 2.5 V", "Rest for 1 hour", "Charge at 2.5 A until 4.2 V", "Hold at 4.2 V until 0.1 A", "Rest for 30 minutes"))
    return cyc, rpt


def run(nblocks=10, quick=False):
    """Run the ageing reference and return the dict saved by ``scripts/ageing_reference.py``. ``quick``: RPT + one ageing cycle only."""
    import pybamm
    cyc, rpt = cycle_definitions()
    cycles = [rpt, cyc] if quick else ([rpt] + [cyc] * 9) * nblocks + [rpt]
    opts = {"SEI": "reaction limited", "SEI film resistance": "distributed", "loss of active material": ("current-driven", "none"),
            "total interfacial current density as a state": "true", "thermal": "isothermal"}
    model = pybamm.lithium_ion.DFN(options=opts)
    sim = pybamm.Simulation(model, parameter_values=make_params(), experiment=pybamm.Experiment(cycles, period="0.5 seconds"), var_pts=VAR_PTS,
                            solver=pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-10, output_variables=NAMES + PROFILES))
    t0 = time.time(); sol = sim.solve(); wall = time.time() - t0
    d = {"t": sol["Time [s]"].entries}
    for n in NAMES: d[n] = sol[n].entries
    nmax = max(len(c) for c in cycles)
    d["step_end_t"] = np.array([[st.t[-1] for st in c.steps] + [np.nan] * (nmax - len(c.steps)) for c in sol.cycles])
    tt = d["t"]; idx = np.unique(np.searchsorted(tt, np.arange(0, tt[-1], 600.0))); d["prof_t"] = tt[idx]
    for n in PROFILES: d["prof_" + n.split(" [")[0].replace(" ", "_")] = np.asarray(sol[n].entries)[:, idx]
    d["cycle_start_t"] = np.array([c.t[0] for c in sol.cycles]); d["is_rpt"] = np.array([len(c_) > len(cyc) for c_ in cycles]); d["wall_s"] = wall
    return d
