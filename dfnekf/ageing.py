"""Pre-processing of the dynamic ageing reference run (Fig. 8).

The PyBaMM run (``scripts/ageing_reference.py``) is stored with the solver's own output times; ``load`` resamples it to a uniform 0.5 s grid, one
sample per instant, step by step (the experiment steps are separated by duplicated times), and derives the effective ageing parameters of the
truth at every sample:

    Rf  = L_sei / 5e-9                  film resistance relative to the nominal cell (x-averaged SEI thickness)
    LAM = 1 - eps_s / 0.75              loss of active material of the anode
    LLI = (Li_tot(0) - Li_tot) / window loss of lithium inventory, in units of the anode window (WINDOW_MOL)

together with the electrode-averaged lithium concentrations used to initialise the filter.
"""
import numpy as np

CMAX_N = 33133.0                 # maximum anode concentration [mol/m3]
DX_N = 0.8735                    # stoichiometric width of the anode window
EPS0 = 0.75                      # nominal anode active-material volume fraction
LN = 85.2e-6                     # anode thickness [m]
AREA = 0.065 * 1.58              # electrode area [m2]
WINDOW_MOL = DX_N * CMAX_N * EPS0 * LN * AREA          # moles in the anode window (denominator of LLI)


def load(path, dt=0.5):
    """Resample the reference run stored in ``path`` (.npz) to a uniform grid of step ``dt`` [s].

    Returns a dict with the time ``t``, current ``I`` and terminal voltage ``V`` on the grid, the experiment step index of every sample ``step``,
    the effective parameters ``Rf``, ``LAM``, ``LLI``, the electrode-averaged concentrations ``cn_avg``, ``cp_avg``, the total lithium ``Ltot``
    and the cycle bookkeeping of the file (``cycle_start_t``, ``is_rpt``).
    """
    d = np.load(path)
    t = d["t"]; V = d["Terminal voltage [V]"]; I = d["Current [A]"]
    # a step is a stretch between duplicated times (last sample of step A and first sample of step B)
    dup = np.where(np.diff(t) <= 1e-9)[0] + 1
    starts = np.concatenate([[0], dup]); ends = np.concatenate([dup, [len(t)]])
    tk, Ik, Vk, sid = [], [], [], []
    for s, (a, b) in enumerate(zip(starts, ends)):
        ts, te = t[a], t[b - 1]
        grid = np.arange(np.ceil(ts / dt - 1e-9) * dt, te + 1e-9, dt)
        grid = grid[(grid >= ts - 1e-9)]
        if len(grid) == 0: continue
        tk.append(grid); Ik.append(np.interp(grid, t[a:b], I[a:b])); Vk.append(np.interp(grid, t[a:b], V[a:b])); sid.append(np.full(len(grid), s))
    tk, Ik, Vk, sid = map(np.concatenate, (tk, Ik, Vk, sid))
    keep = np.concatenate([[True], np.diff(tk) > 1e-9]); tk, Ik, Vk, sid = tk[keep], Ik[keep], Vk[keep], sid[keep]      # one sample per instant

    def at(name): return np.interp(tk, t, d[name])           # effective quantities on the uniform grid, same interpolation
    Lsei = at("X-averaged negative SEI thickness [m]"); eps = at("X-averaged negative electrode active material volume fraction"); Ltot = at("Total lithium [mol]")
    Rf = Lsei / 5e-9; LAM = 1 - eps / EPS0; LLI = (Ltot[0] - Ltot) / WINDOW_MOL
    nn = at("Total lithium in negative electrode [mol]"); npos = at("Total lithium in positive electrode [mol]")
    cn_avg = nn / (eps * LN * AREA); cp_avg = npos / (0.665 * 75.6e-6 * AREA)
    return dict(t=tk, I=Ik, V=Vk, step=sid, Rf=Rf, LAM=LAM, LLI=LLI, cn_avg=cn_avg, cp_avg=cp_avg, cycle_start_t=d["cycle_start_t"], is_rpt=d["is_rpt"], Ltot=Ltot)
