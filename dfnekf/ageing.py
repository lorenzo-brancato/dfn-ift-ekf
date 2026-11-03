"""Pre-processing del run di aging dinamico: ricampionamento a 0.5 s uniforme (V, I per step dell'Experiment), theta_eff(t), indici RPT."""
import numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAX_N = 33133.0; DX_N = 0.8735; EPS0 = 0.75; LN = 85.2e-6; AREA = 0.065 * 1.58
WINDOW_MOL = DX_N * CMAX_N * EPS0 * LN * AREA          # moli della finestra anodica (denominatore di LLI)


def load(path=None, dt=0.5):
    d = np.load(path or f"{ROOT}/data/c/aging_dyn.npz")
    t = d["t"]; V = d["Terminal voltage [V]"]; I = d["Current [A]"]
    # step = tratti separati da tempi duplicati (campione finale dello step A e iniziale dello step B)
    dup = np.where(np.diff(t) <= 1e-9)[0] + 1
    starts = np.concatenate([[0], dup]); ends = np.concatenate([dup, [len(t)]])
    tk, Ik, Vk, sid = [], [], [], []
    tau = 0.0
    for s, (a, b) in enumerate(zip(starts, ends)):
        ts, te = t[a], t[b - 1]
        grid = np.arange(np.ceil(ts / dt - 1e-9) * dt, te + 1e-9, dt)
        grid = grid[(grid >= ts - 1e-9)]
        if len(grid) == 0: continue
        tk.append(grid); Ik.append(np.interp(grid, t[a:b], I[a:b])); Vk.append(np.interp(grid, t[a:b], V[a:b])); sid.append(np.full(len(grid), s))
    tk, Ik, Vk, sid = map(np.concatenate, (tk, Ik, Vk, sid))
    keep = np.concatenate([[True], np.diff(tk) > 1e-9]); tk, Ik, Vk, sid = tk[keep], Ik[keep], Vk[keep], sid[keep]      # un campione per istante
    # theta_eff(t) (campionato a ogni istante uniforme dallo stesso interp)
    def at(name): return np.interp(tk, t, d[name])
    Lsei = at("X-averaged negative SEI thickness [m]"); eps = at("X-averaged negative electrode active material volume fraction"); Ltot = at("Total lithium [mol]")
    Rf = Lsei / 5e-9; LAM = 1 - eps / EPS0; LLI = (Ltot[0] - Ltot) / WINDOW_MOL
    nn = at("Total lithium in negative electrode [mol]"); npos = at("Total lithium in positive electrode [mol]")
    cn_avg = nn / (eps * LN * AREA); cp_avg = npos / (0.665 * 75.6e-6 * AREA)
    return dict(t=tk, I=Ik, V=Vk, step=sid, Rf=Rf, LAM=LAM, LLI=LLI, cn_avg=cn_avg, cp_avg=cp_avg, cycle_start_t=d["cycle_start_t"], is_rpt=d["is_rpt"], Ltot=Ltot)
