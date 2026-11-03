"""Lettura dei riferimenti data/c con la convenzione CONVENTIONS.md 0: campione k (k = 1..N) = stato a t_{k-1} con la corrente I_k (ultimo campione allo stesso t)."""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(proto, tag, mesh=(40, 20, 40), period=1.0, grid_ratio=None):
    ms = "-".join(str(m) for m in mesh) + (f"nu{grid_ratio:g}" if grid_ratio else "")
    return f"{ROOT}/data/c/ref_{proto}_{tag}_m{ms}" + ("" if period == 1.0 else f"_dt{period:g}") + ".npz"


def load(proto, tag, mesh=(40, 20, 40), period=1.0, grid_ratio=None):
    d = np.load(path(proto, tag, mesh, period, grid_ratio))
    t = d["t"]; n = int(round(t[-1] / period))
    tk = np.arange(n) * period
    idx = np.array([np.where(np.isclose(t, x, atol=1e-9))[0][-1] for x in tk])
    out = {k: (d[k][idx] if d[k].ndim >= 1 and d[k].shape[0] == len(t) and k not in ("theta", "c0", "mesh") else d[k]) for k in d.files}
    out["t"] = t[idx]; out["N"] = n; out["period"] = period
    th = d["theta"]; out["Rf"], out["LAM"], out["LLI"], out["Ds"] = [float(x) for x in th]
    out["as_"] = 1 - out["LAM"]
    c0 = [float(x) for x in d["c0"]]
    out["c0_eff"] = (c0[0] - out["LLI"] * 0.8735 * 33133.0, c0[1])        # stato iniziale uniforme effettivo (LLI applicato all'anodo)
    return out


def load_uniform(proto, tag, mesh=(40, 20, 40), period=0.5):
    """M2 block C: references whose time grid is shifted by a voltage-event termination (step ends at a non-grid instant). Resample on the uniform grid k*period:
    continuous variables by linear interpolation (duplicate times collapsed to the last sample), current = mean over [t_k, t_k+period] of the piecewise-constant current (sample value holds until the next sample)."""
    from scipy.interpolate import interp1d
    d = np.load(path(proto, tag, mesh, period)); t = d["t"]; Iraw = d["I"]
    keep = np.r_[t[1:] > t[:-1] + 1e-9, True]; tu = t[keep]                      # last sample at each distinct time
    n = int(np.floor(t[-1] / period + 1e-9)); tk = np.arange(n) * period
    out = {}
    for k in d.files:
        a = d[k]
        if k in ("theta", "c0", "mesh", "wall_s", "period") or a.ndim < 1 or a.shape[0] != len(t): out[k] = a; continue
        out[k] = interp1d(tu, a[keep], axis=0, kind="linear", bounds_error=False, fill_value=(a[keep][0], a[keep][-1]))(tk)
    # current: forward-held piecewise constant evaluated at the midpoint of [t_k, t_k+period] (an off-grid event is snapped to the nearest grid time, charge error <= 0.25 s of current);
    # voltage at an event interval whose event falls before the midpoint (snapped to the new step): value just after the event, so that (V_k, I_k) are paired as in the other samples
    Ik = np.zeros(n)
    for i in range(n):
        tm = tk[i] + 0.5 * period; j = np.where(t <= tm)[0][-1]; Ik[i] = Iraw[j]
    dup = np.where((t[1:] - t[:-1]) < 1e-9)[0]
    snaps = []
    for jd in dup:
        te = t[jd]
        if abs(te / period - round(te / period)) > 1e-6:                    # off-grid event
            kk = int(np.floor(te / period))
            if kk < n and te < tk[kk] + 0.5 * period:
                for key in d.files:                                                    # all time-dependent fields: the sample is taken just after the event
                    if key not in ("theta", "c0", "mesh", "wall_s", "period", "t", "I") and d[key].ndim >= 1 and d[key].shape[0] == len(t): out[key][kk] = d[key][jd + 1]
            snaps.append((float(te), float(round(te / period) * period - te)))
    out["event_snaps"] = np.array(snaps)
    out["I"] = Ik; out["t"] = tk; out["N"] = n; out["period"] = period
    th = d["theta"]; out["Rf"], out["LAM"], out["LLI"], out["Ds"] = [float(x) for x in th]; out["as_"] = 1 - out["LAM"]
    c0 = [float(x) for x in d["c0"]]; out["c0_eff"] = (c0[0] - out["LLI"] * 0.8735 * 33133.0, c0[1])
    return out
