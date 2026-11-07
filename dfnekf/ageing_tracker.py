"""Ageing tracker (Sec. III-D, Fig. 8): the guarded EKF with four parameters (Rf, LAM, Ds, LLI) run over the whole ageing reference, and its evaluation.

One code path for every use: ``Tracker`` builds the filter (4:1 or uniform grid, N_m = 12, dt = 0.5 s, closure 'newton', sigma_V = 2 mV, optional state guard),
``track`` runs it in jitted chunks of 20000 steps (resumable), ``evaluate_rpt`` / ``cv_table`` reduce the stored trajectory to the numbers of the paper.

Conventions (identical to the campaign that produced the paper):
  * noise y = V + 2 mV * default_rng(seed).normal(size=N), drawn for the whole run (truncated runs use the first samples of the same sequence);
  * initial cell state = true electrode averages at t = 0 plus an offset of 1.5 % c_max on both electrodes (anode +, cathode -), P0 = 3 % c_max on the averages;
  * initial parameters [Rf, LAM, Ds, LLI] = [1.5, 0.02, 0.9, 0.02] (truth 1, 0, 1, 0), P0 = 30 % of the ageing range, random-walk process noise 1e-4 range sqrt(dt);
  * the stored series are the estimate and its standard deviation every 10 s (every 20th step), the innovation and a few diagnostics at every step.
"""
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from . import ageing as ag, params as prm
from .ekf import LLIEKF, RANGE, LOGIT
from .step import DFN

CHUNK = 20000                         # steps per jitted chunk (= 10000 s); also the window of the innovation criterion
SIGMA_V = 2e-3
COORDS = ("Rf", "LAM", "Ds", "LLI")
THETA0 = (1.5, 0.02, 0.9, 0.02)       # initial parameter estimates
STATE_OFFSET = 0.5 * 0.03             # initial state offset, fraction of c_max
DT = 0.5
FLOORS = dict(Rf=3.90, LAM=3.10, Ds=5.37, LLI_inv=3.00)      # criterion A: error within bias floor + 3 % (points of range)
POINTS = ("p6l", "disc", "precv", "end")                      # read-out points of every RPT (step_end_t index 12, 14, 16, last)
RES_LIM = 1e-2                        # closure residual limit of the guard / of the breakdown detector
SER = ("vals", "sd_vals", "li", "eta", "ce_min", "corr_ut")  # stored every 20 steps
PS = ("innov", "K_th", "csn_max", "rejected")                # stored at every step ("frozen" is added when a freeze is active)
FREEZE_MODES = ("none", "voltage", "phase")                  # see ``Tracker``
VIOL = ("viol_ce", "viol_cs", "viol_F", "viol_nf")           # guard conditions: electrolyte depletion, surface concentration out of range, closure residual, non-finite


class Tracker:
    """Filter + reference data. ``P`` is the dict of ``ageing.load``; ``guard`` switches the state guard; ``seed`` seeds the measurement noise."""

    def __init__(self, P, grid, seed=0, guard=True, nsteps=None, freeze="none", freeze_above=4.195, cv_update="state", R0=None, cv_scope="rpt"):
        """CV freeze (default ``"none"``: the filter of the paper, bit for bit). ``freeze="voltage"``: the update is restricted at the steps where the MEASURED voltage is >= ``freeze_above`` and the
        current is a charge (rule of the first remedy tried, campaign M9). ``freeze="phase"``: restricted during the constant-voltage segment of the protocol (step 17 of every RPT; with
        ``cv_scope="all"`` also the CV of the ageing cycles, step 3), known to a BMS from its charger state, never from the measurement; needs ``R0`` (the reference file, for the step schedule).
        ``cv_update="state"``: the parameter rows of the gain are zero, the cell state is still updated and the parameter variances are not reduced; ``"none"``: no measurement update (prediction only)."""
        assert freeze in FREEZE_MODES and cv_update in ("state", "none") and cv_scope in ("rpt", "all")
        self.P, self.guard, self.seed, self.grid, self.freeze = P, guard, seed, grid, freeze
        self.I, self.t = np.asarray(P["I"]), np.asarray(P["t"])
        N = len(self.t); self.y = P["V"] + SIGMA_V * np.random.default_rng(seed).normal(size=N)
        self.N = N if nsteps is None else min(int(nsteps), N)
        P0 = prm.base(); self.P0 = P0
        self.m = m = DFN(P0, 20, 10, 20, 12, 2, DT, grid); self.ek = ek = LLIEKF(m, closure="newton", coords=COORDS); nel = m.nel
        z0 = np.asarray(m.init_state(P["cn_avg"][0] + STATE_OFFSET * P0["cmax_n"], P["cp_avg"][0] - STATE_OFFSET * P0["cmax_p"]))
        c0 = ek.coord_of(np.array(THETA0)); self.x0 = np.concatenate([z0, c0]); dc = np.asarray(ek.dval_dc(jnp.asarray(c0)))
        rng_th = np.array([RANGE[c][1] - RANGE[c][0] for c in COORDS])
        self.P0d = np.concatenate([np.full(m.Nn, 0.03 * P0["cmax_n"]), np.full(m.Np, 0.03 * P0["cmax_p"]), np.full(m.Nm * nel, 1.0), np.full(m.K * nel, 1.0), np.full(m.N, 3.0), 0.3 * rng_th / dc])
        self.Qd = np.concatenate([np.zeros(m.Nn + m.Np), np.zeros(m.Nm * nel), np.zeros(m.K * nel), np.zeros(m.N), 1e-4 * rng_th * np.sqrt(DT) / dc])
        step0, self.scale = ek.make_step_g(self.x0, self.P0d, self.Qd, SIGMA_V ** 2, guard=guard, freeze=freeze != "none")
        if freeze == "none":
            self.fcode = None
            self._chunk = jax.jit(lambda carry, Ic, yc, In: jax.lax.scan(step0, carry, (Ic, yc, In)))
        else:
            if freeze == "voltage": mask = (self.y >= freeze_above) & (self.I < 0)
            else:
                assert R0 is not None, "freeze='phase' needs the reference file (R0) for the step schedule"
                cyc, sti = step_map(P, R0); rpt_c = np.asarray(R0["is_rpt"], bool)[cyc]
                mask = (rpt_c & (sti == 17)) | ((~rpt_c) & (sti == 3) & (cv_scope == "all"))
            self.fcode = np.where(mask, 1 if cv_update == "state" else 2, 0).astype(np.int32)
            self._chunk = jax.jit(lambda carry, Ic, yc, In, Fc: jax.lax.scan(step0, carry, (Ic, yc, In, Fc)))
        x0 = jnp.asarray(self.x0); g0 = ek.init_g(ek.zeff(x0), jnp.asarray(self.I[0]), ek.th_of(x0[m.nz:]))
        self.carry0 = (jnp.zeros(ek.n), jnp.diag((jnp.asarray(self.P0d) / self.scale) ** 2), g0)


def track(tr, ckpt=None, resume=False, skip=None, stop_on_breakdown=False, chunk=CHUNK, log=print):
    """Run the tracker over the first ``tr.N`` samples. Returns the dict that ``save_run`` stores.

    skip: boolean mask (length N) of samples without measurement update. They are fed as a NaN measurement, which the guard turns into a rejected correction: state and covariance
          go back to the prediction (pure time update with the known current); ``innov``, ``rejected`` and the violation counts are then not meaningful on these samples.
    stop_on_breakdown: stop after the chunk in which the closure breaks down (non-finite estimate, closure residual above 1e-2); the first such sample is returned as ``breakdown``.
    """
    assert chunk % 20 == 0, "the chunk must be a multiple of 20 (the series are stored every 20th step)"
    I, y, N, t = tr.I, tr.y.copy(), tr.N, tr.t
    if skip is not None: y[:N][np.asarray(skip[:N], bool)] = np.nan
    PSk = PS + (("frozen",) if tr.fcode is not None else ())
    S = dict(out={k: [] for k in SER}, ps={k: [] for k in PSk}, innov_rms=[], resF=[], nrej=[], nviol=[], nkind={k: [] for k in VIOL}, nonfinite=[], first={}, next_ci=0)
    carry = tr.carry0; breakdown = -1; wall0 = time.time(); cpu0 = time.process_time()
    if resume and ckpt and os.path.exists(ckpt):
        st = pickle.load(open(ckpt, "rb")); carry = tuple(jnp.asarray(a) for a in st["carry"]); S = st["S"]; breakdown = st["breakdown"]; wall0 -= st.get("wall", 0.0); log(f"resumed at chunk {S['next_ci'] // chunk}")
    for ci in range(S["next_ci"], N, chunk):
        if breakdown >= 0 and stop_on_breakdown: break
        n_ok = min(chunk, N - ci); sl = slice(ci, ci + chunk)
        Ic, yc = I[sl][:n_ok], y[sl][:n_ok]
        In = np.concatenate([I[ci + 1:ci + n_ok + 1], I[-1:]])[:n_ok]
        if len(In) < n_ok: In = np.concatenate([In, np.full(n_ok - len(In), I[-1])])
        pad = chunk - n_ok
        if pad:
            Ic = np.concatenate([Ic, np.zeros(pad)]); yc = np.concatenate([yc, np.full(pad, yc[-1])]); In = np.concatenate([In, np.zeros(pad)])
        if tr.fcode is None: carry, H = tr._chunk(carry, jnp.asarray(Ic), jnp.asarray(yc), jnp.asarray(In))
        else:
            Fc = np.concatenate([tr.fcode[sl][:n_ok], np.zeros(pad, np.int32)]); carry, H = tr._chunk(carry, jnp.asarray(Ic), jnp.asarray(yc), jnp.asarray(In), jnp.asarray(Fc))
        H = jax.tree_util.tree_map(np.asarray, H)
        meas = np.ones(n_ok, bool) if skip is None else ~np.asarray(skip[ci:ci + n_ok], bool)             # samples with a real measurement update
        for k in SER: S["out"][k].append(H[k][::20])
        for k in PSk: S["ps"][k].append(np.asarray(H[k][:n_ok], dtype=np.float32 if k not in ("rejected", "frozen") else np.int8))
        S["innov_rms"].append(float(np.sqrt(np.mean(H["innov"][:n_ok][meas] ** 2))) if meas.any() else float("nan")); S["resF"].append(float(np.max(H["resF"][:n_ok])))
        rej = H["rejected"][:n_ok] & meas; S["nrej"].append(int(rej.sum()))
        vk = {k: H[k][:n_ok] & meas for k in VIOL}
        S["nviol"].append(int(np.any([vk[k] for k in VIOL], axis=0).sum()))
        for k in VIOL: S["nkind"][k].append(int(vk[k].sum()))
        finite = np.all(np.isfinite(H["vals"][:n_ok]), axis=1) & np.isfinite(H["resF"][:n_ok]); S["nonfinite"].append(int((~finite).sum()))
        # breakdown of the closure / filter: non-finite estimate, closure residual above the limit (at this step or at the next one), non-finite state
        bd = ~finite | ~(H["resF"][:n_ok] <= RES_LIM) | ~(H["F_next"][:n_ok] <= RES_LIM) | H["viol_nf"][:n_ok]
        if skip is not None: bd &= meas
        for name, flag in dict(non_finite=~finite, resF=~(H["resF"][:n_ok] <= RES_LIM), F_next=~(H["F_next"][:n_ok] <= RES_LIM), **{k: vk[k] for k in VIOL}).items():
            if name not in S["first"] and flag.any(): S["first"][name] = ci + int(np.argmax(flag))
        if breakdown < 0 and bd.any(): breakdown = ci + int(np.argmax(bd))
        S["next_ci"] = ci + chunk
        log(f"chunk {ci // chunk} t {t[min(ci + chunk, N) - 1] / 3600:.2f} h innov rms mV {1e3 * S['innov_rms'][-1]:.2f} rejected {S['nrej'][-1]} theta {np.round(S['out']['vals'][-1][-1], 3)} wall {time.time() - wall0:.0f} s", flush=True)
        if ckpt and (ci // chunk) % 10 == 9:
            pickle.dump(dict(carry=[np.asarray(a) for a in carry], S=S, breakdown=breakdown, wall=time.time() - wall0), open(ckpt, "wb"))
    res = {k: np.concatenate(v) for k, v in S["out"].items()}; n10 = (N - 1) // 20 + 1
    res = {k: v[:n10] for k, v in res.items()}
    d = dict(res, **{"ps_" + k: np.concatenate(v) for k, v in S["ps"].items()})
    d.update(t=t[::20][:n10], innov_rms=np.array(S["innov_rms"]), resF_max=np.array(S["resF"]), n_rejected=np.array(S["nrej"]), n_would_reject=np.array(S["nviol"]),
             coords=np.array(COORDS), chunk=chunk, n_steps=N, guard=tr.guard, seed=tr.seed, breakdown=breakdown, n_nonfinite=np.array(S["nonfinite"]),
             wall=time.time() - wall0, cpu=time.process_time() - cpu0, **{"n_" + k: np.array(v) for k, v in S["nkind"].items()}, **{"first_" + k: v for k, v in S["first"].items()})
    return d


def save_run(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); np.savez(path, **d)


# ----------------------------------------------------------------------------------------------------------------- reference bookkeeping
def inventory0(m, P):
    """Lithium inventory of the true initial state, per unit area x electrode area [mol]."""
    z0 = m.init_state(float(P["cn_avg"][0]), float(P["cp_avg"][0]))
    return float(m.lithium(z0, jnp.array([1.0, 1.0, 1.0]))) * ag.AREA


def inventory_loss(m, P, li):
    """Estimated lithium-inventory loss in units of the anode window from the stored total lithium ``li``."""
    return (inventory0(m, P) - li * ag.AREA) / ag.WINDOW_MOL


def step_map(P, R0):
    """(cycle index, step-in-cycle index) of every sample, from the step ids of ``ageing.load`` and the step counts of the reference file."""
    se = R0["step_end_t"]; counts = (~np.isnan(se)).sum(1); sid = P["step"]; us = np.unique(sid)
    assert len(us) == counts.sum(), (len(us), counts.sum())
    flat = np.concatenate([[(c, j) for j in range(n)] for c, n in enumerate(counts)])
    pos = np.searchsorted(us, sid)
    return flat[pos, 0], flat[pos, 1]


def true_value(P, name, k):
    """True effective parameter at sample k (the diffusivity does not age in the reference: 1)."""
    return 1.0 if name == "Ds" else P[name][k]


# ----------------------------------------------------------------------------------------------------------------- evaluation
def evaluate_rpt(d, P, R0, m, label, variant="final", points=POINTS):
    """One row per RPT x read-out point: truth, estimate, error (% of range), sd, z, distance from the logit edge, inventory error, criteria A, B, z.
    Read-out points: end of P6L ('p6l'), end of the slow discharge ('disc'), last sample of the slow charge before the CV ('precv'), end of the CV hold ('cvend', optional), end of the RPT ('end').
    Only points reached by the run (d['n_steps']) are evaluated."""
    vals, sd, li, cu = d["vals"], d["sd_vals"], d["li"], d["corr_ut"]; coords = list(d["coords"]); chunk = int(d["chunk"]) if "chunk" in d else CHUNK; n_done = int(d["n_steps"]) if "n_steps" in d else len(P["t"])
    pair = {f"{coords[a]}_{coords[b]}": k for k, (a, b) in enumerate(zip(*np.triu_indices(len(coords), 1)))}
    Linv = inventory_loss(m, P, li); se = R0["step_end_t"]; rpt_c = np.where(R0["is_rpt"])[0]; last = len(P["Rf"]) - 1; rows = []
    for kr, c in enumerate(rpt_c):
        pts = {"p6l": se[c][12], "disc": se[c][14], "precv": se[c][16], "cvend": se[c][17], "end": np.nanmax(se[c])}
        for lab in points:
            tr_ = pts[lab] if lab in pts else se[c][int(lab)]
            ks = min(int(round(tr_ / DT)), last)
            if ks >= n_done: continue
            kd = min(int(round(tr_ / 10)), len(vals) - 1); ch = min(ks // chunk, len(d["innov_rms"]) - 1)
            r = dict(label=label, variant=variant, rpt=kr, point=lab, t_h=tr_ / 3600, innov_rms_mV_chunk=1e3 * float(d["innov_rms"][ch])); A = B = Z = True
            for i, cc in enumerate(coords):
                t_ = float(true_value(P, cc, ks)); est = vals[kd, i]; rg = RANGE[cc][1] - RANGE[cc][0]; e = 100 * (est - t_) / rg; sdp = 100 * sd[kd, i] / rg
                lo, hi = LOGIT[cc]; p = (est - lo) / (hi - lo); ed = 100 * min(p, 1 - p)
                r.update({f"{cc}_true": t_, f"{cc}_est": est, f"{cc}_err": e, f"{cc}_sd": sdp, f"{cc}_z": e / sdp if sdp > 1e-9 else np.nan, f"{cc}_edge_pct": ed})
                if cc != "LLI": A &= bool(abs(e) <= FLOORS[cc]); Z &= bool(abs(e / sdp) <= 3)
                B &= bool(ed > 2)
            e_inv = 100 * (Linv[kd] - float(true_value(P, "LLI", ks))) / 0.2
            r.update(LLI_inv_err=e_inv, LLI_inv_z=e_inv / r["LLI_sd"]); A &= bool(abs(e_inv) <= FLOORS["LLI_inv"]); Z &= bool(abs(r["LLI_inv_z"]) <= 3)
            r.update(A_ok=A, B_ok=B, z_ok=Z, ABZ_ok=bool(A and B and Z))
            for k_, v_ in pair.items(): r[f"corr_{k_}"] = cu[kd, v_]
            rows.append(r)
    return pd.DataFrame(rows)


def summary(R, d, R0, label, variant="final"):
    """Gates of the paper: criteria A, B, z at the read-out before the CV of RPT 1-3 (retro) and 4-10 (pro), innovation criterion C; counts of rejected updates."""
    rpt_c = np.where(R0["is_rpt"])[0]; se = R0["step_end_t"]; chunk = int(d["chunk"]) if "chunk" in d else CHUNK; nch_all = len(d["innov_rms"])
    t3 = np.nanmax(se[rpt_c[3]]) if len(rpt_c) > 3 else np.inf; nch_retro = min(int(np.ceil(t3 / DT / chunk)), nch_all)
    S = dict(label=label, variant=variant, n_chunks=nch_all, C_all=bool(np.all(d["innov_rms"] <= 3e-3)), innov_rms_max_mV=1e3 * float(np.nanmax(d["innov_rms"])),
             C_retro=bool(np.all(d["innov_rms"][:nch_retro] <= 3e-3)), n_rejected=int(d["n_rejected"].sum()), n_would_reject=int(d["n_would_reject"].sum()))
    def gate(lo, hi, Cok):
        g = R[(R.rpt >= lo) & (R.rpt <= hi) & (R.point == "precv")]; return bool(len(g) and g.ABZ_ok.all() and Cok)
    S["G_retro"] = gate(1, 3, S["C_retro"]); S["G_pro"] = gate(4, 10, S["C_all"])
    for pt in POINTS:
        for lab, lo, hi in [("retro", 1, 3), ("pro", 4, 10)]:
            g = R[(R.rpt >= lo) & (R.rpt <= hi) & (R.point == pt)]
            if not len(g): continue
            S[f"{lab}_{pt}_n_pass_of_{len(g)}"] = int(g.ABZ_ok.sum())
            for c_, n_ in [("Rf", "Rf"), ("LAM", "LAM"), ("Ds", "Ds"), ("LLI_inv", "inv")]: S[f"{lab}_{pt}_maxabs_{n_}"] = float(g[f"{c_}_err"].abs().max())
    return S


def cv_table(d, P, R0, m, label):
    """Diagnostics of the constant-voltage holds: the CV of every RPT (step 17, from the end of the slow charge) and of the first ageing cycle after each RPT (step 3).
    Per CV: change of Rf, LAM, Ds and of the inventory (points of range / of 0.2) between start and end, mean / RMS of the innovation (mV; a non-zero mean = model bias),
    largest estimated anode surface concentration over c_max, guard rejections, mean norm of the parameter rows of the gain."""
    se = R0["step_end_t"]; rpt_c = np.where(R0["is_rpt"])[0]; vals, li = d["vals"], d["li"]; inn = d["ps_innov"].astype(float); Kth = d["ps_K_th"]; cs = d["ps_csn_max"]; rej = d["ps_rejected"]
    Linv = inventory_loss(m, P, li); cvs = []
    for kr, c in enumerate(rpt_c):
        cvs.append(("RPT", kr, c, se[c][16], se[c][17]))
        if c + 1 < len(se) and kr < len(rpt_c) - 1: cvs.append(("P4", kr, c + 1, se[c + 1][2], se[c + 1][3]))
    rows = []
    for kind, kr, c, t0, t1 in cvs:
        a, b = int(round(t0 / DT)), min(int(round(t1 / DT)), len(inn)); ka, kb = int(round(t0 / 10)), min(int(round(t1 / 10)), len(vals) - 1)
        if b - a < 20 or b > len(inn): continue
        x = inn[a:b] * 1e3; r = dict(label=label, kind=kind, block=kr, cycle=c, t_start_h=t0 / 3600, duration_s=t1 - t0, n_steps=b - a)
        for i, nme in enumerate(["Rf", "LAM", "Ds"]): r[f"d_{nme}_pts"] = 100 * (vals[kb, i] - vals[ka, i]) / (RANGE[nme][1] - RANGE[nme][0])
        r["d_inventory_pts"] = 100 * (Linv[kb] - Linv[ka]) / 0.2
        r.update(innov_mean_mV=x.mean(), innov_rms_mV=np.sqrt((x ** 2).mean()), innov_mean_z=x.mean() / (x.std() / np.sqrt(len(x))), innov_mean_sign="+" if x.mean() > 0 else "-",
                 csn_max_over_cmax=float(cs[a:b].max()) / prm.base()["cmax_n"], guard_rejections=int(rej[a:b].sum()), Kth_mean=float(Kth[a:b].mean()))
        rows.append(r)
    return pd.DataFrame(rows)
