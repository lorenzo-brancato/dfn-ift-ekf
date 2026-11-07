"""Cramer-Rao lower bounds of the ageing parameters from the exact voltage sensitivities (Fig. 5 grey bars, Sec. III-B, Appendix D).

Fisher information of a voltage record y_k = V_k(theta) + e_k, e_k ~ N(0, sigma_V^2) independent:  I = sigma_V^-2 sum_k (dV_k/dtheta)^T (dV_k/dtheta),
with the sensitivities dV_k/dtheta propagated along the trajectory by the structured Jacobians (``sens.traj_sens``, implicit function theorem).

Three bounds are produced from the same sensitivities (the version names are the values of the ``version`` column of the output tables):

``conditional_on_LLI``  THE bound of the paper (grey bars of Fig. 5). The filter estimates (R_f, LAM, D_s) with the lithium-inventory offset LLI fixed
                        at its true value. The bound of the three estimated parameters with LLI known is the inverse of the 3 x 3 sub-block of the
                        4-parameter Fisher matrix (R_f, LAM, LLI, D_s) obtained by deleting the LLI row and column; it equals the Schur complement
                        of the 4 x 4 covariance with respect to LLI (``schur_complement``, checked numerically in ``scripts/crlb.py``).
``known_state``         4-parameter marginal bound (R_f, LAM, LLI, D_s), the initial state known: the diagonal of the inverse of the full 4 x 4 matrix.
                        It is larger than the conditional one for R_f and LAM, which are strongly correlated with LLI (|corr(LAM, LLI)| = 0.89-0.98 on pulses).
``unknown_soc``         initial SOC unknown: the initial stoichiometry shifts of the two electrodes (dc_n0, dc_p0) are two extra nuisance parameters; LLI is
                        removed because it is degenerate with dc_n0 (both shift the initial anode concentration). Parameters (R_f, LAM, D_s, dc_n0, dc_p0).
``known_state+k0``      optional: the anode exchange-current constant k0 as fifth parameter (finite-difference column, h = 2 %).

Correspondence with the column ``version`` of the earlier tables (``tab_crlb_m17.csv``): known_state = stato_noto, unknown_soc = SOC_incognito,
known_state+k0 = stato_noto+k0; ``conditional_on_LLI`` is new (it was assembled by hand from stato_noto).
"""
import numpy as np

RANGE_WIDTH = dict(Rf=9.0, LAM=0.3, LLI=0.2, Ds=0.7, k0=1.0)           # width of the ageing ranges (R_f in [1,10], LAM in [0,0.3], LLI in [0,0.2], D_s in [0.3,1])
TABLE_NAME = dict(Rf="R_f", LAM="LAM_n", LLI="LLI", Ds="D_s", k0="k0", dcn="dc_n0", dcp="dc_p0")
COLUMN = dict(Rf=0, LAM=1, LLI=2, Ds=3, dcn=4, dcp=5)                  # columns of the sensitivity matrix returned by sens.traj_sens(par=(Rf, LAM, LLI, Ds), nuis=True)
VERSIONS = {
    "known_state": dict(names=("Rf", "LAM", "LLI", "Ds"), keep=None),
    "conditional_on_LLI": dict(names=("Rf", "LAM", "LLI", "Ds"), keep=("Rf", "LAM", "Ds")),
    "unknown_soc": dict(names=("Rf", "LAM", "Ds", "dcn", "dcp"), keep=None),
}
SOC_START = {"P6Lx2": 0.5, "P6L": 0.5, "P2": 1.0, "D1": 0.8}            # initial SOC of the protocols; th3 starts at 0.75 in the P6L protocols
SIGMA_V = 2e-3                                                           # measurement noise standard deviation [V]


def fisher(dV, sigma=SIGMA_V):
    """Fisher matrix sigma^-2 dV^T dV of the sensitivity matrix dV [n_samples, n_par] (independent Gaussian noise)."""
    dV = np.asarray(dV, float)
    return dV.T @ dV / sigma ** 2


def invert(F):
    """Inverse of a Fisher matrix (pseudo-inverse if it is numerically singular, cond > 1e15)."""
    return np.linalg.pinv(F) if np.linalg.cond(F) > 1e15 else np.linalg.inv(F)


def sd_corr(C):
    """Standard deviations and correlation matrix of a covariance matrix."""
    sd = np.sqrt(np.diag(C))
    return sd, C / np.outer(sd, sd)


def conditional(F, keep):
    """Covariance of the parameters ``keep`` (indices into F) when all the others are known: inverse of the sub-block F[keep, keep]."""
    keep = list(keep)
    return invert(F[np.ix_(keep, keep)])


def schur_complement(C, keep):
    """Schur complement of the covariance C with respect to the parameters not in ``keep``: C_kk - C_kf C_ff^-1 C_fk.
    It equals the inverse of the sub-block of the Fisher matrix, ``conditional(inv(C), keep)``."""
    keep = list(keep); free = [i for i in range(C.shape[0]) if i not in keep]
    if not free: return C[np.ix_(keep, keep)]
    return C[np.ix_(keep, keep)] - C[np.ix_(keep, free)] @ np.linalg.solve(C[np.ix_(free, free)], C[np.ix_(free, keep)])


def version_rows(dV, version, sigma=SIGMA_V):
    """Bound of one version from the sensitivity matrix dV [n, 6] (columns Rf, LAM, LLI, Ds, dc_n0, dc_p0). Returns (names, sd, corr, cond_fisher)."""
    spec = VERSIONS[version]; names = list(spec["names"])
    F = fisher(dV[:, [COLUMN[n] for n in names]], sigma)
    if spec["keep"] is None:
        C = invert(F); cond = float(np.linalg.cond(F)); out = names
    else:
        keep = [names.index(n) for n in spec["keep"]]
        C = conditional(F, keep); cond = float(np.linalg.cond(F[np.ix_(keep, keep)])); out = list(spec["keep"])
    sd, corr = sd_corr(C)
    return out, sd, corr, cond


def table_rows(dV, protocol, tag, theta, versions=("known_state", "conditional_on_LLI", "unknown_soc"), sigma=SIGMA_V, dk0=None):
    """Rows of the output table (one per version and parameter): sd in the parameter's units, sd as percent of the range, correlation row, Fisher condition number.
    ``dk0``: optional finite-difference sensitivity column of k0 (adds the version known_state+k0)."""
    rows = []
    for v in versions:
        names, sd, corr, cond = version_rows(dV, v, sigma)
        rows += _rows(protocol, tag, theta, v, names, sd, corr, cond, len(dV))
    if dk0 is not None:
        D5 = np.concatenate([dV[:, :4], np.asarray(dk0)[:, None]], 1); F = fisher(D5, sigma); C = invert(F); sd, corr = sd_corr(C)
        rows += _rows(protocol, tag, theta, "known_state+k0", ["Rf", "LAM", "LLI", "Ds", "k0"], sd, corr, float(np.linalg.cond(F)), len(dV))
    return rows


def _rows(protocol, tag, theta, version, names, sd, corr, cond, n):
    rows = []
    for i, nm in enumerate(names):
        w = RANGE_WIDTH.get(nm, np.nan)
        row = dict(protocol=protocol, theta_tag=tag, theta=str(list(theta)), version=version, parameter=TABLE_NAME[nm], sd=float(sd[i]), range_width=w, sd_pct_range=float(100 * sd[i] / w) if nm in RANGE_WIDTH else np.nan)
        for j, nj in enumerate(names): row[f"corr_{TABLE_NAME[nj]}"] = float(corr[i, j])
        row.update(cond_fisher=cond, n_samples=n)
        rows.append(row)
    return rows


def soc_effect(rows):
    """Effect of an unknown initial SOC on the bound of the estimated parameters: for each of (R_f, LAM_n, D_s), the bound with unknown SOC, the conditional bound
    and their difference [percent of range] (the paper: LAM at most 0.5 %, the other parameters change by at most 0.12 %)."""
    import pandas as pd
    d = pd.DataFrame(rows); out = []
    for (proto, tag), g in d.groupby(["protocol", "theta_tag"], sort=False):
        for p in ("R_f", "LAM_n", "D_s"):
            a = g[(g.version == "conditional_on_LLI") & (g.parameter == p)].sd_pct_range.iloc[0]; b = g[(g.version == "unknown_soc") & (g.parameter == p)].sd_pct_range.iloc[0]
            out.append(dict(protocol=proto, theta_tag=tag, parameter=p, crlb_conditional_pct=a, crlb_unknown_soc_pct=b, change_pct=b - a))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- sensitivities along a protocol (needs jax)
def initial_concentrations(protocol, tag, data_dir="data/sim"):
    """Initial uniform concentrations (anode, cathode) of the cell at the start of the protocol for the ageing state ``tag`` (before the LLI shift).
    For the P6L protocols they are read from the stored PyBaMM reference (th3 starts at SOC 0.75, a value that otherwise needs a PyBaMM solve); otherwise from
    ``reference.c0_for_soc`` (tabulated for SOC 0.5 and 1.0; other values need PyBaMM)."""
    import os
    from . import reference
    if protocol in ("P6Lx2", "P6L"):
        f = reference.fname(protocol, tag, (40, 20, 40), 0.5, None, data_dir)
        if os.path.exists(f):
            d = np.load(f); return tuple(float(x) for x in d["c0"])
    soc = 0.75 if (tag == "th3" and protocol in ("P6Lx2", "P6L")) else SOC_START[protocol]
    return tuple(float(x) for x in reference.c0_for_soc(soc))


def protocol_current(protocol, dt):
    from .protocols import current_series, d1_current
    return np.repeat(d1_current(), int(round(1.0 / dt))) if protocol == "D1" else current_series(protocol, dt)


def sensitivities(m, theta, protocol, tag, data_dir="data/sim", I=None):
    """dV/d(Rf, LAM, LLI, Ds, dc_n0, dc_p0) along the protocol for the model ``m`` (a ``step.DFN``; the grid is the one of the model) at the true state theta = (Rf, LAM, LLI, Ds).
    Returns the array [n_samples, 6]; the initial state is the uniform concentration of the protocol with the anode shifted by LLI."""
    import jax
    import jax.numpy as jnp
    from .ageing import DX_N
    from .sens import traj_sens
    Rf, LAM, LLI, Ds = [float(x) for x in theta]
    th = jnp.array([Rf, 1 - LAM, Ds], dtype=float)
    c0 = initial_concentrations(protocol, tag, data_dir)
    z0 = m.init_state(c0[0] - LLI * DX_N * m.P["cmax_n"], c0[1])
    I = jnp.asarray(protocol_current(protocol, m.dt) if I is None else I)
    f = jax.jit(lambda th_, z0_, I_: traj_sens(m, th_, z0_, I_, par=("Rf", "LAM", "LLI", "Ds"), nuis=True))
    return np.asarray(f(th, z0, I))


def k0_column(m, theta, protocol, tag, data_dir="data/sim", I=None, h=0.02):
    """Central finite-difference sensitivity of V to the relative anode exchange-current constant k0 (exact closed-loop model, h = 2 %)."""
    import jax
    import jax.numpy as jnp
    from .ageing import DX_N
    from .rom import simulate
    from .step import DFN
    Rf, LAM, LLI, Ds = [float(x) for x in theta]
    th = jnp.array([Rf, 1 - LAM, Ds], dtype=float)
    c0 = initial_concentrations(protocol, tag, data_dir)
    z0 = m.init_state(c0[0] - LLI * DX_N * m.P["cmax_n"], c0[1])
    I = jnp.asarray(protocol_current(protocol, m.dt) if I is None else I); Vs = []
    for s in (1 + h, 1 - h):
        P = dict(m.P); P["k0_n"] = m.P["k0_n"] * s; mm = DFN(P, m.Nn, m.Ns, m.Np, m.Nm, m.K, m.dt, m.grid)
        Vs.append(np.asarray(jax.jit(lambda z0_, I_, th_: simulate(mm, z0_, I_, th_, k=None)["V"])(z0, I, th)))
    return (Vs[0] - Vs[1]) / (2 * h)
