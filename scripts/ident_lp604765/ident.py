"""Identifiability analysis of the LP604765 parameter groups at a fitted point (local, sensitivity based).

    python ident.py --fit <results>/fit_v2/best.json --params all --workers 26 --tag v2

Which groups can the three HPPCs determine, and which must come from the literature? Standard, citable criteria:
  1. Sensitivity measure (Brun et al., Water Res. 35 (2001) 1015):  delta_j = ||s_j|| / sqrt(N), with s_j the time-weighted
     residual derivative w.r.t. ln(theta_j) [mV per unit relative change] (physical value for linear boxes, scaled by |theta|).
  2. Collinearity index of a subset K (same ref.): gamma_K = 1 / sqrt(lambda_min(S_K^T S_K)), S = columns normalised to
     unit length; gamma_K < 10-15 = identifiable subset (a change of the other parameters can compensate at most
     1/gamma of a unit change). Every subset of the candidate groups is evaluated; the largest identifiable subset is reported.
  3. Orthogonalisation ranking (Yao et al., Polym. React. Eng. 11 (2003) 563): greedy, each step adds the group with the
     largest sensitivity residual after projection on the groups already chosen; the residual norm is reported.
  4. Effective sample size: N_eff = sum_tests N_k / tau_k, tau_k = 1 + 2 sum_l rho_k(l) (integrated autocorrelation time of
     the residual, summed to its first zero crossing). The residual is model error, strongly correlated in time; the
     white-noise Fisher bound is optimistic by sqrt(tau). Reported: Cramer-Rao relative sd with N_eff, and the profile
     threshold dRMSE/RMSE = sqrt(1 + chi2_{1,0.95} / N_eff) - 1 used by profile_lik.py.
Output: <fit folder>/ident_<tag>.json, ident_<tag>.npz (J, r0), ident_<tag>.png/.pdf.
"""
import argparse

import json
import os

import numpy as np

import common as C
import fit as Fm

HERE = os.path.dirname(os.path.abspath(__file__))
CHI2_95 = 3.841


def jacobian(p0, names, workers, h):
    """Central differences in unit coordinates, converted to d r / d ln(theta)."""
    Fm.FREE[:] = names
    Fm.FIXED.clear(); Fm.FIXED.update({k: v for k, v in p0.items() if k in C.BOUNDS})
    u0 = np.array([C.to_unit(n, p0[n]) for n in names])
    prof = C.profiles(Fm.TESTS)
    with C.pool(Fm.TESTS, workers) as pl:
        pb = Fm.Problem(pl, Fm.TESTS, prof, lambda s: None)
        U = []
        hs = []
        for j in range(len(u0)):
            hp = min(h, 1.0 - u0[j]); hm = min(h, u0[j])     # one-sided at a bound
            for s, hh in ((+1, hp), (-1, hm)):
                v = u0.copy(); v[j] += s * hh
                U.append(v)
            hs.append(hp + hm)
        R = pb.eval_many([u0] + U)
    r0 = R[0]
    J = np.zeros((len(r0), len(names)))
    for j, nm in enumerate(names):
        du = (R[1 + 2 * j] - R[2 + 2 * j]) / hs[j]
        lo, hi, kind = C.BOUNDS[nm]
        J[:, j] = du * (1.0 / np.log(hi / lo) if kind == "log" else max(abs(p0[nm]), 1e-12) / (hi - lo))
    return J, r0, prof


def jacobian_ift(p0, names, workers):
    """Exact d r / d ln(theta) by AD with the IFT (common._vjac), one worker per test: theta dV/dtheta, weighted,
    zero where the voltage clip is active."""
    p0 = dict(C.NOMINAL, **p0)
    prof = C.profiles(Fm.TESTS)
    w = {k: C.weights(prof[k]) / np.sqrt(len(Fm.TESTS)) for k in Fm.TESTS}
    th = np.array([max(abs(p0[n]), 1e-12) for n in names])
    with C.pool(Fm.TESTS, workers) as pl:
        out = pl.map(C._vjac, [(k, list(names), p0, False) for k in Fm.TESTS], chunksize=1)
    R, J = [], []
    for (res, err), k in zip(out, Fm.TESTS):
        if res is None:
            raise RuntimeError("%s: %s" % (k, err))
        V, Jp = res
        Vc = np.clip(np.nan_to_num(V, nan=Fm.V_CLIP[0]), *Fm.V_CLIP)
        act = np.isfinite(V) & (V >= Fm.V_CLIP[0]) & (V <= Fm.V_CLIP[1])
        R.append(1e3 * w[k] * (Vc - prof[k]["V"]))
        J.append((1e3 * w[k] * act)[:, None] * Jp * th[None, :])
    return np.concatenate(J), np.concatenate(R), prof


def tau_int(x):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    k0 = int(np.argmax(ac < 0)) if np.any(ac < 0) else n
    return float(1.0 + 2.0 * ac[1:k0].sum())


def collinearity(Sn, idx):
    ev = np.linalg.eigvalsh(Sn[:, idx].T @ Sn[:, idx])
    return float(1.0 / np.sqrt(max(ev[0], 1e-300)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--params", default="all", help="'all' = every group of common.BOUNDS, 'free' = the fit's free groups, or a list")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--h", type=float, default=2e-3)
    ap.add_argument("--gamma", type=float, default=15.0, help="collinearity threshold of an identifiable subset")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--reuse", action="store_true", help="reuse the saved Jacobian of this tag")
    ap.add_argument("--sets", default="", help="candidate free sets: lab=a,b,c;lab2=...")
    ap.add_argument("--jac", default="fd", choices=("fd", "ift"), help="Jacobian: central differences or exact (IFT)")
    ap.add_argument("--tests", default="hppc", help="fit set: hppc, B, or a list (as fit.py)")
    ap.add_argument("--ocp-p", default=None, help="cathode OCP table (ocp_sub.py json), as in the fit")
    a = ap.parse_args()
    Fm.TESTS[:] = C.TEST_SETS.get(a.tests, a.tests.split(","))
    if a.ocp_p:
        os.environ["LP_OCP_P"] = os.path.abspath(a.ocp_p)
    out = os.path.dirname(os.path.abspath(a.fit))
    best = json.load(open(a.fit))
    p0 = dict(C.NOMINAL, **best["params"])
    names = list(C.BOUNDS) if a.params == "all" else (best.get("free", Fm.FREE) if a.params == "free" else a.params.split(","))
    fj = os.path.join(out, "ident_%s.npz" % a.tag)
    if a.reuse and os.path.exists(fj):
        z = np.load(fj)
        assert list(z["names"]) == names, "saved Jacobian has other parameters"
        J, r0 = z["J"], z["r0"]
        prof = C.profiles(Fm.TESTS)
    else:
        J, r0, prof = jacobian_ift(p0, names, a.workers) if a.jac == "ift" else jacobian(p0, names, a.workers, a.h)
        np.savez(fj, J=J, r0=r0, names=np.array(names))
    # per-test bookkeeping (residual r = w e, w = sqrt(dt / (T_k n_tests)); e [mV] = r / w)
    nk = [len(prof[k]["V"]) for k in Fm.TESTS]
    cut = np.cumsum([0] + nk)
    taus, neff = {}, 0.0
    for i, k in enumerate(Fm.TESTS):
        w = C.weights(prof[k]) / np.sqrt(len(Fm.TESTS))
        e = r0[cut[i]:cut[i + 1]] / w
        # autocorrelation on the 1-s part (the residual sampled per model step; rest steps are long and few)
        taus[k] = tau_int(e[prof[k]["dt"] == 1.0])
        T_k = prof[k]["dt"].sum()
        neff += T_k / taus[k]                      # 1-Hz samples / correlation time
    rmse = float(np.sqrt(np.sum(r0 ** 2)))
    # sensitivity measure per group: the RMS of s_j over time, in mV per unit relative change
    delta = np.sqrt(np.sum(J ** 2, axis=0))
    Sn = J / np.linalg.norm(J, axis=0)
    # Fisher with N_eff (sigma = the fit's RMSE): F = N_eff J^T J / sigma^2 (J already time-averaged)
    F = neff * J.T @ J / rmse ** 2
    Cov = np.linalg.pinv(F)
    sd = np.sqrt(np.abs(np.diag(Cov)))
    corr = Cov / np.outer(sd, sd)
    # orthogonalisation ranking
    rank, resid, rem = [], [], list(range(len(names)))
    X = np.zeros((J.shape[0], 0))
    while rem:
        if X.shape[1]:
            Q, _ = np.linalg.qr(X)
            Rr = J[:, rem] - Q @ (Q.T @ J[:, rem])
        else:
            Rr = J[:, rem]
        nr = np.linalg.norm(Rr, axis=0)
        j = int(np.argmax(nr))
        rank.append(names[rem[j]]); resid.append(float(nr[j]))
        X = np.column_stack([X, J[:, rem[j]]]); rem.pop(j)
    n = len(names)
    # collinearity of the ranked prefixes (the first k groups of the orthogonal ranking) and of named candidate sets
    gam_prefix = [collinearity(Sn, [names.index(r_) for r_ in rank[:k]]) for k in range(1, n + 1)]
    sets = {}
    for spec in (s for s in a.sets.split(";") if s):
        lab, lst = spec.split("=")
        idx = [names.index(x) for x in lst.split(",")]
        sd_k = np.sqrt(np.abs(np.diag(np.linalg.pinv(F[np.ix_(idx, idx)]))))
        sets[lab] = dict(params=lst.split(","), gamma=collinearity(Sn, idx),
                         rel_sd_crlb_Neff=dict(zip(lst.split(","), map(float, sd_k))))
        print("set %s (%d groups): gamma %.1f" % (lab, len(idx), sets[lab]["gamma"]))
    best_sub = dict(prefix_gamma=dict(zip(rank, gam_prefix)), sets=sets)
    res = dict(point=a.fit, params=names, rmse_mV=rmse, tau_s=taus, N_eff=neff,
               profile_threshold_rel=float(np.sqrt(1.0 + CHI2_95 / neff) - 1.0),
               delta_mV=dict(zip(names, map(float, delta))),
               rel_sd_crlb_Neff=dict(zip(names, map(float, sd))),
               gamma_all=collinearity(Sn, list(range(n))), gamma_threshold=a.gamma,
               largest_identifiable=best_sub, orth_rank=rank, orth_resid_mV=resid,
               high_corr=[(names[i], names[j], float(corr[i, j])) for i in range(n) for j in range(i + 1, n) if abs(corr[i, j]) > 0.9])
    json.dump(res, open(os.path.join(out, "ident_%s.json" % a.tag), "w"), indent=1)

    print("RMSE %.2f mV, N_eff %.0f (tau %s s), profile threshold +%.3f %% of the RMSE" %
          (rmse, neff, {k: round(v) for k, v in taus.items()}, 100 * res["profile_threshold_rel"]))
    print("gamma(all %d) = %.0f" % (n, res["gamma_all"]))
    for k, (nm, rr) in enumerate(zip(rank, resid)):
        print("  %2d %-10s delta %8.2f  orth. resid %8.2f mV  gamma(first %2d) %8.1f" % (k + 1, nm, res["delta_mV"][nm], rr, k + 1, gam_prefix[k]))
    print("|corr| > 0.9 (all groups free):", [(i, j, round(c, 3)) for i, j, c in res["high_corr"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 70 * mm), layout="constrained")
    o = [names.index(r_) for r_ in rank]
    ax[0].barh(np.arange(n), delta[o], color="0.7", label=r"$\delta_j$")
    ax[0].barh(np.arange(n), np.array(resid), color="C0", label="orth. residual")
    ax[0].set_yticks(np.arange(n), rank); ax[0].invert_yaxis(); ax[0].set_xscale("log")
    ax[0].set_xlabel("mV per unit relative change"); ax[0].legend(frameon=False, fontsize=7)
    sizes = sorted(best_sub)
    gam = [collinearity(Sn, [names.index(r_) for r_ in rank[:s]]) for s in range(1, n + 1)]
    ax[1].semilogy(np.arange(1, n + 1), gam, "o-", ms=3, label="first $k$ of the ranking")
    ax[1].axhline(a.gamma, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("subset size $k$"); ax[1].set_ylabel(r"collinearity index $\gamma_K$"); ax[1].legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, "ident_%s.png" % a.tag), dpi=200); fig.savefig(os.path.join(out, "ident_%s.pdf" % a.tag))


if __name__ == "__main__":
    main()
