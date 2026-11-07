"""Step 4 for the LP604765: local sensitivity and Fisher identifiability at a fitted set (FREE parameters, the three HPPCs).

    python fisher.py --fit results/fit_v1/best.json --workers 16

J = d r / d log(theta) (r = the fit's weighted residual [mV]; log for every parameter whose box is log, the physical
value otherwise, scaled by |theta| so all columns are relative), central differences in unit coordinates.
FIM over 1-Hz samples with white noise of the fit's RMSE (optimistic lower bound). Reported: relative std lower bound
(Cramer-Rao), |correlation| > 0.9 pairs, collinearity index of the normalised columns, local importance ||J_i||.
Output: <fit folder>/fisher.json, fisher.png/.pdf.
"""
import argparse
import json
import os

import numpy as np

import common as C
import fit as Fm

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(C.RES, "fit_v1", "best.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--h", type=float, default=2e-3)
    a = ap.parse_args()
    out = os.path.dirname(os.path.abspath(a.fit))
    best = json.load(open(a.fit))
    free = best.get("free", Fm.FREE)
    Fm.FREE[:] = free
    u0 = np.array([C.to_unit(n, best["params"][n]) for n in free])
    prof = C.profiles(Fm.TESTS)
    with C.pool(Fm.TESTS, a.workers) as pl:
        pb = Fm.Problem(pl, Fm.TESTS, prof, lambda s: None)
        U = []
        for j in range(len(u0)):
            for s in (+1, -1):
                v = u0.copy(); v[j] = v[j] + s * a.h
                U.append(v)
        R = pb.eval_many([u0] + U)
    r0 = R[0]
    n_eff = len(r0)
    # d u / d (relative change of theta): log box -> 1 / ln(hi/lo); linear -> |theta| / (hi - lo)
    J = np.zeros((len(r0), len(free)))
    for j, nm in enumerate(free):
        du = (R[1 + 2 * j] - R[2 + 2 * j]) / (2 * a.h)
        lo, hi, kind = C.BOUNDS[nm]
        th = best["params"][nm]
        J[:, j] = du * (1.0 / np.log(hi / lo) if kind == "log" else max(abs(th), 1e-12) / (hi - lo))
    # r = w e with w^2 = dt / (T_test n_tests): one 1-Hz sample of noise sigma per second of test,
    # F = sum_tests (T_test n_tests) J_test^T J_test / sigma^2, sigma = the fit's RMSE (optimistic: white noise)
    sigma2 = float(np.sum(r0 ** 2))
    scale = np.concatenate([np.full(len(prof[k]["V"]), prof[k]["dt"].sum() * len(Fm.TESTS)) for k in Fm.TESTS])
    F = (J * scale[:, None]).T @ J / sigma2
    try:
        Cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        Cov = np.linalg.pinv(F)
    sd = np.sqrt(np.abs(np.diag(Cov)))
    corr = Cov / np.outer(sd, sd)
    Jn = J / np.linalg.norm(J, axis=0)
    ci = float(1.0 / np.sqrt(np.min(np.linalg.eigvalsh(Jn.T @ Jn))))
    pairs = [(free[i], free[j], float(corr[i, j])) for i in range(len(free)) for j in range(i + 1, len(free))
             if abs(corr[i, j]) > 0.9]
    imp = np.linalg.norm(J, axis=0)
    res = dict(free=free, rel_std=dict(zip(free, map(float, sd))), importance_mV=dict(zip(free, map(float, imp))),
               collinearity_index=ci, high_corr=pairs, sigma_mV=float(np.sqrt(sigma2)), note=
               "rel_std: Cramer-Rao lower bound of the relative std, 1-Hz samples with white noise of the fit's RMSE "
               "(optimistic: the residual is model error, correlated in time)")
    json.dump(res, open(os.path.join(out, "fisher.json"), "w"), indent=1)
    print("sigma %.2f mV, collinearity index %.1f" % (res["sigma_mV"], ci))
    for nm in sorted(free, key=lambda n: -res["importance_mV"][n]):
        print("  %-10s ||dr/dlog|| %8.2f mV   rel std %7.2f %%" % (nm, res["importance_mV"][nm], 100 * res["rel_std"][nm]))
    print("|corr| > 0.9:", [(i, j, round(c, 3)) for i, j, c in pairs])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(HERE, os.pardir, os.pardir, "journal.mplstyle"))
    mm = 1 / 25.4
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 75 * mm), layout="constrained")
    o = np.argsort(-imp)
    ax[0].barh(np.arange(len(free)), imp[o], color="C0"); ax[0].set_yticks(np.arange(len(free)), [free[i] for i in o])
    ax[0].invert_yaxis(); ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\|\partial r / \partial \ln\theta\|$ [mV]")
    im = ax[1].imshow(np.abs(corr[np.ix_(o, o)]), vmin=0, vmax=1, cmap="viridis")
    ax[1].set_xticks(np.arange(len(free)), [free[i] for i in o], rotation=90, fontsize=6)
    ax[1].set_yticks(np.arange(len(free)), [free[i] for i in o], fontsize=6)
    fig.colorbar(im, ax=ax[1], label="|correlation|")
    ax[0].spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(out, "fisher.png")); fig.savefig(os.path.join(out, "fisher.pdf"))


if __name__ == "__main__":
    main()
