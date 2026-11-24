"""Selection rule of the ApEn draft (Section 2.4.2) on the local sensitivities of ident.py.

    python select_groups.py --ident <fit folder>/ident_<tag>.npz [--smin 1.0] [--gamma 15]

S_j = RMS over the calibration tests of theta_j dV/dtheta_j [mV per unit relative change] (ident.py: delta_j).
Rule: every group with S_j < smin is fixed; then, while the collinearity index gamma_K (Brun et al. 2001) of the
remaining groups is above the threshold, the group whose removal lowers gamma_K the most is fixed (ties: the less
sensitive first). Prints the steps; writes <ident stem>_select.json.
"""
import argparse
import json

import numpy as np


def gamma(Sn, idx):
    ev = np.linalg.eigvalsh(Sn[:, idx].T @ Sn[:, idx])
    return float(1.0 / np.sqrt(max(ev[0], 1e-300)))


def select(J, names, smin=1.0, gmax=15.0, log=print):
    S = np.sqrt(np.sum(J ** 2, axis=0))
    Sn = J / np.where(S > 0, S, 1.0)
    keep = [j for j in range(len(names)) if S[j] >= smin]
    steps = [dict(step="S_j < %.1f mV" % smin, fixed=[names[j] for j in range(len(names)) if S[j] < smin])]
    log("S_j [mV]: " + ", ".join("%s %.2f" % (names[j], S[j]) for j in np.argsort(-S)))
    log("fixed (S_j < %.1f mV): %s" % (smin, steps[0]["fixed"]))
    g = gamma(Sn, keep)
    log("gamma of the %d remaining: %.1f" % (len(keep), g))
    while len(keep) > 1 and g > gmax:
        cand = sorted(((gamma(Sn, [k for k in keep if k != j]), S[j], j) for j in keep))
        g, _, j = cand[0]
        keep.remove(j)
        steps.append(dict(step="gamma", fixed=[names[j]], gamma_after=g))
        log("  fix %-8s (S %.2f mV) -> gamma %.1f with %d groups" % (names[j], S[j], g, len(keep)))
    return dict(S_mV=dict(zip(names, map(float, S))), selected=[names[j] for j in keep], gamma=g, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", required=True)
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=15.0)
    ap.add_argument("--names", default="", help="restrict to these groups (comma list)")
    a = ap.parse_args()
    z = np.load(a.ident)
    names = list(z["names"]); J = z["J"]
    if a.names:
        idx = [names.index(n) for n in a.names.split(",")]
        J = J[:, idx]; names = [names[i] for i in idx]
    res = select(J, names, a.smin, a.gamma)
    print("selected (%d): %s, gamma %.1f" % (len(res["selected"]), res["selected"], res["gamma"]))
    json.dump(res, open(a.ident.replace(".npz", "_select.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
