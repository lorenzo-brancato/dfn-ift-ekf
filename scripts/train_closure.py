"""Appendix E ('The network'): training set and training of the learned closure (MIONet, 56802 trainable parameters).

Training set: 80000 states of the grey-box grid (16000 base states x 5 variants), drawn from an EXPLICIT list of PyBaMM reference simulations (``TRAINING_REFS`` below),
projected onto the grid cells, perturbed, and labelled with the converged Newton solver (tolerance 1e-12) (``dfnekf.closure_data``). Training: Adam with cosine decay,
80000 iterations of batch 512, seed 0; validation split of 20 % of the base states (``default_rng(1)``), so that a state and its perturbed variants never straddle the split.
Seeds, split and hyper-parameters are those of the first version of the network; only the list of source references changed.

EVALUATION PROTOCOLS ARE NOT USED FOR TRAINING. No reference of the evaluation protocols F6, F6C, F6Q, F6Q3 (any state) and no reference at th2 (the state of the ablation, Table VIII,
and of Fig. 11) enters the training set; the script refuses to run if the list violates this (``check_refs``). The first version of the network drew 35 references selected by file
date, which included the F6 reference at th2. The exact list used is printed and written to ``<results>/closure_training_refs.csv``.

Outputs (``<results>`` = --results; with --quick everything goes to ``<results>/quick``):
  closure_4to1.pkl              network (keys params, norm, grid, hist, tr, va, refs, ...); closure_uniform.pkl for --grid uniform
  closure_training_refs.csv     the source references actually used (and closure_training_refs_excluded.csv: the candidates that were removed, with the reason)
  closure_validation.csv        validation errors (median, 99th percentile of the relative error of the reaction currents), overall and per current band
  closure_training_history.csv  loss and validation errors during training; closure_training.log: the printed log
  traj/closure_training_set.npz the labelled set (only with --save-set; about 135 MB, not committed)

Commands (about 10 min wall on the machine of the paper: 628 s wall, 1181 s CPU; plus a few minutes for the labels):
  python scripts/train_closure.py --grid 4:1 --data data --results results
  python scripts/train_closure.py --quick --grid 4:1 --data data --results results       # smoke test: 4 references, 2500 samples, 200 iterations
Paper numbers: validation median 1.5 % and 99th percentile 13.5 % (first version of the network; the retrained network on the list below gives the numbers of this log).
"""
import os
import sys
import time
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime                                             # no jax import here

# Source references of the training set: '<protocol>_<state>' files <data>/ref_<protocol>_<state>_m40-20-40_dt0.5.npz (mesh 40/20/40, period 0.5 s).
# Nominal / ageing-direction / single-parameter-bias references of the identification protocols, 1C and 3C discharges, pulse trains and a drive cycle; states th1, th3, th4, th5 of the
# identification protocol P6Lx2 (the th2 reference of P6Lx2 and all F6 references are excluded, see the module docstring).
TRAINING_REFS = [
    "D1_age", "D1_nom",
    "P1_1C_age", "P1_1C_nom",
    "P1_3C_age", "P1_3C_bDs", "P1_3C_bLAM", "P1_3C_bLLI", "P1_3C_bRf", "P1_3C_nom", "P1_3C_nomVT",
    "P2C300_nomVT",
    "P2_age", "P2_bDs", "P2_bLAM", "P2_bLLI", "P2_bRf", "P2_nom", "P2_nomVT",
    "P6L_age", "P6L_bDs", "P6L_bLAM", "P6L_bLLI", "P6L_bRf", "P6L_nom",
    "P6Lx2_age", "P6Lx2_nom", "P6Lx2_th1", "P6Lx2_th3", "P6Lx2_th4", "P6Lx2_th5",
]
# First version: the 35 references listed here minus the ones removed by the project decision (names kept for the audit trail).
REMOVED_REFS = {"F6_th2": "evaluation protocol F6 and state th2", "F6_th4": "evaluation protocol F6", "F6_th5": "evaluation protocol F6", "P6Lx2_th2": "state th2 (Table VIII, Fig. 11)"}
EVAL_PROTOCOLS = ("F6", "F6C", "F6Q", "F6Q3")                           # evaluation protocols, excluded for every state
EVAL_STATE = "th2"                                                      # state of Table VIII / Fig. 11, excluded for every protocol
QUICK_REFS = ["P6Lx2_th1", "P2_nom", "P1_3C_nom", "P6L_nom"]

N_SAMPLES, N_ITERS, LOG_EVERY = 80000, 80000, 5000
QUICK_SAMPLES, QUICK_ITERS = 2500, 200


def check_refs(refs):
    """Refuse a list that contains an evaluation protocol or the evaluation state."""
    bad = [r for r in refs if r.rsplit("_", 1)[0] in EVAL_PROTOCOLS or r.rsplit("_", 1)[1] == EVAL_STATE]
    if bad: raise SystemExit(f"training references must exclude the evaluation protocols {EVAL_PROTOCOLS} and the state {EVAL_STATE}: {bad}")
    if len(set(refs)) != len(refs): raise SystemExit("duplicate training references")


class Tee:
    """Write to several streams (console and log file)."""
    def __init__(self, *s): self.s = s
    def write(self, x):
        for f in self.s: f.write(x)
    def flush(self):
        for f in self.s: f.flush()


def main():
    p = cli.parser("Appendix E (network)", "training set and training of the learned closure on an explicit list of references, excluding the evaluation protocols")
    p.add_argument("--iters", type=int, default=None, help=f"training iterations (default {N_ITERS}; --quick {QUICK_ITERS})")
    p.add_argument("--n-samples", type=int, default=None, help=f"number of training samples (default {N_SAMPLES}; --quick {QUICK_SAMPLES})")
    p.add_argument("--save-set", action="store_true", help="also save the labelled training set to <results>/traj/closure_training_set.npz")
    a = p.parse_args()
    runtime.setup(a.core)
    import numpy as np, pandas as pd, jax
    import jax.numpy as jnp
    from dfnekf import closure_data as cd, closure_net as cn, params as prm
    if a.quick: a.results = os.path.join(a.results, "quick")
    os.makedirs(a.results, exist_ok=True)
    refs = QUICK_REFS if a.quick else TRAINING_REFS
    check_refs(TRAINING_REFS); check_refs(refs)
    n_samples = a.n_samples or (QUICK_SAMPLES if a.quick else N_SAMPLES); iters = a.iters or (QUICK_ITERS if a.quick else N_ITERS)
    tag = "uniform" if a.grid == "uniform" else "4to1"; clock = cli.Clock(a, "train_closure.py")
    log = open(cli.out_path(a, "closure_training.log"), "w", encoding="utf-8"); out = Tee(sys.stdout, log)
    with contextlib.redirect_stdout(out):
        files = sorted(refs, key=lambda n: f"ref_{n}_m40-20-40_dt0.5.npz")          # order of the file names (fixes the order of the source states)
        print(f"training references ({len(files)}), excluded evaluation protocols {EVAL_PROTOCOLS} and state {EVAL_STATE}:")
        print("  " + " ".join(files), flush=True)
        t0 = time.time(); c0 = time.process_time()
        S = cd.coarse_states(files, a.data, a.grid)
        D = cd.label(cd.build(S, n_samples, seed=1, hiI=0.5, ce_amp=0.4), a.grid, tol=1e-12)
        res = D["res"]
        print(f"{len(res)} samples from {len(S['csn'])} source states; label residual: max {res.max():.3e}, fraction <= 1e-12 {np.mean(res <= 1e-12):.4f}, <= 1e-9 {np.mean(res <= 1e-9):.4f}; "
              f"max Newton iterations {int(D['nit'].max())}; labels {time.process_time() - c0:.0f} s CPU", flush=True)
        if a.save_set:
            os.makedirs(os.path.join(a.results, "traj"), exist_ok=True); np.savez_compressed(os.path.join(a.results, "traj", "closure_training_set.npz"), **D)
        bid = D["base_id"]; nb = bid.max() + 1
        rng = np.random.default_rng(1); vb = rng.choice(nb, nb // 5, replace=False); isv = np.isin(bid, vb); tr, va = np.where(~isv)[0], np.where(isv)[0]
        cfg = cn.make_config(cd.NN, cd.NS, cd.NP, a.grid)
        npar = cn.n_params(cn.init(jax.random.PRNGKey(0), cfg)); print(f"train {len(tr)} val {len(va)} params {npar}", flush=True)
        c1 = time.process_time(); t1 = time.time()
        params, norm, hist = cn.train(cfg, D, tr, va, iters=iters, log_every=max(1, min(LOG_EVERY, iters // 4)))
        cpu, wall = time.process_time() - c1, time.time() - t1
        cn.save_net(cli.out_path(a, f"closure_{tag}.pkl"), params, norm, a.grid, hist=hist, tr=tr, va=va, refs=files, dims=(cd.NN, cd.NS, cd.NP), train_time=wall, iters=iters, n_samples=len(res))
        err = np.asarray(cn.rel_err(cfg, jnp.asarray(cn.predict(params, norm, cfg, D, va)), jnp.asarray(D["g"][va, :cd.NN + cd.NP])))
        Ih = np.abs(D["I"][va]) / prm.C_RATE_A
        bands = [("rest (I = 0)", Ih == 0), ("0-0.5C", (Ih > 0) & (Ih <= 0.5)), ("0.5-1.5C", (Ih > 0.5) & (Ih <= 1.5)), ("1.5-3C", (Ih > 1.5) & (Ih <= 3)), ("3-5C", Ih > 3)]
        rows = [dict(band="all validation", n=len(va), median_pct=100 * float(np.median(err)), p99_pct=100 * float(np.quantile(err, .99)), n_params=npar, n_samples_total=len(res), n_train=len(tr), n_val=len(va), iters=iters,
                     train_wall_s=wall, train_cpu_s=cpu, res_label_max=float(res.max()), frac_res_le_1e12=float(np.mean(res <= 1e-12)), final_train_loss=hist[-1][1])]
        for nm, mk in bands: rows.append(dict(band=nm, n=int(mk.sum()), median_pct=100 * float(np.median(err[mk])) if mk.any() else np.nan, p99_pct=100 * float(np.quantile(err[mk], .99)) if mk.any() else np.nan))
        V = pd.DataFrame(rows); print(V.round(3).to_string(), flush=True)
        print(f"VALIDATION (relative error of the reaction currents, held-out base states): median {rows[0]['median_pct']:.2f} %, 99th percentile {rows[0]['p99_pct']:.2f} %  (first version of the network: 1.5 % / 13.5 %)", flush=True)
    note = "relative error of the reaction currents with a floor of 1 % of the 1C current scale; validation = 20 % of the base states"
    cli.write_csv(V, a, "closure_validation.csv", note=note, float_format=None)
    cli.write_csv(pd.DataFrame(hist, columns=["iteration", "train_loss", "val_median_rel_err", "val_p99_rel_err"]), a, "closure_training_history.csv", note="training history", float_format=None)
    R = pd.DataFrame([dict(reference=n, protocol=cd.split_ref_name(n)[0], state=cd.split_ref_name(n)[1], file=f"ref_{n}_m40-20-40_dt0.5.npz") for n in files])
    cli.write_csv(R, a, "closure_training_refs.csv", note="explicit list of the source references of the training set (no F6/F6C/F6Q/F6Q3 reference, no reference at th2)", float_format=None)
    cli.write_csv(pd.DataFrame([dict(reference=k, reason=v) for k, v in REMOVED_REFS.items()]), a, "closure_training_refs_excluded.csv", note="references of the first version of the training set that are not used", float_format=None)
    clock.log(f"val median {rows[0]['median_pct']:.3f} % p99 {rows[0]['p99_pct']:.3f} %")
    print("wrote", cli.out_path(a, f"closure_{tag}.pkl")); log.close()


if __name__ == "__main__":
    main()
