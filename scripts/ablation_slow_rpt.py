"""Ablation of the slow segment of the RPT (Sec. III-D): "with pulses alone the filter confused film resistance with loss of active material; the measured capacity tells them apart".

The final tracker (ageing_track.py: 4 parameters, guard, 4:1 grid, same noise sequence) is fed ONLY the pulse part of every RPT. In every RPT the samples of steps 14-18 of the cycle
(slow C/2 discharge to 2.5 V, rest, slow C/2 charge to 4.2 V, CV hold to C/50, final rest: about 5 h of the 7.5 h) receive NO measurement update: the voltage is replaced by NaN, which the
guard of the tracker turns into a rejected correction, i.e. the state and the covariance are advanced by the prediction alone with the known current (the parameters stay at their last
values; their covariance grows with the process noise). Everything else is unchanged: RPT steps 0-13 (discharge to 3.65 V, rest, pulses P6L, rest after the pulses) and all ordinary
ageing cycles between the RPTs are measured as in the final tracker. Run to --hours (default 100 h: RPT 0-3, i.e. the retrospective window of the paper).

Read-out points compared at every RPT, with errors in % of the ageing range against the truth (columns of ageing_track): p6l = end of the pulses; rest = end of the rest after the pulses (the last
measured sample of the pulses-only run); precv = last sample of the slow charge (main point of the final tracker; in the pulses-only run it is a pure prediction from the rest point); end = end of the RPT.
The reference run (variant with_slow) is the final tracker over the same horizon: read from <tracking>/traj/ageing_track_seed<s>.npz (the full run of ageing_track.py), or recomputed
with --with-reference (costs as much again).

  python scripts/ablation_slow_rpt.py --grid 4:1 --data data --results results/ablation_slow_rpt --seeds 0 --hours 100 --tracking results/ageing --core 0     # about 5 h CPU (100 h simulated at the ~20 simulated h per CPU h of the tracker)
  python scripts/ablation_slow_rpt.py --quick --data data --results results/quick       # 0.1 h; tests the skipping mechanism on a window of the pulses

Outputs: traj/ablation_slow_rpt_seed<s>.npz, ablation_slow_rpt.csv (variant x RPT x point: truth, errors, z, corr(Rf, LAM), criteria), ablation_slow_rpt_summary.csv (largest errors per variant and point, RPT >= 1).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

REF_NAME = "aging_dyn_rptslow_full.npz"
POINTS = ("p6l", "13", "precv", "end")
FIRST_SKIPPED_STEP = 14


def slow_mask(P, R0):
    """True on the samples of RPT steps >= 14 (slow discharge, rest, slow charge, CV, rest)."""
    from dfnekf import ageing_tracker as at
    cyc, sti = at.step_map(P, R0)
    return R0["is_rpt"][cyc] & (sti >= FIRST_SKIPPED_STEP)


def evaluate(at, d, P, R0, m, variant, seed):
    R = at.evaluate_rpt(d, P, R0, m, f"seed{seed}", variant=variant, points=POINTS)
    if not len(R): return R
    R["point"] = R["point"].replace({"13": "rest"})
    R.insert(0, "seed", seed); return R.drop(columns=["label"])


def main():
    p = cli.parser("Sec. III-D (slow segment of the RPT)", "final tracker fed with the pulse part of the RPT only vs with the slow segment", seeds="0")
    p.add_argument("--hours", type=float, default=100.0); p.add_argument("--tracking", default="results/ageing")
    p.add_argument("--with-reference", action="store_true", help="recompute the with_slow variant instead of reading it")
    p.add_argument("--chunk", type=int, default=None); p.add_argument("--ref", default=None); p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    runtime.setup(a.core)
    import numpy as np, pandas as pd
    from dfnekf import ageing, ageing_tracker as at
    ref = a.ref or os.path.join(a.data, REF_NAME); P = ageing.load(ref); R0 = np.load(ref); clock = cli.Clock(a, "ablation_slow_rpt")
    n = 360 if a.quick else int(a.hours * 3600 / at.DT) + 1; chunk = a.chunk or (120 if a.quick else at.CHUNK)
    skip = slow_mask(P, R0)
    if a.quick: skip = np.zeros(len(P["t"]), bool); skip[200:300] = True            # test of the mechanism on a window of the pulses
    tables = []
    for s in cli.parse_ids(a.seeds):
        tr = at.Tracker(P, a.grid, seed=s, guard=True, nsteps=n); name = f"ablation_slow_rpt_seed{s}"
        os.makedirs(os.path.join(a.results, "traj"), exist_ok=True); ck = os.path.join(a.results, "traj", name + "_ckpt.pkl")
        d = at.track(tr, ckpt=ck, resume=a.resume, skip=skip, chunk=chunk); at.save_run(os.path.join(a.results, "traj", name + ".npz"), d)
        tables.append(evaluate(at, dict(d), P, R0, tr.m, "pulses_only", s))
        if a.quick:
            dm = d["vals"]; kk = np.arange(len(dm)) * 20; w = (kk >= 200) & (kk < 300)
            print(f"mechanism check: parameters constant over the skipped window: {bool(np.allclose(dm[w][:, :4], dm[w][0, :4]))}; rejected updates on measured samples: {int(d['n_rejected'].sum())}", flush=True)
        ref_f = os.path.join(a.tracking, "traj", f"ageing_track_seed{s}.npz")
        if a.with_reference or a.quick:
            tr2 = at.Tracker(P, a.grid, seed=s, guard=True, nsteps=n); d2 = at.track(tr2, chunk=chunk); at.save_run(os.path.join(a.results, "traj", f"{name}_with_slow.npz"), d2)
            tables.append(evaluate(at, d2, P, R0, tr.m, "with_slow", s))
        elif os.path.exists(ref_f):
            d2 = dict(np.load(ref_f)); d2["n_steps"] = n; tables.append(evaluate(at, d2, P, R0, tr.m, "with_slow", s))
        else: print(f"no reference run {ref_f}: only the pulses-only variant is written (use --with-reference)")
        if os.path.exists(ck) and not a.quick: os.remove(ck)
    T = pd.concat(tables, ignore_index=True)
    keep = ["variant", "seed", "rpt", "point", "t_h", "Rf_true", "Rf_est", "Rf_err", "Rf_z", "LAM_true", "LAM_est", "LAM_err", "LAM_z", "Ds_err", "LLI_inv_err", "LLI_inv_z", "corr_Rf_LAM", "A_ok", "ABZ_ok"]
    if len(T):
        cli.write_csv(T[keep], a, "ablation_slow_rpt.csv", note="errors in % of the ageing range; variant pulses_only = no measurement update in RPT steps >= 14")
        g = T[T.rpt >= 1].groupby(["variant", "seed", "point"])
        S = g.agg(n_rpt=("rpt", "size"), maxabs_Rf=("Rf_err", lambda x: x.abs().max()), maxabs_LAM=("LAM_err", lambda x: x.abs().max()), maxabs_inv=("LLI_inv_err", lambda x: x.abs().max()), mean_corr_Rf_LAM=("corr_Rf_LAM", "mean")).reset_index()
        cli.write_csv(S, a, "ablation_slow_rpt_summary.csv", note="RPT >= 1 inside the horizon"); print(S.to_string())
    else: print("no read-out point reached in this horizon")
    clock.log(f"{n} steps")


if __name__ == "__main__":
    main()
