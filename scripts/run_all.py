"""Run the whole reproduction in stages, cheapest first, with at most ``--procs`` processes at a time (one thread each).

    python scripts/run_all.py --list                       # print the stages and jobs
    python scripts/run_all.py --stage model --procs 6      # run one stage
    python scripts/run_all.py --stage all --procs 6        # every stage in order (days of CPU, see README)

Stages (in order): refs, ageing_ref, model, static, protocols, twin, closure_train, closure, ageing, ukf, timing, aggregate. 'ageing' only needs 'ageing_ref'
and contains the two longest jobs (13-14 h each): start it in a second runner next to the rest, e.g. --stage ageing --procs 3 and --stage model,static,protocols,twin,closure_train,closure --procs 3.
A job is skipped if its marker file ``<results>/.done/<job>`` exists, so an interrupted run resumes where it stopped.
The ``ukf`` stage (full unscented filter, the longest), the ``ukf_pilot`` stage (4 runs at 4 closure iterations) and the ``timing`` stage (cost measurements, which need an otherwise idle machine) are only run when named.
Outputs: ``<results>/4to1`` (main), logs in ``<results>/logs``.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
S = ["static_mc", "twin_nees"]
STATES = ["th1", "th2", "th3", "th4", "th5"]


def job(name, script, *args, procs=1, stage=None):
    return dict(name=name, argv=[PY, f"scripts/{script}.py", *args], procs=procs, stage=stage)


def stages(a):
    R, D = os.path.join(a.results, "4to1"), a.data
    g = ["--grid", "4:1", "--data", D, "--results", R]
    jobs = {s: [] for s in ["refs", "ageing_ref", "model", "static", "protocols", "twin", "closure_train", "closure", "ageing", "ukf", "ukf_pilot", "timing", "aggregate"]}
    jobs["refs"].append(dict(name="references", argv=[PY, "-m", "dfnekf.reference", "standard", "--workers", "4", "--data", D], procs=4))
    jobs["ageing_ref"].append(job("ageing_reference", "ageing_reference", *g))
    # model-level checks, grid study, CRLB, model bias
    jobs["model"] += [job("jacobian_agreement", "jacobian_check", *g, "--parts", "agreement"), job("model_checks", "model_checks", *g, "--parts", "algebra,vt,lithium,statements", "--with-pybamm"),
                      job("crlb", "crlb", *g, "--states", ",".join(STATES), "--protocols", "P6Lx2")]
    jobs["model"] += [job(f"grid_{r}", "grid_study", *g, "--rows", r) for r in ["10-5-10_Nm8_uniform", "20-10-20_Nm12_uniform", "40-20-40_Nm16_uniform", "20-10-20_Nm12_nu4"]]
    jobs["model"].append(job("grid_modes", "grid_study", *g, "--rows", "modes"))
    jobs["model"] += [job(f"bias_{t}", "bias", *g, "--states", t) for t in STATES]
    # static Monte Carlo (EKF, five states, ten seeds)
    jobs["static"] += [job(f"static_ekf_{t}", "static_mc", *g, "--states", t, "--seeds", "0-9") for t in STATES]
    # charge (F6Q) and discharge (F6, F6C): ten seeds, one code path
    jobs["protocols"] += [job(f"charge_{t}_{r}", "charge", *g, "--states", t, "--seeds", r) for t in STATES for r in ["0-4", "5-9"]]
    jobs["protocols"] += [job(f"discharge_{t}_{r}", "discharge", *g, "--states", t, "--seeds", r) for t in ["th2", "th4", "th5"] for r in ["0-4", "5-9"]]
    # twin experiments of the EKF (20 runs each) and the Appendix D variants at th1
    for t in ["th1", "th2"]:
        for r in ["0-9", "10-19"]: jobs["twin"].append(job(f"twin_ekf_{t}_{r}", "twin_nees", *g, "--states", t, "--seeds", r))
    for r in ["0-9", "10-19"]:
        jobs["twin"].append(job(f"twin_p0rf2_{r}", "twin_nees", *g, "--states", "th1", "--seeds", r, "--p0-scale-rf", "2"))
        jobs["twin"].append(job(f"twin_relinit_{r}", "twin_nees", *g, "--states", "th1", "--seeds", r, "--init-error", "relative"))
        jobs["twin"].append(job(f"twin_uniform_{r}", "twin_nees", "--grid", "uniform", "--data", D, "--results", R, "--states", "th1", "--seeds", r))
    # learned closure: training (explicit reference list), ablation (5 closures x 10 seeds), open loop, stress test
    jobs["closure_train"].append(job("train_closure", "train_closure", *g))
    jobs["closure"] += [job(f"ablation_{c}_{r}", "closure_ablation", *g, "--closures", c, "--seeds", r) for c in ["warm", "cold", "net0", "net1", "net1c"] for r in ["0-4", "5-9"]]
    jobs["closure"] += [job("closure_openloop", "closure_ablation", *g, "--openloop"), job("residual_stress", "residual_stress", *g)]
    # ageing: tracker (two noise seeds, the longest jobs), CV model error, ablations
    ar = R
    ga = g
    jobs["ageing"] += [job("track_seed0", "ageing_track", *ga, "--seeds", "0", "--resume"), job("track_seed1", "ageing_track", *ga, "--seeds", "1", "--resume"),
                       job("cv_z1", "cv_model_error", *ga, "--variants", "z1"), job("cv_z2", "cv_model_error", *ga, "--variants", "z2"),
                       job("ablation_guard", "ablation_guard", *ga, "--seeds", "0"),
                       job("ablation_slow_rpt", "ablation_slow_rpt", *ga, "--seeds", "0", "--hours", "100", "--with-reference")]
    jobs["ageing"].append(job("protocols_fig", "protocols_fig", "--data", D, "--results", R))
    # full unscented filter: closure iterations per sigma point as decided (default: 1)
    it = str(a.ukf_iters)
    jobs["ukf"] += [job(f"ukf_twin_{i}", "twin_nees", *g, "--filter", "ukf", "--closure-iters", it, "--states", "th2", "--seeds", str(i)) for i in range(20)]
    jobs["ukf"] += [job(f"ukf_static_{i}", "static_mc", *g, "--filter", "ukf", "--closure-iters", it, "--states", "th2", "--seeds", str(i)) for i in range(10)]
    # pilot: four twin runs of the full UKF with 4 closure iterations per sigma point (a result of its own; about 8.7 h CPU each)
    P = os.path.join(a.results, "ukf_pilot_k4")
    jobs["ukf_pilot"] += [job(f"ukf_pilot_{i}", "ukf_pilot", "--grid", "4:1", "--data", D, "--results", P, "--seeds", str(i)) for i in (0, 3, 14, 5)]
    jobs["ukf_pilot"].append(job("ukf_pilot_agg", "ukf_pilot", "--results", P, "--seeds", "0,3,14,5", "--aggregate", "--k1", R))
    # timings: idle machine, one job at a time (the runner forces procs = 6 for these so that nothing else runs)
    jobs["timing"] += [dict(name="cost_4to1", argv=[PY, "scripts/cost.py", "--grid", "4:1", "--data", D, "--results", R], procs=6),
                       dict(name="cost_uniform", argv=[PY, "scripts/cost.py", "--grid", "uniform", "--data", D, "--results", R], procs=6),
                       dict(name="jacobian_cost", argv=[PY, "scripts/jacobian_check.py", *g, "--parts", "cost,step", "--n-calls", "3600", "--n-jac", "100"], procs=6),
                       dict(name="closure_cost", argv=[PY, "scripts/closure_ablation.py", *g, "--cost"], procs=6)]
    # aggregation: every script writes its tables and source data, then the figure data
    agg = [("static_mc", []), ("twin_nees", ["--static-results", R]), ("charge", []), ("discharge", []), ("closure_ablation", []), ("grid_study", []), ("cost", []), ("crlb", []), ("jacobian_check", [])]
    jobs["aggregate"] += [job(f"agg_{s}", s, *g, "--aggregate", *x) for s, x in agg]
    jobs["aggregate"] += [job("agg_ageing_track", "ageing_track", *ga, "--seeds", "0,1", "--aggregate"), job("agg_cv", "cv_model_error", *ga, "--aggregate", "--tracking", ar)]
    jobs["aggregate"].append(dict(name="figdata", argv=[PY, "figdata/make_figdata.py", "--results", R, "--out", "paper/figures/data"], procs=1))
    return jobs


def run(js, procs, results):
    os.makedirs(os.path.join(results, "logs"), exist_ok=True); os.makedirs(os.path.join(results, ".done"), exist_ok=True)
    todo = [j for j in js if not os.path.exists(os.path.join(results, ".done", j["name"]))]
    print(f"{len(todo)} job(s) to run, {len(js) - len(todo)} already done", flush=True)
    slots = [procs]; import threading; cv = threading.Condition()

    def go(j):
        need = min(j["procs"], procs)
        with cv:
            while slots[0] < need: cv.wait()
            slots[0] -= need
        t0 = time.time(); log = os.path.join(results, "logs", j["name"] + ".log")
        try:
            with open(log, "w") as f: rc = subprocess.call(j["argv"], stdout=f, stderr=subprocess.STDOUT)
        finally:
            with cv: slots[0] += need; cv.notify_all()
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {j['name']:30s} {time.time() - t0:8.0f} s", flush=True)
        if rc == 0: open(os.path.join(results, ".done", j["name"]), "w").write(f"{time.time() - t0:.0f}\n")
        return rc
    with ThreadPoolExecutor(procs) as ex: rcs = list(ex.map(go, todo))
    return sum(r != 0 for r in rcs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None, help="stage name, comma-separated list, or 'all'"); ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--data", default="data/sim"); ap.add_argument("--results", default="results"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--closure-iters", "--ukf-iters", dest="ukf_iters", default="1", help="Newton iterations per sigma point of the full UKF (1 = as in the manuscript; 'conv' = to convergence)")
    a = ap.parse_args(); J = stages(a)
    if a.list or not a.stage:
        for s, js in J.items(): print(f"{s:11s} {len(js):3d} jobs: " + ", ".join(j["name"] for j in js)[:300])
        sys.exit(0)
    names = [s for s in J if s not in ("ukf", "ukf_pilot", "timing")] if a.stage == "all" else a.stage.split(",")
    bad = 0
    for s in names:
        print(f"=== stage {s}", flush=True); bad += run(J[s], a.procs, a.results)
    sys.exit(1 if bad else 0)
