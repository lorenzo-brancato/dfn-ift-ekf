"""Discretization study of the embedded model (Python, dfnekf): slices x (modes + tail bands), at the LP604765 fit_v2
parameters without refit, with the filter of export_model.py (Rf free, implicit closure, D_p(y) law).

    python embedded/tools/study.py --jobs 4                    # run every configuration missing, then summarize
    python embedded/tools/study.py --summary-only

Per configuration (results/embedded_study/<tag>.npz):
  open loop HPPC20 on the identification's mixed steps (1 s under current, 5 / 20 s on rests: lpmodel.prepare), from the
  fit's start (dq_20 = 0); CYC20 (9.2 h) on a uniform --dt grid from the OCV-placed start; EKF on CYC20 from the start
  errors --errs [A.h] (SOC read from the cathode, reference: Coulomb counting from the true start).
Summary: states, filter size, time-weighted open-loop RMSE against the measurement, maximum deviation from the full
model (10/6/10, 12 + 2), EKF SOC error (second half rms, max after the first hour, final). Written to
results/embedded_study/summary.csv and printed as a Markdown table.
Extra rows "CYC20 at 1 s" quantify the 5-s step against 1 s (open loop only).
"""
import argparse
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.normpath(os.path.join(HERE, os.pardir))
ROOT = os.path.normpath(os.path.join(EMB, os.pardir))
OUT = os.path.join(ROOT, "results", "embedded_study")
SLICES = [(10, 6, 10), (10, 3, 10), (5, 3, 5), (3, 2, 3)]
MODES = [(12, 2), (3, 1), (2, 1)]
FULL = (10, 6, 10, 12, 2)


def tag_of(dims, dt):
    return "%d-%d-%d_m%dk%d_dt%g" % (dims + (dt,))


def args_ns(dims, a):
    return types.SimpleNamespace(dataset="exp_LP604765", T=20.0, film_share=0.5, soft_eps=1e-3, dims=dims, grid="4:1",
                                 sigma_mV=15.0, p0_q=0.5, p0_shell=0.005, p0_th=0.3, q_th=1e-4)


def job(dims, dt, errs, ekf):
    sys.path.insert(0, EMB)
    sys.path.insert(0, os.path.join(ROOT, "scripts", "ident_lp604765"))
    import export_model as em
    import jax.numpy as jnp
    from dfnekf import datasets
    from lpmodel import prepare
    a = args_ns(dims, None)
    cs, cell, P, m = em.build_model(a, dt)
    th1 = jnp.array([1.0, 1.0, 1.0])
    out = dict(dims=np.array(dims), dt=dt, nz=m.nz, ng=m.A.ng)
    t0 = time.time()
    # HPPC20, mixed steps
    pr = prepare(datasets.lp_cut("HPPC20"), start_step=0)
    d_h = datasets.load(a.dataset, "HPPC20")
    q_h = datasets.initial_charge(a.dataset, d_h, cs)
    from dfnekf import cellset
    z_of = lambda q: np.asarray(m.init_state(*cellset.c0_at(cs, cell, q)))
    Hh = em.openloop(m, z_of(q_h), pr["I"], pr["idx"].astype(int), [1.0, 5.0, 20.0], th1, 10 ** 9)
    out.update(hppc_V=Hh["V"], hppc_Vm=pr["V"], hppc_dt=pr["dt"], hppc_t=pr["t"], hppc_res=Hh["res"])
    # CYC20, uniform dt
    d = datasets.load(a.dataset, "CYC20", dt=dt)
    fs = em.filter_setup(a, m, cs, cell, P, d)
    Hc = em.openloop(m, fs.z0, d["I"], np.zeros(len(d["I"]), int), [dt], th1, 10 ** 9)
    out.update(cyc_V=Hc["V"], cyc_Vm=d["V"], cyc_t=d["t"], cyc_res=Hc["res"])
    if ekf:
        for e in errs:
            soc, sd, soc_ref, H, wall = em.ekf_soc(fs, d, e)
            k = "ekf%+.2f_" % e
            out.update({k + "soc": soc, k + "sd": sd, k + "ref": soc_ref, k + "innov": H["innov"], k + "Rf": H["vals"][:, 0],
                        k + "wall": wall})
    out["wall"] = time.time() - t0
    np.savez(os.path.join(OUT, tag_of(dims, dt) + ".npz"), **out)
    return tag_of(dims, dt), out["wall"]


def rmse_w(e, w):
    return float(np.sqrt(np.sum(w * e ** 2) / np.sum(w)))


def summarize(errs):
    import csv
    full = np.load(os.path.join(OUT, tag_of(FULL[:3] + FULL[3:], 5.0) + ".npz"))
    rows = []
    for sl in SLICES:
        for md in MODES:
            for dt in (5.0, 1.0):
                f = os.path.join(OUT, tag_of(sl + md, dt) + ".npz")
                if not os.path.exists(f):
                    continue
                r = np.load(f)
                nx = int(r["nz"]) + 1
                row = dict(slices="%d/%d/%d" % sl, modes="%d+%d" % md, dt_s=dt, states=int(r["nz"]), filter=nx,
                           closure=int(r["ng"]),
                           cov_kB_float_UD=round((nx + 1) * nx * 4 / 1024, 1))
                eh = r["hppc_V"] - r["hppc_Vm"]
                row["HPPC20_rmse_mV"] = round(1e3 * rmse_w(eh, r["hppc_dt"]), 1)
                row["HPPC20_maxdev_full_mV"] = round(1e3 * float(np.max(np.abs(r["hppc_V"] - full["hppc_V"]))), 1)
                ec = r["cyc_V"] - r["cyc_Vm"]
                row["CYC20_rmse_mV"] = round(1e3 * float(np.sqrt(np.mean(ec ** 2))), 1)
                if dt == 5.0:               # (1-s rows: compare the RMSE; their samples are not on the 5-s instants)
                    row["CYC20_maxdev_full_mV"] = round(1e3 * float(np.max(np.abs(r["cyc_V"] - full["cyc_V"]))), 1)
                for e in errs:
                    k = "ekf%+.2f_" % e
                    if k + "soc" not in r.files:
                        continue
                    es = 100 * (r[k + "soc"] - r[k + "ref"])
                    t = r["cyc_t"]
                    half = len(es) // 2
                    row["EKF%+.1f_rms2nd_pct" % e] = round(float(np.sqrt(np.mean(es[half:] ** 2))), 2)
                    row["EKF%+.1f_max_after1h_pct" % e] = round(float(np.max(np.abs(es[t > 3600]))), 2)
                    row["EKF%+.1f_innov_mV" % e] = round(1e3 * float(np.sqrt(np.mean(r[k + "innov"] ** 2))), 1)
                rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    print("wrote", os.path.join(OUT, "summary.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--errs", default="-0.4,0.4")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of tags to (re)run, e.g. 3-2-3_m2k1_dt5")
    a = ap.parse_args()
    errs = [float(v) for v in a.errs.split(",")]
    os.makedirs(OUT, exist_ok=True)
    if not a.summary_only:
        todo = [(sl + md, a.dt, True) for sl in SLICES for md in MODES]
        todo += [((10, 6, 10, 3, 1), 1.0, False), ((10, 6, 10, 12, 2), 1.0, False)]
        todo = [t for t in todo if (a.only and tag_of(t[0], t[1]) in a.only.split(","))
                or (not a.only and not os.path.exists(os.path.join(OUT, tag_of(t[0], t[1]) + ".npz")))]
        print("running %d configurations with %d processes" % (len(todo), a.jobs), flush=True)
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(job, d, dt, errs, e) for d, dt, e in todo]
            for f in futs:
                tg, wall = f.result()
                print("done %s (%.0f s)" % (tg, wall), flush=True)
    summarize(errs)


if __name__ == "__main__":
    main()
