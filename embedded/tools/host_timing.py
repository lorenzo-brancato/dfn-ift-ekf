"""C++ (float, UD) time per step and filter memory of every exported model, through its host replay stand-in.

    python embedded/tools/host_timing.py --profile embedded/generated/lp604765_T20_10-6-10_m3k1/golden/ekf_CYC20_e+0.40.csv

Runs build/replay_<tag>.exe on the profile (EKF from --dq, then the open loop) and prints a Markdown table: filter
states, sizeof(EKF) (the static RAM of the filter on the board, same layout on the M4F), median and max time per step.
"""
import argparse
import glob
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.normpath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from replay import HostLink, read_profile  # noqa: E402


def run(exe, prof, mode, dq, n):
    t, I, V, T, _, _ = prof
    link = HostLink(exe)
    info = link.ask("?")
    link.ask("R %s %.10g" % (mode, dq))
    us = []
    for k in range(n):
        dt = (t[k] - t[k - 1]) if k else (t[1] - t[0])
        f = link.ask("S %.10g %.10g 20 %.6g" % (I[k], V[k], dt)).split()
        us.append(int(f[9]))
    link.close()
    return info, us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--build", default=os.path.join(EMB, "build"))
    ap.add_argument("--dq", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    prof = read_profile(a.profile)
    n = min(a.steps, len(prof[0]))
    print("| model | NZ | NX | filter RAM (sizeof EKF) | EKF median / max [ms] | open loop median [ms] |")
    print("|---|---|---|---|---|---|")
    for exe in sorted(glob.glob(os.path.join(a.build, "replay_*.exe")) + glob.glob(os.path.join(a.build, "replay_*[!e]"))):
        if not os.access(exe, os.X_OK) or os.path.isdir(exe):
            continue
        info, ue = run(exe, prof, "e", a.dq, n)
        _, uo = run(exe, prof, "o", a.dq, n)
        kv = dict(x.split("=", 1) for x in info.split()[1:] if "=" in x)
        print("| %s | %s | %s | %.1f kB | %.2f / %.2f | %.3f |" % (
            kv["tag"], kv["NZ"], kv["NX"], int(kv["bytes"]) / 1024, statistics.median(ue) / 1e3, max(ue) / 1e3,
            statistics.median(uo) / 1e3), flush=True)


if __name__ == "__main__":
    main()
