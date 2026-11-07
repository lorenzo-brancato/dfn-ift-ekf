"""Streams a recorded profile to the board (USB serial) or to its host stand-in, and logs and checks the replies.

    python embedded/tools/replay.py --port COM5 --profile embedded/generated/<tag>/golden/ekf_CYC20_e+0.40.csv --dq 0.4
    python embedded/tools/replay.py --host-exe embedded/build/replay_<tag>.exe --profile ... --dq 0.4
    python embedded/tools/replay.py --port COM5 --mode o --profile .../golden/openloop_CYC20.csv --dq <start - q0>

Protocol: include/dfn/replay.h. The profile is a CSV with t [s] and I [A] (> 0 on discharge) and the measured voltage
in y or V_meas [V]; optional T [degC] (else --T). The step of each row is the difference of t (first row: the second
row's). When the CSV is a golden file, the replies are compared with it (columns V and soc). Only the standard library,
plus pyserial for --port (PlatformIO's Python has it: ~/.platformio/penv/Scripts/python.exe).
"""
import argparse
import csv
import math
import subprocess
import sys
import time


def read_profile(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    vkey = "y" if "y" in cols else ("V_meas" if "V_meas" in cols else None)
    t = [float(r["t"]) for r in rows]
    I = [float(r["I"]) for r in rows]
    V = [float(r[vkey]) if vkey else float("nan") for r in rows]
    T = [float(r["T"]) for r in rows] if "T" in cols else None
    ref_V = [float(r["V"]) for r in rows] if "V" in cols else None
    ref_soc = [float(r["soc"]) for r in rows] if "soc" in cols else None
    return t, I, V, T, ref_V, ref_soc


class HostLink:
    def __init__(self, exe):
        import os
        self.p = subprocess.Popen([os.path.abspath(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  bufsize=1)

    def ask(self, line):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        self.p.stdin.close(); self.p.wait(5)


class SerialLink:
    def __init__(self, port, timeout):
        import serial
        self.s = serial.Serial(port, 115200, timeout=timeout)
        time.sleep(0.5)
        self.s.reset_input_buffer()

    def ask(self, line):
        self.s.write((line + "\n").encode())
        while True:
            r = self.s.readline().decode(errors="replace").strip()
            if not r:
                raise TimeoutError("no reply to %r" % line)
            if r[0] in "O#!":
                return r

    def close(self):
        self.s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--port")
    g.add_argument("--host-exe")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--mode", default="e", choices=("e", "o"), help="e: EKF, o: open loop")
    ap.add_argument("--dq", type=float, default=0.0, help="start [A.h] relative to the model's q0 (CYC20 start)")
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="CSV of the replies")
    a = ap.parse_args()
    t, I, V, T, ref_V, ref_soc = read_profile(a.profile)
    n = len(t) if a.max_steps is None else min(a.max_steps, len(t))
    link = SerialLink(a.port, a.timeout) if a.port else HostLink(a.host_exe)
    print(link.ask("?"))
    print(link.ask("R %s %.10g" % (a.mode, a.dq)))
    out = []
    dV = dS = 0.0
    us = []
    t0 = time.time()
    for k in range(n):
        dt = (t[k] - t[k - 1]) if k else (t[1] - t[0])
        r = link.ask("S %.10g %.10g %.6g %.6g" % (I[k], V[k], T[k] if T else a.T, dt))
        if not r.startswith("O"):
            sys.exit("step %d: %s" % (k, r))
        f = r.split()
        Vp, soc, sd, Rf, sdRf, res = (float(x) for x in f[2:8])
        nit, u = int(f[8]), int(f[9])
        us.append(u)
        if ref_V:
            dV = max(dV, abs(Vp - ref_V[k]))
        if ref_soc:                      # EKF golden: SOC after the update; open-loop golden: before the step
            kk = k if a.mode == "e" else k + 1
            if kk < n:
                dS = max(dS, abs(soc - ref_soc[kk]))
        out.append((t[k], I[k], V[k], Vp, soc, sd, Rf, sdRf, res, nit, u))
        if k % 500 == 0:
            print("  step %6d  t %8.0f s  V %.4f  SOC %6.2f %% +- %.2f  Rf %.3f  %6.1f ms" % (
                k, t[k], Vp, 100 * soc, 100 * sd, Rf, u / 1e3), flush=True)
    link.close()
    wall = time.time() - t0
    us_s = sorted(us)
    print("%d steps in %.1f s; step time median %.2f ms, max %.2f ms" % (n, wall, us_s[n // 2] / 1e3, us_s[-1] / 1e3))
    if ref_V:
        print("max |V - V_golden| %.3f mV" % (1e3 * dV))
    if ref_soc:
        print("max |SOC - SOC_golden| %.4f %%" % (100 * dS))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "I", "V_meas", "V_pred", "soc", "sd_soc", "Rf", "sd_Rf", "res", "nit", "us"])
            w.writerows(out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
