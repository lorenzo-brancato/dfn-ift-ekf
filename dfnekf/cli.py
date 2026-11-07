"""Command-line conventions shared by every script in ``scripts/``.

Every script: ``--grid`` (explicit, 'uniform' or '4:1'), ``--data`` (folder with the PyBaMM references), ``--results`` (output folder), ``--quick``
(one run on a short protocol, for testing) and ``--core`` (optional CPU pinning). Output CSV files start with a header comment naming the paper item.
This module does not import jax: scripts call ``runtime.single_thread()`` first, then import the numerical modules.
"""
import argparse
import os
import time


def parser(item, description, seeds="0-9", states="th1,th2,th3,th4,th5"):
    """ArgumentParser with the common options; ``item`` is the paper item the script reproduces (e.g. 'Fig. 5, Table III')."""
    p = argparse.ArgumentParser(description=f"[{item}] {description}")
    p.add_argument("--grid", default="4:1", choices=["uniform", "4:1"], help="spatial grid of the grey-box model")
    p.add_argument("--data", default="data/sim", help="folder with the PyBaMM reference simulations")
    p.add_argument("--results", default="results", help="output folder")
    p.add_argument("--seeds", default=seeds, help="seed list, e.g. '0-9' or '0,3,5'")
    p.add_argument("--states", default=states, help="ageing states, comma-separated (th1..th5)")
    p.add_argument("--quick", action="store_true", help="smoke test: one run, short protocol")
    p.add_argument("--core", type=int, default=None, help="pin the process to this CPU core (optional)")
    p.set_defaults(item=item)
    return p


def parse_ids(s):
    """'0-9' -> [0..9]; '0,3,5' -> [0, 3, 5]; '2-4,9' -> [2, 3, 4, 9]."""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part: out.append(int(part))
    return out


def states(args):
    return [s.strip() for s in args.states.split(",") if s.strip()]


def out_path(args, name):
    os.makedirs(args.results, exist_ok=True)
    return os.path.join(args.results, name)


def write_csv(df, args, name, note="", float_format="%.10g"):
    """Write ``df`` to ``<results>/<name>`` with a header comment naming the paper item; returns the path."""
    path = out_path(args, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# paper item: {args.item}; grid: {getattr(args, 'grid', 'n/a')}" + (f"; {note}" if note else "") + "\n")
        df.to_csv(f, index=False, float_format=float_format)
    return path


class Clock:
    """Wall and CPU time of a script, logged to ``<results>/cpu_log.csv`` (one row per script run)."""
    def __init__(self, args, script):
        self.args, self.script = args, script; self.t0, self.c0 = time.time(), time.process_time()

    def log(self, extra=""):
        import csv
        path = out_path(self.args, "cpu_log.csv"); new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new: w.writerow(["script", "item", "grid", "quick", "wall_s", "cpu_s", "note"])
            w.writerow([self.script, self.args.item, getattr(self.args, "grid", ""), getattr(self.args, "quick", ""), round(time.time() - self.t0, 2), round(time.process_time() - self.c0, 2), extra])
