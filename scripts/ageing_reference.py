"""Reference of the dynamic ageing run, input of Fig. 8 (tracking, panels a-f) and of the guard / slow-RPT ablations.

Computes with PyBaMM the 101-cycle (294 h) ageing history: SEI growth accelerated 450x, current-driven loss of active material, film resistance
varying along the electrode, a reference performance test (RPT) every 10 cycles. The enriched RPT is the pulse protocol P6L followed by a C/2 discharge to
2.5 V, a C/2 charge to 4.2 V and a constant-voltage hold to C/50 (about 7.5 h each). Output: ``<data>/aging_dyn_rptslow_full.npz`` (about 74 MB,
2.1 million samples at 0.5 s), read by ``dfnekf.ageing.load`` and by the tracking scripts. Expected cost: about 0.7 h on one core, PyBaMM 26.9.

  python scripts/ageing_reference.py --data data/sim [--core 3]
  python scripts/ageing_reference.py --quick --results results/quick      # RPT + one ageing cycle (a few minutes)

The grid option of the common command line is irrelevant here (the reference is the full PyBaMM DFN) and is ignored.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dfnekf import cli, runtime

NAME = "aging_dyn_rptslow_full.npz"


def main():
    p = cli.parser("Fig. 8 (ageing reference)", "PyBaMM ageing run: 100 cycles, SEI x450, current-driven LAM, distributed film, enriched RPT every 10 cycles")
    p.add_argument("--blocks", type=int, default=10, help="number of [RPT + 9 ageing cycles] blocks (default 10 -> 101 cycles)")
    p.add_argument("--out", default=None, help="output file (default <data>/aging_dyn_rptslow_full.npz; with --quick <results>/aging_quick.npz)")
    a = p.parse_args()
    runtime.setup(a.core)
    import numpy as np
    from dfnekf import ageing_reference
    clock = cli.Clock(a, "ageing_reference")
    out = a.out or (cli.out_path(a, "aging_quick.npz") if a.quick else os.path.join(a.data, NAME))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    d = ageing_reference.run(a.blocks, quick=a.quick)
    np.savez_compressed(out, **d)
    print(f"done: {out}; wall {float(d['wall_s']):.0f} s, {len(d['t'])} samples, {len(d['cycle_start_t'])} cycles, t_end {d['t'][-1] / 3600:.2f} h", flush=True)
    clock.log(f"{len(d['t'])} samples")


if __name__ == "__main__":
    main()
