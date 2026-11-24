"""SOC reference for the LP604765 ageing-block tests: Coulomb counting re-anchored at 100 % at the end of every CC-CV
charge (fig_soh_nom.soc_reference), with the full scale = the cell's actual capacity on the stoichiometric window:

    Q(t) = Q_window,BOL x Q_1C(t) / Q_1C,BOL

Q_window,BOL is the OCV-window capacity of the cell set (4.2-3.0 V); Q_1C(t) is interpolated linearly in time across the
block between the 1C capacity checks before and after it (dfnekf.datasets.LP_CAPCHECK); Q_1C,BOL = 1.934 A.h (the
2026-10-17 check). The per-cycle 1C discharge capacities are not used: in the 2026-10-22 block the discharges are cut
at 1 h, before the 2.75 V cut-off.

    from soc_ref import reference
    ref, anchors, Q = reference(t, I, "AGE22V", q_window_bol=2.2257)
"""
import functools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir, os.pardir)))
sys.path.insert(0, HERE)
from dfnekf import datasets  # noqa: E402
from fig_soh_nom import soc_reference  # noqa: E402

Q1C_BOL = 1.934
# capacity checks (LP_CAPCHECK dates) before and after each ageing block
BLOCK_CHECKS = {"20261022_Ageing": ("2026-10-18", "2026-10-26"),
                "20261027_Ageing": ("2026-10-26", "2026-10-29"),
                "20261030_Ageing": ("2026-10-29", "2026-11-02")}


@functools.lru_cache(maxsize=None)
def _block_times(test):
    """(offset of the test's first sample from the start of its block [s], block duration [s], block folder)."""
    folder = datasets.LP_TESTS[test]["test"]
    t0 = float(datasets.lp_cut(test)["t"][0])
    blk = datasets.lp_load_folder(folder)
    return t0 - float(blk["t"][0]), float(blk["t"][-1] - blk["t"][0]), folder


def q_actual(t, test, q_window_bol):
    """Actual capacity on the stoichiometric window [A.h] at the test's times t [s from the test's start]."""
    off, dur, folder = _block_times(test)
    if folder not in BLOCK_CHECKS:
        return np.full(len(t), q_window_bol)
    qb, qa = (datasets.LP_CAPCHECK[k] for k in BLOCK_CHECKS[folder])
    q1c = qb + (qa - qb) * np.clip((off + np.asarray(t)) / dur, 0.0, 1.0)
    return q_window_bol * q1c / Q1C_BOL


def reference(t, I, test, q_window_bol):
    """SOC reference [0-1], the anchor indices (ends of the CC-CV charges) and Q(t) [A.h]."""
    Q = q_actual(t, test, q_window_bol)
    ref, anchors, _ = soc_reference(np.asarray(t), np.asarray(I), q_window=Q)
    return ref, anchors, Q


if __name__ == "__main__":
    for tst in ("CYC20", "AGE22V", "AGE22", "AGE27", "AGE30"):
        off, dur, folder = _block_times(tst)
        Q = q_actual(np.array([0.0, dur - off]), tst, 2.2257)
        print("%-7s offset %6.0f s in %s (%.1f h): Q %.3f -> %.3f A.h" % (tst, off, folder, dur / 3600, Q[0], Q[1]))
