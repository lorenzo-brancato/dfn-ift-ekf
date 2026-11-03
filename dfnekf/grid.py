"""Spatial grid of the grey-box DFN: the electrode cells are uniform or shrink geometrically towards the separator.

``grid`` is always an explicit argument ("4:1" or "uniform"); no environment variable decides it.
"""
import numpy as np

GRIDS = {"uniform": None, "4:1": 4.0}


def ratio_of(grid):
    """Collector-to-separator cell-width ratio for a grid name ('uniform' -> None, '4:1' -> 4.0); a number is passed through."""
    if isinstance(grid, str):
        if grid not in GRIDS: raise ValueError(f"unknown grid {grid!r}; use 'uniform' or '4:1'")
        return GRIDS[grid]
    return None if grid is None else float(grid)


def electrode_widths(n, L, ratio, collector_left):
    """Electrode cell widths: uniform, or geometric with width(collector cell) / width(separator cell) = ratio."""
    if ratio is None or n < 2: return np.full(n, L / n)
    q = ratio ** (-1.0 / (n - 1)); w = q ** np.arange(n)                      # decreasing from the collector
    w = L * w / w.sum()
    return w if collector_left else w[::-1]
