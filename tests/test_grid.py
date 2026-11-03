"""The grid is an explicit argument; the 4:1 progression shrinks the electrode cells towards the separator (7.8 to 2.0 um in the negative electrode)."""
import numpy as np
import pytest
from dfnekf import params as prm
from dfnekf.grid import ratio_of, electrode_widths
from dfnekf.step import DFN


def test_ratio_names():
    assert ratio_of("uniform") is None and ratio_of("4:1") == 4.0
    with pytest.raises(ValueError): ratio_of("3:1")


def test_cell_widths_4to1():
    P = prm.base(); wn = electrode_widths(20, P["Ln"], 4.0, True)
    assert abs(wn.sum() - P["Ln"]) < 1e-15 and abs(wn[0] / wn[-1] - 4.0) < 1e-12
    assert abs(wn[0] * 1e6 - 7.8) < 0.05 and abs(wn[-1] * 1e6 - 2.0) < 0.05
    m = DFN(P, 20, 10, 20, 12, 2, 0.5, "4:1"); mu = DFN(P, 20, 10, 20, 12, 2, 0.5, "uniform")
    assert m.nz + 3 == 653 and mu.nz == m.nz and not np.allclose(m.A.dx, mu.A.dx)
