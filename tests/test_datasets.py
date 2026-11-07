"""dfnekf.datasets: uniform binning, OCV inversion on the committed cell sets, and the loaders (skipped without the data)."""
import os

import numpy as np
import pytest

from dfnekf import datasets as D


def test_bin_uniform_means_and_last_sample():
    t = np.arange(0.0, 3.0, 0.1)                       # 10 samples per second, 3 s
    d = dict(t=t, I=np.repeat([1.0, 2.0, 3.0], 10), V=t.copy(), step=np.zeros(len(t), int))
    b = D.bin_uniform(d, 1.0)
    assert np.allclose(b["t"], [1.0, 2.0, 3.0])
    assert np.allclose(b["I"], [1.0, 2.0, 3.0])
    assert np.allclose(b["V"], [0.9, 1.9, 2.9])        # last sample of each bin


@pytest.mark.parametrize("dataset", ["exp_LP604765", "exp_Mohtat"])
def test_ocv_inversion_roundtrip(dataset):
    if not os.path.exists(D.cellset_path(dataset)):
        pytest.skip("cell set not present")
    cs = D.cellset(dataset)
    cell = D.default_cell(dataset)
    T = cs["T_ref_K"] + 5.0
    for q in (0.05, 0.2, 0.5):                         # inside the inversion grid [-0.4, 0.6] A.h
        V = float(D.ocv(cs, cell, q, T))
        assert abs(D.q_from_ocv(cs, cell, V, T) - q) < 1e-3


def _have(dataset):
    return os.path.isdir(D.LP_RAW) if dataset == "exp_LP604765" else os.path.isdir(D.MOHTAT_TESTS)


@pytest.mark.parametrize("dataset,test", [("exp_LP604765", "CAP20"), ("exp_Mohtat", "C01_VAL")])
def test_load_interface(dataset, test):
    if not _have(dataset):
        pytest.skip("measured data not present (git-ignored)")
    d = D.load(dataset, test, dt=1.0 if dataset == "exp_LP604765" else None)
    n = len(d["t"])
    for k in ("I", "V", "T", "valid"):
        assert len(d[k]) == n
    assert d["t"][0] >= 0 and np.all(np.diff(d["t"]) > 0)
    assert 2.0 < d["V"].min() and d["V"].max() < 4.5
    q0 = D.initial_charge(dataset, d, D.cellset(dataset))
    assert -0.5 < q0 < 5.0
    assert test in D.tests(dataset)
