"""Loader of the LP604765 (CNR STEMS, BATTAGE) raw data, now in ``data/exp_LP604765/raw`` (see dfnekf.datasets): one test
folder = numbered step files 'NNN_<Type>[_...].txt' with a CSStudio header line and columns E(V), i(A/cm2) [in fact A], T(s).

    from lpdata import load_test, list_tests
    d = load_test("20261011_HPPC_20°C")      # dict: t [s], V [V], I [A, discharge > 0], step [int], names [list]

Sign convention returned: I > 0 on discharge (the files store discharge as negative current).
A thin wrapper of dfnekf.datasets (lp_* functions), kept for the identification scripts.
"""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
from dfnekf import datasets as _D  # noqa: E402

ROOT = _D.LP_RAW
list_tests = _D.lp_list_folders
read_step = _D.lp_read_step
step_files = _D.lp_step_files
load_test = _D.lp_load_folder
temperature_of = _D.lp_temperature_of
