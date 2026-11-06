"""The learned-closure weights shipped with the package load, have the size stated in Appendix E, and are the default when nothing was retrained."""
import os
import numpy as np
import jax
from dfnekf import closure_net as cn


def test_shipped_weights_are_default_and_have_56802_parameters(tmp_path):
    path = cn.default_net_path(str(tmp_path), "4:1")                          # empty results folder: no retrained network
    assert os.path.samefile(path, os.path.join(cn.ASSET_DIR, "closure_4to1.pkl"))
    S = __import__("pickle").load(open(path, "rb"))
    assert sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(S["params"])) == 56802
    assert S["n_samples"] == 80000 and S["grid"] == "4:1"
    net, cfg = cn.load_net(path, "4:1")
    assert callable(net)


def test_retrained_network_takes_precedence(tmp_path):
    (tmp_path / "closure_4to1.pkl").write_bytes(b"x")
    assert cn.default_net_path(str(tmp_path), "4:1") == str(tmp_path / "closure_4to1.pkl")
