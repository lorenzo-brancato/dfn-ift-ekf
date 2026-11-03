import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyb, protocols

def go(j):
    try:
        proto, tag, theta, mesh, period = j["proto"], j["tag"], tuple(j["theta"]), tuple(j["mesh"]), j.get("period", 1.0)
        gr = j.get("grid_ratio"); fn = pyb.fname(proto, tag, mesh, period, gr)
        if os.path.exists(fn) and not j.get("force"): return ("skip", fn)
        kw = dict(theta=theta, mesh=mesh, period=period, soc=j.get("soc"), var_transport=bool(j.get("var_transport", False)), grid_ratio=gr)
        if proto == "D1": kw["I_custom"] = protocols.d1_current(); kw["soc"] = 0.8
        o = pyb.run(proto if proto != "D1" else "D1", **kw)
        np.savez_compressed(fn, **o)
        return ("ok", fn, round(o["wall_s"], 1), len(o["t"]), float(o["V"][-1]), float(o["ce"].min()))
    except Exception as e:
        return ("FAIL", j, str(e)[:200])
