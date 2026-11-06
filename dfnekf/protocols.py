"""Protocols of the pilot. Each protocol = list of segments (current [A], positive = discharge, duration [s]),
plus initial state (soc). 1C = 5 A. Sampling 1 s."""
import numpy as np
from .params import C_RATE_A as C1


def seg(c_rate, dur):
    return (c_rate * C1, float(dur))


PROTOCOLS = {
    # discharge down to the 2.5 V cut-off (maximum safety duration)
    "P1_1C": dict(segs=[seg(1, 4000)], soc=1.0, cutoff=2.5),
    "P1_3C": dict(segs=[seg(3, 30)], soc=1.0, cutoff=2.5),
    # 3C pulses of 60 s, 10 min rest (4 repetitions)
    "P2": dict(segs=sum([[seg(3, 30), seg(0, 600)] for _ in range(4)], []), soc=1.0, cutoff=None),
    # reduced P6 (1800 s): +-1C/+-3C pulses of 10 s (20 s rest), 1C discharge 300 s, rest 600 s, C/5 780 s
    "P6": dict(segs=sum([[seg(c, 10), seg(0, 20)] for c in (1, -1, 3, -3)], [])
               + [seg(1, 300), seg(0, 600), seg(0.2, 780)], soc=0.5, cutoff=None),
}
# long P6, 1 h (two repetitions) for the filter
PROTOCOLS["P6x2"] = dict(segs=PROTOCOLS["P6"]["segs"] * 2, soc=0.5, cutoff=None)
# P6 with long pulses (remedy C if CRLB > 10 %)
PROTOCOLS["P6L"] = dict(segs=sum([[seg(c, 60), seg(0, 60)] for c in (1, -1, 3, -3)], [])
                        + [seg(1, 300), seg(0, 600), seg(0.2, 540)], soc=0.5, cutoff=None)


def current_series(name, dt=1.0):
    """ZOH current at step dt: I[k] applied over (t_k, t_k+dt]. Returns an array of length n_steps."""
    segs = PROTOCOLS[name]["segs"]
    out = []
    for I, d in segs:
        out += [I] * int(round(d / dt))
    return np.array(out)

# ---- campaign (BRIEF v4)
PROTOCOLS["P6Lbrief"] = dict(segs=sum([[seg(c, 60), seg(0, 60)] for c in (1, -1, 3, -3)], []) + [seg(1, 300), seg(0, 600), seg(0.2, 780)], soc=0.5, cutoff=None)

# P6L campaign (corrected from the brief, NOTES #32): +-1C pulses of 60 s, +-3C pulses of 30 s (at 3C for 60 s c_e,min = 5.2 mol/m3 in the brief's P6L)
PROTOCOLS["P6L"] = dict(segs=[seg(1, 60), seg(0, 60), seg(-1, 60), seg(0, 60), seg(3, 30), seg(0, 90), seg(-3, 30), seg(0, 90)]
                        + [seg(1, 300), seg(0, 600), seg(0.2, 780)], soc=0.5, cutoff=None)
PROTOCOLS["P6Lx2"] = dict(segs=PROTOCOLS["P6L"]["segs"] * 2, soc=0.5, cutoff=None)
# P5: out of distribution, 5C for 10 s from 100 % with constant transport (7C for 5 s does not hold: c_e); rest 60 s, repeated
PROTOCOLS["P5"] = dict(segs=sum([[seg(5, 10), seg(0, 120)] for _ in range(3)], []), soc=0.9, cutoff=None)


def d1_current(seed=7, dur=1800):
    """Synthetic WLTP-like driving cycle (4 phases), power -> current, discharge peak 3C (15 A), regeneration limited to 1.5C.
    Returns an array [dur] in A (positive = discharge) at 1 s."""
    rng = np.random.default_rng(seed)
    phases = [(589, 56.5), (433, 76.6), (455, 97.4), (323, 131.3)]
    v = []
    for L, vmax in phases:
        t = 0; seg_v = []
        while t < L:
            stop = int(rng.integers(5, 25)); run = int(rng.integers(30, 90)); peak = vmax * rng.uniform(0.45, 1.0)
            seg_v += [0.0] * stop
            ramp = np.sin(np.linspace(0, np.pi, run)) ** 0.8 * peak
            seg_v += list(ramp); t += stop + run
        v += seg_v[:L]
    v = np.array(v, float) / 3.6                        # m/s
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    a = np.gradient(v)
    m, cd, cr = 1500.0, 0.4, 0.012 * 1500 * 9.81
    P = m * v * a + cd * v ** 3 + cr * v * (v > 0.1)     # W
    P = np.where(P < 0, 0.6 * P, P)                      # partial recovery
    I = P / 3.7                                          # equivalent A (single scaled cell)
    I = I / np.max(I) * 15.0
    I = np.maximum(I, -7.5)
    I = I[:dur]; nb = len(I) // 10 * 10
    I = np.repeat(I[:nb].reshape(-1, 10).mean(1), 10)           # blocks of 10 s (PyBaMM's Experiment handles ~200 steps)
    I = I / np.max(I) * 15.0
    I[(np.abs(I) < 0.05)] = 0.0
    return np.round(I, 3)

# F6: evaluation protocol of the baselines = P6L x2 (identification) + P2b (1C, 2C, 3C pulses of 30 s with 300 s rests) for the estimate of the anode eta and c_e,min
PROTOCOLS["P2b"] = dict(segs=[seg(1, 30), seg(0, 300), seg(2, 30), seg(0, 300), seg(3, 30), seg(0, 300)], soc=0.5, cutoff=None)
PROTOCOLS["F6"] = dict(segs=PROTOCOLS["P6L"]["segs"] * 2 + PROTOCOLS["P2b"]["segs"], soc=0.5, cutoff=None)

# F8: 2C for 300 s from 100 % (variable transport)
PROTOCOLS["P2C300"] = dict(segs=[seg(2, 300)], soc=1.0, cutoff=2.5)

# M2 block C: F6 with the last 3C (30 s) terminated at a voltage event (2.5 V; remedy 2.7 V), followed by the usual 300 s rest
for _nm, _v in (("F6C", 2.5), ("F6C27", 2.7)):
    PROTOCOLS[_nm] = dict(segs=PROTOCOLS["F6"]["segs"], soc=0.5, cutoff=None, cutoff_idx=len(PROTOCOLS["F6"]["segs"]) - 2, cutoff_V=_v)


# M5 block B0: charge-pulse scan from rest (10 s of rest, then a 30 s charge pulse at -2C / -3C terminated by the 4.2 V event); initial SOC is given per job
for _nm, _c in (("CH2", 2.0), ("CH3", 3.0)):
    PROTOCOLS[_nm] = dict(segs=[seg(0, 10), seg(-_c, 30)], soc=0.7, cutoff=None, cutoff_idx=[1], cutoff_V=4.2)


# M5 block B1: F6Q = P6Lx2 (identification) + 1C CC charge to the reference SOC (duration from the nominal 5 Ah accounting: net charge removed by P6Lx2 = 25.33 % of 5 Ah),
# 600 s rest, two -2C pulses of 30 s (300 s rest between and after), the charge and the pulses terminated by the 4.2 V event. SOC 0.6, -2C chosen in the B0 scan (NOTES #102).
_P = PROTOCOLS["P6Lx2"]["segs"]; _n = len(_P)
for _nm, _soc0 in (("F6Q", 0.5), ("F6Q3", 0.75)):                       # F6Q3: th3, which starts from SOC 0.75 (NOTES #36)
    _tch = round((0.6 - (_soc0 - 0.2533333)) * 3600.0)
    PROTOCOLS[_nm] = dict(segs=_P + [seg(-1, _tch), seg(0, 600), seg(-2, 30), seg(0, 300), seg(-2, 30), seg(0, 300)], soc=_soc0, cutoff=None, cutoff_idx=[_n, _n + 2, _n + 4], cutoff_V=4.2)
