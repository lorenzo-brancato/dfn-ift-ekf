"""Protocolli del pilot. Ogni protocollo = lista di segmenti (corrente [A] positiva = scarica, durata [s]),
piu stato iniziale (soc). 1C = 5 A. Campionamento 1 s."""
import numpy as np
from .params import C_RATE_A as C1


def seg(c_rate, dur):
    return (c_rate * C1, float(dur))


PROTOCOLS = {
    # scarica fino al cut-off 2.5 V (durata massima di sicurezza)
    "P1_1C": dict(segs=[seg(1, 4000)], soc=1.0, cutoff=2.5),
    "P1_3C": dict(segs=[seg(3, 30)], soc=1.0, cutoff=2.5),
    # impulsi 3C 60 s, riposo 10 min (4 ripetizioni)
    "P2": dict(segs=sum([[seg(3, 30), seg(0, 600)] for _ in range(4)], []), soc=1.0, cutoff=None),
    # P6 ridotto (1800 s): impulsi +-1C/+-3C 10 s (riposo 20 s), scarica 1C 300 s, riposo 600 s, C/5 780 s
    "P6": dict(segs=sum([[seg(c, 10), seg(0, 20)] for c in (1, -1, 3, -3)], [])
               + [seg(1, 300), seg(0, 600), seg(0.2, 780)], soc=0.5, cutoff=None),
}
# P6 lungo 1 h (due ripetizioni) per il filtro
PROTOCOLS["P6x2"] = dict(segs=PROTOCOLS["P6"]["segs"] * 2, soc=0.5, cutoff=None)
# P6 con impulsi lunghi (rimedio C se CRLB > 10 %)
PROTOCOLS["P6L"] = dict(segs=sum([[seg(c, 60), seg(0, 60)] for c in (1, -1, 3, -3)], [])
                        + [seg(1, 300), seg(0, 600), seg(0.2, 540)], soc=0.5, cutoff=None)


def current_series(name, dt=1.0):
    """Corrente ZOH a passo dt: I[k] applicata su (t_k, t_k+dt]. Ritorna array lungo n_steps."""
    segs = PROTOCOLS[name]["segs"]
    out = []
    for I, d in segs:
        out += [I] * int(round(d / dt))
    return np.array(out)

# ---- campagna (BRIEF v4)
PROTOCOLS["P6Lbrief"] = dict(segs=sum([[seg(c, 60), seg(0, 60)] for c in (1, -1, 3, -3)], []) + [seg(1, 300), seg(0, 600), seg(0.2, 780)], soc=0.5, cutoff=None)

# P6L campagna (corretto dal brief, NOTES #32): impulsi +-1C da 60 s, +-3C da 30 s (a 3C 60 s c_e,min = 5.2 mol/m3 nel P6L del brief)
PROTOCOLS["P6L"] = dict(segs=[seg(1, 60), seg(0, 60), seg(-1, 60), seg(0, 60), seg(3, 30), seg(0, 90), seg(-3, 30), seg(0, 90)]
                        + [seg(1, 300), seg(0, 600), seg(0.2, 780)], soc=0.5, cutoff=None)
PROTOCOLS["P6Lx2"] = dict(segs=PROTOCOLS["P6L"]["segs"] * 2, soc=0.5, cutoff=None)
# P5: fuori distribuzione, 5C 10 s da 100 % con trasporto costante (7C 5 s non regge: c_e); riposo 60 s, ripetuto
PROTOCOLS["P5"] = dict(segs=sum([[seg(5, 10), seg(0, 120)] for _ in range(3)], []), soc=0.9, cutoff=None)


def d1_current(seed=7, dur=1800):
    """Ciclo di guida sintetico tipo WLTP (4 fasi), potenza -> corrente, picco di scarica 3C (15 A), rigenerazione limitata a 1.5C.
    Ritorna array [dur] in A (positivo = scarica) a 1 s."""
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
    P = np.where(P < 0, 0.6 * P, P)                      # recupero parziale
    I = P / 3.7                                          # A equivalenti (cella unica scalata)
    I = I / np.max(I) * 15.0
    I = np.maximum(I, -7.5)
    I = I[:dur]; nb = len(I) // 10 * 10
    I = np.repeat(I[:nb].reshape(-1, 10).mean(1), 10)           # blocchi da 10 s (Experiment di PyBaMM regge ~200 step)
    I = I / np.max(I) * 15.0
    I[(np.abs(I) < 0.05)] = 0.0
    return np.round(I, 3)

# F6: protocollo di valutazione dei baseline = P6L x2 (identificazione) + P2b (impulsi 1C, 2C, 3C da 30 s con riposi da 300 s) per la stima di eta anodico e c_e,min
PROTOCOLS["P2b"] = dict(segs=[seg(1, 30), seg(0, 300), seg(2, 30), seg(0, 300), seg(3, 30), seg(0, 300)], soc=0.5, cutoff=None)
PROTOCOLS["F6"] = dict(segs=PROTOCOLS["P6L"]["segs"] * 2 + PROTOCOLS["P2b"]["segs"], soc=0.5, cutoff=None)

# F8: 2C per 300 s da 100 % (trasporto variabile)
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
