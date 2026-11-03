"""M5: charge-pulse metrics (definitions of BRIEF_M5_charge.md section 1; same node and phi_pl / eta definitions as M3-C)."""
import numpy as np


def pulses(I, thr=4.9, cc_s=100.0, dt=0.5):
    """Contiguous runs with I <= -thr. Returns list of dict(label, kind, a, b): label '-<n>C' by the minimum current (1C = 5 A), kind 'pulse' or 'CC' (duration > cc_s), index range [a, b)."""
    m = np.asarray(I) <= -thr; out = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            imin = float(np.min(I[i:j])); lab = f"-{int(round(-imin / 5.0))}C"; kind = "CC" if (j - i) * dt > cc_s else "pulse"
            out.append(dict(label=lab, kind=kind, a=i, b=j)); i = j
        else: i += 1
    return out


def windows(I, p, rest_s=90.0, dt=0.5):
    """pulse window and the rest window (contiguous I = 0 samples after the pulse, up to rest_s)."""
    a, b = p["a"], p["b"]; k = b; nmax = int(rest_s / dt)
    while k < len(I) and k - b < nmax and abs(I[k]) < 1e-9: k += 1
    return [("pulse", a, b), ("rest90", b, k)]


def metrics(ref_eta, ref_ppl, ref_ce, est_eta, est_ppl, est_ce, a, b):
    """errors in mV (eta, phi_pl), ce_min in %; signed phi_pl error mean(est - ref); fraction of samples with est - ref > +5 mV (unsafe side); error on the minimum of phi_pl."""
    if b <= a: return {}
    sl = slice(a, b); e_pl = 1e3 * (est_ppl[sl] - ref_ppl[sl]); e_eta = 1e3 * (est_eta[sl] - ref_eta[sl]); ec = 100 * np.abs(est_ce[sl] - ref_ce[sl]) / ref_ce[sl]
    return dict(n=b - a, E_phipl_rms_mV=float(np.sqrt(np.mean(e_pl ** 2))), E_phipl_max_mV=float(np.abs(e_pl).max()), phipl_err_signed_mean_mV=float(e_pl.mean()), frac_unsafe_gt5mV=float(np.mean(e_pl > 5.0)), phipl_min_ref_mV=float(1e3 * ref_ppl[sl].min()),
                phipl_min_err_mV=float(1e3 * (est_ppl[sl].min() - ref_ppl[sl].min())), E_eta_rms_mV=float(np.sqrt(np.mean(e_eta ** 2))), ce_min_relerr_max_pct=float(ec.max()))


def detect(ref_ppl, est_ppl, thr_mV, t, a, b):
    """threshold crossing (downwards) within [a, b): reference time, delay of the estimate (+: late, -: early), missed detection, false alarm."""
    thr = thr_mV * 1e-3; r = np.where(ref_ppl[a:b] < thr)[0]; e = np.where(est_ppl[a:b] < thr)[0]
    if len(r) == 0: return dict(ref_crossed=False, est_crossed=len(e) > 0, delay_s=np.nan, missed=False, false_alarm=len(e) > 0, t_ref_s=np.nan)
    tr = t[a + r[0]]
    if len(e) == 0: return dict(ref_crossed=True, est_crossed=False, delay_s=np.nan, missed=True, false_alarm=False, t_ref_s=tr)
    return dict(ref_crossed=True, est_crossed=True, delay_s=float(t[a + e[0]] - tr), missed=False, false_alarm=False, t_ref_s=tr)
