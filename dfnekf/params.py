"""Chen2020 with constant parameters (constant D_e, kappa, t+) and applicable theta multipliers.
Single source of the numbers: PyBaMM 'Chen2020' (values extracted at run time, see dump_params_csv)."""
import numpy as np

F = 96485.33212
R = 8.314462618
T = 298.15
A_CELL = 0.065 * 1.58          # Electrode height x width [m2]
C_RATE_A = 5.0                 # 1C = 5 A (Nominal cell capacity)

# ageing range (dimensionless, relative to the nominal value)
THETA_NOM = dict(Rf=1.0, as_=1.0, Ds=1.0)
RANGE = dict(Rf=(1.0, 10.0), Ds=(0.3, 1.0))

# SOC 50 % (from PyBaMM initial_soc=0.5, DFN Chen2020)
C0_N_50, C0_P_50 = 15522.211404385791, 35269.4534602329
# SOC 100 % (default Chen2020)
C0_N_100, C0_P_100 = 29866.0, 17038.0


def base():
    """Nominal physical parameters (SI), PyBaMM Chen2020 with constant electrolyte transport."""
    return _base_chen()


def _base_chen():
    return dict(
        Ln=85.2e-6, Ls=12e-6, Lp=75.6e-6,
        Rn=5.86e-6, Rp=5.22e-6,
        eps_en=0.25, eps_es=0.47, eps_ep=0.335,
        eps_sn=0.75, eps_sp=0.665,
        brug=1.5,                      # Bruggeman elettrolita (n, s, p); elettrodo: 0
        sig_n=215.0, sig_p=0.18,
        cmax_n=33133.0, cmax_p=63104.0,
        k0_n=6.48e-7, k0_p=3.42e-6, alpha=0.5,
        Ds_n=3.3e-14, Ds_p=4e-15,
        De=1.7694e-10, kappa=0.9487, tplus=0.2594, tdf=1.0, ce0=1000.0,
        Rf0=5e-9 * 2e5,                # L_sei * rho_sei = 1e-3 Ohm m2
        area=A_CELL,
    )


def apply_theta(P, Rf=1.0, as_=1.0, Ds=1.0):
    """theta: Rf = R_f/R_f0 ; as_ = a_s,n/a_s,n0 (eps_s,n scala; eps_e invariata);
    Ds = D_s,n/D_s,n0."""
    Q = dict(P)
    Q["Rf"] = P["Rf0"] * Rf
    Q["eps_sn"] = P["eps_sn"] * as_
    Q["Ds_n"] = P["Ds_n"] * Ds
    return Q


def dump_params_csv(path):
    import pandas as pd
    P = base()
    pd.DataFrame({"name": list(P), "value": [P[k] for k in P]}).to_csv(path, index=False)
