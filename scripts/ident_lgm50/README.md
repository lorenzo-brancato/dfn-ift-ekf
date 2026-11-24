# LG M50: the LP604765 identification repeated on a richer dataset

The cell is an LG M50 21700 (NMC811 / graphite-SiOx, 5 Ah). The data are the Kirdaky dataset. Expt 4 has 8 cells (A–H) with beginning-of-life check-ups, all run in a 25 °C chamber (the 10 / 25 / 40 °C in the cell names is the ageing temperature). Expt 5 has 8 cells (5A–5H) cycled at 10 / 25 / 40 °C; its first ageing cycle gives the temperature range. They sit outside the repository (`C:\Users\alexa\Datasets Batteries\Kirdaky DS LG50M`, env `M50_ROOT`). Results go to `results/exp_LGM50/` (git-ignored).

This is the same pipeline and the same model as LP604765. `scripts/ident_lp604765/common.py` and `lpmodel.py` switch to this cell when `LP_CELL=lgm50`. That setting also sets `LP_BASE_SET`, `LP_Q_CELL`, `LP_BASE_Q`, `LP_T_REF_C=25` and `LP_DQ_RANGE`. The per-cell windows are taken from `windows.json`.

**Why this dataset.** Mohtat logs every 10 s and has no short pulses. Here:
- the beginning-of-life tests log every 0.1 s during the 10 s pulses (binned to 1 s by `lpmodel.prepare`);
- there is a GITT with 1 h rests, and a C/10 discharge and charge;
- the surface temperature is measured. The model uses it as a time-varying input, uniform in the cell (`LP_TMEAS=1`, no thermal model).

Chen2020 / OKane2022 parametrise this exact cell, so the dV/dQ selection of the anode has a known answer.

**Split.**
- Fit on cells A, D and F (tests P05, P10 and GITT, at 25 °C), plus the first 1C discharge of 5A, 5D and 5F (10 / 25 / 40 °C, surface 13–49 °C). These carry the activation energies.
- Hold out cells B, E, G and 5B, 5E, 5G, plus the C10, CC05 and G20 tests of the fit cells.
- Cells C, H, 5C and 5H are spare.

Every cell gets its own window from its own GITT rests, because capacity and balance are properties of the cell. The dynamics and the Arrhenius laws are shared and come from the fit cells only.

## Tests (BoL, `m50data.py`)

Every test starts after a CC-CV charge and a 2 h rest.

| Kind | File | What |
|---|---|---|
| C10 | fast_01 | C/10 discharge, 6 h rest, C/10 charge (1 s) |
| P05 | fast_02 | 0.5C base current with 10 s pulses at 0.25–1.5C, discharge then charge (0.1 s) |
| P10 | fast_03 | 1C base current with 10 s pulses up to 1.5C (7.5 A), discharge (0.1 s) |
| CC05 | slow_02 | 0.5C discharge and charge |
| GITT | slow_03 | 25 × 288 s at 0.5C, 1 h rests |
| G20 | slow_04 | 5 × 1440 s at 0.5C, 1 h rests |

The GITT starts at the top of the cell's window (dq = 0). The other tests start by OCV inversion of their rested voltage, because the CV ends differ by up to 7 mV at the top.

The WLTP drive cycles are not used yet. The processed cycle files have a broken time base, and the raw ageing records are binary `.mpr`.

## Pipeline

| Step | Script | Result (2026-11-17) |
|---|---|---|
| base | `make_baseset.py` | `base_chen2020.json`: Chen2020 geometry, kinetics and OCPs; ORegan2022 electrolyte; area 0.1027 m² |
| 1a | `dva_select.py --qn-max 6.5 --span-min 0.7 --tag _phys` | Anode OCP by dV/dQ on the C/10 of A, D and F. With a physical window, Chen2020 (measured on this cell) is ranked 1st, 3rd and 2nd. It is tied with NCA_Kim2011 (dV/dQ RMS 0.0334 vs 0.0333 V/Ah). Without the window bound, the best fits use 8–20 Ah anodes. |
| 1b | `ocp_sub.py --cell D` | Cathode OCP by subtraction (U_p = OCV + U_n). The C/10 discharge is corrected to rest with the GITT rests (no model); beyond the knee the GITT rests are used directly. Below the knee the raw subtraction is not monotone (U_n is steep there, so it rose by 85 mV at y 0.88–0.90); a monotone projection (isotonic, minimum slope 0.2 V per unit y) fixes it (max change 78 mV, RMS 5 mV). It is 7 mV RMS from Chen2020 on y 0.29–0.90. |
| 1c | `ocv_window.py` | Per-cell windows (x100, y100, Q_n, Q_p; dU/dT = 0). Expt 4 uses the GITT rests above 3.2 V (22 of 25; the split below the knee is not known): 0.8–2.2 mV RMS. Expt 5 uses the RPT0 C/10 corrected with the C/10 overpotential of A, D and F: 0.6–0.7 mV. Q_n is 5.84–5.95 Ah (Chen2020: 5.83) and Q_p is 7.66–7.79 Ah (Chen2020: 8.73). The windows are fixed in the dynamic fit. |
| 0 | `../ident_lp604765/sens_morris.py --r 8` (LP_CELL=lgm50) | Morris screening of 12 dynamic parameters on the 12 fit tests. Mean mu [mV]: Ds_p 854, drop 662, tau 473, Ea_Dp 345, shape 328, Ea_r 283, k0_p 252, Ds_n 249, Ea_Dn 234, k0_n 189, R_contact 30, Ea_s 5.6. Ea_s is fixed at 0. |
| 2 | `../ident_lp604765/fit.py --tests fit --jac ift --tag m50_v2 --free Ds_n,Ds_p,drop,shape,k0_n,k0_p,R_contact,tau,Ea_Dn,Ea_Dp,Ea_r` | Joint fit on the 12 fit tests (Chen nominal 71.9 mV RMS; m50_v1 stopped at 38.5 mV on the non-monotone U_p, with artefacts in the rests at the bottom; m50_v2 restarts from it at 33.9 mV) |
| 3 | `../ident_lp604765/validate.py` | Held-out cells and tests |
