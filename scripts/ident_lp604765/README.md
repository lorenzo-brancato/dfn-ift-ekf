# LP604765 (office cell): grey-box DFN identification on dfnekf

Cell: LP604765 LiPo pouch, about 1.9 Ah at 1C, 20 °C (2.09 Ah at 40 °C, 1.55 Ah at 0 °C), 4.2 / 2.75 V. Data from CNR STEMS (BATTAGE),
`data/exp_LP604765/raw/01. Electric Characterization/01. ID01` (git-ignored; see `data/exp_LP604765/README.md`), read through
`dfnekf.datasets`. Results in `results/exp_LP604765/` (git-ignored); the exported set and the fit are also in `data/exp_LP604765/params/` (committed). Model: the grey-box DFN of
this repository (`dfnekf`: exact modal particles and electrolyte, implicit closure, Newton) with a variable
time step (1 s under current, 5 / 20 s on rests). Run from this folder with the repository's venv:

```bash
../../.venv/Scripts/python.exe step1_ocv.py
```

Moved here on 2026-11-07 from `MIONet/ParamIdentification/lp604765` (scripts, results) and `MIONet/ParamIdentification/data/01_LP604765` (data);
then on 2026-11-07 from `ParamIdentification/` to `scripts/ident_lp604765/` (data to `data/exp_LP604765/raw`, results to `results/exp_LP604765`). The test table
(FIT / VAL) and the raw reader now live in `dfnekf.datasets`; `common.py` and `lpdata.py` re-export them. Paths `results/...` below are relative to `results/exp_LP604765/`.

## Pipeline (2026-11-06)

| Step | Script | What |
|---|---|---|
| data | `lpdata.py`, `survey.py` | raw step files → t, V, I (I > 0 on discharge) |
| 1 | `step1_ocv.py` | rested OCV points (HPPC 1 h, EIS 3 h) and window fits for candidate OCP pairs |
| model | `lpmodel.py`, `common.py` | variable-step DFN on dfnekf, datasets, parameter boxes, process pool |
| 0 | `sens_morris.py`, `plot_morris.py` | Morris screening on the three HPPCs (16 trajectories, 21 parameters) |
| 2 | `fit.py` | least squares (TRF) on the HPPCs at 0/20/40 °C, parallel FD Jacobian, multi-start |
| 3 | `validate.py` | open loop on the fit tests and the held-out tests, figures |
| 4 | `fisher.py` | local sensitivity, Cramér–Rao bound, correlations (white noise; superseded by `ident.py`) |
| 5 | `ident.py`, `profile_lik.py` | identifiability at an optimum: orthogonal ranking, collinearity index, N_eff from the residual autocorrelation; profile likelihood |
| figs | `fig_ident.py` | `results/exp_LP604765/figs/fig_ident_*` (screening, residual, profiles, fits) |
| export | `export_cellset.py` | `results/fit_v3b/lp604765_cellset.json` for `dfnekf.cellset` (copied to `data/exp_LP604765/params/`) |

**Reported set: fit_v3b** (2026-11-07). 12 fitted groups; Ds_p, shape, Ea_r, the entropic terms and the anode groups
come from the literature. The evidence for each choice is in [IDENTIFIABILITY.md](IDENTIFIABILITY.md). The
fit_v2 notes below are the history that led to it.

## Choices and findings

- **Chemistry.** No datasheet was available, so I compared OCP pairs on the 3 h rests. The lm3c NMC532 table with Mohtat2020 graphite fits at
  2.7 mV RMS; LCO pairs (Ai2020, Marquis2019) fit at 30-50 mV. Treated as NMC/graphite. Geometry, porosities, radii and the electrolyte material are
  those of lm3c, with the area scaled to 1.95 Ah.
- **Screening** (`results/sens/morris.png`). Normal-operation effects: y100, tau, Q_p, k0_p, drop, Ds_p are strong; shape, dUdT0, the start offsets,
  x100, Ea_Dp, Ea_r, Q_n and R_contact are moderate. Ds_n, Ea_Dn and k0_n are invisible (≤ 5 mV), the same anode non-identifiability as on Mohtat,
  so they are fixed at the lm3c values. About 45 % of the prior box collapses at the end of discharge (cathode saturation through the D_p(y) law).
- **Fit** (`results/fit_v2`, 17 free parameters, 4 starts, 3 at the same optimum): 15.94 mV on the HPPCs (14.8 / 16.2 / 17.7 mV at 20 / 40 / 0 °C).
  Ea_r sits at its 90 kJ/mol bound and dUdT_n at -3 mV/K.
- **Held out**, whole-test RMSE: 1C cycling at 20 °C (3 cycles with CC-CV, 9.2 h) 23.9 mV (21.5 above 3 V); 1C capacity discharges
  35.7 / 24.4 / 30.5 mV at 20 / 40 / 0 °C (28.0 / 20.4 / 20.6 above 3 V). A held-out test starts at the end of its first rest, with
  the state placed by inverting the model OCV at that voltage.
- **Identifiability** (`results/fit_v2/fisher.json`). Collinearity index 185; Ds_p, drop and shape are fully correlated (|ρ| up to 1.0); the window pairs
  x100–Q_n and y100–Q_p are at 0.93-0.96. One start found a second optimum 0.3 mV worse: R_contact 15 mΩ with Ea_s 60 / Ea_r 5 kJ/mol
  instead of 38 mΩ with 9 / 90 kJ/mol. The series and charge-transfer resistances are not separable on these data. That second set
  predicts worse on every held-out test (cycling 26.5 mV).
- **Time step.** 5 s steps under current err up to 48 mV near the end of discharge against all-1-s runs. The fit used them (cost
  difference 0.06 mV); validation and Fisher use 1 s under current.
- **Known limits.** The end-of-discharge knee is too soft. The CV phase of the charge is about 30 mV low. The 20 °C capacity test is biased by
  about -20 mV over the whole discharge, while the same 1C discharge inside the cycling test fits within ±15 mV.

## Cost

0.27 ms per step on this PC (one thread, CPU shared with other jobs): 9.2 h of cycling at 1 s takes 8.6 s. State size 326,
closure 66 unknowns (dense Newton). That is the budget an embedded port has to meet.

## EKF on the 20 °C cycling (2026-11-07)

`scripts/ekf_exp.py --dataset exp_LP604765 --test CYC20` (formerly `ekf_lp.py`) runs the structured EKF of dfnekf (`dfnekf.ekf.EKF`, implicit closure, D_p(y) law) with the exported cell set on CYC20 (9.2 h,
3 cycles with CC-CV charges), on a 1-s grid. `plot_ekf.py` draws the figures. The output is in `results/ekf/` (csv, npz, `ekf_CYC20.png`, `ekf_CYC20_zoom.png`).

- **Filter.** State: the DFN (326) plus the film multiplier Rf (film = 50 % of R_contact). LAM and Ds are frozen at the identified values.
- **Prior.** The initial covariance is a lithium-conserving charge offset (sd 0.5 Ah) plus 0.5 % independent shell noise.
- **Noise.** σ = 15 mV. Rf random walk of 1e-4 of its range per √s.
- **Start errors.** −0.4, 0 and +0.4 Ah (about ±20 % SOC). The reference is Coulomb counting from the rested start.

| Start error | Open-loop RMSE | EKF innovation RMSE | SOC error, 2nd half (rms / max) | Final Rf | ms per step |
|---|---|---|---|---|---|
| −0.4 Ah | 223 mV | 19.4 mV | 0.24 / 0.46 % | 4.2 | 4.5 |
| 0 | 32 mV | 19.2 mV | 0.21 / 0.44 % | 4.2 | 2.9 |
| +0.4 Ah | 510 mV | 19.3 mV | 0.24 / 0.46 % | 4.2 | 2.7 |

- **Convergence.** The ±20 % start error drops to about 3 % within the first seconds (one 300 mV innovation). It stays within 2-4 % during the
  first discharge and closes at the first end-of-discharge knee (about 1 h). After that, all three runs agree within 0.5 % of Coulomb counting. SOC is read from the cathode mean stoichiometry.
- **Rf is a model-error sink, not a resistance estimate.** It swings between 0.5 and 7 within each cycle, highest in the CV phase (the model is about 30 mV
  low there) and at the end of discharge (soft knee).
- **Lithium is not conserved by the corrections.** The total lithium drifts by −9 %, and the anode-read SOC ends 21 % below the cathode-read SOC. The
  dfnekf lithium pseudo-measurement (`li_sigma` in `EKF`) is the available fix; it is not used here.
- **Overconfident band.** The ±2σ band covers the SOC error only 1-4 % of the time (σ of order 0.1 %). The 0.2 % agreement is real, but
  the filter's own error bar is not.
- **Open loop at the right start: 32 mV**, against 23.9 mV in `validate.py`. The difference is the 50 % film split (the film acts in the kinetics) and the 1-s grid on rests.

**With fit_v3b** (same filter settings; `results/exp_LP604765/ekf_v3b/`), the v2 values are in brackets:
- Innovation RMSE 17.1 mV (19.3); open loop at the right start 31.5 mV (32.0).
- SOC error in the 2nd half 0.31-0.36 % rms, max 0.50 % (0.21-0.24 %, max 0.46 %); settling below 2 % after 0.93 h (0.79 h).
- Lithium drift −6.3 % (−9.1 %); anode-read SOC minus cathode-read SOC −12.9 % (−21.3 %).
- Rf in the 2nd half 0.35-4.95 (0.48-6.95), median 1.5 in both.
- The ±2σ band still covers 1-6 % of the time.

The physically admissible set leaves less model error for Rf and the state to absorb. The SOC accuracy is the same within
0.1 %. ms per step is not comparable: the run shared the machine with other jobs.

## Not done yet

The window and OCP come from only about 6 rested points per temperature; a C/20 (or GITT) test on this cell would pin the OCV. The
ageing blocks (20261022/27/30) are not used. The EKF still needs: the lithium pseudo-measurement, a σ that makes the band honest, the 0/40 °C tests, and an embedded port
(2.7 ms per step on the PC with a 327-state dense covariance).
