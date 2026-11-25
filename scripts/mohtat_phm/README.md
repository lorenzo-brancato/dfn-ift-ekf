# mohtat_phm: DFN observer and LDP-aware charging on the Mohtat cells (PHM Society 2027)

Working folder for the conference paper (abstract: `paper/PHM Society 2027/`): the embedded DFN observer tested
hardware-in-the-loop on a literature cell, the LDP reading, and the simulated ageing comparison CC-CV vs LDP-aware
charging. Outputs go to `results/mohtat_phm/` (git-ignored).

## Parameters: fit B

MPM-LE identification, case B (`mohtat_B_fin3`, the reference set since 2026-11-24 evening: generic graphite OCP Ai2020,
common window of C07-C09, calibration on the 2C first cycles C07, C08, C09 plus the HPPC and C/20 of C07, eps_n fixed
0.389; 16.6 mV on the calibration tests; held-out VAL 6-31 mV, C11 now held out, see `heldout_val_rmse.csv`). It
replaces `mohtat_B_fin` (calibration C07-C09 + C11, 16.9 mV), on which the open-loop table and the EKF tuning below were
made; the EKF runs on fin3 are in `results/mohtat_phm/ekf/f3_*`:

```
C:\Users\alexa\OneDrive - Politecnico di Milano\Desktop\Batteries\Python codes\MIONet\mpmle_identification\
    configs\mohtat_B_fin3.json                 study config (model options)
    runs\studies\mohtat_B_fin3\dynamics.json   groups (theta_n, theta_p given at 0 degC: theta_ref_C = 0)
    runs\studies\mohtat_B_fin3\windows.json    electrode windows of every cell
    runs\studies\mohtat_B_fin3\ocp_correction.json   cathode OCP (step 1)
```

Not to be confused with `data/exp_Mohtat/params/mohtat_lm3c.json` (the older lm3c set) or with the LP604765
"fit set B" (`LP_FIT_B`).

## Signals

- Beginning-of-life tests of the 21 cells (C20, CRATE1C, HPPC, VAL): `data/exp_Mohtat/tests/DB_REAL_<cell>_<test>_BOL/`
  (git-ignored copy), loaded by `dfnekf.datasets.load("exp_Mohtat", "C01_VAL")`. VAL is the first ageing cycle of the
  cell (protocols in the table below, from the dataset READ_ME): for C19 (25 degC), C20 (cold), C21 (hot) it is the
  **drive-cycle discharge** (0-50 % DOD) followed by a 1.5C CC-CV charge. The cycler logs every 10 s (point samples of
  the drive cycle), so the prepared current is the mean over each interval taken from the cycler's coulomb count.
- Ageing cycles: `C:\Users\alexa\Datasets Batteries\Mohtat NMC\processed\Mohtat2020\real_cell_NN\Plant`
  (raw data in `...\Mohtat NMC\data`).

## Fit B in dfnekf

`export_fitB.py` (needs PyBaMM; the project's `.venv` or any env with pybamm) writes
`data/exp_Mohtat/params/mohtat_B_fin3.json`. The dataset default is still `mohtat_lm3c.json`: pass the set explicitly,

```
.venv/Scripts/python.exe scripts/mohtat_openloop.py --set data/exp_Mohtat/params/mohtat_B_fin3.json --cell C13 --tests VAL --results results/mohtat_phm/openloop
.venv/Scripts/python.exe scripts/ekf_exp.py --dataset exp_Mohtat --test C13_VAL --set data/exp_Mohtat/params/mohtat_B_fin3.json --start soc
```

Case B needed three options in dfnekf (2026-11-24):
- cathode D_p(y) law at `y_mean + wsurf (y_surf - y_mean)` (`dp_law.wsurf` = MPM-LE `Dp_wsurf` = 1 here;
  `step.DFN.cp_law_of`, `structjac.Struct.law_sel`; the exact Jacobian is tested in `tests/test_cellset_options.py`);
- chi(c_e0, T) tabulated with kappa and D_e (`temperature.electrolyte.tdf`, MPM-LE `tdf_T`);
- D_e = `De_f` D_e,set with the extra activation energy `dEa_De`, folded into the D_e(T) table by the exporter.
theta_n, theta_p (identified at 0 degC) are taken back to 25 degC by the exporter.

Not yet in the C++ port (`embedded/`): the surface weight and chi(T).

## Check: open loop on the VAL tests (2026-11-24)

`mohtat_openloop.py` runs a test at its mean temperature; `openloop_T.py` follows the measured temperature sample by
sample (0.5 K bins), as MPM-LE does. Results in `results/mohtat_phm/openloop/` and `openloop_T/`. RMSE [mV]:

| Cell | T [degC] | VAL: charge, discharge, DOD | dfnekf, mean T | dfnekf, measured T | MPM-LE fit B |
|---|---|---|---|---|---|
| C01 | 23 | C/5, C/5 | 12.7 | | 12.3 |
| C02 | -4 | C/5, C/5 | 30.4 | | 30.6 |
| C03 | 44 | C/5, C/5 | 12.5 | | 18.7 |
| C04 | 30 | 1.5C, 1.5C | 17.9 | | 19.0 |
| C05 | -1 | 1.5C, 1.5C | 91.7 | 19.8 | 21.9 |
| C06 | 48 | 1.5C, 1.5C | 11.8 | | 11.9 |
| C07 (cal) | 31 | 2C, 2C | 19.3 | | 18.4 |
| C08 (cal) | -1 | 2C, 2C | 488 | 28.3 | 15.6 |
| C09 (cal) | 50 | 2C, 2C | 18.9 | | 19.0 |
| C10 | 25 | C/5, 1.5C | 15.1 | | 15.6 |
| C11 (cal) | -3 | C/5, 1.5C | 297 | 14.3 | 15.7 |
| C12 | 45 | C/5, 1.5C | 9.7 | | 10.4 |
| C13 | 25 | C/5, C/5, 50 % | 8.4 | | 8.7 |
| C14 | -3 | C/5, C/5, 50 % | 15.5 | | 15.1 |
| C15 | 46 | C/5, C/5, 50 % | 6.8 | | 7.1 |
| C16 | 26 | C/5, 1.5C, 50 % | 9.9 | | 8.6 |
| C17 | -3 | C/5, 1.5C, 50 % | 23.2 | 18.5 | 17.7 |
| C18 | 46 | C/5, 1.5C, 50 % | 4.2 | | 5.3 |
| C19 | 26 | 1.5C, drive, 50 % | 10.0 | | 9.7 |
| C20 | -3 | 1.5C, drive, 50 % | 26.0 | 19.5 | 17.9 |
| C21 | 46 | 1.5C, drive, 50 % | 9.2 | | 9.3 |

The port reproduces fit B within ~1-2 mV wherever the temperature is steady. The cold, high-current tests self-heat by
~6 K and the model is very sensitive to T there (C08 VAL, first 3000 s at a constant T: 27 mV at +3 degC, a collapse at
-3 degC), so they need the measured temperature; with it they match MPM-LE too, except C08 (28 vs 16 mV, one spike of
0.3 V and one unconverged step), where MPM-LE holds the particles' diffusion at the mean T of constant-current windows.

## EKF for the paper: SOC + LDP (2026-11-24)

`ekf_batch.py` runs `ekf_exp.py` with the PHM configuration:
- start `soc`, no parameter coordinates (SOC only; the LDP is read from the closure at every step, `phi_pl_n` in the npz);
- the measured temperature as a known input for the whole model (`--t-cell measured`: D_s, k0, kappa, D_e, chi, entropic,
  `EKF.set_temperature_law`, chi(T) added);
- 2-s model steps on the 10-s data (each sample held over 5 steps, `datasets` upsampling) with one voltage update per
  sample (`--y-every-s 10`): at 10-s steps the cold 2C tests are not resolved (open loop at 10 s blows up); at 2 s the
  filter's temperature path matches `openloop_T.py` within 2.2 mV rms;
- sigma 15 mV.

SOC + voltage bias (`--coords Vb`) was tried: same SOC at 25 degC, worse at -5 degC (C08 2.1 % vs 0.48 % rms, second half):
the bias absorbs part of the SOC information, so it is not used.

### Tuned configuration on fin3 (2026-11-25)

The filter starts at SOC 50 % (`--soc0 0.5 --p0-q 1.5 --innov-clip-mV 100`).

Noise settings:
- sigma 35 mV, plus 20 mV per C-rate and 15 mV per 10 K below 25 degC (`--r-i-mV 20 --r-cold-mV 15`);
- random walk on the charge along the lithium-conserving direction, `--q-q 0.25` A.h/sqrt(h);
- lithium pseudo-measurement `--li-sigma-Ah 0.05`.

Command:

```
.venv/Scripts/python.exe scripts/mohtat_phm/ekf_batch.py --name f3_final_s35_qq25_li05 --cells C01,...,C21,C13_HPPC --sigma-mV 35 -- --soc0 0.5 --p0-q 1.5 --innov-clip-mV 100 --r-i-mV 20 --r-cold-mV 15 --li-sigma-Ah 0.05 --q-q 0.25
```

Result on the 21 VAL tests and C13 HPPC: SOC rms (second half) 0.6-2.2 % (mean 1.26 %); +-2 sd coverage 80-100 %
(mean 94 %). The weakest coverage is on the slow C/5 charges (C01, C10, C12, C17: 80-84 %), where the model's OCP error
is correlated.

The SOC-only filter (sigma 15 mV, no process noise on the charge) had 1-58 % coverage. Its errors were correlated model
error, not noise.

Lithium weight:
- without the constraint, the state's lithium drifts by up to 0.3 % (C10, ~36 mA.h);
- with sd 0.2 A.h or less the drift is gone;
- the SOC is unchanged.

LDP minimum while charging:
- C08 (2C, -1 degC): 27 mV;
- C05, C20: 52 mV;
- the others: 56-98 mV.
