# mohtat_phm: DFN observer and LDP-aware charging on the Mohtat cells (PHM Society 2027)

Working folder for the conference paper (abstract: `paper/PHM Society 2027/`): the embedded DFN observer tested
hardware-in-the-loop on a literature cell, the LDP reading, and the simulated ageing comparison CC-CV vs LDP-aware
charging. Outputs go to `results/mohtat_phm/` (git-ignored).

## Parameters: fit B

MPM-LE identification, case B (`mohtat_B_fin`: generic graphite OCP Ai2020, common window of C07-C09, calibration
cells C07, C08, C09, C11; 16.9 mV on the calibration tests; held-out VAL 5-31 mV, see `heldout_val_rmse.csv`):

```
C:\Users\alexa\OneDrive - Politecnico di Milano\Desktop\Batteries\Python codes\MIONet\mpmle_identification\
    configs\mohtat_B_fin.json                 study config (model options)
    runs\studies\mohtat_B_fin\dynamics.json   groups (theta_n, theta_p given at 0 degC: theta_ref_C = 0)
    runs\studies\mohtat_B_fin\windows.json    electrode windows of every cell
    runs\studies\mohtat_B_fin\ocp_correction.json   cathode OCP (step 1)
```

Not to be confused with `data/exp_Mohtat/params/mohtat_lm3c.json` (the older lm3c set) or with the LP604765
"fit set B" (`LP_FIT_B`).

## Signals

- Beginning-of-life tests of the 21 cells (C20, CRATE1C, HPPC, VAL): `data/exp_Mohtat/tests/DB_REAL_<cell>_<test>_BOL/`
  (git-ignored copy), loaded by `dfnekf.datasets.load("exp_Mohtat", "C01_VAL")`.
- Ageing cycles: `C:\Users\alexa\Datasets Batteries\Mohtat NMC\processed\Mohtat2020\real_cell_NN\Plant`
  (raw data in `...\Mohtat NMC\data`).

## Fit B in dfnekf

`export_fitB.py` (needs PyBaMM; the project's `.venv` or any env with pybamm) writes
`data/exp_Mohtat/params/mohtat_B_fin.json`. The dataset default is still `mohtat_lm3c.json`: pass the set explicitly,

```
.venv/Scripts/python.exe scripts/mohtat_openloop.py --set data/exp_Mohtat/params/mohtat_B_fin.json --cell C13 --tests VAL --results results/mohtat_phm/openloop
.venv/Scripts/python.exe scripts/ekf_exp.py --dataset exp_Mohtat --test C13_VAL --set data/exp_Mohtat/params/mohtat_B_fin.json --start soc
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

| Cell | T [degC] | dfnekf, mean T | dfnekf, measured T | MPM-LE fit B |
|---|---|---|---|---|
| C01 | 23 | 12.7 | | 12.3 |
| C02 | -4 (1C) | 30.4 | | 30.6 |
| C03 | 44 | 12.5 | | 18.7 |
| C04 | 30 | 17.9 | | 19.0 |
| C05 | -1 (1.5C) | 91.7 | 19.8 | 21.9 |
| C06 | 48 | 11.8 | | 11.9 |
| C07 (cal) | 31 | 19.3 | | 18.4 |
| C08 (cal) | -1 (2C) | 488 | 28.3 | 15.6 |
| C09 (cal) | 50 | 18.9 | | 19.0 |
| C10 | 25 | 15.1 | | 15.6 |
| C11 (cal) | -3 (1.5C) | 297 | 14.3 | 15.7 |
| C12 | 45 | 9.7 | | 10.4 |
| C13 | 25 | 8.4 | | 8.7 |
| C14 | -3 (1C) | 15.5 | | 15.1 |
| C15 | 46 | 6.8 | | 7.1 |
| C16 | 26 | 9.9 | | 8.6 |
| C17 | -3 (1.5C) | 23.2 | 18.5 | 17.7 |
| C18 | 46 | 4.2 | | 5.3 |
| C19 | 26 | 10.0 | | 9.7 |
| C20 | -3 (1.5C) | 26.0 | 19.5 | 17.9 |
| C21 | 46 | 9.2 | | 9.3 |

The port reproduces fit B within ~1-2 mV wherever the temperature is steady. The cold, high-current tests self-heat by
~6 K and the model is very sensitive to T there (C08 VAL, first 3000 s at a constant T: 27 mV at +3 degC, a collapse at
-3 degC), so they need the measured temperature; with it they match MPM-LE too, except C08 (28 vs 16 mV, one spike of
0.3 V and one unconverged step), where MPM-LE holds the particles' diffusion at the mean T of constant-current windows.

The EKF (`ekf_exp.py --set ...mohtat_B_fin.json`) runs on C13 VAL (finite, innovation 4.2 mV) with the dataset's default
coordinates; its configuration for the paper (SOC + LDP, temperature input) is still to be set.
