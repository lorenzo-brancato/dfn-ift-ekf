# exp_Mohtat: Mohtat et al. NMC532 / graphite pouch cells (real data)

**Cells.** 21 cells, C01-C21, at beginning of life (the Mohtat 2021 ageing data set, processed in MIONet's ParamIdentification).

**Files (git-ignored).** `tests/DB_REAL_<cell>_<test>_BOL/` holds `cycles.csv` (t_s, I_A > 0 on discharge, V_V, T_C,
eis = 1 on EIS sweeps during rests), `config.json` (dt_s and the source) and `metadata.json`. The tests were resampled to
10 s for the EM group fitting. They were copied on 2026-11-07 from
`MIONet/ParamIdentification/data/DB_REAL_C??_*_BOL` (84 folders, 45 MB). The `_BOLQ` variants and the ageing cycles
(`<Mohtat NMC>/processed/Mohtat2020/real_cell_NN/Plant`, used by `scripts/mohtat_batch.py --plant-root`) were not copied.

| Test | What | Duration (C01) |
|---|---|---|
| C20 | C/20 charge and discharge (OCV) | 41 h |
| CRATE1C | 1C discharge | 2.9 h |
| HPPC | pulse test | 14.8 h |
| VAL | validation drive cycle | 19.9 h (other cells 3.7-20 h) |

**Identified set (committed).** `params/mohtat_lm3c.json` (schema dfnekf-cellset-1, 21 cells), exported by
`scripts/export_mohtat_lm3c.py` from the MIONet study `mohtat_2step_fv_lm3c` (step 1: per-cell windows on C20;
step 2: dynamics on C01/C04/C07 C20 + HPPC and the VAL of C01-C09; 17.3 mV overall, C01 VAL 11.0 mV).

Load with `datasets.load("exp_Mohtat", "C01_VAL")`. The start is at the top of the window when the test begins under current,
and otherwise by OCV inversion at the first voltage (`cellset.q_start`).
