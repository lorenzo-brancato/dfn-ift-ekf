# exp_LP604765: LP604765 LiPo pouch (CNR STEMS, BATTAGE)

**Cell.** LP604765 LiPo pouch, cell ID01. Capacity at 1C: about 1.9 Ah at 20 °C, 2.09 Ah at 40 °C and 1.55 Ah at 0 °C. Voltage window 4.2 / 2.75 V.
Treated as NMC/graphite: the lm3c NMC532 OCP with Mohtat2020 graphite fits the 3 h rests at 2.7 mV.

**Files (git-ignored).** `raw/` holds the CNR STEMS tree as delivered:
`raw/01. Electric Characterization/01. ID01/<yyyymmdd>_<test>_<T>/NNN_<step>.txt`. Each file has a CSStudio header and
the columns E (V), i (A, discharge negative) and T (s). `raw/02. Thermal …` and `raw/03. Mechanical …` are not used.
They are read by `dfnekf.datasets` (`lp_*` functions; I > 0 on discharge).

**Identified set (committed).** `params/lp604765_cellset.json` (schema dfnekf-cellset-1, cell `LP01`) and
`params/best.json` (the fit it was exported from). Both come from `scripts/ident_lp604765` (fit_v2: 17 free parameters,
15.9 mV on the HPPCs). The working copies and every other output are in `results/exp_LP604765/`.

**Tests** (`datasets.load("exp_LP604765", <key>)`):

| Key | Folder | T | Role | Start of the record |
|---|---|---|---|---|
| HPPC20 | 20261011_HPPC_20°C | 20 °C | fit | first pulse, after the 1 h rest at the top (dq = 0) |
| HPPC40 | 20261014_HPPC_40°C | 40 °C | fit | same, dq_40 fitted |
| HPPC00 | 20261021_HPPC_00°C | 0 °C | fit | same, dq_00 fitted |
| CYC20 | 20261022_Ageing (files 001-008) | 20 °C | validation | t = 5345 s (end of the first rest); 3 1C cycles with CC-CV, 9.2 h |
| CAP20 / CAP40 / CAP00 | 202610{09,13,16}_Capacity_* | 20 / 40 / 0 °C | validation | step 2 to 4 (1C discharge) |

A validation test starts at the end of its first rest. It is placed by inverting the OCV of the set at that rested voltage
(`datasets.initial_charge`). SOC = 1 - q / 1.95 Ah. No temperature is logged: `T` is the chamber value.

Other folders (wake-up cycles, later capacity checks, EIS, the later ageing blocks) are not used yet.
