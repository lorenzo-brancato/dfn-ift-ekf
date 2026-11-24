# embedded/: the DFN + EKF of dfnekf in portable C++ (Seeed XIAO nRF52840)

The grey-box DFN of `dfnekf` (exact modal particles with the cathode D_p(y) law, exact ZOH electrolyte, implicit
closure solved by Newton) and its structured EKF (step Jacobian from the implicit function theorem, A = Λ + U Vᵀ,
never formed; rank-1 voltage update; Rf free, LAM and Ds frozen), as header-only C++: no heap, no STL on the device
path, `real_t` = float or double, the discretization fixed at compile time by a generated header.

```
embedded/
  export_model.py          Python: cell set + dims + step lengths -> generated/<tag>/model.h, golden/*.csv
  include/dfn/             config.h (precision), linalg.h (LU, equilibration), dfn.h (model, closure, open loop),
                           ekf.h (filter), replay.h (serial protocol)
  host/                    CMake tests against the golden files; replay_host.cpp (the board's protocol on stdin/stdout)
  board/xiao_nrf52840/     PlatformIO project (one environment per model)
  tools/replay.py          streams a profile to the board (or the host stand-in), logs and checks the replies
  tools/study.py           discretization study in Python
  generated/               (git-ignored) model headers and golden files: the golden files hold measured data
```

## Use

```bash
python embedded/export_model.py --dataset exp_LP604765 --dims 10,6,10,3,1 --dt 1,5,20      # header + golden files
cmake -S embedded/host -B embedded/build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build embedded/build -j
ctest --test-dir embedded/build --output-on-failure
```

On this PC there is no compiler on PATH: the tests were built with MATLAB's MinGW-w64 g++ 8.1
(`C:\ProgramData\MATLAB\SupportPackages\R2026aPrerelease\3P.instrset\mingw_w64.instrset\bin`, add it to PATH) and
CMake from the venv (`pip install cmake`). Any C++17 compiler works (MSVC: drop `-G`).

Board: `cd embedded/board/xiao_nrf52840 && pio run -e m10_6_10_k31 -t upload`, then
`python embedded/tools/replay.py --port COMx --profile embedded/generated/<tag>/golden/ekf_CYC20_e+0.40.csv --dq 0.4`.
Without the board, `--host-exe embedded/build/replay_<tag>.exe` runs the same protocol on the PC.

## Export format

`generated/<tag>/model.h` (`<tag>` = `lp604765_T20_<Nn>-<Ns>-<Np>_m<Nm>k<K>`): `constexpr` sizes and `static const
real_t` arrays (flash on the board), every number printed with 17 digits and rounded once by the compiler:

| Group | Content |
|---|---|
| sizes | NN, NS, NP, NM, K, N, NEL, NZ, NG (closure unknowns), NDT, DT[] (step lengths), EKF_IDT |
| cell at the test temperature | c_max, k0 (Arrhenius applied), α, F/2RT, χ, a_n, a_p, area, σ_p, R_contact, Rf0, soft-edge widths, entropic coefficients |
| operators | cell widths, electrolyte edge conductances Kd, edge weights, solid-phase edge conductances (flux form) |
| OCP tables | the cell set's uniform 8001 / 4001 knots, linear interpolation (as `jnp.interp`) |
| electrolyte | E (N × N) and G (N × NEL) per step length |
| particles | anode e, gain, ρ, W per step length; cathode e, gain, ρ, W and d/ds per step length on 129 knots of s = (u + 1e-12)^shape (cubic Hermite; max. relative error 2e-8) |
| filter | z0 at the CYC20 start q0, dz/dq (the lithium-conserving charge direction), P0 (sd + p0_q), Q, R, the scaling, SOC read-out weights, window (Q_n, Q_p, x100, y100, Q_ref) |

Golden files: `openloop_HPPC20.csv` (mixed 1 / 5 / 20 s steps), `openloop_CYC20.csv` (2 h at the EKF step),
`states_*.csv` (full state every 200 steps), `jac.csv` (dense `jac_full` at three states), `ekf_CYC20_e+0.40.csv`
(9.2 h at 5 s from +0.4 A.h).

## Numerics that float needed (all also in the double build; results unchanged there)

1. **Potentials as offsets.** φ_e and φ_s,p are solved relative to two references re-centred every step: at ~4 V a
   float resolves 0.5 µV, while the conductance σ/dx ≈ 3e7 S/m² turns that into ~15 A/m² of error (the current is 29 A/m²).
2. **Charge-balance rows.** The last cathode solid-phase row and the last electrolyte row are replaced by the sums of
   their blocks (the cathode's and the cell's charge balance), scaled to A/m². The insulated cathode leaves a uniform
   shift of φ_s,p nearly singular: equilibrated condition number 3.9e6 → 6.7e3. The scaling makes the Newton
   tolerance bound the charge error itself (a potential row allows ~1e4 A/m² per unit of residual).
3. **Equilibration** of the closure Jacobian by powers of two before the LU (raw condition number ~1e11).
4. **Soft-edge gap without cancellation.** Past c_max (the separator-side anode in the CV phase) the gap
   c_max − c_s is ~1e-3 mol/m³, below float's resolution of c_max: it is formed from the softplus terms directly.
5. **Newton tolerance** 1e-4 in float (the residual floor is ~5e-5), with a stagnation stop below 2e-4; 1e-10 in double, as dfnekf.
6. **UD covariance** (Bierman update, Thornton prediction by modified weighted Gram–Schmidt, in place). The Joseph form
   of dfnekf (kept as `-DDFN_COV_UD=0`) goes indefinite in float at the first top of charge of CYC20 (step 1814) and the
   filter returns NaN. UD keeps every D > 0 (floored at dfnekf's jitter, 1e-12).

## Host results (10/6/10, 3 + 1 modes; i7-9700, one thread, g++ -O2)

| Test | Double | Float |
|---|---|---|
| Open loop HPPC20 (5885 mixed steps), max \|V − V_py\| | < 1e-4 mV | 0.004 mV |
| Open loop CYC20 (2 h at 5 s), max \|V − V_py\| | < 1e-4 mV | 0.001 mV |
| Step Jacobian vs `jac_full`, max relative error (per row) | 5e-7 | 1.3e-2, in the Rf column |
| EKF CYC20 from +0.4 A.h (6645 steps of 5 s), max \|SOC − SOC_py\| | < 1e-4 % | 0.004 % |
| EKF covariance | D > 0 every step | D > 0 every step |
| Time per step: open loop / EKF | 0.19 / 2.6 ms | 0.08 / 2.1 ms |

SOC against Coulomb counting on the same run (second half, rms): 0.50 %, as the Python filter.

## Discretization study (`tools/study.py`, `tools/host_timing.py`)

**Cell set: fit_v2** (`lp604765_cellset_v2.json`; the dataset's current set is fit_v3b since commit 8fb542a, see below),
without refit, film share 0.5, soft edges 1e-3, 20 °C. Headers record their set since 2026-11-09 (`DFN_MODEL_SET`,
`export_model.py --set`, the replay's `?` reply); older ones answer `set=unrecorded`. Open loop: HPPC20 on the mixed steps
(time-weighted RMSE, as the identification reports it), CYC20 on a 5-s grid (9.2 h). Maximum deviation: from the full
model (10/6/10, 12 + 2). EKF: CYC20 at 5 s from −0.4 / +0.4 A.h, SOC error against Coulomb counting. C++: float, UD,
one thread on the i7-9700; RAM = `sizeof(EKF)`, the filter's static memory, the same on the M4F.

| Slices | Modes | Filter states | HPPC20 RMSE / max dev. [mV] | CYC20 RMSE / max dev. [mV] | EKF SOC err., 2nd half rms (−0.4 / +0.4) [%] | EKF max after 1 h [%] | C++ RAM [kB] | C++ EKF step [ms] |
|---|---|---|---|---|---|---|---|---|
| 10/6/10 | 12+2 | 327 | 16.0 / 0 | 30.6 / 0 | 0.51 / 0.49 | 0.82 | 555 (does not fit) | 30.8 |
| 10/6/10 | 3+1 | 127 | 15.9 / 19.0 | 30.6 / 5.9 | 0.51 / 0.50 | 0.83 | 138 | 2.04 |
| 10/6/10 | 2+1 | 107 | 15.7 / 28.2 | 30.5 / 11.1 | 0.52 / 0.51 | 0.84 | 114 | 1.48 |
| 10/3/10 | 12+2 | 324 | 16.6 / 3.3 | 30.7 / 4.1 | 0.50 / 0.49 | 0.82 | 544 (does not fit) | 27.3 |
| 10/3/10 | 3+1 | 124 | 16.4 / 17.1 | 30.7 / 5.0 | 0.51 / 0.49 | 0.83 | 132 | 2.15 |
| 10/3/10 | 2+1 | 104 | 16.3 / 26.4 | 30.7 / 10.2 | 0.52 / 0.50 | 0.84 | 108 | 1.42 |
| 5/3/5 | 12+2 | 164 | 16.3 / 3.5 | 30.5 / 5.1 | 0.57 / 0.56 | 0.88 | 147 | 2.49 |
| 5/3/5 | 3+1 | 64 | 16.2 / 16.5 | 30.5 / 5.1 | 0.57 / 0.57 | 0.89 | 38 | 0.25 |
| 5/3/5 | 2+1 | 54 | 16.0 / 25.8 | 30.5 / 9.9 | 0.58 / 0.58 | 0.90 | 32 | 0.18 |
| 3/2/3 | 12+2 | 99 | 17.3 / 11.9 | 30.9 / 15.2 | 1.16 / 1.24 | 1.45 | 57 | 0.61 |
| 3/2/3 | 3+1 | 39 | 17.2 / 12.7 | 30.8 / 15.2 | 1.16 / 1.24 | 1.45 | 16 | 0.07 |
| 3/2/3 | 2+1 | 33 | 17.1 / 20.7 | 30.8 / 15.2 | 1.16 / 1.23 | 1.44 | 13 | 0.06 |

- Down to 5/3/5 the filter's SOC is unchanged within 0.1 %, and the modes cost more than the slices: the 19-28 mV of
  the reduced modes are short spikes on the HPPC pulses that the time-weighted RMSE hardly sees. 3/2/3 doubles the SOC error.
- 5/3/5, 3 + 1 (64 states, 38 kB, 0.25 ms on the PC) keeps the SOC of the full model within 0.06 % and leaves most of the
  board's RAM free; 10/6/10, 3 + 1 (138 kB) is the closest to the Python reference that fits.
- 1-s against 5-s steps (10/6/10, open loop CYC20): RMSE 31.9 against 30.6 mV, so 5 s is not worse on this test.
- The CYC20 RMSE here (30.6 mV) is above the 23.9 mV of the identification: this model moves half of R_contact into the
  anode film (film share 0.5, so that Rf has a baseline) and starts CYC20 from the OCV placement at 5-s binning.
- Not done: a refit at the reduced sizes.

With the **fit_v3b** set (10/6/10, CYC20 at 5 s, ±0.4 A.h) the EKF's SOC error does not depend on the modes either:
0.82 % (2 + 1), 0.82-0.83 % (3 + 1), 0.83-0.84 % (4 + 1, 5 + 1, 8 + 1), 0.84-0.85 % (12 + 2), a bias of +0.8 %, innovation
21 mV (fit_v2: 25 mV, bias -0.44 %). The open loop converges to 12 + 2 with the modes (CYC20, first 2 h: 22, 13, 7.5,
4.7, 2.1, 0.3 mV at 3, 4, 5, 6, 8, 12 modes + 1 band). The SOC difference between the two sets is the parameters', not
the discretization's.

## Board (XIAO nRF52840, Seeed Adafruit core, `-O2`, float, UD)

| Model (fit_v2) | RAM | Flash | EKF step | Accuracy against Python |
|---|---|---|---|---|
| 10/6/10, 3 + 1 (127 filter states) | 149 KB of 232 KB (63 %) | 203 KB of 792 KB | 0.59 s median, 0.63 s p95, 0.77 s max | full CYC20 (6645 steps of 5 s, from +0.4 A.h): V 0.043 mV, SOC 0.004 % max |

RAM of the other builds (no BLE): 10/6/10 4 + 1 177 KB (75 %), 5 + 1 209 KB (88 %); 5/3/5 3 + 1 47 KB; 3/2/3 3 + 1 24 KB.
## Ageing filter (Rf + LLI)

`export_model.py --coords Rf,LLI --start params --test AGE30 --li-sigma 1e-4 --sigma-mV 14 --q-th 1e-4` (tag suffix
`_RfLLI_AGE30`): the set-up of `scripts/ekf_exp.py --start params --coords Rf,LLI`. LLI is a second coordinate (linear, in
units of the anode window: the closure sees the anode means lowered by LLI dx_window c_max,n; its Jacobian column is
−dshift times the sum of the anode mean columns). The lithium pseudo-measurement (total lithium of the state = that of
the beginning-of-life window, relative sd 1e-4) is a second scalar update after the voltage (Bierman in the UD form).
Diagonal P0 (shell noise, 0.3 of the range on Rf and LLI), start at Rf = 1, LLI = 0.

AGE30 (66.1 h, 25 cycles of 1C CC-CV / 1C discharge / 5 min rest, 47 609 steps of 5 s), fit_v3b, 10/6/10, 3 + 1:
Python ends at Rf = 8.61 ± 0.18 (film multiplier), LLI = 0.312 ± 0.0003 of the anode window, innovation 34.9 mV,
lithium drift 2e-7. C++ float (UD) against it over the whole run: V 0.25 mV, SOC 0.003 %, Rf 0.008, LLI 8e-5;
2.3 ms per step on the PC. Board firmware (`m10_6_10_k31_age`): RAM 151.5 KB (64 %), flash 213 KB.
On the board (2026-11-09/10, 7.8 h for the 66 h record): Rf = 8.60 ± 0.18, LLI = 0.3123 ± 0.0003 at the end; against
Python over all 47 609 steps V 0.21 mV, SOC 0.002 %, Rf 0.007, LLI 5e-5; 0.58 s per step (median), 0.78 s max.

## Paper filter (Rf + LLI + voltage bias Vb, two time scales)

The filter of the paper's EKF runs (`results/exp_LP604765/rfli_an_mohtat`, set fit_an_Mohtat2020), exported with
`export_model.py --set <fit_an_Mohtat2020 cellset> --coords Rf,LLI,Vb --li-sigma-Ah 1e-4 --soc-vlim 4.2,3.0
--sigma-mV 30 --q-vb-mV 1 --q-shell 1e-4 --y-every-s 300 --y-burn-s 600 --th-hold-s 7200 --p0-th-release 0.3
--q-th 1e-5 --innov-clip-mV 90 --soft-eps 1e-2 --hold-vb` (tag `lp604765_T20_10-6-10_m3k1_RfLLIVb_CYC20`, the board's
model since 2026-11-18). In the C++ (`ekf.h`):
- Vb: a third coordinate, V = V_model + Vb (Jacobian column 1 on the voltage row), random walk 1 mV per sqrt(s);
- shell noise on the particle means: extra noise rows of the UD time update (header `NQZ`, `IQZ`);
- hold: the coordinates start at 1e-8 of their prior variance with no random walk, so only the SOC is corrected;
  `EKF::release()` after `hold_s` gives them 0.3 of the prior sd and zeroes their cross-covariances, in UD form by MWGS
  on [M U | e_h] (a dense re-factorization of P failed: after 2 h with the lithium constraint P is near singular);
- innovation clip (90 mV) on the state correction; the voltage update once per 300 s after a 600 s burn-in (the
  firmware's schedule, `main.cpp`; the golden's `upd` column in the host test).
CYC20 from a 0.4 A.h start error, 6645 steps of 5 s, 228 voltage updates: C++ float (UD) against Python SOC 0.0007 %,
V 0.10 mV, Rf 9e-5, LLI 2.5e-6, Vb 0.011 mV; SOC vs Coulomb counting 0.42 % rms over the second half. The temperature
goldens of this set stop at Python's first unconverged closure step (Ea_r, Ea_Dp at 90 kJ/mol: a 62 mV jump at
constant current at the end of the 1C discharge, at 40 degC); before it C++ and Python agree (0.0006 mV at 40 degC).

After flashing, the board enumerates on a different COM port (USB ID 2886:8044, the stock firmware is 2886:8045).
PlatformIO links the USB serial only when the sketch includes `Adafruit_TinyUSB.h`.
