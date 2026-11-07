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

fit_v2 parameters without refit, film share 0.5, soft edges 1e-3, 20 °C. Open loop: HPPC20 on the mixed steps
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
- The board's time per step is not measured yet (the PlatformIO platform was not installed in this session).
- Not done: a refit at the reduced sizes.
