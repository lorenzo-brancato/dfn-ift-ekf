# LP604765: which parameter groups are fitted, and why (draft for the paper)

This note records the evidence behind each choice of the reported parameter set (**fit_v3b**). It is written to become
the "Parameter identification" subsection of the paper. Results are in `results/exp_LP604765/`, in `fit_v2b/`
(ident_v2b.json, profile_v2b.json) and `fit_v3b/` (ident_v3b.json). Figures are in `figs/fig_ident_*.pdf`.
Scripts: `sens_morris.py`, `fit.py`, `ident.py`, `profile_lik.py`, `validate.py`, `fig_ident.py`.

## Principle

The observer (the structured EKF of `dfnekf`) needs a DFN that runs, not a full physical parameterisation. We therefore
estimate **lumped groups**, the combinations the terminal voltage actually depends on, and never their constituents.
Examples: the electrode capacities Q_n = ε_s,n A L_n c_max,n F and Q_p; the stoichiometry window (x100, y100); the
electrolyte transport factor τ_e, which divides κ and D_e and replaces Bruggeman exponent, porosity and tortuosity; the
cathode exchange-current multiplier k0_p; the cathode diffusivity law D_p(y) = D_s,p 10^(−d_p u^s_p); the lumped series
resistance R_c; the activation energies; the entropic coefficients; and the start offsets Δq of the 0/40 °C tests.
Everything structural comes from the literature and is never fitted:
- Geometry, porosities, particle radii and electrolyte material: Mohtat2020 [PyBaMM] and the lm3c identification of the
  Mohtat cells, with the area scaled to the 1.95 Ah capacity of this cell.
- Electrolyte κ(c, T) and D_e(c, T): ORegan2022.
- OCP curves: lm3c NMC532 and Mohtat2020 graphite. The pair was chosen on the rested points, at 2.7 mV RMS.

A group is estimated only if the data determine it (identifiability) **and** its estimate is physically admissible. A
group that fails either test is taken from the literature. The cost of that choice is measured on the fit data and,
decisively, on held-out tests.

## Step 1: global screening (Morris; Fig. fig_ident_screening a)

Elementary effects of 20 groups on the three HPPCs (0/20/40 °C), over a plausible prior box (r = 16 trajectories,
p = 4 levels). μ* ranges from 880 mV (d_p) down to 5 mV (E_a,s, E_a,Dn). Screening only orders the groups; the
decisions are taken at the optimum (Steps 2-3).

## Step 2: a converged reference optimum, and local identifiability there (Fig. fig_ident_screening b)

**Reference.** The earlier multi-start fit (fit_v2: 17 groups, 15.94 mV) stopped at its evaluation cap. A profile
refit (Step 3) found a better point, in a different basin (Q_n 4.63 → 3.49 Ah, s_p 0.57 → 1.74). Re-optimised from
there to convergence (TRF, x-tolerance), it gives the reference fit_v2b: 17 groups, **14.24 mV** (HPPC 14.6 / 11.9 /
15.9 mV at 20 / 40 / 0 °C). All local statements below are made at fit_v2b. *The profile likelihood thus also served as
a check of the multi-start: an unconverged optimum makes every downstream identifiability statement wrong.*

**Correlated residual (Fig. fig_ident_residual).** The residual is model error and is correlated in time. Its
integrated autocorrelation time is τ = 114 / 91 / 248 s (HPPC 20 / 40 / 0 °C). So the effective number of independent
samples is N_eff = Σ T_k / τ_k = **544**, not about 69 000 1-Hz samples. All variances and profile thresholds use
N_eff. A white-noise analysis would be optimistic by a factor of 10 to 16 in standard deviation.

**Criteria.** Sensitivity measure δ_j and collinearity index γ_K (Brun et al., 2001), and orthogonalisation ranking
(Yao et al., 2003), on s_j = ∂r/∂ln θ_j (central differences, all 20 groups):

| rank | group | δ_j (mV) | orthogonal residual (mV) | γ (first k) | class |
|---:|---|---:|---:|---:|---|
| 1 | y100 | 688 | 688 | 1.0 | identifiable |
| 2 | Q_p | 688 | 323 | 2.9 | identifiable |
| 3 | d_p | 176 | 138 | 3.2 | identifiable |
| 4 | x100 | 175 | 122 | 3.3 | identifiable |
| 5 | E_a,Dp | 47.5 | 45.9 | 3.3 | identifiable |
| 6 | E_a,r | 34.9 | 32.8 | 3.4 | identifiable, **at bound (90 kJ/mol)** |
| 7 | τ_e | 40.4 | 25.7 | 3.7 | identifiable |
| 8 | Q_n | 130 | 23.5 | 7.9 | identifiable |
| 9 | Δq_0 | 29.9 | 16.3 | 8.4 | identifiable |
| 10 | R_c | 30.2 | 14.4 | 8.7 | identifiable |
| 11 | ∂_T U_n | 11.7 | 8.7 | 8.7 | identifiable, **at bound (−3 mV/K)** |
| 12 | ∂_T U_0 | 12.6 | 5.0 | 9.2 | identifiable |
| 13 | s_p | 32.4 | 3.8 | 12.0 | weak → profile |
| 14 | E_a,s | 8.4 | 3.2 | 12.0 | weak → profile |
| 15 | Δq_40 | 12.0 | 3.0 | 12.1 | weak (test offset, kept) |
| 16 | k0_p | 12.5 | 1.9 | 12.6 | weak → profile |
| 17 | k0_n | 2.0 | 0.89 | 12.6 | not identifiable |
| 18 | D_s,p | 30.9 | **0.36** | **117** | redundant with d_p (ρ = 0.998) |
| 19 | D_s,n | 0.06 | 0.04 | 119 | not identifiable |
| 20 | E_a,Dn | 0.05 | 0.03 | 122 | not identifiable |

**Findings.**
1. D_s,n, E_a,Dn and k0_n are not identifiable (orthogonal residual ≤ 0.9 mV). They are fixed at lm3c, the same anode
   non-identifiability as on the Mohtat cells.
2. D_s,p and d_p are redundant (ρ = 0.998; adding D_s,p raises γ from 13 to 117).
3. s_p, E_a,s and k0_p are weak (orthogonal residual 1.9-3.8 mV) and are decided by their profiles.
4. E_a,r and ∂_T U_n are identifiable but end at the box bounds, at values outside the literature (Step 3).

## Step 3: profile likelihood (Fig. fig_ident_profiles)

Each group is fixed on a grid, every other group is re-optimised to convergence (≤ 80 evaluations; all refits
converged), and the RMSE is recorded (Raue et al., 2009). The 95 % threshold with N_eff samples is
ΔRMSE / RMSE < √(1 + χ²₁,₀.₉₅ / N_eff) − 1 = **0.35 %**.

| group | grid | ΔRMSE | verdict |
|---|---|---|---|
| D_s,p | ×0.18 (lm3c), ×0.5, ×1.8 | +0.26, +0.01, +0.27 % | **flat → practically non-identifiable; fixed at lm3c 9.65e-14 m²/s** |
| s_p | 1.0, 1.37 (lm3c), 2.5 | +0.29, +0.08, +0.22 % | **flat → fixed at lm3c 1.37** |
| k0_p | ×0.5, ×2 | +1.0, +1.0 % | identifiable within a factor of 2 → free |
| E_a,s | 0, 30 kJ/mol | +1.4, +5.3 % | identifiable → free |
| E_a,r | 20, 40, 60 kJ/mol (literature range 18-54) | +4.8, +4.8, +3.2 % | literature values cost 3-5 % of the HPPC fit |
| ∂_T U_n | 0, −1 mV/K (literature |·| ≲ 0.3) | +8.8, +5.5 % | literature values cost 5-9 % |
| ∂_T U_0 | 0 | +3.8 % | |

The last three are the price of physical admissibility *on the fit data*. Whether to pay it is a model-selection
question, settled on held-out data in Step 4.

## Step 4: reduced sets and held-out validation (Fig. fig_ident_fits)

Two reduced sets were fitted to convergence, warm-started from the converged profile point at D_s,p = lm3c:
- **v3a**, 15 groups: D_s,p and s_p at lm3c.
- **v3b**, 12 groups: v3a, plus E_a,r = 39.57 kJ/mol (Mohtat2020, NMC) and the entropic coefficients from O'Regan et al.
  (2022). Their graphite (measured) and NMC entropic changes, evaluated along this cell's window and projected onto
  the model's linear law, give ∂_T U_0 = 0.008 mV/K and ∂_T U_n = 0.285 mV/K (residual of the projection 0.013 mV/K).

All values are RMSE in mV.

| set | groups | at bound | HPPC 0/20/40 °C | mean fit | CYC20 | CAP 0/20/40 °C | mean held out | held out, V ≥ 3 V |
|---|---:|---|---|---:|---:|---|---:|---:|
| v2 (not converged) | 17 | E_a,r, ∂_T U_n | 17.7 / 14.8 / 16.2 | 16.2 | 23.9 | 30.5 / 35.7 / 24.4 | 28.6 | 22.6 |
| v2b (reference) | 17 | E_a,r, ∂_T U_n | 15.9 / 14.6 / 11.9 | 14.1 | 26.3 | 27.0 / 35.1 / 28.9 | 29.3 | 21.6 |
| v3a | 15 | E_a,r, ∂_T U_n | 15.8 / 14.5 / 12.3 | 14.2 | 27.0 | 30.9 / 32.1 / 30.3 | 30.0 | 21.1 |
| **v3b** | **12** | **none** | 19.5 / 15.6 / 14.2 | 16.5 | 25.6 | 29.7 / 36.4 / 24.0 | **28.9** | 21.8 |

**Reading.** The literature temperature laws cost 2.3 mV on the HPPC fit, mostly at 0 °C. They cost nothing on the
held-out tests: v3b predicts as well as the 17-group sets (28.9 against 29.3 mV) with 5 fewer groups, and every
estimate is physically admissible. The lower HPPC cost of v2b and v3a was bought with non-physical temperature
parameters that do not transfer to other tests. In the paper this is over-fitting, made visible by the held-out split.
The residual temperature error is now in the residual (HPPC00 19.5 mV), where it belongs, not hidden in E_a,r.

**Conditioning of v3b (ident_v3b.json, at the v3b optimum, 16.6 mV, N_eff = 374).** γ of the 12 free groups is
**8.6** (v2's 17 groups at the same point: 48.5). Cramér-Rao relative sd with N_eff:
y100 0.4 %, Q_p 0.6 %, d_p 1.1 %, x100 1.8 %, E_a,Dp 2.4 %, Q_n 2.8 %, τ_e 5.6 %, E_a,s 5.6 %, Δq_0 6.8 %, R_c 12.9 %,
**k0_p 48 %**, Δq_40 50 %. At the literature values the entropic groups carry almost no information (orthogonal
residual 0.73 and 0.03 mV). Their earlier identifiability was a consequence of their non-physical size.

**Caveats to state.**
- k0_p is weakly determined at v3b (48 %). E_a,s and E_a,r are correlated (ρ = −0.95 when both are free): with E_a,r
  fixed, E_a,s carries the remaining thermal sensitivity (26.7 kJ/mol). Together with R_c, these three describe the
  ohmic plus charge-transfer resistance and its temperature law, not separate physical processes. The same
  non-separability was seen at fit_v2.
- Δq_40 and Δq_0 are nuisance parameters (start of the 0/40 °C HPPCs). They are not used by the observer.
- The entropic projection uses an NMC811 cathode (O'Regan) for an NMC532 cell. Its contribution is ≤ 0.07 mV/K, an
  order of magnitude below the graphite term.

## Reported set

**fit_v3b**: 12 fitted groups and 8 groups from literature. HPPC 16.5 mV, held out 28.9 mV (21.8 mV above 3 V).
No estimate is at a bound, and the free set has γ = 8.6.

## References

- Brun, R., Reichert, P., Künsch, H.R. (2001). Practical identifiability analysis of large environmental simulation models. *Water Resources Research* 37(4), 1015-1030.
- Yao, K.Z., Shaw, B.M., Kou, B., McAuley, K.B., Bacon, D.W. (2003). Modeling ethylene/butene copolymerization with multi-site catalysts: parameter estimability and experimental design. *Polymer Reaction Engineering* 11(3), 563-588.
- Raue, A., et al. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25(15), 1923-1929.
- Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics* 33(2), 161-174.
- Chen, C.-H., et al. (2020). Development of experimental techniques for parameterization of multi-scale lithium-ion battery models. *J. Electrochem. Soc.* 167, 080534.
- O'Regan, K., Brosa Planella, F., Widanage, W.D., Kendrick, E. (2022). Thermal-electrochemical parameters of a high energy lithium-ion cylindrical battery. *Electrochimica Acta* 425, 140700.
- Mohtat, P., et al.: PyBaMM parameter set "Mohtat2020" (NMC532 / graphite).
- Ecker, M., et al. (2015). Parameterization of a physico-chemical model of a lithium-ion battery. *J. Electrochem. Soc.* 162, A1836 / A1849.
- Thomas, K.E., Newman, J. (2003). Heats of mixing and entropy in porous insertion electrodes. *J. Power Sources* 119-121, 844-849.
- Le, W., et al. (2006). *J. Electrochem. Soc.* 153, A2147. (The NMC111 entropy polynomial used by Mohtat2020 diverges for y > 0.8, below −1.7 mV/K at the end of discharge. It was therefore not used.)
- NMC532/graphite full-cell entropy dominated by graphite: ChemRxiv 10.26434/chemrxiv-2022-d2q4n; graphite −0.2 to +0.3 mV/K: J. Electrochem. Soc. 10.1149/1945-7111/ae352a. **Check against the full texts before citing.**
