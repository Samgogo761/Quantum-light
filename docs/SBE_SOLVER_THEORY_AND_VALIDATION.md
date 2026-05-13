# SBE solver theory, code map, and validation notes

This note summarizes the current new SBE solver in
`C:\Users\26507\Documents\New_SBEs\Quantum-light`, the tests already
performed, and the remaining theoretical/numerical risks. It is written as a
handoff document for independent review.

## 1. Physical target and input data

System: bilayer antiferromagnetic CrI3 with spin-orbit coupling, described by
a Wannier90 tight-binding model generated from QE + Wannier90.

Main data path used locally:

```text
C:\Users\26507\Documents\量子光研究\CrI3_tb.dat
```

The solver uses the Wannier90 `*_tb.dat` file, not only `*_hr.dat`, because the
SBE coupling may need both the real-space Hamiltonian matrix elements and the
real-space position matrix elements:

```math
H_{mn}(R) = <0,m|H|R,n>
```

```math
r^a_{mn}(R) = <0,m|r_a|R,n>, a=x,y,z
```

The parser in `src/mod_wannier.f90` supports the standard Wannier90 `*_tb.dat`
header containing three lattice vectors before `num_wann` and `nrpts`. The
input lattice vectors are treated as authoritative; a mismatch between the
input lattice and the `*_tb.dat` header is reported so the user can catch data
set inconsistencies.

## 2. Band indexing and truncation logic

QE full-space band count: 200 spinor bands.

Electron count: 140 spinor occupied states. Therefore, in the QE full space,
the Fermi-edge pair is:

```text
VBM = band 140
CBM = band 141
```

The Wannier model keeps 112 bands after removing 56 deep bands below the
Wannier disentanglement window. Therefore, inside the 112-band Wannier/SBE
space:

```text
Wannier-space VBM index = 140 - 56 = 84
Wannier-space CBM index = 85
```

The new solver uses `nv_orig = 84` and then maps it into the selected truncated
band window:

```text
nv = nv_orig - nb_start + 1
```

Current tested windows:

| label | `nb_start:nb_end` | band count | occupied count in truncation |
|---|---:|---:|---:|
| 20 bands | 75:94 | 20 | 10 |
| 30 bands | 70:99 | 30 | 15 |
| 40 bands | 65:104 | 40 | 20 |

This is a physically sensible Fermi-window truncation for near-gap dynamics,
but it is not automatically gauge converged in velocity gauge. More remote
filled and empty bands can still contribute to optical matrix-element sum
rules and off-resonant polarization.

## 3. Wannier Fourier reconstruction

Implemented mainly in:

```text
src/mod_wannier.f90
src/mod_crystal.f90
```

For each crystal momentum `k`, the code reconstructs:

```math
H(k) = sum_R exp(i k.R) H(R) / ndeg(R)
```

```math
D_a(k) = sum_R exp(i k.R) r_a(R) / ndeg(R)
```

and the Hamiltonian-gradient velocity matrix:

```math
v_a(k) = partial H(k) / partial k_a
       = sum_R i R_a exp(i k.R) H(R) / ndeg(R)
```

Units:

```text
H: eV -> Hartree
r, R, lattice vectors: Angstrom -> bohr
E_fermi: subtracted from the R=0 onsite diagonal
```

The full 112-band `H(k)` is diagonalized, and then the selected band window is
kept:

```math
H_diag(k) = U^\dagger(k) H(k) U(k)
```

For the truncated basis, `U(k)` contains only the selected eigenvectors.

## 4. Laser field and vector potential

Implemented mainly in:

```text
src/mod_laser.f90
```

For the current linear-light tests:

```text
wvl_nm = 3200
intensity_Wcm2 = 2.0e11
theta_deg = 0
phi_cep_deg = 90
ncyc = 4
env_type = 2
ellipticity = 0
dt = 0.35 a.u.
```

The current code supports two orthogonal field components for elliptic/circular
polarization:

```math
E_x(t), E_y(t)
```

with input parameters:

```text
ellipticity
delta_phase_deg
```

The vector potential is constructed consistently from the field used by the
solver:

```math
A(t) = - integral^t E(t') dt'
```

This matters in velocity gauge because the Hamiltonian is evaluated at
`k + A(t)`.

## 5. Velocity-gauge SBE used in the current tests

Implemented mainly in:

```text
src/mod_sbe.f90
```

The current production/test path is velocity gauge:

```text
gauge_method = 'vg'
```

For each k point, the code preprojects each real-space Hamiltonian shell:

```math
H_R^{proj}(k) = U^\dagger(k) [exp(i k.R) H(R) / ndeg(R)] U(k)
```

Then the time-dependent velocity-gauge Hamiltonian in the truncated basis is:

```math
H_{VG}(k,t) = sum_R exp(i A(t).R) H_R^{proj}(k)
```

This is the projected version of:

```math
H(k + A(t))
```

The density matrix evolves as:

```math
d rho_k(t) / dt = -i [H_{VG}(k,t), rho_k(t)]
```

Initial condition:

```math
rho_{nn}(t=0) = 1 for n <= nv
rho_{nn}(t=0) = 0 for n > nv
rho_{mn}(t=0) = 0 for m != n
```

Dephasing is applied to off-diagonal density-matrix elements every
`n_dt_deph` steps:

```math
rho_{mn} -> rho_{mn} exp(-Delta t / T2), m != n
```

with `T2_fs = 0.5` in the current tests.

## 6. RK4 time stepping

Implemented in:

```text
src/mod_sbe.f90
```

The corrected RK4 logic evaluates the time-dependent Hamiltonian at the
appropriate RK4 stages:

```text
k1: t
k2: t + dt/2
k3: t + dt/2
k4: t + dt
```

Therefore, in velocity gauge, it uses:

```text
A(t), A(t+dt/2), A(t+dt)
```

This fixed the earlier risk that all RK4 stages could use the same field time.
For the present classical 120x120, dt=0.35 case, the numerical effect was
small: the corrected result almost overlaps the earlier result after applying
the missing HHG `dt^2` normalization.

## 7. Current and HHG spectrum

Implemented mainly in:

```text
src/mod_sbe.f90
src/mod_hhg.f90
```

The velocity-gauge current is computed as:

```math
J_a(t) = - 1 / (N_k A_{cell}) sum_k Tr[ rho_k(t) v_a(k,t) ]
```

with:

```math
v_a(k,t) = partial H(k + A(t)) / partial k_a
         = sum_R i R_a exp(i A(t).R) H_R^{proj}(k)
```

`A_cell` is the in-plane cell area. It is meaningful for converting the
discrete k average into a 2D current density. For bilayer CrI3, using the
in-plane area rather than the full vacuum-containing 3D volume is physically
reasonable for a sheet-like response.

The HHG spectrum is calculated from the windowed current:

```math
J_w(omega) = FFT[ w(t) J(t) ]
```

```math
HHG(omega) = dt^2 [ |J_x(omega)|^2 + |J_y(omega)|^2 ]
```

The `dt^2` factor has been fixed in `src/mod_hhg.f90`. This is essential when
comparing spectra computed with different time steps.

## 8. Length-gauge path status

The code has a length-gauge branch using:

```math
H_{LG}(k,t) = H_0(k) - E(t).D(k)
```

but this branch is not the current validation target. It still requires more
careful review before being treated as production-quality, especially regarding
covariant derivatives, Berry connection consistency, and truncation/gauge
issues.

## 9. BSV path status

Implemented mainly in:

```text
src/mod_quantum_light.f90
```

The BSV path samples intensity and phase, runs a classical SBE trajectory per
sample, and averages spectra. One caveat:

```text
bsv_mean_intensity currently feeds an exponential-like sampling formula whose
mean appears to be 2 * bsv_mean_intensity.
```

Therefore, comparisons between BSV and classical light must explicitly check
the effective sampled mean intensity, not only the input variable name.

## 10. Code architecture

Main execution flow in `src/main.f90`:

1. Parse input namelists with `src/mod_input.f90`.
2. Read Wannier data with `src/mod_wannier.f90`.
3. Build reciprocal lattice and k mesh with `src/mod_crystal.f90`.
4. Compute diagnostic valley assignment near K/K'.
5. Diagonalize the Wannier Hamiltonian and truncate the band window.
6. Precompute projected Hamiltonian/position/velocity matrices.
7. Generate classical laser field with `src/mod_laser.f90`.
8. Initialize `rho_k`.
9. Propagate SBE with `src/mod_sbe.f90`.
10. Write `Jt.dat`, decomposed currents, and valley currents.
11. Compute and write `HHG.dat` with `src/mod_hhg.f90`.
12. If BSV is enabled, repeat trajectories and average spectra.

## 11. Old solver comparison: why exact agreement is not expected

Old solver source used for reference:

```text
C:\Users\26507\Documents\量子光研究\script5
```

Old data paths used in plots:

```text
C:\Users\26507\Documents\量子光研究\Jt\J_tot.txt
C:\Users\26507\Documents\量子光研究\HHG\hhg_tot.txt
```

Important differences:

| item | old solver | new solver current test |
|---|---|---|
| basis/gauge | Houston-like (`var_method="ht"`) | velocity gauge (`gauge_method='vg'`) |
| band space | originally all 112 Wannier bands | truncated 20/30/40 near Fermi level |
| SOC/magnetism | old reference noted as no SOC in comparison plot | current Wannier data includes SOC |
| current output cadence | old `J_tot.txt` has 1010 rows over ~42.7 fs | new `Jt.dat` has 5045 rows over ~42.7 fs |
| current formula | Houston-basis current matrix expression | trace of density matrix with `partial H(k+A)/partial k` |
| HHG normalization | old uses its own FFT/window convention | new has explicit `dt^2` FFT scaling |

Therefore the old/new comparison is useful as a qualitative sanity check, but
not a strict equality test. Differences can come from band truncation, gauge
implementation, current definition, SOC/magnetic model, and FFT/window details.

## 12. Completed validation tests

### 12.1 dt normalization and RK4 time point test

Test: 10x10, 20 bands, classical light, compare `dt=0.35` and `dt=0.20`.

Conclusion:

```text
After adding the missing HHG dt^2 factor, dt=0.35 and dt=0.20 agree at the
percent level through the relevant low-order harmonics.
```

Therefore, the time step itself is unlikely to be the main reason for the
observed H9-H15 smoothness or suppression.

### 12.2 k-grid test: 10x10 vs 120x120, 20 bands

Current corrected 120x120 output:

```text
C:\Users\26507\Documents\量子光研究\新SBEs\Quantum-light-wannier90-fortran-integration\output_120x120_current_dt035
```

Reference 10x10 output:

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\local_test_10x10_current_dt035
```

Key result:

| harmonic | 10x10 20 bands | 120x120 20 bands | ratio 120/10 |
|---:|---:|---:|---:|
| H1 | 3.535e+01 | 3.534e+01 | 0.9998 |
| H3 | 7.693e-01 | 7.688e-01 | 0.9994 |
| H5 | 2.714e-03 | 2.710e-03 | 0.9983 |
| H7 | 3.734e-06 | 3.696e-06 | 0.9899 |
| H9 | 4.677e-10 | 7.749e-10 | 1.657 |
| H11 | 8.125e-11 | 6.283e-11 | 0.773 |
| H15 | 1.978e-10 | 2.663e-14 | 1.35e-04 |
| H20 | 5.601e-11 | 1.255e-14 | 2.24e-04 |
| H25 | 1.150e-12 | 7.159e-16 | 6.23e-04 |
| H35 | 5.748e-18 | 4.777e-20 | 8.31e-03 |

Conclusion:

```text
H1-H7 are already k-grid stable. The high-order tail from a 10x10 grid is not
reliable and can create a false plateau-like tail. A 120x120 grid suppresses
that tail strongly.
```

The time-domain current amplitude itself is stable between 10x10 and 120x120:

```text
new 10x10 max |Jx| ~ 0.013019
new 120x120 max |Jx| ~ 0.013017
```

So the k-grid issue is mainly in high-frequency spectral cancellation, not in
the gross time-domain current amplitude.

### 12.3 current corrected 120x120 vs older pre-fix 120x120

The older May-01 120x120 spectrum agrees closely with the current corrected
120x120 spectrum after applying the missing `dt^2` factor to the older data.

Conclusion:

```text
For classical light at dt=0.35, the RK4 time-point correction does not strongly
change the result. The large visual change was mainly the HHG normalization
when comparing spectra across time steps.
```

### 12.4 BSV 5-task output

Existing BSV plot:

```text
C:\Users\26507\Documents\量子光研究\新SBEs\Quantum-light-wannier90-fortran-integration\output_bsv\HHG_bsv_spectrum_5tasks.png
```

Observation:

```text
The BSV high-order tail is stronger than the classical 120x120 tail, but up to
H35 it appears more like a long decay than a clearly resolved plateau.
```

Caveat:

```text
The BSV and classical intensity definitions must be checked before making a
physics claim, because `bsv_mean_intensity` may not equal the actual sample
mean intensity.
```

### 12.5 New local 10x10 band-window convergence test

Generated outputs:

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\band_convergence_10x10
```

Plot:

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\band_convergence_10x10\band_convergence_10x10_to35.png
```

CSV:

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\band_convergence_10x10\band_convergence_10x10_key_harmonics.csv
```

Key result:

| harmonic | 20 bands | 30 bands | 40 bands | 30/20 | 40/20 |
|---:|---:|---:|---:|---:|---:|
| H1 | 3.535e+01 | 5.673e+01 | 8.084e+01 | 1.60 | 2.29 |
| H3 | 7.693e-01 | 1.230e+00 | 1.689e+00 | 1.60 | 2.20 |
| H5 | 2.714e-03 | 4.224e-03 | 5.220e-03 | 1.56 | 1.92 |
| H7 | 3.734e-06 | 5.180e-06 | 4.681e-06 | 1.39 | 1.25 |
| H9 | 4.677e-10 | 2.268e-09 | 3.679e-09 | 4.85 | 7.86 |
| H11 | 8.125e-11 | 1.888e-10 | 2.563e-10 | 2.32 | 3.15 |
| H15 | 1.978e-10 | 1.927e-10 | 2.895e-10 | 0.97 | 1.46 |
| H20 | 5.601e-11 | 1.266e-10 | 3.855e-10 | 2.26 | 6.88 |
| H25 | 1.150e-12 | 4.568e-12 | 9.199e-12 | 3.97 | 8.00 |
| H35 | 5.748e-18 | 4.695e-18 | 9.661e-18 | 0.82 | 1.68 |

Conclusion:

```text
At 10x10, increasing the band window from 20 to 30/40 bands substantially
changes even H1-H5. This suggests the 20-band velocity-gauge truncation is not
fully converged for absolute response amplitudes.
```

However:

```text
The same 10x10 data cannot be used to judge high-order plateau/cutoff
reliably, because the 20-band 120x120 test already showed strong k-grid
cancellation above roughly H9.
```

## 13. Current best interpretation

1. The core SBE flow is structurally normal for a velocity-gauge density-matrix
   solver: Wannier reconstruction, band projection, initial occupied density
   matrix, RK4 propagation, current evaluation, FFT spectrum.
2. The `dt^2` HHG normalization issue has been fixed.
3. The RK4 time-point issue has been fixed.
4. The current 120x120 20-band result is stable in H1-H7 and shows strong
   suppression in H9+ compared with 10x10.
5. The new 30/40-band 10x10 test indicates that the 20-band truncation may
   undercount low-order optical response amplitudes.
6. The strongest remaining theoretical concern is finite-band velocity-gauge
   convergence: a too-small band window can violate optical sum-rule/gauge
   convergence even when the near-Fermi bands are the main physical carriers.
7. The old solver is not a strict benchmark because it differs in gauge/basis,
   band space, SOC/magnetism, output cadence, and current/FFT conventions.

Current conclusion after the 40x40 / 30-band run:

```text
No obvious implementation bug has been isolated in the tested velocity-gauge
classical-light path. The most important open question is not "does the code
run incorrectly?", but "how much of the old/new discrepancy is caused by
finite-band velocity-gauge truncation and methodological differences?"
```

Therefore the next most diagnostic old/new comparison is:

```text
10x10 / full 112 Wannier bands / velocity gauge / same laser and T2.
```

This is not a production-quality HHG convergence test, because 10x10 is too
coarse for the high-order tail. Its purpose is narrower: determine whether
the new solver approaches the old full-Wannier-band result when the band-window
truncation is removed. If the 112-band new run still differs strongly from the
old solver, the remaining discrepancy is more likely due to gauge/basis,
current definition, SOC/magnetism, or normalization conventions rather than
near-Fermi band truncation alone.

## 14. Recommended next controlled checks

Priority 1:

```text
Run 20/30/40 bands on a moderately denser grid, e.g. 20x20 or 40x40, before
attempting expensive 120x120 30/40-band production.
```

Purpose:

```text
Separate band-window convergence from k-grid convergence.
```

Priority 2:

```text
For the chosen band window, repeat a small T2 sweep, e.g. T2=0.5 fs, 1.0 fs,
2.0 fs, with the same window function.
```

Purpose:

```text
Check whether H9-H15 smoothness is physical dephasing/window smoothing rather
than a code bug.
```

Update on 2026-05-13:

```text
A server template for the Priority 1 run was added:
deploy/input_40x40_30bands.nml
deploy/run_hhg_40x40_30bands.sh

This uses nkx=nky=40, nb_start=70, nb_end=99, nv_orig=84, T2=0.5 fs,
dt=0.35 a.u., and velocity gauge.
```

The Priority 2 local T2 sweep has been run for 10x10 / 30 bands:

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\t2_sweep_10x10_30bands
```

Key HHG values:

| harmonic | T2=0.5 fs | T2=1.0 fs | T2=2.0 fs |
|---:|---:|---:|---:|
| H1 | 5.673e+01 | 6.217e+01 | 6.645e+01 |
| H3 | 1.230e+00 | 1.280e+00 | 1.280e+00 |
| H5 | 4.224e-03 | 4.212e-03 | 3.641e-03 |
| H7 | 5.180e-06 | 5.193e-06 | 4.866e-06 |
| H9 | 2.268e-09 | 1.972e-08 | 7.798e-08 |
| H11 | 1.888e-10 | 1.424e-09 | 1.088e-08 |
| H13 | 7.235e-11 | 1.687e-10 | 8.987e-10 |
| H15 | 1.927e-10 | 1.928e-10 | 1.668e-10 |
| H20 | 1.266e-10 | 1.212e-10 | 1.076e-10 |
| H25 | 4.568e-12 | 4.542e-12 | 3.817e-12 |
| H35 | 4.695e-18 | 5.884e-18 | 1.435e-18 |

Interpretation:

```text
Changing T2 mainly affects the transition region around H9-H13 and slightly
changes the time-domain current amplitude. Low harmonics H1-H7 are much less
sensitive, while H15+ remains mostly shaped by k-grid/band-window cancellation
and spectral noise in this 10x10 test. Therefore T2 is a plausible contributor
to the detailed smoothness around H9-H13, but it is not by itself a full
explanation of the high-order behavior.
```

Priority 1 update after the server 40x40 / 30-band run:

```text
C:\Users\26507\Documents\New_SBEs\Quantum-light\output_40x40_30bands_dt035_T2fs05
```

Run parameters:

```text
nkx=nky=40
nb_start=70
nb_end=99
n_trunc=30
nv=15
T2=0.5 fs
dt=0.35 a.u.
velocity gauge
```

The run completed normally. The log reports:

```text
Pre-projection memory estimate: 12722 MB
real wall time: 35m18.713s
```

Selected HHG values:

| harmonic | 10x10 / 30 bands | 40x40 / 30 bands | 120x120 / 20 bands |
|---:|---:|---:|---:|
| H1 | 5.673e+01 | 5.673e+01 | 3.534e+01 |
| H3 | 1.230e+00 | 1.229e+00 | 7.688e-01 |
| H5 | 4.224e-03 | 4.220e-03 | 2.710e-03 |
| H7 | 5.180e-06 | 5.103e-06 | 3.696e-06 |
| H9 | 2.268e-09 | 2.501e-09 | 7.749e-10 |
| H11 | 1.888e-10 | 7.055e-11 | 6.283e-11 |
| H13 | 7.235e-11 | 1.485e-12 | 2.250e-12 |
| H15 | 1.927e-10 | 2.321e-13 | 2.663e-14 |
| H20 | 1.266e-10 | 2.173e-14 | 1.255e-14 |
| H25 | 4.568e-12 | 1.718e-15 | 7.159e-16 |
| H35 | 4.695e-18 | 1.812e-20 | 4.777e-20 |

Interpretation:

```text
For the 30-band window, the time-domain current and H1-H9 are already almost
unchanged between 10x10 and 40x40. However, H13-H25 are suppressed by several
orders of magnitude when the k grid is increased to 40x40. This strongly
suggests that the apparent high-order plateau/tail in the 10x10 spectrum is a
coarse-k-grid residual rather than a converged physical plateau. The 40x40 /
30-band high-order tail is much closer to the previously obtained 120x120 /
20-band result, although the low-order amplitude remains larger because of the
wider band window.
```

Priority 3:

```text
Verify BSV effective intensity distribution by printing the sampled mean and
variance of intensity.
```

Purpose:

```text
Make classical-vs-BSV comparisons physically normalized.
```

Priority 4:

```text
If absolute amplitudes remain important, compare velocity gauge against a
carefully reviewed length-gauge or Houston-gauge implementation in the same
truncated band space.
```

Purpose:

```text
Diagnose finite-band gauge dependence.
```

## 15. CrI3-specific considerations

For bilayer AFM CrI3, the following physical features can influence HHG:

1. Spin-orbit coupling and magnetic order can alter selection rules and valley
   responses.
2. Bilayer AFM symmetry can suppress or reshape even-order harmonics depending
   on stacking, inversion/time-reversal related symmetries, and the light
   polarization direction.
3. The Wannier interpolation was already checked against QE/PRB-like bands, so
   the single-particle band structure is likely not the first suspect.
4. HHG is more sensitive than band interpolation to dipole/velocity matrix
   elements and gauge convergence, so passing the band comparison is necessary
   but not sufficient.
5. The in-plane area normalization is appropriate for sheet current density;
   using the full 25 Angstrom vacuum cell volume would artificially dilute the
   response.
