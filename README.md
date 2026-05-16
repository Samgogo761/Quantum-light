# Quantum-light HHG-SBE Solver

Wannier90-based semiconductor Bloch equation (SBE) solver for high-harmonic
generation (HHG) calculations in bilayer antiferromagnetic CrI3. The current
validated production path uses a velocity-gauge density-matrix propagation with
Wannier band windows around the Fermi level.

The code reads Wannier90 `*_tb.dat` files so that both real-space Hamiltonian
matrix elements and real-space position matrix elements can be kept available.
The present classical-light validation focuses on the velocity-gauge path.

## Current Code Status

The current solver includes the numerical fixes used in the latest validation:

- `src/mod_hhg.f90`: HHG spectra include the continuous-time FFT scaling
  factor `dt^2`.
- `src/mod_sbe.f90`: RK4 evaluates the time-dependent Hamiltonian at
  `t`, `t + dt/2`, and `t + dt`.
- `src/mod_laser.f90`: the field is represented as vector components, with
  support for ellipticity and phase difference. The current validation still
  uses linear polarization.
- `src/mod_wannier.f90`: supports the standard Wannier90 `*_tb.dat` header with
  lattice vectors and checks lattice consistency against the input file.
- `src/mod_sbe.f90`: adds `gauge_method = 'matrix_vg'`, a velocity-gauge path
  that uses the real-space position matrix elements `rmn_R` to construct the
  covariant momentum/velocity matrix before band truncation.
- `src/mod_sbe.f90`: applies T2 dephasing in the instantaneous eigenbasis of
  the Hamiltonian used at the end of each RK4 step, instead of damping
  off-diagonal elements in a fixed truncated basis.

No current validation result points to an obvious implementation bug in the
velocity-gauge classical-light path. The main unresolved physics/numerics issue
is convergence with respect to band-window size and k-grid density.

## Gauge Methods

The input variable `gauge_method` currently supports:

| value | mathematical form | status |
|---|---|---|
| `'vg'` | Peierls-only velocity gauge, `H(k,t)=H(k+A(t))` | existing validated path |
| `'matrix_vg'` | matrix velocity gauge, `H(k,t)=H0(k)+sum_a A_a(t) P_a(k)` | new rmn_R-based path |
| `'lg'` | length gauge, `H(k,t)=H0(k)-sum_a E_a(t) D_a(k)` | available, less validated |

For `matrix_vg`, the code first constructs in the full Wannier space:

```math
D_a(k) = \sum_R e^{i k\cdot R} r_a(R) / n_{\rm deg}(R)
```

```math
P_a(k) = \partial_{k_a}H(k) - i [D_a(k), H(k)] .
```

It then projects `H(k)` and `P_a(k)` into the selected band window using the
field-free eigenvectors `U_S(k)`:

```math
H^S_0(k)=U_S^\dagger(k)H(k)U_S(k), \quad
P^S_a(k)=U_S^\dagger(k)P_a(k)U_S(k).
```

The propagated Hamiltonian is:

```math
H^S_{\rm matrix\_vg}(k,t)=H^S_0(k)+\sum_a A_a(t)P^S_a(k).
```

The T2 dephasing convention is also gauge-aware at the propagation level. At
each dephasing step, the code diagonalizes the instantaneous Hamiltonian,
`H(t+dt) W = W epsilon`, transforms `rho` to this basis, damps only
off-diagonal coherences by `exp(-n_dt_deph dt / T2)`, and transforms back.

The current uses the corresponding minimal-coupling velocity:

```math
J_a(t) = -\frac{1}{N_k A_{\rm cell}}\sum_k
{\rm Tr}\{\rho^S_k(t)[P^S_a(k)+A_a(t)I]\}.
```

This branch is intended for controlled comparison with the old solver's
velocity-gauge implementation. The older `'vg'` branch is kept unchanged as the
Peierls-only path.

## Repository Layout

| path | purpose |
|---|---|
| `src/` | Fortran source code |
| `deploy/` | server input files and SLURM helper scripts |
| `input/` | default/local input examples |
| `docs/` | theory notes, validation summaries, BSV notes |
| `results/` | map of local result/data directories; large outputs are ignored |
| `tests/` | small standalone tests |
| `examples/` | example legacy-style input files |
| `patches/` | old integration snippets kept for reference |

Main modules:

| module | role |
|---|---|
| `mod_wannier.f90` | read `_tb.dat`, `_hr.dat`, and position matrices |
| `mod_crystal.f90` | reciprocal lattice, k-grid, band truncation, valley labels |
| `mod_laser.f90` | electric field and vector potential |
| `mod_sbe.f90` | density-matrix propagation and current evaluation |
| `mod_hhg.f90` | FFT and HHG spectrum output |
| `mod_quantum_light.f90` | BSV sampling and ensemble runs |
| `main.f90` | classical and BSV execution flow |

For the detailed theory and validation log, see:

```text
docs/SBE_SOLVER_THEORY_AND_VALIDATION.md
```

## Required External Data

The CrI3 Wannier data are large and should not be committed to git.

Typical local/server file:

```text
CrI3_tb.dat
```

The server scripts rewrite `wannier_tb_file` to the absolute path configured by
`TB_FILE`. Local runs need the input file to point to a valid local path.

## Build

On the server, the deployment scripts compile automatically.

On Windows/MSYS2, a local build can be done from PowerShell with:

```powershell
C:\msys64\usr\bin\bash.exe -lc 'PATH=/mingw64/bin:$PATH; export PATH; cd "/c/Users/26507/Documents/New_SBEs/Quantum-light" && mingw32-make clean && mingw32-make'
```

Generated binaries, module files, logs, plots, CSV files, and output data are
ignored by `.gitignore`.

## Production and Validation Runs

### Main 120x120 / 20-band production template

```bash
sbatch deploy/run_hhg.sh
```

Uses `deploy/input_production.nml` by default:

```text
nkx = 120
nky = 120
nb_start = 75
nb_end = 94
nv_orig = 84
T2_fs = 0.5
dt = 0.35
gauge_method = 'vg'
```

### 40x40 / 30-band validation

```bash
sbatch deploy/run_hhg_40x40_30bands.sh
```

Uses:

```text
deploy/input_40x40_30bands.nml
output_40x40_30bands_dt035_T2fs05/
```

This run is the current preferred intermediate convergence check because it is
much cheaper than 120x120 but already suppresses the false high-order tail seen
on a 10x10 grid.

### 10x10 / 112-band full-window comparison

```bash
sbatch deploy/run_hhg_10x10_112bands.sh
```

Uses:

```text
deploy/input_10x10_112bands.nml
output_10x10_112bands_dt035_T2fs05/
```

The Wannier model has 112 bands. This test is the cleanest way to isolate
band-truncation effects when comparing the new solver to the older solver that
used the full Wannier band space.

If scripts copied from Windows trigger a SLURM line-ending error, run:

```bash
sed -i 's/\r$//' deploy/*.sh deploy/*.nml
```

## Current Validation Conclusions

Band indexing:

```text
QE full space: VBM = 140, CBM = 141
Wannier 112-band space: VBM = 84, CBM = 85
```

Current tested windows:

| label | bands | occupied count |
|---|---:|---:|
| 20-band window | 75-94 | 10 |
| 30-band window | 70-99 | 15 |
| 40-band window | 65-104 | 20 |
| full Wannier window | 1-112 | 84 |

Numerical summary:

- `dt = 0.35` and `dt = 0.20` agree well after the `dt^2` HHG normalization
  fix, so time step is not the leading issue for the current classical runs.
- `10x10` k-grid spectra can show a false high-order tail or plateau-like
  feature. This tail is strongly suppressed on denser k grids.
- `40x40 / 30 bands` matches `10x10 / 30 bands` very well for H1-H9 but
  suppresses H13-H25 by several orders of magnitude.
- Increasing the band window from 20 to 30/40 bands changes low-order
  amplitudes, so a 20-band velocity-gauge truncation is efficient but not fully
  converged for absolute intensity.
- The `10x10 / 112 bands` full-Wannier-window diagnostic does not continue the
  20->30->40 monotonic increase. Instead, low-order amplitudes are strongly
  suppressed, while H9-H25 are enhanced relative to 20/30/40. This points to a
  finite-band velocity-gauge cancellation/gauge-truncation issue rather than a
  simple "more bands gives a monotonic limit" picture.
- T2 mainly affects the H9-H13 transition region; it does not by itself explain
  the full high-order behavior. The 2026-05-14 `matrix_vg` local sweep with
  instantaneous-basis dephasing gives, for 10x10/30 bands: H1 = 17.33, 17.07,
  16.72 for T2 = 0.5, 1.0, 2.0 fs, respectively, while H9-H15 vary much more
  strongly.
- The old Houston-basis solver is not a strict equality benchmark because it
  differs in gauge/basis, current definition, band space, dephasing-unit
  convention, output cadence, and FFT/window conventions. In the old code,
  `deph_t2 = 0.5` is interpreted as 0.5 optical cycles, not 0.5 fs. For the
  3200 nm pulse this is about 5.34 fs.

Current best interpretation:

```text
The new velocity-gauge solver does not show an obvious implementation failure
in the tested classical-light path. The remaining old/new discrepancy is most
likely dominated by finite-band velocity-gauge behavior, k-grid cancellation in
the high-order tail, and methodological differences between the new
velocity-gauge solver and the old Houston-basis solver. The full 112-band
diagnostic shows that the discrepancy is not explained by near-Fermi band
truncation alone. The old solver should be treated as a historical qualitative
reference until its exact Wannier input, SOC setting, current normalization, and
dephasing convention are reproduced.
```

## Output Files

Each run directory usually contains:

| file | content |
|---|---|
| `HHG.dat` | harmonic order, angular frequency, HHG_x, HHG_y, HHG_total |
| `Jt.dat` | total time-domain current |
| `Jt_decomposed.dat` | instantaneous-basis intra/inter diagnostic |
| `Jt_valley.dat` | K/K' separated current diagnostic |
| `bands.dat` | truncated-band interpolation on the SBE grid |
| `berry_curvature.dat` | Berry-curvature diagnostic |
| `run.log` | parameter summary and runtime log |

These files are generated data and are intentionally ignored by git.

## Recommended Next Checks

1. If needed, run `40x40 / 40 bands` or a low-cost `10x10` band-window scan
   beyond 40 bands to map the non-monotonic band-window behavior.
2. Compare old `var_method="vg"` against new `gauge_method='matrix_vg'` with
   the same Wannier data, band window, dephasing convention, and current
   normalization.
3. Use an external bilayer-CrI3 HHG/SHG reference as the physics benchmark,
   preferably with matched stacking, AFM configuration, SOC, laser field, and
   spectrum convention. A directly relevant HHG paper is Liu/Zhang/Si,
   Europhysics Letters 140, 25001 (2022), DOI `10.1209/0295-5075/ac9c26`.
4. For absolute HHG amplitudes, review the length-gauge/Houston path or a more
   gauge-consistent truncation strategy before committing to an expensive dense
   k-grid production run.
5. Keep BSV and length-gauge validation separate until the classical
   velocity-gauge reference is fixed.
