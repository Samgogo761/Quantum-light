# Quantum-light HHG-SBE Solver

This repository contains a Wannier90-based semiconductor Bloch equation solver for high-harmonic generation (HHG) calculations in bilayer AFM CrI3. The current production path uses the velocity gauge with a truncated Wannier band window around the Fermi level.

## Current Status

The current code includes the two numerical fixes used for the latest validation runs:

- HHG Fourier spectra include the continuous-time scaling factor `dt^2` in `src/mod_hhg.f90`.
- The RK4 propagator evaluates the time-dependent Hamiltonian at `t`, `t + dt/2`, and `t + dt` in `src/mod_sbe.f90`.

The default production input in `deploy/input_production.nml` is set for:

- `120 x 120` k-grid
- velocity gauge, `gauge_method = 'vg'`
- 20-band truncated window, Wannier bands `75` to `94`
- `nv_orig = 84`, so the truncated occupied count is `nv = 10`
- `dt = 0.35` a.u.
- `T2 = 0.5` fs
- linearly polarized classical light at `3200 nm`, `2.0e11 W/cm^2`

## Repository Layout

- `src/`: Fortran source code.
- `deploy/`: server input files and SLURM helper scripts.
- `input/`: small local/example inputs and older test outputs.
- `tests/`: standalone test code for the quantum-light sampler.
- `Makefile`: local and server build entry point.

Main source modules:

- `mod_wannier.f90`: reads Wannier90 `_tb.dat`, `_hr.dat`, and position matrix data.
- `mod_crystal.f90`: builds k-grid, band structure, truncation, and valley assignment.
- `mod_laser.f90`: builds electric field `E(t)` and vector potential `A(t)`.
- `mod_sbe.f90`: density-matrix propagation and current evaluation.
- `mod_hhg.f90`: HHG FFT and spectrum output.
- `mod_quantum_light.f90`: BSV sampling.
- `main.f90`: classical and BSV execution paths.

## Server Run

Before submitting, check these variables near the top of `deploy/run_hhg.sh`:

```bash
NTHREADS=36
COMPILER="intel"
TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat"
WORKDIR="/public/home/wangjs/project/Quantum-light"
OUTDIR="${WORKDIR}/output_120x120_current_dt035"
```

Then submit from the project root on the server:

```bash
sbatch ./deploy/run_hhg.sh
```

The script will:

1. Load the Intel oneAPI environment.
2. Compile `hhg_sbe`.
3. Generate `${OUTDIR}/input.nml` from `deploy/input_production.nml`.
4. Run `hhg_sbe input.nml`.
5. Write outputs under `${OUTDIR}`.

Important output files:

- `HHG.dat`: harmonic order, angular frequency, `HHG_x`, `HHG_y`, `HHG_total`.
- `Jt.dat`: time-domain total current.
- `Jt_decomposed.dat`: intraband/interband decomposition in the instantaneous band basis.
- `Jt_valley.dat`: K/K' separated currents.
- `bands.dat`: truncated-band interpolation on the SBE k-grid.
- `berry_curvature.dat`: Kubo-formula Berry curvature diagnostic.
- `run.log`: full runtime log and parameter summary.

## Local 10x10 Check

A local `10 x 10` run is useful as a smoke test after modifying source code. It should not be used to judge final plateau/cutoff convergence.

On Windows with MSYS2 installed:

```powershell
C:\msys64\usr\bin\bash.exe -lc 'PATH=/mingw64/bin:$PATH; export PATH; cd "/c/Users/26507/Documents/New_SBEs/Quantum-light" && mingw32-make clean && mingw32-make'
```

For local execution, make sure the `wannier_tb_file` path in the input points to an existing `CrI3_tb.dat`. The server script rewrites this path automatically, but local runs need a valid local path.

Expected smoke-test behavior:

- Standard output, or `run.log` if output is redirected, reports `Wannier TB loaded: nwann=112, nrpts=579`.
- `Band window : 75 to 94`, `n_trunc = 20`, `nv = 10`.
- `nt` is consistent with `ncyc = 4` and the chosen `dt`.
- `HHG.dat` is generated and uses the fixed `dt^2` scaling.

## Physical and Numerical Caveats

The current production path is a velocity-gauge implementation. It is not expected to be numerically identical to the older Houston-basis solver unless gauge, current definition, band space, dephasing, window function, and normalization are made consistent.

Known points to keep in mind:

- The 20-band truncation is efficient and physically focused near the Fermi level, but high-order HHG can still be sensitive to remote bands.
- The length-gauge path exists but has not been the priority for current validation.
- The current is normalized by the 2D cell area `A_cell`; older solver outputs may use a different normalization convention.
- `10 x 10` k-grids are useful for quick checks, but high-order spectra and apparent plateau structure require denser k-grid convergence tests.
- Existing BSV output should be compared carefully with classical output only when the mean intensity and code version match.

## Recommended Validation Workflow

1. Run a local or server `10 x 10` smoke test after source edits.
2. Compare `dt = 0.35` and `dt = 0.20` at `10 x 10` if changing propagation or HHG post-processing.
3. Run the production `120 x 120` classical calculation with the current code.
4. Only after the current classical reference is fixed, run BSV ensembles with matching model and intensity settings.
