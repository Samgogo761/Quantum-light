# hBN Two-Band Model Benchmark

This folder is a lightweight, model-based HHG benchmark for the new
Quantum-light SBE solver. It deliberately avoids the first-principles hBN slab
Wannier route, because the vacuum states in a large-c slab made the low
conduction manifold hard to localize in the quick test.

The file `generate_hBN_model_tb.py` writes the analytic model into the
Wannier90-style `tb.dat` format read by `src/mod_wannier.f90`.

## Model

The Hamiltonian is the standard minimal two-band gapped honeycomb form:

```text
H11 = +M0
H22 = -M0
H12(k) = t [1 + exp(i k.a1) + exp(-i k.a2)]
H21(k) = conj(H12(k))
```

Parameters:

```text
t  = 2.33 eV
M0 = 1.96 eV
a0 = 2.50 Ang
c  = 20.00 Ang
```

The direct gap at the K point `(1/3, 1/3)` in reciprocal fractional
coordinates is `2*M0 = 3.92 eV`.

## SBE Input

`input_hBN_model_lgcov_k30_yue_gaarde.nml` uses a Yue/Gaarde-style benchmark
laser setup:

```text
gauge_method = lg_cov
nkx = nky = 30
bands = 1 valence + 1 conduction
lambda = 1600 nm
intensity = 3.5e12 W/cm^2
ncyc = 11
T2_fs = 1e30, T2_cycles = -1  (no dephasing)
```

This is not a first-principles hBN model. It is meant to validate the SBE time
propagation, lg_cov path, current/HHG output, and qualitative HHG selection
behavior against the standard two-band hBN literature model.

## Reproduce

From the repository root:

```powershell
python examples/hBN_model_tb/generate_hBN_model_tb.py
```

Then run the solver from this folder or pass the input file as a command-line
argument from the repository root.

On Windows/MSYS, one tested local command is:

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "export PATH=/mingw64/bin:/usr/bin:$PATH; cd '/c/Users/26507/Documents/New_SBEs/Quantum-light/examples/hBN_model_tb' && ../../hhg_sbe.exe input_hBN_model_lgcov_k30_yue_gaarde.nml > run_hBN_model_lgcov_k30.log 2>&1"
```

The run produces:

```text
Et.dat
Jt.dat
HHG.dat
bands.dat
Jt_decomposed.dat
Jt_valley.dat
quantum_geometry.dat
harmonics_summary.dat
```

Use `analyze_hBN_model_hhg.py` to extract harmonic-window peak values:

```powershell
python analyze_hBN_model_hhg.py --hhg HHG.dat --max-order 15
```

Current sanity-check result from the local run:

```text
Initial current |J(t=0)| = 2.6652e-19
TB Hermiticity diagnostics = 0 within printed precision
Wall time = 9.62 s for lg_cov / 30x30 / 2 bands
```

## Literature-Style Scan

For comparison with the hBN SBE literature, the local k-grid scan script runs
`T2 = 5 fs`, `lambda = 1600 nm`, `I = 3.5 TW/cm^2`:

```powershell
python run_local_k_scan.py --k 30 60 90 120 --t2-fs 5.0 --ncyc 11.0
python compare_hBN_k_scan.py --k 30 60 90 120 --t2-label 5p0 --max-order 25
```

The main outputs are:

```text
runs/lgcov_k30_T2_5p0fs/
runs/lgcov_k60_T2_5p0fs/
runs/lgcov_k90_T2_5p0fs/
runs/lgcov_k120_T2_5p0fs/
analysis/k_scan_harmonics_summary.dat
analysis/k_scan_spectrum_T2_5fs.svg
```

An additional 35 fs pulse-duration check, closer to the wording in
Yue/Gaarde's hBN paper, was run as:

```powershell
python run_local_k_scan.py --k 120 --t2-fs 5.0 --ncyc 6.557 --tag ncyc6p557
```

That run is in:

```text
runs/lgcov_k120_T2_5p0fs_ncyc6p557/
```

The robust qualitative comparison point is:

```text
parallel Jx      -> mainly odd harmonics
perpendicular Jy -> mainly even harmonics
tail extends toward H25, but high-order absolute intensities become very small
```

This agrees with the central hBN SBE selection pattern in Yue/Gaarde. It should
not be presented as a point-by-point reproduction of their Fig. 2, because this
folder uses a minimal analytic TB model rather than their pseudopotential
band/dipole/Berry data. A strict graphical reproduction should be treated as a
separate follow-up task: align the exact band/dipole/Berry data, pulse
definition, Fourier-window convention, and plotted spectral normalization.
