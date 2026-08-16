# Results and local data

Tracked small reports may live here. Large numerical outputs and user-provided Wannier data should stay out of git.

| tracked | purpose |
|---|---|
| `gh5_quadrature_diag_20260816/` | GH3→GH5 diagnostic archive (v1=FAIL permanent; Jones / A22 / tail / CEP‑π Jones-only) |
This directory is only a map for where validation data should live locally.

Recommended local result directories:

| directory | purpose |
|---|---|
| `output_10x10_112bands_dt035_T2fs05/` | full 112-band comparison run against the old solver |
| `output_40x40_30bands_dt035_T2fs05/` | current 40x40 / 30-band validation run |
| `band_convergence_10x10/` | local 20/30/40-band convergence test |
| `t2_sweep_10x10_30bands/` | local T2 sweep for 10x10 / 30 bands |
| `compare_refs/` | copied old-solver or external reference data for plotting |

Recommended external data locations:

| file | recommended handling |
|---|---|
| `CrI3_tb.dat` | keep outside git; point input files or server scripts to its absolute path |
| `CrI3_hr.dat`, `CrI3_r.dat` | keep outside git unless tiny example files are created |
| `Jt.dat`, `HHG.dat`, `bands.dat`, `berry_curvature.dat` | generated outputs; do not commit |

The `.gitignore` file is configured to ignore generated result folders, large
Wannier data, compiled binaries, logs, plots, and CSV tables.
