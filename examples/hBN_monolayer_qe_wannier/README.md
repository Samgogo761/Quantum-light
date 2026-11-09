# Monolayer hBN QE + Wannier90 + SBE benchmark

This folder contains a monolayer hBN benchmark input suite for validating the `Quantum-light` 2D `lg_cov` SBE path.

## Reference positioning

Primary SBE/gauge reference:

- Lun Yue and Mette B. Gaarde, *Structure- and laser-gauges for the semiconductor Bloch equations in high-harmonic generation in solids*, Phys. Rev. A 101, 053411 (2020), arXiv:2003.02961.

Structure and scale reference:

- Guillaume Le Breton, Angel Rubio, and Nicolas Tancogne-Dejean, *High-harmonic generation from few layer hexagonal boron nitride*, arXiv:1807.02961.

## Geometry and electronic model

- Monolayer hBN, 1 B + 1 N.
- B-N bond length: 2.73 Bohr = 1.4449 Angstrom.
- Hexagonal lattice constant: a = sqrt(3) * 2.73 Bohr = 2.5026 Angstrom.
- Vacuum: c = 20.0 Angstrom.
- PBE, no SOC, no magnetism.
- Wannier basis: B:s;p + N:s;p, 8 WFs.
- QE bands: 24 bands.
- Expected occupied bands for SBE: nv_orig = 4.

## Workflow

1. Copy this folder to the server, for example `/public/home/wangjs/project/hBN_TB`.
2. Put the B and N pseudopotentials in `pseudo/`.
3. Run `bash run_all.sh` or override executable paths as shown in the script header.
4. `run_all.sh` runs `bands.x` immediately after `hBN_bands.in`, before the NSCF step overwrites `tmp/hBN.save`; this produces `hBN_qe_bands_path.dat` for QE-vs-Wannier band validation.
5. After `hBN.eig` is generated, choose `dis_win_*` / `dis_froz_*` in `hBN.win`.
6. Run final `wannier90.x hBN > logs/hBN_wannier90.out`.
7. Download `hBN_tb.dat`, `hBN_hr.dat`, `hBN_band.dat`, `hBN.wout`, `hBN.eig`, and `hBN_qe_bands_path.dat` for validation.

## Why the first hBN SBE run uses 1600 nm

The hBN SBE benchmark follows Yue and Gaarde's hBN scale rather than the CrI3 production laser:

- `wvl_nm = 1600.0`
- `A0 = 0.35 a.u.` in the reference paper
- `E0 ~= omega0 * A0 ~= 0.00998 a.u.`
- `intensity_Wcm2 ~= 3.5e12`
- `ncyc = 11.0`, matching a total pulse length close to `2*tau = 58.8 fs`

The pulse envelope in `Quantum-light` is an electric-field cos^2 envelope, so it is not mathematically identical to the reference vector-potential cos^2 envelope. It is still suitable as a solver benchmark because the central frequency, field scale, and total pulse duration are aligned.
