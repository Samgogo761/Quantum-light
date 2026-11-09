# hBN Benchmark Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a self-contained monolayer hBN QE + Wannier90 + SBE benchmark input suite aligned with the approved 2026-06-20 hBN benchmark design.

**Architecture:** Add a new focused example directory under `examples/hBN_monolayer_qe_wannier/`. Keep QE/Wannier inputs, runner script, README, pseudo notes, and SBE templates together so the benchmark can be uploaded to the server as one folder.

**Tech Stack:** Quantum ESPRESSO input files, Wannier90 `.win`, shell runner, `Quantum-light` SBE namelist.

---

### Task 1: Create hBN QE/Wannier Input Suite

**Files:**
- Create: `examples/hBN_monolayer_qe_wannier/hBN_scf.in`
- Create: `examples/hBN_monolayer_qe_wannier/hBN_bands.in`
- Create: `examples/hBN_monolayer_qe_wannier/hBN_bands_pp.in`
- Create: `examples/hBN_monolayer_qe_wannier/hBN_nscf.in`
- Create: `examples/hBN_monolayer_qe_wannier/hBN.pw2wan`
- Create: `examples/hBN_monolayer_qe_wannier/hBN.win`
- Create: `examples/hBN_monolayer_qe_wannier/run_all.sh`
- Create: `examples/hBN_monolayer_qe_wannier/README.md`
- Create: `examples/hBN_monolayer_qe_wannier/pseudo/README.md`

- [ ] **Step 1: Create the example directory**

Run: `New-Item -ItemType Directory -Force -Path examples/hBN_monolayer_qe_wannier,pseudo`

Expected: the hBN example directory and nested pseudo directory exist.

- [ ] **Step 2: Add QE SCF input**

Create `hBN_scf.in` with PBE, no SOC, no magnetism, `CELL_PARAMETERS angstrom`, `a=2.5026 Å`, `c=20 Å`, `18 18 1` SCF grid, and B/N pseudopotential names.

- [ ] **Step 3: Add QE bands input**

Create `hBN_bands.in` with the same structure and a high-symmetry path `G -> M -> K -> G`, using `calculation='bands'`.

- [ ] **Step 4: Add QE NSCF input**

Create `hBN_nscf.in` with `nosym=.true.`, `noinv=.true.`, `nbnd=24`, and an explicit `18×18×1` `K_POINTS crystal` list.

- [ ] **Step 5: Add Wannier90 inputs**

Create `hBN.win` with `num_bands=24`, `num_wann=8`, `mp_grid=18 18 1`, projections `B:s;p` and `N:s;p`, the same explicit k-point list as NSCF, path bands, and no first-pass disentanglement windows. Create `hBN.pw2wan` with `write_mmn`, `write_amn`, and `write_eig` enabled.

- [ ] **Step 6: Add runner and documentation**

Create `run_all.sh` using QE 7.2 paths consistent with the ZnO example, and README instructions explaining the two-stage Wannier window workflow.

### Task 2: Add SBE Input Template

**Files:**
- Create: `examples/hBN_monolayer_qe_wannier/sbe_inputs/input_hbn_lgcov_k20_yue_gaarde.nml`
- Create: `examples/hBN_monolayer_qe_wannier/sbe_inputs/input_hbn_lgcov_k42_yue_gaarde.nml`

- [ ] **Step 1: Add the k20 SBE template**

Create a `lg_cov` namelist using `hBN_tb.dat`, `nkx=nky=20`, `nb_start=1`, `nb_end=8`, `E_fermi_eV` documented as a post-Wannier value to confirm after QE output, `wvl_nm=1600`, `intensity_Wcm2=3.5e12`, `ncyc=11`, and no dephasing.

- [ ] **Step 2: Add the k42 SBE template**

Copy the same template with `nkx=nky=42` for the literature-scale hBN grid.

### Task 3: Verify File Consistency

**Files:**
- Inspect: all files created in `examples/hBN_monolayer_qe_wannier/`

- [ ] **Step 1: Check k-point list consistency**

Run a small read-only script to verify `hBN_nscf.in` and `hBN.win` contain the same 324 k points in the same order.

- [ ] **Step 2: Check no unresolved markers remain**

Run `rg -n "UNRESOLVED_MARKER|FILL_ME_IN" examples/hBN_monolayer_qe_wannier`.

- [ ] **Step 3: Show the resulting file tree**

Run `Get-ChildItem -Recurse examples/hBN_monolayer_qe_wannier | Select-Object FullName,Length`.
