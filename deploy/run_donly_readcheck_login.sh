#!/bin/bash
# Quick Task2: read CrI3_donly_tb.dat + lg_cov precompute (login node, ~5-15 min)
# Does NOT sbatch. Safe: writes only under output_donly_readcheck/
set -eo pipefail

WORKDIR=/public/home/wangjs/project/New_SBEs/Quantum-light
TB=/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf/CrI3_donly_tb.dat
OUT=${WORKDIR}/output_donly_readcheck/k10_b40_nv22
NK=10

cd "${WORKDIR}"
test -x ./hhg_sbe
test -f "${TB}"

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u; source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh; set -u

mkdir -p "${OUT}"
cat > "${OUT}/input.nml" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB}"
  wannier_hr_file = ""
  wannier_r_file  = ""
/
&kgrid
  nkx = ${NK}
  nky = ${NK}
/
&bands
  nv_orig = 22
  nb_start = 1
  nb_end = 40
/
&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = 2.0e11
  theta_deg = 0.0
  phi_cep_deg = 90.0
  ncyc = 4.0
  env_type = 2
  ellipticity = 0.0
  delta_phase_deg = 90.0
/
&laser2
  wvl_nm_2 = 0.0
  intensity_Wcm2_2 = 0.0
/
&external_field
  use_external_A = .false.
/
&timestep
  dt = 0.35
  n_dt_deph = 5
/
&dephasing
  T2_fs = 1.0e30
  T2_cycles = 0.5
/
&bsv
  bsv_enabled = .false.
/
&method
  gauge_method = 'lg_cov'
/
&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .false.
/
EOF

cd "${OUT}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_STACKSIZE=256M
echo "START $(date) host=$(hostname)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "DONE $(date)"
grep -E 'ERROR|error|Band window|nwann|J\(0\)|Total wall' run.log | tail -20 || true
ls -lh HHG.dat Jt.dat bands.dat 2>/dev/null || true
