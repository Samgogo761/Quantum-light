#!/bin/bash
#=============================================================================
# Task 3 pilot: d-only 40-WF lg_cov HHG (match production laser / k40 / T2=0.5)
#
# Submit (after user approval):
#   cd /public/home/wangjs/project/New_SBEs/Quantum-light
#   sbatch deploy/run_donly_lg_cov_pilot.sh
#
# Est. wall time: ~25-50 min on part_1/36 cores (vs nb104 ~115 min reference)
#=============================================================================

#SBATCH --job-name=donly_hhg
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=donly_hhg_%j.out
#SBATCH --error=donly_hhg_%j.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf/CrI3_donly_tb.dat}"
WORKDIR="${WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
OUTDIR="${OUTROOT:-${WORKDIR}/output_donly_lg_cov_T2_0p5cycle}/lgcov_k40_b40_nv22"
NK=40
NB_START=1
NB_END=40
NV_ORIG=22
T2_CYCLES=0.5
DT_AU=0.35

cd "${WORKDIR}" || exit 1
echo "=== d-only HHG pilot $(date) host=$(hostname) ==="
echo "TB=${TB_FILE} OUT=${OUTDIR}"

test -f "${TB_FILE}" || { echo "ERROR: missing TB"; exit 1; }
test -x "${WORKDIR}/hhg_sbe" || { echo "ERROR: missing hhg_sbe"; exit 1; }

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

mkdir -p "${OUTDIR}"
cat > "${OUTDIR}/input.nml" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB_FILE}"
  wannier_hr_file = ""
  wannier_r_file  = ""
/
&kgrid
  nkx = ${NK}
  nky = ${NK}
/
&bands
  nv_orig = ${NV_ORIG}
  nb_start = ${NB_START}
  nb_end = ${NB_END}
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
  dt = ${DT_AU}
  n_dt_deph = 5
/
&dephasing
  T2_fs = 1.0e30
  T2_cycles = ${T2_CYCLES}
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

cd "${OUTDIR}"
export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

grep -E 'nkx|nv_orig|nb_start|nb_end|wannier_tb_file|gauge_method|T2_cycles' input.nml
echo "Starting: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished: $(date)"
ls -lh HHG.dat Jt.dat Jt_decomposed.dat 2>/dev/null || true
