#!/bin/bash
#=============================================================================
# Peierls velocity-gauge full112 gauge-consistency check.
#
# Purpose:
#   Compare Peierls VG full112 against lg_cov full112 as k-grid is increased.
#   This is a gauge-invariance diagnostic, not a production route.
#
# Memory estimate from CrI3_tb.dat: num_wann=112, nrpts=579.
# HR_proj alone is about:
#   nk=20: 46.5 GiB
#   nk=30: 104.5 GiB
# plus H/P/rho/RK workspace and OpenMP/LAPACK overhead.
#
# Recommended first submission on a large-memory compute node:
#   sbatch --array=0 deploy/run_peierls_full112_gauge_check.sh
#
# Only if memory/time is comfortable:
#   sbatch --array=1 deploy/run_peierls_full112_gauge_check.sh
#
# Cases:
#   0 vg_k20_b112  nk=20  nb=1-112
#   1 vg_k30_b112  nk=30  nb=1-112
#=============================================================================

#SBATCH --job-name=vg_full112
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=vg_full112_%A_%a.out
#SBATCH --error=vg_full112_%A_%a.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
WORKDIR="${WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_peierls_full112_gauge_check_nodeph}"
T2_FS="${T2_FS:-1.0e30}"
DT_AU="${DT_AU:-0.35}"

CASES=(
  "vg_k20_b112 20 1 112"
  "vg_k30_b112 30 1 112"
)

CASE_INDEX="${CASE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "${CASE_INDEX}" -lt 0 ] || [ "${CASE_INDEX}" -ge "${#CASES[@]}" ]; then
  echo "ERROR: CASE_INDEX=${CASE_INDEX} outside 0..$((${#CASES[@]}-1))"
  exit 1
fi

read -r CASE_NAME NK NB_START NB_END <<< "${CASES[${CASE_INDEX}]}"
N_TRUNC=$((NB_END - NB_START + 1))
OUTDIR="${OUTROOT}/${CASE_NAME}"

cd "${WORKDIR}" || { echo "ERROR: cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  Peierls full112 gauge check"
echo "============================================="
echo "  Date:       $(date)"
echo "  Host:       $(hostname)"
echo "  Workdir:    ${WORKDIR}"
echo "  TB file:    ${TB_FILE}"
echo "  Output:     ${OUTDIR}"
echo "  Case index: ${CASE_INDEX}"
echo "  Case name:  ${CASE_NAME}"
echo "  k-grid:     ${NK} x ${NK}"
echo "  Bands:      ${NB_START} to ${NB_END} (${N_TRUNC})"
echo "  T2_fs:      ${T2_FS}"
echo "  dt:         ${DT_AU} a.u."
echo "  Threads:    ${NTHREADS}"

if [ ! -f "${TB_FILE}" ]; then
  echo "ERROR: TB file not found: ${TB_FILE}"
  exit 1
fi

if command -v free >/dev/null 2>&1; then
  free -h
fi

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

compile_if_needed() {
  if [ -x "${WORKDIR}/hhg_sbe" ] && [ "${FORCE_COMPILE:-0}" != "1" ]; then
    echo "Using existing executable: ${WORKDIR}/hhg_sbe"
    return
  fi

  local lockdir="${WORKDIR}/.compile_peierls_full112.lockdir"
  while ! mkdir "${lockdir}" 2>/dev/null; do
    echo "Waiting for compile lock..."
    sleep 10
  done
  trap 'rmdir "${lockdir}" 2>/dev/null || true' RETURN

  if [ -x "${WORKDIR}/hhg_sbe" ] && [ "${FORCE_COMPILE:-0}" != "1" ]; then
    echo "Executable appeared while waiting for lock; skipping compile."
    return
  fi

  export FC=ifort
  export FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays"
  export LDFLAGS=""
  echo "Compiling with Intel ifort + MKL..."
  make clean
  make FC="${FC}" FFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}"
}

compile_if_needed

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
  nv_orig = 84
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
  external_A_file = ""
/

&timestep
  dt = ${DT_AU}
  n_dt_deph = 5
/

&dephasing
  T2_fs = ${T2_FS}
/

&bsv
  bsv_enabled = .false.
  bsv_n_samples = 100
  bsv_mean_intensity = 2.0e11
  bsv_seed = 42
/

&method
  gauge_method = 'vg'
/

&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .true.
/
EOF

sed -i 's/\r$//' "${OUTDIR}/input.nml"

cd "${OUTDIR}" || exit 1
export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export OMP_STACKSIZE=256M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "--- Effective input ---"
grep -E "nkx|nky|nb_start|nb_end|T2_fs|T2_cycles|dt|gauge_method|wannier_tb_file|bsv_enabled" input.nml
echo "Starting ${CASE_NAME}: $(date)"

{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log

echo "Finished ${CASE_NAME}: $(date)"
