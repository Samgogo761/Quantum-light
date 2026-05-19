#!/bin/bash
#=============================================================================
# LG-covariant valence-depth scan for CrI3.
#
# Purpose:
#   Keep the full conduction side of the current 112-band model
#   (nb_end=112, i.e. all 28 conduction bands) and gradually add deeper
#   valence bands.  This tests whether a cheaper asymmetric window can
#   reproduce lg_cov/full112.
#
# Recommended no-dephasing benchmark:
#   sbatch --array=0-3 deploy/run_lg_cov_valence_depth_scan.sh
#
# Include the full112 reference only if it is not already available:
#   sbatch --array=0-4 deploy/run_lg_cov_valence_depth_scan.sh
#
# Selected-T2 rerun example:
#   T2_FS=5.337 OUTROOT=/public/home/wangjs/project/New_SBEs/Quantum-light/output_lg_cov_valdepth_T2_5p337fs \
#     sbatch --array=0-4 deploy/run_lg_cov_valence_depth_scan.sh
#
# Cases:
#   0 lgcov_k40_nb70_112   nk=40  nb=70-112   15 valence + 28 conduction = 43 bands
#   1 lgcov_k40_nb60_112   nk=40  nb=60-112   25 valence + 28 conduction = 53 bands
#   2 lgcov_k40_nb50_112   nk=40  nb=50-112   35 valence + 28 conduction = 63 bands
#   3 lgcov_k40_nb30_112   nk=40  nb=30-112   55 valence + 28 conduction = 83 bands
#   4 lgcov_k40_nb1_112    nk=40  nb=1-112    84 valence + 28 conduction = 112 bands
#=============================================================================

#SBATCH --job-name=lgcov_vdepth
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=lgcov_vdepth_%A_%a.out
#SBATCH --error=lgcov_vdepth_%A_%a.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
WORKDIR="${WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
NK="${NK:-40}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_lg_cov_valence_depth_nodeph}"
T2_FS="${T2_FS:-1.0e30}"
T2_CYCLES="${T2_CYCLES:--1.0}"
DT_AU="${DT_AU:-0.35}"

CASES=(
  "lgcov_k${NK}_nb70_112 ${NK} 70 112"
  "lgcov_k${NK}_nb60_112 ${NK} 60 112"
  "lgcov_k${NK}_nb50_112 ${NK} 50 112"
  "lgcov_k${NK}_nb30_112 ${NK} 30 112"
  "lgcov_k${NK}_nb1_112 ${NK} 1 112"
)

CASE_INDEX="${CASE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "${CASE_INDEX}" -lt 0 ] || [ "${CASE_INDEX}" -ge "${#CASES[@]}" ]; then
  echo "ERROR: CASE_INDEX=${CASE_INDEX} outside 0..$((${#CASES[@]}-1))"
  exit 1
fi

read -r CASE_NAME NK_CASE NB_START NB_END <<< "${CASES[${CASE_INDEX}]}"
N_TRUNC=$((NB_END - NB_START + 1))
N_VAL=$((84 - NB_START + 1))
if [ "${N_VAL}" -lt 0 ]; then N_VAL=0; fi
if [ "${N_VAL}" -gt "${N_TRUNC}" ]; then N_VAL="${N_TRUNC}"; fi
N_COND=$((N_TRUNC - N_VAL))
OUTDIR="${OUTROOT}/${CASE_NAME}"

cd "${WORKDIR}" || { echo "ERROR: cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  LG-covariant valence-depth scan"
echo "============================================="
echo "  Date:       $(date)"
echo "  Host:       $(hostname)"
echo "  Workdir:    ${WORKDIR}"
echo "  TB file:    ${TB_FILE}"
echo "  Output:     ${OUTDIR}"
echo "  Case index: ${CASE_INDEX}"
echo "  Case name:  ${CASE_NAME}"
echo "  k-grid:     ${NK_CASE} x ${NK_CASE}"
echo "  Bands:      ${NB_START} to ${NB_END} (${N_TRUNC})"
echo "  Occupied:   ${N_VAL} valence bands kept"
echo "  Empty:      ${N_COND} conduction bands kept"
echo "  T2_fs:      ${T2_FS}"
echo "  T2_cycles:  ${T2_CYCLES}"
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

  local lockdir="${WORKDIR}/.compile_lg_cov_valdepth.lockdir"
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
  nkx = ${NK_CASE}
  nky = ${NK_CASE}
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
  T2_cycles = ${T2_CYCLES}
/

&bsv
  bsv_enabled = .false.
  bsv_n_samples = 100
  bsv_mean_intensity = 2.0e11
  bsv_seed = 42
/

&method
  gauge_method = 'lg_cov'
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
