#!/bin/bash
#=============================================================================
# Pre-GH5 high-field material-model spot check (plus-N only, CLASSICAL drive).
#
# IMPORTANT: these are fixed classical (I, phi_CEP) trajectories, NOT a
# Gaussian-Q / squeezed_vacuum ensemble.  Job 27752 used w=1 + SV and was
# correctly rejected by the moment gate; keep that failure as a record.
#
# Array 0..3:
#   0: node A × M88-full
#   1: node A × full112
#   2: node B × M88-full
#   3: node B × full112
#
# Node A (from GH5 r=2.5, θ=0° Imax):
#   I = 1.63245553e12 W/cm^2,  phi_CEP = 85.3073968 deg
# Node B (from GH5 r=2.5, θ=180° Imax):
#   I = 1.63245553e12 W/cm^2,  phi_CEP = 4.69260335 deg
#
# Requires save_complex_hhg -> HHG_complex.dat
# Does NOT submit GH5.
#=============================================================================
#SBATCH --job-name=a0_hf_class
#SBATCH --partition=part_1
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=a0_hf_class_%A_%a.out
#SBATCH --error=a0_hf_class_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

source_tree_digest() {
  (
    cd "${REPO}"
    sha256sum Makefile
    find src -maxdepth 1 -type f -name '*.f90' -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
# New OUTROOT: do not reuse the failed 27752 tree.
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_highfield_classical_m88_vs_full112_k20}"
NK=20

TB_M88="${TB_M88:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb.dat}"
TB_F112="${TB_F112:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 3 ] || die "bad index ${IDX}"

INODE=$(( IDX / 2 ))
IMODEL=$(( IDX % 2 ))

NODE_TAGS=(A B)
# Classical fixed-field parameters (NOT quantum-light weights)
I_LIST=(1.63245553e12 1.63245553e12)
PHI_CEP_DEG_LIST=(85.3073968 4.69260335)
# Provenance labels only (which GH5 tail the classical drive was taken from)
SRC_R=(2.5 2.5)
SRC_TH=(0 180)

if [ "${IMODEL}" -eq 0 ]; then
  MODEL="M88full"
  TB="${TB_M88}"
  NV=58
  NB_START=1
  NB_END=88
else
  MODEL="full112"
  TB="${TB_F112}"
  NV=84
  NB_START=1
  NB_END=112
fi

NTAG="${NODE_TAGS[${INODE}]}"
I_DRIVE="${I_LIST[${INODE}]}"
PHI_CEP_DEG="${PHI_CEP_DEG_LIST[${INODE}]}"
RVAL="${SRC_R[${INODE}]}"
THVAL="${SRC_TH[${INODE}]}"
CASE=$(printf "hfclass_node%s_%s_k%02d_plusN" "${NTAG}" "${MODEL}" "${NK}")
OUTDIR="${OUTROOT}/${CASE}"

for c in sha256sum git find xargs awk grep mktemp; do require_cmd "${c}"; done
[ -f "${TB}" ] || die "missing TB ${TB}"
[ -f "${REPO}/Makefile" ] || die "missing Makefile"

if [ -d "${OUTDIR}" ] && [ -n "$(find "${OUTDIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  die "non-empty OUTDIR ${OUTDIR}"
fi
mkdir -p "${OUTDIR}"

LOCK=""
LOCK_HELD=0
RUN_COMPLETED=0
on_exit() {
  rc=$?
  if [ "${LOCK_HELD}" -eq 1 ] && [ -n "${LOCK}" ]; then
    rmdir "${LOCK}" 2>/dev/null || true
  fi
  if [ "${RUN_COMPLETED}" -ne 1 ]; then
    printf 'status=FAILED\nexit_code=%s\n' "${rc}" > "${OUTDIR}/run_status.txt" 2>/dev/null || true
  fi
}
trap on_exit EXIT

echo "============================================="
echo " Classical high-field spot ${CASE}"
echo " I=${I_DRIVE} W/cm2  phi_CEP=${PHI_CEP_DEG} deg"
echo " TB=${TB}  nv=${NV} bands=${NB_START}-${NB_END}"
echo " mode=classical fixed field (bsv_enabled=.false.)"
echo " NOTE: plus-N only; cannot form Jo without full112 -N TB"
echo " NOTE: job 27752 failure retained as incorrect SV+w=1 config"
echo " host=$(hostname) date=$(date)"
echo "============================================="

export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

BUILD_FC="ifort"
BUILD_FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays"
BUILD_LDFLAGS=""
require_cmd "${BUILD_FC}"
FC_PATH="$(command -v "${BUILD_FC}")"
COMPILER_VERSION="$(${BUILD_FC} --version 2>&1 | head -n 1)"
SOURCE_SHA256="$(source_tree_digest)"
BUILD_KEY="$({
  printf 'source_sha256=%s\n' "${SOURCE_SHA256}"
  printf 'FC=%s\n' "${BUILD_FC}"
  printf 'FC_path=%s\n' "${FC_PATH}"
  printf 'compiler_version=%s\n' "${COMPILER_VERSION}"
  printf 'FFLAGS=%s\n' "${BUILD_FFLAGS}"
  printf 'LDFLAGS=%s\n' "${BUILD_LDFLAGS}"
} | sha256sum | awk '{print $1}')"

BUILD_ROOT="${REPO}/.a0_build_cache_v2"
LOCK="${BUILD_ROOT}/.compile.lockdir"
CACHE_DIR="${BUILD_ROOT}/${BUILD_KEY}"
CACHED_BIN="${CACHE_DIR}/hhg_sbe"
mkdir -p "${BUILD_ROOT}"

if [ ! -x "${CACHED_BIN}" ]; then
  WAITED=0
  while true; do
    if mkdir "${LOCK}" 2>/dev/null; then LOCK_HELD=1; break; fi
    [ -x "${CACHED_BIN}" ] && break
    sleep 10
    WAITED=$((WAITED+10))
    [ "${WAITED}" -le 2400 ] || die "compile lock timeout"
  done
  if [ "${LOCK_HELD}" -eq 1 ]; then
    (
      cd "${REPO}"
      make clean
      make FC="${BUILD_FC}" FFLAGS="${BUILD_FFLAGS}" LDFLAGS="${BUILD_LDFLAGS}"
    ) 2>&1 | tee "${OUTDIR}/build.log"
    [ -x "${REPO}/hhg_sbe" ] || die "build failed"
    mkdir -p "${CACHE_DIR}"
    cp -f "${REPO}/hhg_sbe" "${CACHED_BIN}"
    chmod 0555 "${CACHED_BIN}"
    {
      echo "build_key=${BUILD_KEY}"
      echo "source_sha256=${SOURCE_SHA256}"
      echo "binary_sha256=$(sha256_file "${CACHED_BIN}")"
      echo "compiler_version=${COMPILER_VERSION}"
    } > "${CACHE_DIR}/build_metadata.txt"
    rmdir "${LOCK}" 2>/dev/null || true
    LOCK_HELD=0
  fi
fi
[ -x "${CACHED_BIN}" ] || die "missing ${CACHED_BIN}"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"

cat > "${OUTDIR}/input.nml" <<EOF
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
  nv_orig = ${NV}
  nb_start = ${NB_START}
  nb_end = ${NB_END}
/

&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = ${I_DRIVE}
  theta_deg = 0.0
  phi_cep_deg = ${PHI_CEP_DEG}
  ncyc = 4.0
  env_type = 2
  ellipticity = 0.0
  delta_phase_deg = 90.0
/

&timestep
  dt = 0.35
  n_dt_deph = 5
/

&dephasing
  T2_cycles = 0.5
/

&bsv
  bsv_enabled = .false.
/

&method
  gauge_method = 'lg_cov'
/

&output
  save_geometry = .false.
  save_occupation = .false.
  save_complex_hhg = .true.
/
EOF

TB_SHA256="$(sha256_file "${TB}")"
{
  echo "case=${CASE}"
  echo "campaign=highfield_classical_m88_vs_full112_k20_plusN"
  echo "drive_mode=classical_fixed_field"
  echo "bsv_enabled=false"
  echo "node_tag=${NTAG}"
  echo "model=${MODEL}"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
  echo "source_gh5_r=${RVAL}"
  echo "source_gh5_theta_deg=${THVAL}"
  echo "nk=${NK}"
  echo "nkx=${NK}"
  echo "nky=${NK}"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "nv_orig=${NV}"
  echo "nb_start=${NB_START}"
  echo "nb_end=${NB_END}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "tb_file=${TB}"
  echo "tb_sha256=${TB_SHA256}"
  echo "domain=plusN"
  echo "note=plusN_only_cannot_form_Jo_without_full112_mN"
  echo "note2=not_a_GaussianQ_distribution_fixed_classical_drive"
  echo "failed_predecessor_job=27752"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
} > "${OUTDIR}/run_metadata.txt"

mkdir -p "${OUTDIR}/source_snapshot"
cp -f "${REPO}/Makefile" "${OUTDIR}/source_snapshot/"
cp -f "${REPO}"/src/*.f90 "${OUTDIR}/source_snapshot/"
(
  cd "${OUTDIR}/source_snapshot"
  sha256sum Makefile *.f90 | tee source_files_sha256.txt
)

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

for f in HHG.dat HHG_complex.dat Et.dat run.log; do
  [ -s "${f}" ] || die "missing ${f}"
done
# Must NOT have gone through the quantum-light ensemble path.
if [ -f nodes_moment_check.txt ]; then
  die "unexpected nodes_moment_check.txt in classical spot run"
fi
if grep -qi 'quantum-light ensemble' run.log; then
  die "run.log suggests quantum-light ensemble path was taken"
fi
grep -q 'Wrote HHG_complex.dat' run.log \
  || die "HHG_complex.dat write confirmation missing in run.log"

sha256sum HHG.dat HHG_complex.dat Et.dat run.log run_metadata.txt input.nml \
  > output_sha256.txt
echo "status=PASS" > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "SUCCESS ${CASE}"
