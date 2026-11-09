#!/bin/bash
#=============================================================================
# A0 hardened exploration: M88-full @ k20, squeezed-vacuum GH3, shared ±N nodes
# Partition: part_1
#
# Array index encoding:
#   idx = ((ir * NTH + ith) * 2 + idom)
#   ir   = 0..NR-1    (r = 0.5, 1.5, 2.5)
#   ith  = 0..NTH-1  (theta = 0, 90, 180, 270 degrees)
#   idom = 0 (+N), 1 (-N)
# Total: 3*4*2 = 24 tasks
#
# This is the hardened successor of sbatch_explore_k20.sh.  It does not alter
# already-running job 27674.  Use a fresh OUTROOT when submitting this version.
#=============================================================================
#SBATCH --job-name=a0_m88_k20_h
#SBATCH --partition=part_1
#SBATCH --array=0-23%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=a0_m88_k20_h_%A_%a.out
#SBATCH --error=a0_m88_k20_h_%A_%a.err

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

# Hash every build input.  Reuse is allowed only for this exact content digest;
# a stale timestamp marker such as .compile_a0.done is intentionally ignored.
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
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb.dat}"
TB_MINUS="${TB_MINUS:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb_mN_Pconj.dat}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_m88full_k20_hardened}"
NODES_DIR="${NODES_DIR:-${WORKDIR}/nodes_gh3}"
NK=20

R_LIST=(0.5 1.5 2.5)
TH_LIST=(0 90 180 270)
NR=${#R_LIST[@]}
NTH=${#TH_LIST[@]}

IDX="${SLURM_ARRAY_TASK_ID:-0}"
IDOM=$(( IDX % 2 ))
TMP=$(( IDX / 2 ))
ITH=$(( TMP % NTH ))
IR=$(( TMP / NTH ))
[ "${IR}" -lt "${NR}" ] || die "bad array index ${IDX}"

RVAL="${R_LIST[${IR}]}"
THVAL="${TH_LIST[${ITH}]}"
if [ "${IDOM}" -eq 0 ]; then
  DOM="plusN"
  TB="${TB_PLUS}"
else
  DOM="minusN"
  TB="${TB_MINUS}"
fi

RTAG=$(echo "${RVAL}" | tr '.' 'p')
NODES_FILE="${NODES_DIR}/nodes_sv_r${RTAG}_th${THVAL}_gh3.dat"
CASE=$(printf "sv_r%s_th%03d_%s" "${RTAG}" "${THVAL}" "${DOM}")
OUTDIR="${OUTROOT}/${CASE}"

require_cmd sha256sum
require_cmd git
require_cmd find
require_cmd xargs
require_cmd awk
require_cmd grep
require_cmd mktemp

# Never validate a new run against stale files from an older run.
if [ -d "${OUTDIR}" ] && \
   [ -n "$(find "${OUTDIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  die "output directory is not empty; choose a fresh OUTROOT: ${OUTDIR}"
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
    printf 'status=FAILED\nexit_code=%s\ntime=%s\n' \
      "${rc}" "$(date --iso-8601=seconds 2>/dev/null || date)" \
      > "${OUTDIR}/run_status.txt" 2>/dev/null || true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "============================================="
echo " A0 hardened explore M88-full k${NK} case=${CASE}"
echo " nodes=${NODES_FILE}"
echo " TB=${TB}"
echo " host=$(hostname) date=$(date)"
echo "============================================="

[ -f "${NODES_FILE}" ] || die "missing shared nodes file ${NODES_FILE}"
[ -f "${TB}" ] || die "TB not found: ${TB}"
[ -f "${REPO}/Makefile" ] || die "Makefile not found under REPO=${REPO}"

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

# A content-addressed cache plus one global lock prevents simultaneous array
# tasks from running make clean in the same repository.
BUILD_ROOT="${REPO}/.a0_build_cache"
LOCK="${BUILD_ROOT}/.compile.lockdir"
mkdir -p "${BUILD_ROOT}"
SOURCE_SHA256="$(source_tree_digest)"
CACHE_DIR="${BUILD_ROOT}/${SOURCE_SHA256}"
CACHED_BIN="${CACHE_DIR}/hhg_sbe"

if [ ! -x "${CACHED_BIN}" ]; then
  WAITED=0
  while true; do
    if mkdir "${LOCK}" 2>/dev/null; then
      LOCK_HELD=1
      break
    fi
    if [ -x "${CACHED_BIN}" ]; then
      break
    fi
    sleep 10
    WAITED=$((WAITED+10))
    [ "${WAITED}" -le 2400 ] || die "compile lock timeout"
  done

  if [ "${LOCK_HELD}" -eq 1 ]; then
    CURRENT_SOURCE_SHA256="$(source_tree_digest)"
    [ "${CURRENT_SOURCE_SHA256}" = "${SOURCE_SHA256}" ] \
      || die "source tree changed while waiting for compile lock; resubmit task"

    echo "Compiling hhg_sbe for source digest ${SOURCE_SHA256}..."
    TMP_CACHE="$(mktemp -d "${BUILD_ROOT}/build.${SOURCE_SHA256}.XXXXXX")"
    (
      cd "${REPO}"
      make clean
      make FC="${BUILD_FC}" FFLAGS="${BUILD_FFLAGS}" LDFLAGS="${BUILD_LDFLAGS}"
    ) 2>&1 | tee "${TMP_CACHE}/build.log"

    POST_BUILD_SOURCE_SHA256="$(source_tree_digest)"
    [ "${POST_BUILD_SOURCE_SHA256}" = "${SOURCE_SHA256}" ] \
      || die "source tree changed during compilation; refusing mixed build"
    [ -x "${REPO}/hhg_sbe" ] \
      || die "compiler returned without executable ${REPO}/hhg_sbe"

    cp -f "${REPO}/hhg_sbe" "${TMP_CACHE}/hhg_sbe"
    chmod 0555 "${TMP_CACHE}/hhg_sbe"
    COMPILER_VERSION="$(${BUILD_FC} --version 2>&1 | head -n 1)"
    {
      echo "source_sha256=${SOURCE_SHA256}"
      echo "compiler=${COMPILER_VERSION}"
      echo "FC=${BUILD_FC}"
      echo "FFLAGS=${BUILD_FFLAGS}"
      echo "LDFLAGS=${BUILD_LDFLAGS}"
      echo "binary_sha256=$(sha256_file "${TMP_CACHE}/hhg_sbe")"
      echo "built_at=$(date --iso-8601=seconds 2>/dev/null || date)"
    } > "${TMP_CACHE}/build_metadata.txt"
    mv "${TMP_CACHE}" "${CACHE_DIR}"
    rmdir "${LOCK}" 2>/dev/null || true
    LOCK_HELD=0
  fi
fi

[ -x "${CACHED_BIN}" ] || die "missing source-keyed executable ${CACHED_BIN}"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"
COMPILER_VERSION="$(${BUILD_FC} --version 2>&1 | head -n 1)"

TMPL="${REPO}/deploy/a0_layerA_m88full/input_template.nml"
[ -f "${TMPL}" ] || die "input template not found: ${TMPL}"
sed -e "s|TB_PLACEHOLDER|${TB}|g" \
    -e "s|NK_PLACEHOLDER|${NK}|g" \
    -e "s|R_PLACEHOLDER|${RVAL}|g" \
    -e "s|TH_PLACEHOLDER|${THVAL}|g" \
    -e "s|NODES_PLACEHOLDER|${NODES_FILE}|g" \
    "${TMPL}" > "${OUTDIR}/input.nml"
cp -f "${NODES_FILE}" "${OUTDIR}/nodes_manifest.input.dat"
cp -f "${TMPL}" "${OUTDIR}/input_template.snapshot.nml"
cp -f "${CACHE_DIR}/build_metadata.txt" "${OUTDIR}/build_metadata.snapshot.txt"

NODES_SHA256="$(sha256_file "${NODES_FILE}")"
COPIED_NODES_SHA256="$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")"
[ "${NODES_SHA256}" = "${COPIED_NODES_SHA256}" ] \
  || die "copied nodes manifest hash differs from original"
TB_SHA256="$(sha256_file "${TB}")"
INPUT_SHA256="$(sha256_file "${OUTDIR}/input.nml")"
TEMPLATE_SHA256="$(sha256_file "${TMPL}")"
GIT_HEAD="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)"
git -C "${REPO}" status --porcelain --untracked-files=normal \
  > "${OUTDIR}/git_status_porcelain.txt" 2>&1 || true
if [ -s "${OUTDIR}/git_status_porcelain.txt" ]; then
  GIT_DIRTY="yes"
else
  GIT_DIRTY="no"
fi

{
  echo "case=${CASE}"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-NA}"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "uname=$(uname -a)"
  echo "repo=${REPO}"
  echo "git_head=${GIT_HEAD}"
  echo "git_dirty=${GIT_DIRTY}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "binary=${CACHED_BIN}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "compiler=${COMPILER_VERSION}"
  echo "FC=${BUILD_FC}"
  echo "FFLAGS=${BUILD_FFLAGS}"
  echo "LDFLAGS=${BUILD_LDFLAGS}"
  echo "nodes_file=${NODES_FILE}"
  echo "nodes_sha256=${NODES_SHA256}"
  echo "tb_file=${TB}"
  echo "tb_sha256=${TB_SHA256}"
  echo "input_sha256=${INPUT_SHA256}"
  echo "template_sha256=${TEMPLATE_SHA256}"
  echo "nk=${NK}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

# Process exit=0 is necessary but not sufficient.  Reject incomplete,
# non-finite, or moment-invalid output before emitting SUCCESS.
CRITICAL_OUTPUTS=(
  HHG_bsv.dat
  HHG_ics_cs.dat
  HHG_nodes_modes.dat
  nodes_moment_check.txt
  run.log
)
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${output_file}" ] \
    || die "missing or empty critical output: ${OUTDIR}/${output_file}"
done

grep -Eq '^pass[[:space:]]*=[[:space:]]*T[[:space:]]*$' nodes_moment_check.txt \
  || die "node moment validation did not report pass = T"

NONFINITE_RE='(^|[^[:alpha:]])[+-]?(nan|inf(inity)?)([^[:alpha:]]|$)'
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  if LC_ALL=C grep -Eiq "${NONFINITE_RE}" "${output_file}"; then
    die "NaN/Inf detected in critical output: ${OUTDIR}/${output_file}"
  fi
done

NODE_COUNT=$(awk '!/^#/ && NF {n++} END {print n+0}' nodes_manifest.input.dat)
MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
EXPECTED_MODE_ROWS=$((NODE_COUNT * 5))
[ "${MODE_ROWS}" -eq "${EXPECTED_MODE_ROWS}" ] \
  || die "HHG_nodes_modes rows=${MODE_ROWS}, expected ${EXPECTED_MODE_ROWS}"

sha256sum "${CRITICAL_OUTPUTS[@]}" > output_sha256.txt
{
  echo "status=PASS"
  echo "exit_code=0"
  echo "completed_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "nodes_sha256=${NODES_SHA256}"
  echo "tb_sha256=${TB_SHA256}"
  echo "input_sha256=${INPUT_SHA256}"
  echo "node_count=${NODE_COUNT}"
  echo "mode_row_count=${MODE_ROWS}"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} $(date)"
