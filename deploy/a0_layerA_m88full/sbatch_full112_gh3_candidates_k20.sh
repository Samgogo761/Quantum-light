#!/bin/bash
#=============================================================================
# full112 @ k20, squeezed_vacuum GH3, THREE candidates × ±N  => 6 array tasks.
#
# Purpose: re-rank states with full112 (NOT M88). Selection only — not GH5.
# Candidates:
#   0: r=2.5, theta=0
#   1: r=2.5, theta=180
#   2: r=0.5, theta=180
#
# 3 states × 2 domains × 9 nodes = 54 trajectories.
# M88 formal GH5 / k50 remain PAUSED.
# Do NOT auto-submit from this file; sbatch explicitly when the queue is free.
#=============================================================================
#SBATCH --job-name=a0_f112_gh3
#SBATCH --partition=part_1
#SBATCH --array=0-5%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_f112_gh3_%A_%a.out
#SBATCH --error=a0_f112_gh3_%A_%a.err

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
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
TB_MINUS="${TB_MINUS:-/public/home/wangjs/project/CrI3_TB_mN/wannier/CrI3_tb_mN_Pconj.dat}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_full112_k20_gh3_candidates}"
NODES_DIR="${NODES_DIR:-${REPO}/deploy/a0_layerA_m88full/nodes_gh3_full112_candidates}"
NK=20
I_BAR=1.0e11
HARMONICS_CSV="2,5,7,9,10"
HARMONIC_COUNT=5

NODE_VALIDATOR="${REPO}/tools/analysis/validate_qlight_nodes_production.py"
RUN_VALIDATOR="${REPO}/tools/analysis/validate_a0_run_strict.py"
NODE_VALIDATOR_CORE="${REPO}/tools/analysis/validate_qlight_nodes_strict.py"

# Explicit candidate table: index 0..2
CAND_R=(2.5 2.5 0.5)
CAND_TH=(0 180 180)
CAND_RTAG=(2p5 2p5 0p5)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 5 ] || die "bad array index ${IDX}"
ICAND=$(( IDX / 2 ))
IDOM=$(( IDX % 2 ))
RVAL="${CAND_R[${ICAND}]}"
THVAL="${CAND_TH[${ICAND}]}"
RTAG="${CAND_RTAG[${ICAND}]}"

if [ "${IDOM}" -eq 0 ]; then
  DOM="plusN"
  TB="${TB_PLUS}"
else
  DOM="minusN"
  TB="${TB_MINUS}"
fi

NODES_FILE="${NODES_DIR}/nodes_sv_r${RTAG}_th${THVAL}_gh3.dat"
CASE=$(printf "sv_r%s_th%03d_%s" "${RTAG}" "${THVAL}" "${DOM}")
OUTDIR="${OUTROOT}/${CASE}"

for command_name in sha256sum git find xargs awk grep mktemp; do
  require_cmd "${command_name}"
done
# shellcheck disable=SC1091
source "${REPO}/deploy/a0_layerA_m88full/resolve_python3.sh"
PYTHON3="${A0_PYTHON3}"
[ -x "${PYTHON3}" ] || die "A0_PYTHON3 not executable: ${PYTHON3}"

[ -f "${NODES_FILE}" ] || die "missing shared nodes file ${NODES_FILE}"
[ -f "${TB}" ] || die "TB not found: ${TB}"
[ -f "${REPO}/Makefile" ] || die "Makefile not found under REPO=${REPO}"
[ -f "${NODE_VALIDATOR}" ] || die "missing node validator: ${NODE_VALIDATOR}"
[ -f "${RUN_VALIDATOR}" ] || die "missing run validator: ${RUN_VALIDATOR}"
[ -f "${NODE_VALIDATOR_CORE}" ] || die "missing validator dependency: ${NODE_VALIDATOR_CORE}"

# Idempotent completed check + refuse non-empty incomplete dirs.
if [ -f "${OUTDIR}/SUCCESS" ] && [ -s "${OUTDIR}/HHG_nodes_modes.dat" ] \
   && grep -q 'status=PASS' "${OUTDIR}/run_status.txt" 2>/dev/null; then
  echo "ALREADY_COMPLETE ${CASE}; skipping"
  exit 0
fi
if [ -d "${OUTDIR}" ] && \
   [ -n "$(find "${OUTDIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  die "output directory is not empty without SUCCESS: ${OUTDIR}"
fi
mkdir -p "${OUTDIR}/validator_snapshot"

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
echo " A0 full112 GH3@k20 candidate selection case=${CASE}"
echo " nodes=${NODES_FILE}"
echo " TB=${TB}  (full112 ±N)"
echo " NOTE: NOT formal M88 GH5 / NOT k50; selection only"
echo " host=$(hostname) date=$(date)"
echo "============================================="

# Archive the exact Python hard gates before invoking them.  Both validators
# import validate_qlight_nodes_strict.py from their own directory.
cp -f "${NODE_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
cp -f "${RUN_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
cp -f "${NODE_VALIDATOR_CORE}" "${OUTDIR}/validator_snapshot/"
NODE_VALIDATOR_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_qlight_nodes_production.py"
RUN_VALIDATOR_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_a0_run_strict.py"
VALIDATOR_CORE_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_qlight_nodes_strict.py"
VALIDATOR_BUNDLE_SHA256="$({
  sha256sum "${NODE_VALIDATOR_SNAPSHOT}"
  sha256sum "${RUN_VALIDATOR_SNAPSHOT}"
  sha256sum "${VALIDATOR_CORE_SNAPSHOT}"
} | sha256sum | awk '{print $1}')"

cp -f "${NODES_FILE}" "${OUTDIR}/nodes_manifest.input.dat"
NODES_SHA256="$(sha256_file "${NODES_FILE}")"
COPIED_NODES_SHA256="$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")"
[ "${NODES_SHA256}" = "${COPIED_NODES_SHA256}" ] \
  || die "copied nodes manifest hash differs from original"

# Hard preflight.  The production wrapper accepts the harmless 2*pi boundary
# serialization and normalizes phase modulo 2*pi with ~1e-6 rad tolerance.
"${PYTHON3}" "${NODE_VALIDATOR_SNAPSHOT}" "${OUTDIR}/nodes_manifest.input.dat" \
  --r "${RVAL}" \
  --theta-deg "${THVAL}" \
  --I-bar "${I_BAR}" \
  --report "${OUTDIR}/node_preflight.json" \
  2>&1 | tee "${OUTDIR}/node_preflight.log"
[ -s "${OUTDIR}/node_preflight.json" ] || die "node preflight report is empty"

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

# Cache identity includes every requested build dimension, not just source.
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

    echo "Compiling hhg_sbe for build key ${BUILD_KEY}..."
    TMP_CACHE="$(mktemp -d "${BUILD_ROOT}/build.${BUILD_KEY}.XXXXXX")"
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
    {
      echo "build_key=${BUILD_KEY}"
      echo "source_sha256=${SOURCE_SHA256}"
      echo "FC=${BUILD_FC}"
      echo "FC_path=${FC_PATH}"
      echo "compiler_version=${COMPILER_VERSION}"
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

[ -x "${CACHED_BIN}" ] || die "missing build-keyed executable ${CACHED_BIN}"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"

TMPL="${REPO}/deploy/a0_layerA_m88full/input_template_full112.nml"
[ -f "${TMPL}" ] || die "input template not found: ${TMPL}"
sed -e "s|TB_PLACEHOLDER|${TB}|g" \
    -e "s|NK_PLACEHOLDER|${NK}|g" \
    -e "s|R_PLACEHOLDER|${RVAL}|g" \
    -e "s|TH_PLACEHOLDER|${THVAL}|g" \
    -e "s|NODES_PLACEHOLDER|${NODES_FILE}|g" \
    -e "s|PROPAGATE_IDS_PLACEHOLDER||g" \
    "${TMPL}" > "${OUTDIR}/input.nml"
cp -f "${TMPL}" "${OUTDIR}/input_template.snapshot.nml"
cp -f "${CACHE_DIR}/build_metadata.txt" "${OUTDIR}/build_metadata.snapshot.txt"

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
PYTHON_VERSION="$("${PYTHON3}" --version 2>&1)"

{
  echo "case=${CASE}"
  echo "campaign=full112_gh3_candidates_k20_selection"
  echo "model=full112"
  echo "nv_orig=84"
  echo "nb_start=1"
  echo "nb_end=112"
  echo "note=NOT_formal_M88_GH5_NOT_k50_state_selection_only"
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
  echo "build_key=${BUILD_KEY}"
  echo "binary=${CACHED_BIN}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "FC=${BUILD_FC}"
  echo "FC_path=${FC_PATH}"
  echo "compiler_version=${COMPILER_VERSION}"
  echo "FFLAGS=${BUILD_FFLAGS}"
  echo "LDFLAGS=${BUILD_LDFLAGS}"
  echo "python=${PYTHON_VERSION}"
  echo "validator_bundle_sha256=${VALIDATOR_BUNDLE_SHA256}"
  echo "nodes_file=${NODES_FILE}"
  echo "nodes_sha256=${NODES_SHA256}"
  echo "tb_file=${TB}"
  echo "tb_sha256=${TB_SHA256}"
  echo "input_sha256=${INPUT_SHA256}"
  echo "template_sha256=${TEMPLATE_SHA256}"
  echo "nk=${NK}"
  echo "nkx=${NK}"
  echo "nky=${NK}"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "squeeze_r=${RVAL}"
  echo "squeeze_theta_deg=${THVAL}"
  echo "I_bar=${I_BAR}"
  echo "harmonics=${HARMONICS_CSV}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

# Independent reconstruction of node coverage, ICS, CS and node metadata.
"${PYTHON3}" "${RUN_VALIDATOR_SNAPSHOT}" "${OUTDIR}" \
  --manifest "${OUTDIR}/nodes_manifest.input.dat" \
  --harmonics "${HARMONICS_CSV}" \
  --report "${OUTDIR}/a0_run_validation.json" \
  2>&1 | tee "${OUTDIR}/a0_run_validation.log"
[ -s "${OUTDIR}/a0_run_validation.json" ] \
  || die "strict post-run report is empty"

CRITICAL_OUTPUTS=(
  HHG_bsv.dat
  HHG_ics_cs.dat
  HHG_nodes_modes.dat
  nodes_moment_check.txt
  run.log
  node_preflight.json
  node_preflight.log
  a0_run_validation.json
  a0_run_validation.log
)
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${output_file}" ] \
    || die "missing or empty critical output: ${OUTDIR}/${output_file}"
done

grep -Eq '^pass[[:space:]]*=[[:space:]]*T[[:space:]]*$' nodes_moment_check.txt \
  || die "node moment validation did not report pass = T"
grep -q 'NODE_PREFLIGHT=PASS' node_preflight.log \
  || die "preflight log lacks NODE_PREFLIGHT=PASS"
grep -q 'A0_RUN_VALIDATION=PASS' a0_run_validation.log \
  || die "post-run log lacks A0_RUN_VALIDATION=PASS"

NONFINITE_RE='(^|[^[:alpha:]])[+-]?(nan|inf(inity)?)([^[:alpha:]]|$)'
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  if LC_ALL=C grep -Eiq "${NONFINITE_RE}" "${output_file}"; then
    die "NaN/Inf detected in critical output: ${OUTDIR}/${output_file}"
  fi
done

NODE_COUNT=$(awk '!/^#/ && NF {n++} END {print n+0}' nodes_manifest.input.dat)
MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
EXPECTED_MODE_ROWS=$((NODE_COUNT * HARMONIC_COUNT))
[ "${MODE_ROWS}" -eq "${EXPECTED_MODE_ROWS}" ] \
  || die "HHG_nodes_modes rows=${MODE_ROWS}, expected ${EXPECTED_MODE_ROWS}"

sha256sum "${CRITICAL_OUTPUTS[@]}" > output_sha256.txt
{
  echo "status=PASS"
  echo "exit_code=0"
  echo "completed_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "validator_bundle_sha256=${VALIDATOR_BUNDLE_SHA256}"
  echo "nodes_sha256=${NODES_SHA256}"
  echo "tb_sha256=${TB_SHA256}"
  echo "input_sha256=${INPUT_SHA256}"
  echo "node_count=${NODE_COUNT}"
  echo "mode_row_count=${MODE_ROWS}"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} $(date)"
