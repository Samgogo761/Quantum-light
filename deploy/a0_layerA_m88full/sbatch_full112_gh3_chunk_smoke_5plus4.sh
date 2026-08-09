#!/bin/bash
#=============================================================================
# full112 GH3@k20 REAL chunk smoke: 5+4 split vs Job 27992 unchunked.
#
# Scope (smoke only — NOT production GH5):
#   state = sv_r2p5_th180
#   domain = plusN
#   chunks = ids 1-5 and 6-9  (array 0-1)
#
# Unique OUTROOT; after both chunks COMPLETED, merge and compare to 27992:
#   tools/analysis/merge_chunked_ensemble.py
#   tools/analysis/compare_chunked_vs_full.py
#
# DO NOT auto-submit. DO NOT treat this as GH5 clearance.
#=============================================================================
#SBATCH --job-name=a0_f112_gh3_smoke
#SBATCH --partition=part_1
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_f112_gh3_smoke_%A_%a.out
#SBATCH --error=a0_f112_gh3_smoke_%A_%a.err

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

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
TB="${TB:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
# Unique OUTROOT for this smoke campaign — do not mix with 27992 or GH5.
# Fresh unique OUTROOT by default (override only if you intentionally resume).
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_full112_k20_gh3_chunk_smoke_$(date +%Y%m%d_%H%M%S)}"
NODES_DIR="${NODES_DIR:-${REPO}/deploy/a0_layerA_m88full/nodes_gh3_full112_candidates}"
NK=20
I_BAR=1.0e11
HARMONICS_CSV="2,5,7,9,10"
HARMONIC_COUNT=5
RVAL=2.5
THVAL=180
RTAG=2p5
DOM="plusN"

NODE_VALIDATOR="${REPO}/tools/analysis/validate_qlight_nodes_production.py"
NODE_VALIDATOR_CORE="${REPO}/tools/analysis/validate_qlight_nodes_strict.py"
RUN_VALIDATOR="${REPO}/tools/analysis/validate_a0_run_strict.py"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 1 ] || die "bad array index ${IDX}"
if [ "${IDX}" -eq 0 ]; then
  PROP_IDS="1,2,3,4,5"
  CHUNK_SIZE=5
  ICHUNK=0
else
  PROP_IDS="6,7,8,9"
  CHUNK_SIZE=4
  ICHUNK=1
fi

NODES_FILE="${NODES_DIR}/nodes_sv_r${RTAG}_th${THVAL}_gh3.dat"
CASE=$(printf "sv_r%s_th%03d_%s_c%02d" "${RTAG}" "${THVAL}" "${DOM}" "${ICHUNK}")
OUTDIR="${OUTROOT}/${CASE}"

for command_name in sha256sum git find xargs awk grep mktemp; do
  require_cmd "${command_name}"
done
# shellcheck disable=SC1091
source "${REPO}/deploy/a0_layerA_m88full/resolve_python3.sh"
PYTHON3="${A0_PYTHON3}"
[ -x "${PYTHON3}" ] || die "A0_PYTHON3 not executable: ${PYTHON3}"

[ -f "${NODES_FILE}" ] || die "missing GH3 manifest ${NODES_FILE}"
[ -f "${TB}" ] || die "TB not found: ${TB}"
[ -f "${REPO}/Makefile" ] || die "Makefile not found under REPO=${REPO}"

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
echo " A0 full112 GH3@k20 CHUNK SMOKE case=${CASE}"
echo " nodes=${NODES_FILE}"
echo " propagate_ids=${PROP_IDS}"
echo " OUTROOT=${OUTROOT}"
echo " host=$(hostname) date=$(date)"
echo "============================================="

export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
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
    [ -x "${REPO}/hhg_sbe" ] || die "compiler returned without executable"
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

NODES_SHA256_EXPECTED="$(sha256_file "${NODES_FILE}")"
TB_SHA256_EXPECTED="$(sha256_file "${TB}")"
TMPL_EXPECTED="${REPO}/deploy/a0_layerA_m88full/input_template_full112.nml"
TEMPLATE_SHA256_EXPECTED="$(sha256_file "${TMPL_EXPECTED}")"
if [ -f "${OUTDIR}/SUCCESS" ] && [ -s "${OUTDIR}/HHG_nodes_modes.dat" ] \
   && [ -f "${OUTDIR}/run_metadata.txt" ] \
   && grep -q 'status=PASS' "${OUTDIR}/run_status.txt" 2>/dev/null; then
  meta_get() { awk -F= -v k="$1" '$1==k{print $2; exit}' "${OUTDIR}/run_metadata.txt"; }
  if [ "$(meta_get source_sha256)" = "${SOURCE_SHA256}" ] \
     && [ "$(meta_get binary_sha256)" = "${BINARY_SHA256}" ] \
     && [ "$(meta_get nodes_sha256)" = "${NODES_SHA256_EXPECTED}" ] \
     && [ "$(meta_get tb_sha256)" = "${TB_SHA256_EXPECTED}" ] \
     && [ "$(meta_get template_sha256)" = "${TEMPLATE_SHA256_EXPECTED}" ] \
     && [ "$(meta_get nk)" = "${NK}" ] \
     && [ "$(meta_get dt)" = "0.35" ] \
     && [ "$(meta_get T2_cycles)" = "0.5" ] \
     && [ "$(meta_get wvl_nm)" = "3200.0" ] \
     && [ "$(meta_get squeeze_r)" = "${RVAL}" ] \
     && [ "$(meta_get squeeze_theta_deg)" = "${THVAL}" ] \
     && [ "$(meta_get I_bar)" = "${I_BAR}" ] \
     && [ "$(meta_get harmonics)" = "${HARMONICS_CSV}" ]; then
    echo "ALREADY_COMPLETE ${CASE}; full provenance matches; skipping"
    RUN_COMPLETED=1
    exit 0
  fi
  die "OUTDIR has SUCCESS but provenance mismatch; use fresh OUTROOT: ${OUTDIR}"
fi
if [ -d "${OUTDIR}" ] && \
   [ -n "$(find "${OUTDIR}" -mindepth 1 -maxdepth 1 ! -name validator_snapshot -print -quit)" ]; then
  if [ -f "${OUTDIR}/run_status.txt" ] || [ -f "${OUTDIR}/input.nml" ] || [ -f "${OUTDIR}/HHG_nodes_modes.dat" ]; then
    die "output directory is not empty without matching SUCCESS: ${OUTDIR}"
  fi
fi

cp -f "${NODE_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
cp -f "${NODE_VALIDATOR_CORE}" "${OUTDIR}/validator_snapshot/"
cp -f "${RUN_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
NODE_VALIDATOR_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_qlight_nodes_production.py"
VALIDATOR_BUNDLE_SHA256="$({
  sha256sum "${OUTDIR}/validator_snapshot/validate_qlight_nodes_production.py"
  sha256sum "${OUTDIR}/validator_snapshot/validate_qlight_nodes_strict.py"
  sha256sum "${OUTDIR}/validator_snapshot/validate_a0_run_strict.py"
} | sha256sum | awk '{print $1}')"

cp -f "${NODES_FILE}" "${OUTDIR}/nodes_manifest.input.dat"
NODES_SHA256="${NODES_SHA256_EXPECTED}"
[ "${NODES_SHA256}" = "$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")" ] \
  || die "copied nodes manifest hash differs"

"${PYTHON3}" "${NODE_VALIDATOR_SNAPSHOT}" "${OUTDIR}/nodes_manifest.input.dat" \
  --r "${RVAL}" \
  --theta-deg "${THVAL}" \
  --I-bar "${I_BAR}" \
  --report "${OUTDIR}/node_preflight.json" \
  2>&1 | tee "${OUTDIR}/node_preflight.log"
grep -q 'NODE_PREFLIGHT=PASS' "${OUTDIR}/node_preflight.log" \
  || die "node preflight failed"

TMPL="${TMPL_EXPECTED}"
[ -f "${TMPL}" ] || die "input template not found: ${TMPL}"
sed -e "s|TB_PLACEHOLDER|${TB}|g" \
    -e "s|NK_PLACEHOLDER|${NK}|g" \
    -e "s|R_PLACEHOLDER|${RVAL}|g" \
    -e "s|TH_PLACEHOLDER|${THVAL}|g" \
    -e "s|NODES_PLACEHOLDER|nodes_manifest.input.dat|g" \
    -e "s|PROPAGATE_IDS_PLACEHOLDER|${PROP_IDS}|g" \
    "${TMPL}" > "${OUTDIR}/input.nml"
cp -f "${TMPL}" "${OUTDIR}/input_template.snapshot.nml"
cp -f "${CACHE_DIR}/build_metadata.txt" "${OUTDIR}/build_metadata.snapshot.txt"

TB_SHA256="${TB_SHA256_EXPECTED}"
INPUT_SHA256="$(sha256_file "${OUTDIR}/input.nml")"
TEMPLATE_SHA256="${TEMPLATE_SHA256_EXPECTED}"
GIT_HEAD="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)"
git -C "${REPO}" status --porcelain --untracked-files=normal \
  > "${OUTDIR}/git_status_porcelain.txt" 2>&1 || true
GIT_DIRTY="no"
[ -s "${OUTDIR}/git_status_porcelain.txt" ] && GIT_DIRTY="yes"

{
  echo "case=${CASE}"
  echo "campaign=full112_gh3_chunk_smoke_5plus4"
  echo "model=full112"
  echo "chunk_index=${ICHUNK}"
  echo "propagate_ids=${PROP_IDS}"
  echo "note=partial_chunk_merge_required_compare_to_27992"
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
  echo "python=$("${PYTHON3}" --version 2>&1)"
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
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
  echo "omp_places=${OMP_PLACES}"
  echo "omp_proc_bind=${OMP_PROC_BIND}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

grep -Eq '^pass[[:space:]]*=[[:space:]]*T[[:space:]]*$' nodes_moment_check.txt \
  || die "full-manifest moment check failed"

CRITICAL_OUTPUTS=(
  HHG_nodes_modes.dat
  chunk_info.txt
  chunk_weighted_spectrum.dat
  nodes_moment_check.txt
  run.log
  node_preflight.json
  node_preflight.log
  run_metadata.txt
)
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${output_file}" ] || die "missing or empty: ${OUTDIR}/${output_file}"
done

MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
EXPECTED_MODE_ROWS=$((CHUNK_SIZE * HARMONIC_COUNT))
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
  echo "nodes_sha256=${NODES_SHA256}"
  echo "tb_sha256=${TB_SHA256}"
  echo "input_sha256=${INPUT_SHA256}"
  echo "propagate_ids=${PROP_IDS}"
  echo "chunk_index=${ICHUNK}"
  echo "mode_row_count=${MODE_ROWS}"
  echo "note=partial_chunk_merge_required_compare_to_27992"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} propagate_ids=${PROP_IDS}"
echo "NEXT: after both chunks PASS, merge + compare:"
echo "  python tools/analysis/merge_chunked_ensemble.py \\"
echo "    --manifest ${OUTROOT}/sv_r${RTAG}_th$(printf '%03d' "${THVAL}")_${DOM}_c00/nodes_manifest.input.dat \\"
echo "    --chunk-dirs ${OUTROOT}/sv_r${RTAG}_th$(printf '%03d' "${THVAL}")_${DOM}_c00 ${OUTROOT}/sv_r${RTAG}_th$(printf '%03d' "${THVAL}")_${DOM}_c01 \\"
echo "    --outdir ${OUTROOT}/merged_plusN"
echo "  python tools/analysis/compare_chunked_vs_full.py \\"
echo "    --merged ${OUTROOT}/merged_plusN \\"
echo "    --full-run-dir <27992>/sv_r${RTAG}_th$(printf '%03d' "${THVAL}")_${DOM}"
