#!/bin/bash
#=============================================================================
# full112 @ k20, GH5-v2, TWO selected states × ±N × 5-node chunks => 20 tasks.
#
# Selected states (post 27992 / GH3-v2 29175):
#   0: r=2.5, theta=0
#   1: r=2.5, theta=180
#
# Production pins (hard-fail on mismatch; do NOT rebuild from dirty tree):
#   - binary = 29175 cache, SHA 34edab1d…
#   - +N TB 66382a51… ; −N TB-v2 d61034e1…
#   - GH5 ES25 manifests 9f266680… / aaca2b3b…
# C8↔C36 topology audit PASS (τ=1e-4) → prefer part_1 / OMP=36.
# Unique OUTROOT: output_a0_full112_k20_gh5_tbv2_omp36
# WORKDIR is job cwd (--chdir), never login SLURM_SUBMIT_DIR.
# Formal quadrature: tools/analysis/gh3_gh5_quadrature_gate.py
#   (ICS/CS/Je 5%, Jo mixed 10%, strong phase <0.1 rad; NOT CEP 1e-3).
# Merge must verify each chunk output_sha256.txt and record OMP/MKL.
#=============================================================================
#SBATCH --job-name=a0_f112_gh5
#SBATCH --partition=part_1
#SBATCH --array=0-19%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=14:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_f112_gh5_%A_%a.out
#SBATCH --error=a0_f112_gh5_%A_%a.err

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
# Job cwd is sbatch --chdir / SLURM WorkDir. Do not inherit login WORKDIR
# or SLURM_SUBMIT_DIR (those dumped 29175 into $HOME).
WORKDIR="$(pwd)"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3}"
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
TB_MINUS="${TB_MINUS:-/public/home/wangjs/project/CrI3_TB_mN/wannier/CrI3_tb_mN_Pconj_v2.dat}"
TB_MINUS_PROD_FORBIDDEN="${TB_MINUS_PROD_FORBIDDEN:-/public/home/wangjs/project/CrI3_TB_mN/wannier/CrI3_tb_mN_Pconj.dat}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_full112_k20_gh5_tbv2_omp36}"
NODES_DIR="${NODES_DIR:-${REPO}/deploy/a0_layerA_m88full/nodes_tbv2_es25/gh5}"
PINNED_BINARY_SHA256="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
PINNED_BINARY="${PINNED_BINARY:-${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe}"
PINNED_TB_PLUS_SHA256="66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18"
PINNED_TB_MINUS_V2_SHA256="d61034e18b551dbb33b064c0c311d4ba0ef13be1c8f4a11a88ddbe1e95b66246"
PINNED_GH5_TH0_SHA256="9f266680eef28b6e5c309f467a9529fe7dd275a55fa1be2097adbaf52e1201f3"
PINNED_GH5_TH180_SHA256="aaca2b3b3aff1cd99c46801a103e8b3b13df11e0d6d918b03203c95778d094b2"
NK=20
I_BAR=1.0e11
HARMONICS_CSV="2,5,7,9,10"
HARMONIC_COUNT=5
CHUNK_SIZE=5
N_MANIFEST=25

CAND_R=(2.5 2.5)
CAND_TH=(0 180)
CAND_RTAG=(2p5 2p5)

NODE_VALIDATOR="${REPO}/tools/analysis/validate_qlight_nodes_production.py"
RUN_VALIDATOR="${REPO}/tools/analysis/validate_a0_run_strict.py"
NODE_VALIDATOR_CORE="${REPO}/tools/analysis/validate_qlight_nodes_strict.py"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 19 ] || die "bad array index ${IDX}"
ICAND=$(( IDX / 10 ))
REM=$(( IDX % 10 ))
IDOM=$(( REM / 5 ))
ICHUNK=$(( REM % 5 ))

RVAL="${CAND_R[${ICAND}]}"
THVAL="${CAND_TH[${ICAND}]}"
RTAG="${CAND_RTAG[${ICAND}]}"
if [ "${IDOM}" -eq 0 ]; then
  DOM="plusN"
  TB="${TB_PLUS}"
else
  DOM="minusN"
  TB="${TB_MINUS}"
  case "${TB}" in
    *Pconj_v2.dat) ;;
    *) die "GH5-v2 refuses non-v2 minus TB: ${TB}" ;;
  esac
fi

START=$(( ICHUNK * CHUNK_SIZE + 1 ))
END=$(( START + CHUNK_SIZE - 1 ))
PROP_IDS=""
for id in $(seq "${START}" "${END}"); do
  [ "${id}" -le "${N_MANIFEST}" ] || die "chunk id ${id} exceeds manifest size ${N_MANIFEST}"
  if [ -z "${PROP_IDS}" ]; then PROP_IDS="${id}"; else PROP_IDS="${PROP_IDS},${id}"; fi
done

NODES_FILE="${NODES_DIR}/nodes_sv_r${RTAG}_th${THVAL}_gh5.dat"
CASE=$(printf "sv_r%s_th%03d_%s_c%02d" "${RTAG}" "${THVAL}" "${DOM}" "${ICHUNK}")
OUTDIR="${OUTROOT}/${CASE}"

for command_name in sha256sum git find xargs awk grep mktemp; do
  require_cmd "${command_name}"
done
# shellcheck disable=SC1091
source "${REPO}/deploy/a0_layerA_m88full/resolve_python3.sh"
PYTHON3="${A0_PYTHON3}"
[ -x "${PYTHON3}" ] || die "A0_PYTHON3 not executable: ${PYTHON3}"

[ -f "${NODES_FILE}" ] || die "missing GH5 manifest ${NODES_FILE}"
[ -f "${TB}" ] || die "TB not found: ${TB}"
if [ "${IDOM}" -eq 1 ]; then
  [ "${TB}" != "${TB_MINUS_PROD_FORBIDDEN}" ] \
    || die "GH5-v2 refuses production E18.8 -N TB: ${TB}"
  TB_SHA_NOW="$(sha256_file "${TB}")"
  [ "${TB_SHA_NOW}" = "${PINNED_TB_MINUS_V2_SHA256}" ] \
    || die "minus TB-v2 SHA mismatch: ${TB_SHA_NOW} != ${PINNED_TB_MINUS_V2_SHA256}"
else
  TB_SHA_NOW="$(sha256_file "${TB}")"
  [ "${TB_SHA_NOW}" = "${PINNED_TB_PLUS_SHA256}" ] \
    || die "plus TB SHA mismatch: ${TB_SHA_NOW} != ${PINNED_TB_PLUS_SHA256}"
fi
NODES_SHA_NOW="$(sha256_file "${NODES_FILE}")"
if [ "${THVAL}" -eq 0 ]; then
  [ "${NODES_SHA_NOW}" = "${PINNED_GH5_TH0_SHA256}" ] \
    || die "GH5 th0 manifest SHA mismatch: ${NODES_SHA_NOW}"
else
  [ "${NODES_SHA_NOW}" = "${PINNED_GH5_TH180_SHA256}" ] \
    || die "GH5 th180 manifest SHA mismatch: ${NODES_SHA_NOW}"
fi
[ -f "${REPO}/Makefile" ] || die "Makefile not found under REPO=${REPO}"
[ -f "${NODE_VALIDATOR}" ] || die "missing node validator: ${NODE_VALIDATOR}"
[ -f "${RUN_VALIDATOR}" ] || die "missing run validator: ${RUN_VALIDATOR}"
[ -f "${NODE_VALIDATOR_CORE}" ] || die "missing validator dependency: ${NODE_VALIDATOR_CORE}"

# SUCCESS skip is deferred until source/binary hashes are known (below).
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
echo " A0 full112 GH5@k20 chunked case=${CASE}"
echo " nodes=${NODES_FILE}"
echo " propagate_ids=${PROP_IDS}"
echo " TB=${TB}  (full112 ±N)"
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
WORKTREE_SOURCE_SHA256="$(source_tree_digest)"

# Reuse the 29175 binary. Never compile from the current dirty worktree.
[ -x "${PINNED_BINARY}" ] || die "pinned 29175 binary missing: ${PINNED_BINARY}"
CACHED_BIN="${PINNED_BINARY}"
CACHE_DIR="$(dirname "${CACHED_BIN}")"
[ -f "${CACHE_DIR}/build_metadata.txt" ] \
  || die "pinned binary cache lacks build_metadata.txt: ${CACHE_DIR}"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"
[ "${BINARY_SHA256}" = "${PINNED_BINARY_SHA256}" ] \
  || die "pinned binary SHA mismatch: ${BINARY_SHA256} != ${PINNED_BINARY_SHA256}"
SOURCE_SHA256="$(awk -F= '$1=="source_sha256"{print $2; exit}' "${CACHE_DIR}/build_metadata.txt")"
BUILD_KEY="$(awk -F= '$1=="build_key"{print $2; exit}' "${CACHE_DIR}/build_metadata.txt")"
[ -n "${SOURCE_SHA256}" ] || die "pinned build_metadata missing source_sha256"
[ -n "${BUILD_KEY}" ] || die "pinned build_metadata missing build_key"
echo "USING_PINNED_29175_BINARY sha=${BINARY_SHA256} worktree_source=${WORKTREE_SOURCE_SHA256}"

# Skip only when SUCCESS exists AND provenance+chunk+validator+output hashes match.
NODES_SHA256_EXPECTED="$(sha256_file "${NODES_FILE}")"
TB_SHA256_EXPECTED="$(sha256_file "${TB}")"
TMPL_EXPECTED="${REPO}/deploy/a0_layerA_m88full/input_template_full112.nml"
TEMPLATE_SHA256_EXPECTED="$(sha256_file "${TMPL_EXPECTED}")"
# Content-only hashes (no paths): expected and actual must use the same order.
VALIDATOR_BUNDLE_SHA256_EXPECTED="$({
  sha256_file "${NODE_VALIDATOR}"
  sha256_file "${NODE_VALIDATOR_CORE}"
  sha256_file "${RUN_VALIDATOR}"
} | sha256sum | awk '{print $1}')"
if [ -f "${OUTDIR}/SUCCESS" ] && [ -s "${OUTDIR}/HHG_nodes_modes.dat" ] \
   && [ -f "${OUTDIR}/run_metadata.txt" ] \
   && grep -q 'status=PASS' "${OUTDIR}/run_status.txt" 2>/dev/null; then
  meta_get() { awk -F= -v k="$1" '$1==k{print $2; exit}' "${OUTDIR}/run_metadata.txt"; }
  status_get() { awk -F= -v k="$1" '$1==k{print $2; exit}' "${OUTDIR}/run_status.txt"; }
  if [ "$(meta_get source_sha256)" = "${SOURCE_SHA256}" ] \
     && [ "$(meta_get binary_sha256)" = "${BINARY_SHA256}" ] \
     && [ "$(meta_get nodes_sha256)" = "${NODES_SHA256_EXPECTED}" ] \
     && [ "$(meta_get tb_sha256)" = "${TB_SHA256_EXPECTED}" ] \
     && [ "$(meta_get template_sha256)" = "${TEMPLATE_SHA256_EXPECTED}" ] \
     && [ "$(meta_get validator_bundle_sha256)" = "${VALIDATOR_BUNDLE_SHA256_EXPECTED}" ] \
     && [ "$(meta_get propagate_ids)" = "${PROP_IDS}" ] \
     && [ "$(meta_get chunk_index)" = "${ICHUNK}" ] \
     && [ "$(status_get propagate_ids)" = "${PROP_IDS}" ] \
     && [ "$(status_get chunk_index)" = "${ICHUNK}" ] \
     && [ "$(meta_get nk)" = "${NK}" ] \
     && [ "$(meta_get dt)" = "0.35" ] \
     && [ "$(meta_get T2_cycles)" = "0.5" ] \
     && [ "$(meta_get wvl_nm)" = "3200.0" ] \
     && [ "$(meta_get squeeze_r)" = "${RVAL}" ] \
     && [ "$(meta_get squeeze_theta_deg)" = "${THVAL}" ] \
     && [ "$(meta_get I_bar)" = "${I_BAR}" ] \
     && [ "$(meta_get harmonics)" = "${HARMONICS_CSV}" ]; then
    if [ -f "${OUTDIR}/output_sha256.txt" ]; then
      ( cd "${OUTDIR}" && sha256sum -c output_sha256.txt --status ) \
        || die "SUCCESS reuse failed: output_sha256.txt mismatch in ${OUTDIR}"
    else
      die "SUCCESS present but missing output_sha256.txt: ${OUTDIR}"
    fi
    echo "ALREADY_COMPLETE ${CASE}; provenance+chunk+validator+outputs match; skipping"
    RUN_COMPLETED=1
    exit 0
  fi
  die "OUTDIR has SUCCESS but provenance mismatch; refuse reuse (use fresh OUTROOT): ${OUTDIR}"
fi
if [ -d "${OUTDIR}" ] && \
   [ -n "$(find "${OUTDIR}" -mindepth 1 -maxdepth 1 ! -name validator_snapshot -print -quit)" ]; then
  # Allow empty-ish new dirs that only have validator_snapshot from a prior failed attempt.
  if [ -f "${OUTDIR}/run_status.txt" ] || [ -f "${OUTDIR}/input.nml" ] || [ -f "${OUTDIR}/HHG_nodes_modes.dat" ]; then
    die "output directory is not empty without matching SUCCESS: ${OUTDIR}"
  fi
fi

cp -f "${NODE_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
cp -f "${NODE_VALIDATOR_CORE}" "${OUTDIR}/validator_snapshot/"
cp -f "${RUN_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
NODE_VALIDATOR_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_qlight_nodes_production.py"
VALIDATOR_CORE_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_qlight_nodes_strict.py"
RUN_VALIDATOR_SNAPSHOT="${OUTDIR}/validator_snapshot/validate_a0_run_strict.py"
VALIDATOR_BUNDLE_SHA256="$({
  sha256_file "${NODE_VALIDATOR_SNAPSHOT}"
  sha256_file "${VALIDATOR_CORE_SNAPSHOT}"
  sha256_file "${RUN_VALIDATOR_SNAPSHOT}"
} | sha256sum | awk '{print $1}')"

cp -f "${NODES_FILE}" "${OUTDIR}/nodes_manifest.input.dat"
COPIED_NODES_SHA256="$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")"
[ "${NODES_SHA256_EXPECTED}" = "${COPIED_NODES_SHA256}" ] \
  || die "copied nodes manifest hash differs from original"

"${PYTHON3}" "${NODE_VALIDATOR_SNAPSHOT}" "${OUTDIR}/nodes_manifest.input.dat" \
  --r "${RVAL}" \
  --theta-deg "${THVAL}" \
  --I-bar "${I_BAR}" \
  --report "${OUTDIR}/node_preflight.json" \
  2>&1 | tee "${OUTDIR}/node_preflight.log"
[ -s "${OUTDIR}/node_preflight.json" ] || die "node preflight report is empty"

TMPL="${TMPL_EXPECTED}"
[ -f "${TMPL}" ] || die "input template not found: ${TMPL}"
# Read the OUTDIR snapshot, not the central original nodes file.
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
NODES_SHA256="${NODES_SHA256_EXPECTED}"
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
  echo "campaign=full112_gh5_tbv2_omp36"
  echo "binary_policy=pinned_29175"
  echo "worktree_source_sha256=${WORKTREE_SOURCE_SHA256}"
  echo "model=full112"
  echo "nv_orig=84"
  echo "nb_start=1"
  echo "nb_end=112"
  echo "chunk_index=${ICHUNK}"
  echo "propagate_ids=${PROP_IDS}"
  echo "note=partial_chunk_merge_required"
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
  [ -s "${output_file}" ] \
    || die "missing or empty critical output: ${OUTDIR}/${output_file}"
done

grep -q 'NODE_PREFLIGHT=PASS' node_preflight.log \
  || die "preflight log lacks NODE_PREFLIGHT=PASS"

NONFINITE_RE='(^|[^[:alpha:]])[+-]?(nan|inf(inity)?)([^[:alpha:]]|$)'
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  if LC_ALL=C grep -Eiq "${NONFINITE_RE}" "${output_file}"; then
    die "NaN/Inf detected in critical output: ${OUTDIR}/${output_file}"
  fi
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
  echo "validator_bundle_sha256=${VALIDATOR_BUNDLE_SHA256}"
  echo "nodes_sha256=${NODES_SHA256}"
  echo "tb_sha256=${TB_SHA256}"
  echo "input_sha256=${INPUT_SHA256}"
  echo "propagate_ids=${PROP_IDS}"
  echo "chunk_index=${ICHUNK}"
  echo "mode_row_count=${MODE_ROWS}"
  echo "note=partial_chunk_merge_required"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} propagate_ids=${PROP_IDS}"
