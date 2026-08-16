#!/bin/bash
#=============================================================================
# GH7 tail-model probes: 4 +N wide-axis antinodes @ full112.
# Default NK=20. Do NOT auto-chain k40 or full GH7.
# IDs come from frozen manifests via selector; not hardcoded.
# Full 49-node manifest is used for Q0 / moment check; only one id is propagated.
# Weights are NOT renormalized.
#=============================================================================
#SBATCH --job-name=a0_gh7_tail
#SBATCH --partition=part_1
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_gh7_tail_%A_%a.out
#SBATCH --error=a0_gh7_tail_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
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
WORKDIR="$(pwd)"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3}"
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
NK="${NK:-20}"
I_BAR=1.0e11
HARMONICS_CSV="2,5,7,9,10"
HARMONIC_COUNT=5
OCC_STRIDE="${OCC_STRIDE:-336}"
PINNED_BINARY_SHA256="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
PINNED_BINARY="${PINNED_BINARY:-${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe}"
PINNED_TB_PLUS_SHA256="66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18"
FREEZE="${REPO}/deploy/a0_layerA_m88full/gh7_tail/FREEZE.json"
NODES_DIR="${REPO}/deploy/a0_layerA_m88full/nodes_tbv2_es25/gh7"
TMPL_EXPECTED="${REPO}/deploy/a0_layerA_m88full/input_template_full112_gh7_tail.nml"

if [ "${NK}" = "40" ]; then
  [ "${APPROVE_GH7_TAIL_K40:-}" = "1" ] || die "k40 blocked: set APPROVE_GH7_TAIL_K40=1 after k20 occupation PASS"
  OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_full112_k40_gh7_tail}"
  CAMPAIGN="full112_gh7_tail_k40"
else
  [ "${NK}" = "20" ] || die "NK must be 20 or 40"
  OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_full112_k20_gh7_tail}"
  CAMPAIGN="full112_gh7_tail_k20"
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 3 ] || die "bad array index ${IDX}"

for command_name in sha256sum git find xargs awk grep python3; do
  require_cmd "${command_name}"
done
# shellcheck disable=SC1091
source "${REPO}/deploy/a0_layerA_m88full/resolve_python3.sh"
PYTHON3="${A0_PYTHON3}"
[ -x "${PYTHON3}" ] || die "A0_PYTHON3 not executable: ${PYTHON3}"
[ -f "${FREEZE}" ] || die "missing ${FREEZE}"

PROBE="$("${PYTHON3}" - "${FREEZE}" "${IDX}" <<'PY'
import json, sys
fr = json.loads(open(sys.argv[1], encoding="utf-8").read())
p = fr["plusN_probes"][int(sys.argv[2])]
print(p["theta_deg"], p["id"], p["manifest"], p["manifest_sha256"], f"{p['I_max']:.8e}")
PY
)"
# shellcheck disable=SC2086
set -- ${PROBE}
THVAL="$1"
NID="$2"
MAN_NAME="$3"
PINNED_MAN_SHA="$4"
I_MAX="$5"
RVAL="2.5"
RTAG="2p5"
NODES_FILE="${NODES_DIR}/${MAN_NAME}"
PROP_IDS="${NID}"
CASE=$(printf "sv_r%s_th%03d_plusN_id%02d_k%s" "${RTAG}" "${THVAL}" "${NID}" "${NK}")
OUTDIR="${OUTROOT}/${CASE}"

[ -f "${NODES_FILE}" ] || die "missing GH7 manifest ${NODES_FILE}"
[ -f "${TB_PLUS}" ] || die "TB not found: ${TB_PLUS}"
TB_SHA_NOW="$(sha256_file "${TB_PLUS}")"
[ "${TB_SHA_NOW}" = "${PINNED_TB_PLUS_SHA256}" ] || die "plus TB SHA mismatch"
NODES_SHA_NOW="$(sha256_file "${NODES_FILE}")"
[ "${NODES_SHA_NOW}" = "${PINNED_MAN_SHA}" ] || die "GH7 manifest SHA mismatch: ${NODES_SHA_NOW}"

NODE_VALIDATOR="${REPO}/tools/analysis/validate_qlight_nodes_production.py"
RUN_VALIDATOR="${REPO}/tools/analysis/validate_a0_run_strict.py"
NODE_VALIDATOR_CORE="${REPO}/tools/analysis/validate_qlight_nodes_strict.py"
[ -f "${NODE_VALIDATOR}" ] || die "missing node validator"
[ -f "${RUN_VALIDATOR}" ] || die "missing run validator"

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
echo " A0 full112 GH7 tail NK=${NK} case=${CASE}"
echo " nodes=${NODES_FILE} propagate_ids=${PROP_IDS} I_max=${I_MAX}"
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

[ -x "${PINNED_BINARY}" ] || die "pinned 29175 binary missing: ${PINNED_BINARY}"
CACHED_BIN="${PINNED_BINARY}"
CACHE_DIR="$(dirname "${CACHED_BIN}")"
[ -f "${CACHE_DIR}/build_metadata.txt" ] || die "pinned binary cache lacks build_metadata.txt"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"
[ "${BINARY_SHA256}" = "${PINNED_BINARY_SHA256}" ] || die "pinned binary SHA mismatch"
SOURCE_SHA256="$(awk -F= '$1=="source_sha256"{print $2; exit}' "${CACHE_DIR}/build_metadata.txt")"
BUILD_KEY="$(awk -F= '$1=="build_key"{print $2; exit}' "${CACHE_DIR}/build_metadata.txt")"
WORKTREE_SOURCE_SHA256="$(source_tree_digest)"
echo "USING_PINNED_29175_BINARY sha=${BINARY_SHA256} worktree_source=${WORKTREE_SOURCE_SHA256}"

if [ -f "${OUTDIR}/SUCCESS" ] && [ -s "${OUTDIR}/HHG_nodes_modes.dat" ]; then
  die "OUTDIR already has SUCCESS; use a fresh OUTROOT: ${OUTDIR}"
fi
if [ -d "${OUTDIR}" ] && [ -f "${OUTDIR}/run_status.txt" ]; then
  die "output directory is not empty without matching SUCCESS: ${OUTDIR}"
fi

cp -f "${NODE_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
cp -f "${NODE_VALIDATOR_CORE}" "${OUTDIR}/validator_snapshot/"
cp -f "${RUN_VALIDATOR}" "${OUTDIR}/validator_snapshot/"
cp -f "${NODES_FILE}" "${OUTDIR}/nodes_manifest.input.dat"
[ "$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")" = "${NODES_SHA_NOW}" ] \
  || die "copied nodes manifest hash differs"

"${PYTHON3}" "${OUTDIR}/validator_snapshot/validate_qlight_nodes_production.py" \
  "${OUTDIR}/nodes_manifest.input.dat" \
  --r "${RVAL}" --theta-deg "${THVAL}" --I-bar "${I_BAR}" \
  --report "${OUTDIR}/node_preflight.json" \
  2>&1 | tee "${OUTDIR}/node_preflight.log"
grep -q 'NODE_PREFLIGHT=PASS' "${OUTDIR}/node_preflight.log" || die "preflight failed"

sed -e "s|TB_PLACEHOLDER|${TB_PLUS}|g" \
    -e "s|NK_PLACEHOLDER|${NK}|g" \
    -e "s|R_PLACEHOLDER|${RVAL}|g" \
    -e "s|TH_PLACEHOLDER|${THVAL}|g" \
    -e "s|NODES_PLACEHOLDER|nodes_manifest.input.dat|g" \
    -e "s|PROPAGATE_IDS_PLACEHOLDER|${PROP_IDS}|g" \
    -e "s|OCC_STRIDE_PLACEHOLDER|${OCC_STRIDE}|g" \
    "${TMPL_EXPECTED}" > "${OUTDIR}/input.nml"
cp -f "${TMPL_EXPECTED}" "${OUTDIR}/input_template.snapshot.nml"
cp -f "${CACHE_DIR}/build_metadata.txt" "${OUTDIR}/build_metadata.snapshot.txt"

{
  echo "case=${CASE}"
  echo "campaign=${CAMPAIGN}"
  echo "binary_policy=pinned_29175"
  echo "worktree_source_sha256=${WORKTREE_SOURCE_SHA256}"
  echo "model=full112"
  echo "propagate_ids=${PROP_IDS}"
  echo "note=full_gh7_manifest_moment_check; subset_not_renormalized"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "repo=${REPO}"
  echo "git_head=$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "binary=${CACHED_BIN}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "nodes_file=${NODES_FILE}"
  echo "nodes_sha256=${NODES_SHA_NOW}"
  echo "tb_file=${TB_PLUS}"
  echo "tb_sha256=${TB_SHA_NOW}"
  echo "input_sha256=$(sha256_file "${OUTDIR}/input.nml")"
  echo "template_sha256=$(sha256_file "${TMPL_EXPECTED}")"
  echo "nk=${NK}"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "squeeze_r=${RVAL}"
  echo "squeeze_theta_deg=${THVAL}"
  echo "I_bar=${I_BAR}"
  echo "I_max=${I_MAX}"
  echo "harmonics=${HARMONICS_CSV}"
  echo "occ_stride=${OCC_STRIDE}"
  echo "save_occupation=true"
  echo "occ_band_resolved=true"
  echo "bsv_save_jt=true"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

grep -Eq '^pass[[:space:]]*=[[:space:]]*T[[:space:]]*$' nodes_moment_check.txt \
  || die "full-manifest moment check failed"

CRITICAL_OUTPUTS=(
  HHG_nodes_modes.dat
  nodes_moment_check.txt
  run.log
  node_preflight.json
  run_metadata.txt
  occupation_kt.dat
  occupation_band_kt.dat
)
JT_FILE=$(printf 'Jt_node_%04d.dat' "${NID}")
CRITICAL_OUTPUTS+=("${JT_FILE}")
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${output_file}" ] || die "missing or empty critical output: ${OUTDIR}/${output_file}"
done

MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
[ "${MODE_ROWS}" -eq "${HARMONIC_COUNT}" ] \
  || die "HHG_nodes_modes rows=${MODE_ROWS}, expected ${HARMONIC_COUNT}"

NONFINITE_RE='(^|[^[:alpha:]])[+-]?(nan|inf(inity)?)([^[:alpha:]]|$)'
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  if LC_ALL=C grep -Eiq "${NONFINITE_RE}" "${output_file}"; then
    die "NaN/Inf detected in ${OUTDIR}/${output_file}"
  fi
done

sha256sum "${CRITICAL_OUTPUTS[@]}" > output_sha256.txt
{
  echo "status=PASS"
  echo "exit_code=0"
  echo "completed_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "nodes_sha256=${NODES_SHA_NOW}"
  echo "tb_sha256=${TB_SHA_NOW}"
  echo "propagate_ids=${PROP_IDS}"
  echo "mode_row_count=${MODE_ROWS}"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} propagate_ids=${PROP_IDS}"
