#!/bin/bash
#=============================================================================
# Plan B: same-binary CHUNKED C8 on part_2 / OMP=8 vs unchunked U8 (28986).
#
# Isolates ONLY chunking (5+4) under identical thread topology:
#   U8 = job 28986 unchunked 9 nodes @ part_2 / OMP=8
#   C8 = this array (c00: ids 1-5, c01: ids 6-9) @ part_2 / OMP=8
#   same binary / TB / nodes as 28861 c00 (no recompile)
#
# After both chunks PASS:
#   merge → compare C8_merged vs U8  (STRICT_CHUNK_EQUIVALENCE @ 8 threads)
# Optional later: C8 vs C36(28861) = cross-workspace reproducibility.
#
# Labels (do not conflate):
#   PHYSICS_STABILITY (C36 vs U8 / 28986): already PASS (strong modes)
#   STRICT_CHUNK_EQUIVALENCE (requires same OMP): C8 vs U8 here
#
# NOT GH5. CEP-π / GH5 remain closed.
#=============================================================================
#SBATCH --job-name=a0_f112_gh3_c8
#SBATCH --partition=part_2
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_f112_gh3_c8_%A_%a.out
#SBATCH --error=a0_f112_gh3_c8_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
meta_get_file() {
  awk -F= -v k="$1" '$1==k{print $2; exit}' "$2"
}

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# Binary / nodes provenance anchor (28861 c00)
SMOKE_OUTROOT="${SMOKE_OUTROOT:-/public/home/wangjs/project/New_SBEs/Quantum-light/runs/a0_full112_gh3_chunk_smoke_689d9a3/output_a0_full112_k20_gh3_chunk_smoke_28861}"
REF_CHUNK_DIR="${REF_CHUNK_DIR:-${SMOKE_OUTROOT}/sv_r2p5_th180_plusN_c00}"
# Unchunked U8 (28986) — compare target after merge
U8_DIR="${U8_DIR:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3/output_a0_full112_k20_gh3_unchunked_ctrl_28986/sv_r2p5_th180_plusN_unchunked}"
# Prefer U8 nml (ids 1..9) so each chunk is a single-line rewrite from the same baseline.
U8_NML_REF="${U8_NML_REF:-${U8_DIR}/input.nml}"
TB="${TB:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"

if [ -n "${OUTROOT:-}" ]; then
  :
elif [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  OUTROOT="${WORKDIR}/output_a0_full112_k20_gh3_chunk_c8_${SLURM_ARRAY_JOB_ID}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  OUTROOT="${WORKDIR}/output_a0_full112_k20_gh3_chunk_c8_${SLURM_JOB_ID}"
else
  die "OUTROOT unset and no SLURM_ARRAY_JOB_ID/SLURM_JOB_ID; export OUTROOT explicitly"
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 1 ] || die "bad array index ${IDX}"
U8_PROP_IDS="1,2,3,4,5,6,7,8,9"
HARMONIC_COUNT=5
if [ "${IDX}" -eq 0 ]; then
  PROP_IDS="1,2,3,4,5"
  CHUNK_SIZE=5
  ICHUNK=0
else
  PROP_IDS="6,7,8,9"
  CHUNK_SIZE=4
  ICHUNK=1
fi

CASE=$(printf "sv_r2p5_th180_plusN_c%02d" "${ICHUNK}")
OUTDIR="${OUTROOT}/${CASE}"

for command_name in sha256sum awk grep sed mktemp diff; do
  require_cmd "${command_name}"
done

[ -f "${REF_CHUNK_DIR}/run_metadata.txt" ] || die "missing REF metadata: ${REF_CHUNK_DIR}/run_metadata.txt"
[ -f "${REF_CHUNK_DIR}/SUCCESS" ] || die "REF chunk missing SUCCESS: ${REF_CHUNK_DIR}"
grep -q '^status=PASS$' "${REF_CHUNK_DIR}/run_status.txt" \
  || die "REF chunk run_status is not PASS"
[ -f "${U8_DIR}/SUCCESS" ] || die "U8 missing SUCCESS: ${U8_DIR}"
grep -q '^status=PASS$' "${U8_DIR}/run_status.txt" || die "U8 run_status is not PASS"
[ -f "${U8_NML_REF}" ] || die "missing U8 nml: ${U8_NML_REF}"

REF_META="${REF_CHUNK_DIR}/run_metadata.txt"
BIN="$(meta_get_file binary "${REF_META}")"
BIN_SHA_REF="$(meta_get_file binary_sha256 "${REF_META}")"
SRC_SHA_REF="$(meta_get_file source_sha256 "${REF_META}")"
BUILD_KEY_REF="$(meta_get_file build_key "${REF_META}")"
NODES_SHA_REF="$(meta_get_file nodes_sha256 "${REF_META}")"
TB_SHA_REF="$(meta_get_file tb_sha256 "${REF_META}")"
TMPL_SHA_REF="$(meta_get_file template_sha256 "${REF_META}")"
REPO_REF="$(meta_get_file repo "${REF_META}")"
U8_BIN_SHA="$(meta_get_file binary_sha256 "${U8_DIR}/run_metadata.txt")"
U8_OMP="$(meta_get_file omp_num_threads "${U8_DIR}/run_metadata.txt")"

[ -n "${BIN}" ] || die "REF metadata missing binary="
[ -x "${BIN}" ] || die "REF binary not executable: ${BIN}"
[ "$(sha256_file "${BIN}")" = "${BIN_SHA_REF}" ] \
  || die "binary sha mismatch vs REF metadata"
[ "${U8_BIN_SHA}" = "${BIN_SHA_REF}" ] \
  || die "U8 binary_sha256 != REF (cannot form matched C8/U8 pair)"
[ "${U8_OMP}" = "8" ] || die "U8 omp_num_threads='${U8_OMP}' expected 8"

mkdir -p "${OUTDIR}"
RUN_COMPLETED=0
on_exit() {
  rc=$?
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
echo " A0 full112 GH3@k20 CHUNKED C8 (Plan B)"
echo " case=${CASE} chunk=${ICHUNK} propagate_ids=${PROP_IDS}"
echo " REF_CHUNK_DIR=${REF_CHUNK_DIR}"
echo " U8_DIR=${U8_DIR}"
echo " binary=${BIN}"
echo " binary_sha256=${BIN_SHA_REF}"
echo " partition=part_2 cpus=${SLURM_CPUS_PER_TASK:-8} mem=56G wall=24h"
echo " OUTDIR=${OUTDIR}"
echo " host=$(hostname) date=$(date)"
echo "============================================="

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${NTHREADS}"
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

# Nodes from REF (same as U8); nml from U8 baseline, only rewrite propagate_ids.
cp -f "${REF_CHUNK_DIR}/nodes_manifest.input.dat" "${OUTDIR}/nodes_manifest.input.dat"
[ "$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")" = "${NODES_SHA_REF}" ] \
  || die "copied nodes_manifest sha drifted"
[ "$(sha256_file "${TB}")" = "${TB_SHA_REF}" ] \
  || die "TB sha mismatch vs REF"

cp -f "${U8_NML_REF}" "${OUTDIR}/input.nml.ref"
if grep -Eq "^\s*bsv_propagate_ids\s*=" "${OUTDIR}/input.nml.ref"; then
  sed -E "s|^([[:space:]]*bsv_propagate_ids[[:space:]]*=[[:space:]]*)['\"]?[^'\"]*['\"]?|\\1'${PROP_IDS}'|" \
    "${OUTDIR}/input.nml.ref" > "${OUTDIR}/input.nml"
else
  die "U8 input.nml missing bsv_propagate_ids"
fi

norm_diff="$(mktemp)"
diff -u \
  <(sed -E 's/\r$//' "${OUTDIR}/input.nml.ref") \
  <(sed -E 's/\r$//' "${OUTDIR}/input.nml") \
  > "${norm_diff}" || true
changed_lines="$(grep -E '^[+-]' "${norm_diff}" | grep -Ev '^[+-]{3}' || true)"
echo "${changed_lines}" > "${OUTDIR}/input_nml_normalized.diff"
n_changed="$(printf '%s\n' "${changed_lines}" | grep -c . || true)"
[ "${n_changed}" -eq 2 ] \
  || die "normalized input.nml diff is not a single-line propagate_ids change (n=${n_changed})"
printf '%s\n' "${changed_lines}" | grep -q "bsv_propagate_ids" \
  || die "normalized diff does not mention bsv_propagate_ids"
printf '%s\n' "${changed_lines}" | grep -E '^-' | grep -Fq "${U8_PROP_IDS}" \
  || die "normalized diff missing old U8 id list ${U8_PROP_IDS}"
printf '%s\n' "${changed_lines}" | grep -E '^\+' | grep -Fq "${PROP_IDS}" \
  || die "normalized diff missing new chunk id list ${PROP_IDS}"
rm -f "${norm_diff}"

ln -sfn "${BIN}" "${OUTDIR}/hhg_sbe.pinned"
[ "$(sha256_file "${OUTDIR}/hhg_sbe.pinned")" = "${BIN_SHA_REF}" ] \
  || die "pinned binary sha mismatch"

{
  echo "case=${CASE}"
  echo "campaign=full112_gh3_chunk_c8_vs_u8"
  echo "label=C8"
  echo "pair_unchunked_label=U8"
  echo "pair_unchunked_dir=${U8_DIR}"
  echo "model=full112"
  echo "chunk_index=${ICHUNK}"
  echo "propagate_ids=${PROP_IDS}"
  echo "note=exact_binary_from_28861_c00_no_recompile_omp8_matched_to_U8_28986"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-NA}"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "repo=${REPO_REF}"
  echo "source_sha256=${SRC_SHA_REF}"
  echo "build_key=${BUILD_KEY_REF}"
  echo "binary=${BIN}"
  echo "binary_sha256=${BIN_SHA_REF}"
  echo "nodes_sha256=${NODES_SHA_REF}"
  echo "tb_file=${TB}"
  echo "tb_sha256=${TB_SHA_REF}"
  echo "template_sha256=${TMPL_SHA_REF}"
  echo "input_sha256=$(sha256_file "${OUTDIR}/input.nml")"
  echo "nk=20"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "squeeze_r=2.5"
  echo "squeeze_theta_deg=180"
  echo "I_bar=1.0e11"
  echo "harmonics=2,5,7,9,10"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
  echo "omp_places=${OMP_PLACES}"
  echo "omp_proc_bind=${OMP_PROC_BIND}"
} > "${OUTDIR}/run_metadata.txt"

for key in binary_sha256 source_sha256 build_key nodes_sha256 tb_sha256; do
  got="$(meta_get_file "${key}" "${OUTDIR}/run_metadata.txt")"
  exp="$(meta_get_file "${key}" "${REF_META}")"
  [ "${got}" = "${exp}" ] || die "metadata pin drift ${key}: ${got} vs ${exp}"
done

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${BIN}" input.nml | tee run.log

grep -Eq '^pass[[:space:]]*=[[:space:]]*T[[:space:]]*$' nodes_moment_check.txt \
  || die "full-manifest moment check failed"

CRITICAL_OUTPUTS=(
  HHG_nodes_modes.dat
  chunk_info.txt
  chunk_weighted_spectrum.dat
  nodes_moment_check.txt
  run.log
  run_metadata.txt
  input.nml
  nodes_manifest.input.dat
)
for f in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${f}" ] || die "missing/empty critical output: ${f}"
done
EXPECTED_MODE_ROWS=$((CHUNK_SIZE * HARMONIC_COUNT))
MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
[ "${MODE_ROWS}" -eq "${EXPECTED_MODE_ROWS}" ] \
  || die "HHG_nodes_modes rows=${MODE_ROWS}, expected ${EXPECTED_MODE_ROWS}"

[ "$(sha256_file "${BIN}")" = "${BIN_SHA_REF}" ] || die "binary sha changed during job"
[ "$(sha256_file nodes_manifest.input.dat)" = "${NODES_SHA_REF}" ] \
  || die "nodes_manifest sha changed during job"

sha256sum "${CRITICAL_OUTPUTS[@]}" > output_sha256.txt
touch SUCCESS
{
  echo "status=PASS"
  echo "propagate_ids=${PROP_IDS}"
  echo "chunk_index=${ICHUNK}"
  echo "binary_sha256=${BIN_SHA_REF}"
  echo "source_sha256=${SRC_SHA_REF}"
  echo "build_key=${BUILD_KEY_REF}"
  echo "finished_at=$(date --iso-8601=seconds 2>/dev/null || date)"
} > run_status.txt
RUN_COMPLETED=1

echo "DONE ${CASE} propagate_ids=${PROP_IDS}"
echo "NEXT (after BOTH chunks PASS):"
echo "  PY=/public/home/wangjs/software/miniconda3/bin/python3.9"
echo "  \${PY} tools/analysis/merge_chunked_ensemble.py \\"
echo "    --manifest ${OUTROOT}/sv_r2p5_th180_plusN_c00/nodes_manifest.input.dat \\"
echo "    --chunk-dirs ${OUTROOT}/sv_r2p5_th180_plusN_c00 ${OUTROOT}/sv_r2p5_th180_plusN_c01 \\"
echo "    --outdir ${OUTROOT}/merged_plusN"
echo "  \${PY} tools/analysis/compare_chunked_vs_full.py \\"
echo "    --merged ${OUTROOT}/merged_plusN \\"
echo "    --full-run-dir ${U8_DIR} \\"
echo "    --rtol-modes 1.0e-12 --rtol-continuum 2.0e-7 \\"
echo "    --report ${OUTROOT}/compare_C8_vs_U8_strict.json"
echo "Gate label: STRICT_CHUNK_EQUIVALENCE @ OMP=8 (Plan B); GH5 still closed."
