#!/bin/bash
#=============================================================================
# Strict chunk-equivalence CONTROL (NOT GH5):
#   Reuse the EXACT binary + nodes + nml from GH3 smoke 28861 c00.
#   Only change: bsv_propagate_ids  1,2,3,4,5  ->  1,2,3,4,5,6,7,8,9
#   Then compare this unchunked OUTDIR <-> 28861 merged_plusN.
#
# Hard rule: do NOT recompile. "Same source" is NOT "same binary".
# DO NOT treat as GH5 clearance. CEP-π / GH5 remain closed.
#=============================================================================
#SBATCH --job-name=a0_f112_gh3_ctrl
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_f112_gh3_ctrl_%j.out
#SBATCH --error=a0_f112_gh3_ctrl_%j.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
meta_get_file() {
  # usage: meta_get_file KEY FILE
  awk -F= -v k="$1" '$1==k{print $2; exit}' "$2"
}

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SMOKE_OUTROOT="${SMOKE_OUTROOT:-/public/home/wangjs/project/New_SBEs/Quantum-light/runs/a0_full112_gh3_chunk_smoke_689d9a3/output_a0_full112_k20_gh3_chunk_smoke_28861}"
REF_CHUNK_DIR="${REF_CHUNK_DIR:-${SMOKE_OUTROOT}/sv_r2p5_th180_plusN_c00}"
MERGED_CHUNKED="${MERGED_CHUNKED:-${SMOKE_OUTROOT}/merged_plusN}"
TB="${TB:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"

if [ -n "${OUTROOT:-}" ]; then
  :
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  OUTROOT="${WORKDIR}/output_a0_full112_k20_gh3_unchunked_ctrl_${SLURM_JOB_ID}"
else
  die "OUTROOT unset and no SLURM_JOB_ID; export OUTROOT explicitly"
fi

CASE="sv_r2p5_th180_plusN_unchunked"
OUTDIR="${OUTROOT}/${CASE}"
PROP_IDS="1,2,3,4,5,6,7,8,9"
REF_PROP_IDS="1,2,3,4,5"
HARMONIC_COUNT=5
N_NODES=9

for command_name in sha256sum awk grep sed mktemp diff; do
  require_cmd "${command_name}"
done

[ -f "${REF_CHUNK_DIR}/run_metadata.txt" ] || die "missing REF metadata: ${REF_CHUNK_DIR}/run_metadata.txt"
[ -f "${REF_CHUNK_DIR}/SUCCESS" ] || die "REF chunk missing SUCCESS: ${REF_CHUNK_DIR}"
grep -q '^status=PASS$' "${REF_CHUNK_DIR}/run_status.txt" \
  || die "REF chunk run_status is not PASS"

REF_META="${REF_CHUNK_DIR}/run_metadata.txt"
BIN="$(meta_get_file binary "${REF_META}")"
BIN_SHA_REF="$(meta_get_file binary_sha256 "${REF_META}")"
SRC_SHA_REF="$(meta_get_file source_sha256 "${REF_META}")"
BUILD_KEY_REF="$(meta_get_file build_key "${REF_META}")"
NODES_SHA_REF="$(meta_get_file nodes_sha256 "${REF_META}")"
TB_SHA_REF="$(meta_get_file tb_sha256 "${REF_META}")"
REPO_REF="$(meta_get_file repo "${REF_META}")"
REF_PROP_META="$(meta_get_file propagate_ids "${REF_META}")"

[ -n "${BIN}" ] || die "REF metadata missing binary="
[ -n "${BIN_SHA_REF}" ] || die "REF metadata missing binary_sha256="
[ -n "${SRC_SHA_REF}" ] || die "REF metadata missing source_sha256="
[ -n "${BUILD_KEY_REF}" ] || die "REF metadata missing build_key="
[ -x "${BIN}" ] || die "REF binary not executable: ${BIN}"
[ "${REF_PROP_META}" = "${REF_PROP_IDS}" ] \
  || die "REF propagate_ids='${REF_PROP_META}' expected '${REF_PROP_IDS}'"

BIN_SHA_NOW="$(sha256_file "${BIN}")"
[ "${BIN_SHA_NOW}" = "${BIN_SHA_REF}" ] \
  || die "binary sha mismatch vs REF metadata: ${BIN_SHA_NOW} vs ${BIN_SHA_REF}"

[ -f "${REF_CHUNK_DIR}/nodes_manifest.input.dat" ] || die "missing REF nodes manifest"
[ -f "${REF_CHUNK_DIR}/input.nml" ] || die "missing REF input.nml"
NODES_SHA_NOW="$(sha256_file "${REF_CHUNK_DIR}/nodes_manifest.input.dat")"
[ "${NODES_SHA_NOW}" = "${NODES_SHA_REF}" ] \
  || die "REF nodes_manifest sha != metadata nodes_sha256"
TB_SHA_NOW="$(sha256_file "${TB}")"
[ "${TB_SHA_NOW}" = "${TB_SHA_REF}" ] \
  || die "TB sha mismatch vs REF metadata: ${TB_SHA_NOW} vs ${TB_SHA_REF}"

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
echo " A0 full112 GH3@k20 UNCHUNKED same-BINARY CONTROL"
echo " REF_CHUNK_DIR=${REF_CHUNK_DIR}"
echo " binary=${BIN}"
echo " binary_sha256=${BIN_SHA_REF}"
echo " source_sha256=${SRC_SHA_REF}"
echo " build_key=${BUILD_KEY_REF}"
echo " propagate_ids: ${REF_PROP_IDS} -> ${PROP_IDS}"
echo " OUTDIR=${OUTDIR}"
echo " host=$(hostname) date=$(date)"
echo "============================================="

# Exact OpenMP/MKL binding as smoke 28861
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

# Copy nodes + nml from REF; only rewrite bsv_propagate_ids.
cp -f "${REF_CHUNK_DIR}/nodes_manifest.input.dat" "${OUTDIR}/nodes_manifest.input.dat"
[ "$(sha256_file "${OUTDIR}/nodes_manifest.input.dat")" = "${NODES_SHA_REF}" ] \
  || die "copied nodes_manifest sha drifted"

cp -f "${REF_CHUNK_DIR}/input.nml" "${OUTDIR}/input.nml.ref"
# Accept both quoted and unquoted namelist forms; force the unchunked id list.
if grep -Eq "^\s*bsv_propagate_ids\s*=" "${OUTDIR}/input.nml.ref"; then
  sed -E "s|^([[:space:]]*bsv_propagate_ids[[:space:]]*=[[:space:]]*)['\"]?[^'\"]*['\"]?|\\1'${PROP_IDS}'|" \
    "${OUTDIR}/input.nml.ref" > "${OUTDIR}/input.nml"
else
  die "REF input.nml missing bsv_propagate_ids"
fi

# Normalized diff must only touch the propagate-ids line.
norm_diff="$(mktemp)"
diff -u \
  <(sed -E 's/\r$//' "${OUTDIR}/input.nml.ref") \
  <(sed -E 's/\r$//' "${OUTDIR}/input.nml") \
  > "${norm_diff}" || true
# Expect exactly one changed content line for bsv_propagate_ids (plus diff headers).
changed_lines="$(grep -E '^[+-]' "${norm_diff}" | grep -Ev '^[+-]{3}' || true)"
echo "${changed_lines}" > "${OUTDIR}/input_nml_normalized.diff"
n_changed="$(printf '%s\n' "${changed_lines}" | grep -c . || true)"
[ "${n_changed}" -eq 2 ] \
  || die "normalized input.nml diff is not a single-line propagate_ids change (n=${n_changed}); see input_nml_normalized.diff"
printf '%s\n' "${changed_lines}" | grep -q "bsv_propagate_ids" \
  || die "normalized diff does not mention bsv_propagate_ids"
printf '%s\n' "${changed_lines}" | grep -E '^-' | grep -Fq "${REF_PROP_IDS}" \
  || die "normalized diff missing old propagate id list ${REF_PROP_IDS}"
printf '%s\n' "${changed_lines}" | grep -E '^\+' | grep -Fq "${PROP_IDS}" \
  || die "normalized diff missing new propagate id list ${PROP_IDS}"
rm -f "${norm_diff}"

# Snapshot the exact binary path/hash used (no copy required; pin by sha).
ln -sfn "${BIN}" "${OUTDIR}/hhg_sbe.pinned"
[ "$(sha256_file "${OUTDIR}/hhg_sbe.pinned")" = "${BIN_SHA_REF}" ] \
  || die "pinned binary sha mismatch"

{
  echo "case=${CASE}"
  echo "campaign=full112_gh3_unchunked_samebin_control"
  echo "model=full112"
  echo "ref_chunk_dir=${REF_CHUNK_DIR}"
  echo "merged_chunked_for_compare=${MERGED_CHUNKED}"
  echo "propagate_ids=${PROP_IDS}"
  echo "ref_propagate_ids=${REF_PROP_IDS}"
  echo "note=exact_binary_from_28861_c00_no_recompile"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
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
  echo "input_sha256=$(sha256_file "${OUTDIR}/input.nml")"
  echo "input_ref_sha256=$(sha256_file "${OUTDIR}/input.nml.ref")"
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

# Hard-check metadata pins match REF for the identity fields.
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
  HHG_ics_cs.dat
  nodes_moment_check.txt
  run.log
  run_metadata.txt
  input.nml
  nodes_manifest.input.dat
)
for f in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${f}" ] || die "missing/empty critical output: ${f}"
done
EXPECTED_MODE_ROWS=$((N_NODES * HARMONIC_COUNT))
MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
[ "${MODE_ROWS}" -eq "${EXPECTED_MODE_ROWS}" ] \
  || die "HHG_nodes_modes rows=${MODE_ROWS}, expected ${EXPECTED_MODE_ROWS}"

# Re-verify binary identity after the run (no swap).
[ "$(sha256_file "${BIN}")" = "${BIN_SHA_REF}" ] \
  || die "binary sha changed during job"
[ "$(sha256_file nodes_manifest.input.dat)" = "${NODES_SHA_REF}" ] \
  || die "nodes_manifest sha changed during job"

sha256sum "${CRITICAL_OUTPUTS[@]}" > output_sha256.txt
touch SUCCESS
{
  echo "status=PASS"
  echo "propagate_ids=${PROP_IDS}"
  echo "binary_sha256=${BIN_SHA_REF}"
  echo "source_sha256=${SRC_SHA_REF}"
  echo "build_key=${BUILD_KEY_REF}"
  echo "finished_at=$(date --iso-8601=seconds 2>/dev/null || date)"
} > run_status.txt
RUN_COMPLETED=1

echo "DONE ${CASE}"
echo "NEXT strict same-binary compare (modes rtol=1e-12, continuum rtol=2e-7):"
echo "  python tools/analysis/compare_chunked_vs_full.py \\"
echo "    --merged ${MERGED_CHUNKED} \\"
echo "    --full-run-dir ${OUTDIR} \\"
echo "    --rtol-modes 1.0e-12 --rtol-continuum 2.0e-7 \\"
echo "    --report ${OUTROOT}/compare_chunked_vs_unchunked_samebin.json"
