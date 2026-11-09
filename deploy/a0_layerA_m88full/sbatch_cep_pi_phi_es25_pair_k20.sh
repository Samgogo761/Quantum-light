#!/bin/bash
#=============================================================================
# CEP-π phase-precision probe: TWO trajectories only (production T2=0.5T).
#
#   task 0: +N , node ALPHA_ID   (α)
#   task 1: -N , node ANTI_ID    (−α)
#
# Uses ES25.17 antipode-enforced manifests (new SHA; does NOT overwrite 27992).
# part_2 / OMP=8 diagnostic (not formal GH5). Wall ~4-5x vs part_1; single-node OK.
# Saves Jt_node_XXXX.dat. GH5 remains HOLD. No dephasing-off yet.
#=============================================================================
#SBATCH --job-name=a0_cep_phi_es25
#SBATCH --partition=part_2
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_cep_phi_es25_%A_%a.out
#SBATCH --error=a0_cep_phi_es25_%A_%a.err

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

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3}"
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
TB_MINUS="${TB_MINUS:-/public/home/wangjs/project/CrI3_TB_mN/wannier/CrI3_tb_mN_Pconj.dat}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_cep_pi_phi_es25_pair}"
NODES_DIR="${NODES_DIR:-${REPO}/deploy/a0_layerA_m88full/nodes_gh3_full112_candidates_es25}"
NK=20
I_BAR=1.0e11
HARMONICS_CSV="2,5,7,9,10"
HARMONIC_COUNT=5

# Strong Im-axis antipode pair on r=2.5, th=0 GH3 (α ↔ −α).
RVAL=2.5
THVAL=0
RTAG=2p5
ALPHA_ID="${ALPHA_ID:-4}"
ANTI_ID="${ANTI_ID:-6}"

NODE_VALIDATOR="${REPO}/tools/analysis/validate_qlight_nodes_production.py"
RUN_VALIDATOR="${REPO}/tools/analysis/validate_a0_run_strict.py"
NODE_VALIDATOR_CORE="${REPO}/tools/analysis/validate_qlight_nodes_strict.py"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -eq 0 ] || [ "${IDX}" -eq 1 ] || die "bad array index ${IDX}"

if [ "${IDX}" -eq 0 ]; then
  DOM="plusN"
  TB="${TB_PLUS}"
  PROP_ID="${ALPHA_ID}"
else
  DOM="minusN"
  TB="${TB_MINUS}"
  PROP_ID="${ANTI_ID}"
fi

NODES_FILE="${NODES_DIR}/nodes_sv_r${RTAG}_th${THVAL}_gh3.dat"
CASE=$(printf "sv_r%s_th%03d_%s_node%04d" "${RTAG}" "${THVAL}" "${DOM}" "${PROP_ID}")
OUTDIR="${OUTROOT}/${CASE}"

for command_name in sha256sum git find xargs awk grep mktemp; do
  require_cmd "${command_name}"
done
# shellcheck disable=SC1091
source "${REPO}/deploy/a0_layerA_m88full/resolve_python3.sh"
PYTHON3="${A0_PYTHON3}"
[ -x "${PYTHON3}" ] || die "A0_PYTHON3 not executable: ${PYTHON3}"

[ -f "${NODES_FILE}" ] || die "missing nodes file ${NODES_FILE}"
[ -f "${TB}" ] || die "TB not found: ${TB}"
[ -f "${REPO}/Makefile" ] || die "Makefile not found under REPO=${REPO}"
[ -f "${NODE_VALIDATOR}" ] || die "missing node validator: ${NODE_VALIDATOR}"
[ -f "${RUN_VALIDATOR}" ] || die "missing run validator: ${RUN_VALIDATOR}"
[ -f "${NODE_VALIDATOR_CORE}" ] || die "missing validator dependency: ${NODE_VALIDATOR_CORE}"

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
echo " CEP-pi phi-ES25 probe case=${CASE}"
echo " nodes=${NODES_FILE}"
echo " TB=${TB}"
echo " propagate_id=${PROP_ID}  (pair ${ALPHA_ID}<->${ANTI_ID})"
echo " bsv_save_jt=true  T2=0.5T  GH5=HOLD"
echo " host=$(hostname) date=$(date)"
echo "============================================="

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
    WAITED=$((WAITED + 10))
    [ "${WAITED}" -lt 1800 ] || die "timed out waiting for build lock"
  done
  if [ ! -x "${CACHED_BIN}" ]; then
    BUILD_TMP="$(mktemp -d "${BUILD_ROOT}/build.XXXXXX")"
    cp -a "${REPO}/Makefile" "${REPO}/src" "${BUILD_TMP}/"
    (
      cd "${BUILD_TMP}"
      make clean
      make FC="${BUILD_FC}" FFLAGS="${BUILD_FFLAGS}" LDFLAGS="${BUILD_LDFLAGS}" -j1
      [ -x ./hhg_sbe ] || die "build failed"
    )
    mkdir -p "${CACHE_DIR}"
    cp -f "${BUILD_TMP}/hhg_sbe" "${CACHED_BIN}"
    chmod +x "${CACHED_BIN}"
    printf '%s\n' "${SOURCE_SHA256}" > "${CACHE_DIR}/source_sha256.txt"
    rm -rf "${BUILD_TMP}"
  fi
  if [ "${LOCK_HELD}" -eq 1 ]; then
    rmdir "${LOCK}" 2>/dev/null || true
    LOCK_HELD=0
  fi
fi
[ -x "${CACHED_BIN}" ] || die "cached binary missing: ${CACHED_BIN}"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"
cp -f "${CACHED_BIN}" "${OUTDIR}/hhg_sbe"
chmod +x "${OUTDIR}/hhg_sbe"

TEMPLATE="${REPO}/deploy/a0_layerA_m88full/input_template_full112.nml"
[ -f "${TEMPLATE}" ] || die "missing template ${TEMPLATE}"
TB_SHA256="$(sha256_file "${TB}")"

# Fill template; force from_file + single-node propagate + save Jt.
sed \
  -e "s|TB_PLACEHOLDER|${TB}|g" \
  -e "s|NK_PLACEHOLDER|${NK}|g" \
  -e "s|NODES_PLACEHOLDER|nodes_manifest.input.dat|g" \
  -e "s|R_PLACEHOLDER|${RVAL}|g" \
  -e "s|TH_PLACEHOLDER|${THVAL}|g" \
  -e "s|PROPAGATE_IDS_PLACEHOLDER|${PROP_ID}|g" \
  "${TEMPLATE}" > "${OUTDIR}/input.nml"

# Ensure bsv_save_jt / save_all_complex / ics_cs overrides.
"${PYTHON3}" - <<PY
from pathlib import Path
p = Path(r"""${OUTDIR}/input.nml""")
t = p.read_text(encoding="utf-8")
repl = {
    "bsv_save_jt": ".true.",
    "bsv_save_complex": ".true.",
    "bsv_save_all_complex": ".true.",
    "bsv_save_ics_cs": ".false.",
    "bsv_write_nodes": ".false.",
}
lines = []
seen_jt = False
for raw in t.splitlines():
    s = raw.strip()
    done = False
    for k, v in repl.items():
        if s.startswith(k + " ") or s.startswith(k + "="):
            indent = raw[: len(raw) - len(raw.lstrip())]
            lines.append(f"{indent}{k} = {v}")
            if k == "bsv_save_jt":
                seen_jt = True
            done = True
            break
    if not done:
        lines.append(raw)
text = "\n".join(lines) + "\n"
if not seen_jt:
    text = text.replace(
        "bsv_save_all_complex = .true.",
        "bsv_save_all_complex = .true.\n  bsv_save_jt = .true.",
        1,
    )
p.write_text(text, encoding="utf-8")
PY

{
  echo "case=${CASE}"
  echo "repo=${REPO}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "tb_sha256=${TB_SHA256}"
  echo "nodes_sha256=${NODES_SHA256}"
  echo "validator_bundle_sha256=${VALIDATOR_BUNDLE_SHA256}"
  echo "nk=${NK}"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "model=full112"
  echo "I_bar=${I_BAR}"
  echo "harmonics=${HARMONICS_CSV}"
  echo "domain=${DOM}"
  echo "propagate_id=${PROP_ID}"
  echo "alpha_id=${ALPHA_ID}"
  echo "anti_id=${ANTI_ID}"
  echo "manifest_precision=ES25.17"
  echo "purpose=cep_pi_phi_rounding_probe"
  echo "gh5=HOLD"
} > "${OUTDIR}/run_metadata.txt"

cd "${OUTDIR}"
./hhg_sbe input.nml 2>&1 | tee run.log

[ -s HHG_nodes_modes.dat ] || die "missing HHG_nodes_modes.dat"
JT_FILE=$(printf "Jt_node_%04d.dat" "${PROP_ID}")
[ -s "${JT_FILE}" ] || die "missing ${JT_FILE} (bsv_save_jt failed?)"

EXPECTED_MODE_ROWS=${HARMONIC_COUNT}
MODE_ROWS=$(awk '!/^#/ && NF {n++} END {print n+0}' HHG_nodes_modes.dat)
[ "${MODE_ROWS}" -eq "${EXPECTED_MODE_ROWS}" ] \
  || die "HHG_nodes_modes rows=${MODE_ROWS}, expected ${EXPECTED_MODE_ROWS}"

# Chunk/single-node probe: skip ICS/CS strict validator (ics_cs disabled).
# Still require modes + Jt + SUCCESS.
printf 'status=PASS_PARTIAL_probe\nnote=ics_cs_disabled_single_node_cep_phi_probe\n' \
  > "${OUTDIR}/a0_run_validation.note"

touch SUCCESS
printf 'status=PASS\nexit_code=0\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > run_status.txt
RUN_COMPLETED=1
echo "DONE ${CASE}"
