#!/bin/bash
# Refuses unless GH7_TAIL_SUBMIT=I_UNDERSTAND_STAGED_K20_ONLY
# Does not approve k40 or full GH7.
# Direct sbatch of the batch file is also refused unless this same token is exported.
set -euo pipefail
if [ "${GH7_TAIL_SUBMIT:-}" != "I_UNDERSTAND_STAGED_K20_ONLY" ]; then
  echo "REFUSED: GH7 tail k20 is staged and not auto-submitted." >&2
  echo "After checkpoint + script audit, export GH7_TAIL_SUBMIT=I_UNDERSTAND_STAGED_K20_ONLY" >&2
  exit 2
fi
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3}"
# shellcheck disable=SC1091
source "${REPO}/deploy/a0_layerA_m88full/resolve_python3.sh"
PYTHON3="${A0_PYTHON3}"
[ -x "${PYTHON3}" ] || { echo "A0_PYTHON3 not executable: ${PYTHON3}" >&2; exit 1; }
SCRIPT="${REPO}/deploy/a0_layerA_m88full/sbatch_full112_gh7_tail_k20.sh"
FREEZE="${REPO}/deploy/a0_layerA_m88full/gh7_tail/FREEZE.json"
FREEZE_SHA_FILE="${REPO}/deploy/a0_layerA_m88full/gh7_tail/FREEZE.sha256"
RUNDIR="${REPO}/runs/a0_full112_gh7_tail_k20"
PINNED_BIN="${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe"
PINNED_SHA="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
[ -f "${SCRIPT}" ] || { echo "missing ${SCRIPT}" >&2; exit 1; }
[ -f "${FREEZE}" ] || { echo "missing ${FREEZE}" >&2; exit 1; }
[ -f "${FREEZE_SHA_FILE}" ] || { echo "missing ${FREEZE_SHA_FILE}" >&2; exit 1; }
[ -x "${PINNED_BIN}" ] || { echo "missing pinned binary" >&2; exit 1; }
got="$(sha256sum "${PINNED_BIN}" | awk '{print $1}')"
[ "${got}" = "${PINNED_SHA}" ] || { echo "binary SHA mismatch" >&2; exit 1; }
HEAD="$(git -C "${REPO}" rev-parse HEAD)"
PIN_BASE="$("${PYTHON3}" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["freeze_pin_base_head"])' "${FREEZE}")"
[ "${PIN_BASE}" != "TO_BE_PINNED" ] || { echo "REFUSED: FREEZE.freeze_pin_base_head is not pinned" >&2; exit 1; }
if [ "${HEAD}" != "${PIN_BASE}" ]; then
  if git -C "${REPO}" diff --quiet "${PIN_BASE}" HEAD -- . \
      ':!deploy/a0_layerA_m88full/gh7_tail/FREEZE.json' \
      ':!deploy/a0_layerA_m88full/gh7_tail/FREEZE.sha256'; then
    echo "HEAD ${HEAD} is a freeze_pin_base_head pin-successor of ${PIN_BASE}"
  else
    echo "REFUSED: git HEAD ${HEAD} is not freeze_pin_base_head ${PIN_BASE} or a FREEZE-only successor" >&2
    exit 1
  fi
fi
SHORT="$(git -C "${REPO}" rev-parse --short=12 HEAD)"
OUTROOT="${RUNDIR}/output_a0_full112_k20_gh7_tail_${SHORT}"
mkdir -p "${RUNDIR}"
if ! mkdir "${OUTROOT}"; then
  echo "REFUSED: campaign OUTROOT already reserved: ${OUTROOT}" >&2
  exit 1
fi
cd "${RUNDIR}"
unset WORKDIR || true
set +e
JOBID="$(sbatch --parsable --chdir="${RUNDIR}" --export=ALL,OUTROOT="${OUTROOT}",REPO="${REPO}",NK=20,GH7_TAIL_SUBMIT="${GH7_TAIL_SUBMIT}",A0_PYTHON3="${PYTHON3}" "${SCRIPT}")"
SBATCH_RC=$?
set -e
if [ "${SBATCH_RC}" -ne 0 ] || [ -z "${JOBID}" ]; then
  if [ -d "${OUTROOT}" ] && [ -z "$(ls -A "${OUTROOT}" 2>/dev/null || true)" ]; then
    rmdir "${OUTROOT}"
  fi
  echo "REFUSED: sbatch failed" >&2
  exit 1
fi
echo "${JOBID}" | tee "${RUNDIR}/job_id.txt"
echo "submitted ${JOBID} chdir=${RUNDIR} outroot=${OUTROOT}"
