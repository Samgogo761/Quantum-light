#!/bin/bash
# Refuses unless GH7_TAIL_SUBMIT=I_UNDERSTAND_STAGED_K20_ONLY
# Does not approve k40 or full GH7.
set -euo pipefail
if [ "${GH7_TAIL_SUBMIT:-}" != "I_UNDERSTAND_STAGED_K20_ONLY" ]; then
  echo "REFUSED: GH7 tail k20 is staged and not auto-submitted." >&2
  echo "After checkpoint + script audit, export GH7_TAIL_SUBMIT=I_UNDERSTAND_STAGED_K20_ONLY" >&2
  exit 2
fi
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3}"
SCRIPT="${REPO}/deploy/a0_layerA_m88full/sbatch_full112_gh7_tail_k20.sh"
RUNDIR="${REPO}/runs/a0_full112_gh7_tail_k20"
OUTROOT="${RUNDIR}/output_a0_full112_k20_gh7_tail"
PINNED_BIN="${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe"
PINNED_SHA="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
[ -f "${SCRIPT}" ] || { echo "missing ${SCRIPT}" >&2; exit 1; }
[ -x "${PINNED_BIN}" ] || { echo "missing pinned binary" >&2; exit 1; }
got="$(sha256sum "${PINNED_BIN}" | awk '{print $1}')"
[ "${got}" = "${PINNED_SHA}" ] || { echo "binary SHA mismatch" >&2; exit 1; }
mkdir -p "${RUNDIR}" "${OUTROOT}"
cd "${RUNDIR}"
unset WORKDIR || true
JOBID="$(sbatch --parsable --chdir="${RUNDIR}" --export=ALL,OUTROOT="${OUTROOT}",REPO="${REPO}",NK=20 "${SCRIPT}")"
echo "${JOBID}" | tee "${RUNDIR}/job_id.txt"
echo "submitted ${JOBID} chdir=${RUNDIR} outroot=${OUTROOT}"
