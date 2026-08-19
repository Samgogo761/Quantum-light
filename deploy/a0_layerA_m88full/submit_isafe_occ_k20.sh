#!/bin/bash
# Submit full112 I_safe occupation scan on part_1 / k20.
# Refuses unless I_SAFE_SUBMIT=I_UNDERSTAND_CLASSICAL_OCC_SCAN_K20_ONLY
# Does not submit GH7, k40, or k50.
set -euo pipefail
if [ "${I_SAFE_SUBMIT:-}" != "I_UNDERSTAND_CLASSICAL_OCC_SCAN_K20_ONLY" ]; then
  echo "REFUSED: I_safe occupation scan is not auto-submitted." >&2
  echo "Export I_SAFE_SUBMIT=I_UNDERSTAND_CLASSICAL_OCC_SCAN_K20_ONLY" >&2
  exit 2
fi
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-d1a1431}"
SCRIPT="${REPO}/deploy/a0_layerA_m88full/sbatch_full112_isafe_occ_k20.sh"
RUNDIR="${REPO}/runs/a0_full112_isafe_occ_k20"
PINNED_BIN="${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe"
PINNED_SHA="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
PINNED_TB="66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18"
[ -f "${SCRIPT}" ] || { echo "missing ${SCRIPT}" >&2; exit 1; }
[ -x "${PINNED_BIN}" ] || { echo "missing pinned binary" >&2; exit 1; }
got="$(sha256sum "${PINNED_BIN}" | awk '{print $1}')"
[ "${got}" = "${PINNED_SHA}" ] || { echo "binary SHA mismatch: ${got}" >&2; exit 1; }
tbgot="$(sha256sum "${TB_PLUS}" | awk '{print $1}')"
[ "${tbgot}" = "${PINNED_TB}" ] || { echo "TB SHA mismatch: ${tbgot}" >&2; exit 1; }
git_in_repo() {
  git --git-dir="${REPO}/.git" --work-tree="${REPO}" "$@"
}
HEAD="$(git_in_repo rev-parse HEAD)"
SHORT="$(git_in_repo rev-parse --short=12 HEAD)"
OUTROOT="${RUNDIR}/output_a0_full112_k20_isafe_occ_${SHORT}"
mkdir -p "${RUNDIR}"
if ! mkdir "${OUTROOT}"; then
  echo "REFUSED: campaign OUTROOT already reserved: ${OUTROOT}" >&2
  exit 1
fi
cd "${RUNDIR}"
unset WORKDIR || true
set +e
JOBID="$(sbatch --parsable --chdir="${RUNDIR}" --export=ALL,OUTROOT="${OUTROOT}",REPO="${REPO}",NK=20,I_SAFE_SUBMIT="${I_SAFE_SUBMIT}",TB_PLUS="${TB_PLUS}" "${SCRIPT}")"
SBATCH_RC=$?
set -e
if [ "${SBATCH_RC}" -ne 0 ] || [ -z "${JOBID}" ]; then
  if [ -d "${OUTROOT}" ] && [ -z "$(ls -A "${OUTROOT}" 2>/dev/null || true)" ]; then
    rmdir "${OUTROOT}"
  fi
  echo "REFUSED: sbatch failed" >&2
  exit 1
fi
{
  echo "${JOBID}"
  echo "campaign=full112_isafe_occ_k20"
  echo "repo=${REPO}"
  echo "git_head=${HEAD}"
  echo "outroot=${OUTROOT}"
  echo "partition=part_1"
  echo "array=0-5"
  echo "nk=20"
  echo "note=classical_I_scan_occupation_only; not_GH7"
} | tee "${RUNDIR}/job_id.txt"
echo "submitted ${JOBID} chdir=${RUNDIR} outroot=${OUTROOT}"
