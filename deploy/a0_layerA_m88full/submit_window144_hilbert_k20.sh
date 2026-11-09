#!/bin/bash
# Submit WINDOW144 k20 classical contrast.
# Default: part_2 (workspace 2) while part_1 has no idle nodes.
# Refuses unless WINDOW144_SUBMIT=I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY
# Does not submit GH7, k40, k50, or ±N.
set -euo pipefail
if [ "${WINDOW144_SUBMIT:-}" != "I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY" ]; then
  echo "REFUSED: WINDOW144 k20 contrast is not auto-submitted." >&2
  echo "Export WINDOW144_SUBMIT=I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY" >&2
  exit 2
fi
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-d1a1431}"
SCRIPT="${SCRIPT:-/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/sbatch_window144_hilbert_k20_part2.sh}"
RUNDIR="${RUNDIR:-/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/runs}"
PINNED_BIN="${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe"
PINNED_SHA="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
TB_W144="${TB_W144:-/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/CrI3_window144_tb.dat}"
GATE_JSON="${GATE_JSON:-/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/build_gate.json}"
[ -f "${SCRIPT}" ] || { echo "missing ${SCRIPT}" >&2; exit 1; }
[ -x "${PINNED_BIN}" ] || { echo "missing pinned binary" >&2; exit 1; }
[ -f "${TB_W144}" ] || { echo "missing WINDOW144 tb: ${TB_W144}" >&2; exit 1; }
got="$(sha256sum "${PINNED_BIN}" | awk '{print $1}')"
[ "${got}" = "${PINNED_SHA}" ] || { echo "binary SHA mismatch: ${got}" >&2; exit 1; }
if [ -f "${GATE_JSON}" ]; then
  if grep -q '"OVERALL": "FAIL"' "${GATE_JSON}"; then
    echo "REFUSED: build_gate OVERALL=FAIL" >&2
    exit 2
  fi
  echo "gate file ${GATE_JSON} present (OVERALL is not FAIL)"
fi
git_in_repo() {
  git --git-dir="${REPO}/.git" --work-tree="${REPO}" "$@"
}
HEAD="$(git_in_repo rev-parse HEAD)"
SHORT="$(git_in_repo rev-parse --short=12 HEAD)"
mkdir -p "${RUNDIR}"
OUTROOT="${RUNDIR}/output_a0_window144_k20_${SHORT}_part2"
if ! mkdir "${OUTROOT}"; then
  echo "REFUSED: campaign OUTROOT already reserved: ${OUTROOT}" >&2
  exit 1
fi
cd "${RUNDIR}"
unset WORKDIR || true
set +e
JOBID="$(sbatch --parsable --chdir="${RUNDIR}" --export=ALL,OUTROOT="${OUTROOT}",REPO="${REPO}",NK=20,WINDOW144_SUBMIT="${WINDOW144_SUBMIT}",TB_W144="${TB_W144}" "${SCRIPT}")"
SBATCH_RC=$?
set -e
if [ "${SBATCH_RC}" -ne 0 ] || [ -z "${JOBID}" ]; then
  rmdir "${OUTROOT}" 2>/dev/null || true
  echo "sbatch failed rc=${SBATCH_RC}" >&2
  exit 1
fi
echo "SUBMITTED job=${JOBID} OUTROOT=${OUTROOT}"
echo "${JOBID}" > "${OUTROOT}/slurm_job_id.txt"
echo "IMPACT_ON_HHG=UNRESOLVED until J2/J5/J7/J9 are compared"
