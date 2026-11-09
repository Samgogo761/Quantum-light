#!/bin/bash
# Submit hardened GH5-v2 on part_1 / OMP=36 into a unique rundir.
# Requires C8↔C36 topology PASS and pinned 29175 binary.
set -euo pipefail
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3}"
SCRIPT="${REPO}/deploy/a0_layerA_m88full/sbatch_full112_gh5_candidates_k20_chunked.sh"
RUNDIR="${REPO}/runs/a0_full112_gh5_tbv2_omp36"
OUTROOT="${RUNDIR}/output_a0_full112_k20_gh5_tbv2_omp36"
PINNED_BIN="${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe"
PINNED_SHA="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"

[ -f "${SCRIPT}" ] || { echo "missing ${SCRIPT}" >&2; exit 1; }
[ -x "${PINNED_BIN}" ] || { echo "missing pinned binary ${PINNED_BIN}" >&2; exit 1; }
got="$(sha256sum "${PINNED_BIN}" | awk '{print $1}')"
[ "${got}" = "${PINNED_SHA}" ] || { echo "binary SHA ${got} != ${PINNED_SHA}" >&2; exit 1; }

mkdir -p "${RUNDIR}" "${OUTROOT}"
cd "${RUNDIR}"
# Do not export login WORKDIR. Job script sets WORKDIR="$(pwd)".
unset WORKDIR || true
JOBID="$(sbatch --parsable --chdir="${RUNDIR}" --export=ALL,OUTROOT="${OUTROOT}",REPO="${REPO}" "${SCRIPT}")"
echo "${JOBID}" | tee "${RUNDIR}/job_id.txt"
echo "submitted ${JOBID} chdir=${RUNDIR} outroot=${OUTROOT}"
