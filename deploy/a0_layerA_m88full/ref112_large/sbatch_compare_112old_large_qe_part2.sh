#!/bin/bash
#SBATCH --job-name=c112L_cmp
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#
# Login-node-safe physics comparison: 112_old vs 112_large vs QE (32911 + mesh eig).
# Reads two *_tb.dat and two .chk; no SBE, no 232, no rerun of 32990.

set -euo pipefail

PY=/public/home/wangjs/software/miniconda3/bin/python
REPO=/public/home/wangjs/project/New_SBEs/Quantum-light
ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/ref112_large"
OUT="${WORKDIR}/logs/compare_112old_large_qe.json"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} compare_112old_large_qe $(date)"
test -x "${PY}"
test -f "${REPO}/tools/analysis/compare_112old_large_qe.py"
test -f "${ROOT}/wannier/CrI3_tb.dat"
test -f "${WORKDIR}/CrI3_112large_tb.dat"
test -f "${ROOT}/wannier/CrI3.chk"
test -f "${WORKDIR}/CrI3_112large.chk"
test -f "${ROOT}/window144_hilbert/ref232_count/logs/CrI3_nscf_soc_count.out"
test -f "${ROOT}/wannier/CrI3.eig"
test -f "${WORKDIR}/CrI3_112large.eig"

# Refuse to treat this as a 232 or A1-H 200-band job.
! grep -q 'num_wann = 232' "${WORKDIR}/CrI3_112large.win"
grep -q 'num_wann = 112' "${WORKDIR}/CrI3_112large.win"
grep -q 'num_bands = 400' "${WORKDIR}/CrI3_112large.win"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" "${REPO}/tools/analysis/compare_112old_large_qe.py" \
  --nscf "${ROOT}/window144_hilbert/ref232_count/logs/CrI3_nscf_soc_count.out" \
  --tb-old "${ROOT}/wannier/CrI3_tb.dat" \
  --tb-large "${WORKDIR}/CrI3_112large_tb.dat" \
  --chk-old "${ROOT}/wannier/CrI3.chk" \
  --chk-large "${WORKDIR}/CrI3_112large.chk" \
  --eig-old "${ROOT}/wannier/CrI3.eig" \
  --eig-large "${WORKDIR}/CrI3_112large.eig" \
  --out "${OUT}"

test -s "${OUT}"
grep -q '"schema": "compare_112old_large_qe_v1"' "${OUT}"
echo "DONE ${OUT} $(date)"
