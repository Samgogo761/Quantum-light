#!/bin/bash
#SBATCH --job-name=c112_13cmp
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_13p2
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#
# 112_[-6,8] vs finished 112_[-6,13.2]. No 232 parent, no HHG, no new QE.

set -euo pipefail

PY=/public/home/wangjs/software/miniconda3/bin/python
REPO=/public/home/wangjs/project/New_SBEs/Quantum-light
ROOT=/public/home/wangjs/project/CrI3_TB
W8="${ROOT}/window144_hilbert/ref112_large"
W13="${ROOT}/window144_hilbert/ref112_13p2"
OUT="${W13}/logs/compare_112_8_vs_13p2.json"

mkdir -p "${W13}/logs"
cd "${W13}"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} compare_112_8_vs_13p2 $(date)"
test -f "${W8}/CrI3_112large_tb.dat"
test -f "${W13}/CrI3_112_13p2_tb.dat"
test -f "${W8}/CrI3_112large.chk"
test -f "${W13}/CrI3_112_13p2.chk"
test -f "${W8}/CrI3_112large.eig"
! test -f "${ROOT}/window144_hilbert/ref232_id/CrI3_232id_tb.dat"
grep -q 'dis_win_max  = 13.2' "${W13}/CrI3_112_13p2.win"
grep -q 'num_wann = 112' "${W13}/CrI3_112_13p2.win"
! grep -q 'I:   d' "${W13}/CrI3_112_13p2.win"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"${PY}" "${REPO}/tools/analysis/compare_112_8_vs_13p2.py" \
  --nscf "${ROOT}/window144_hilbert/ref232_count/logs/CrI3_nscf_soc_count.out" \
  --tb-w8 "${W8}/CrI3_112large_tb.dat" \
  --tb-w132 "${W13}/CrI3_112_13p2_tb.dat" \
  --chk-w8 "${W8}/CrI3_112large.chk" \
  --chk-w132 "${W13}/CrI3_112_13p2.chk" \
  --eig "${W8}/CrI3_112large.eig" \
  --out "${OUT}"
test -s "${OUT}"
grep -q '"schema": "compare_112_8_vs_13p2_v1"' "${OUT}"
echo "DONE ${OUT} $(date)"
