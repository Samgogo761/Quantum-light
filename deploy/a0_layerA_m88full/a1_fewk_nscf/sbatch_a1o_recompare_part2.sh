#!/bin/bash
#SBATCH --job-name=a1o_recompare
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=56G
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=04:00:00
#
# Re-run A1-O compare only. MMN from 32450 is kept. Full-window polar
# alignment had wiped Berry r to ~1e-12; this uses diagonal phase alignment.

set -euo pipefail
cd /public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
test -f CrI3_a1o.mmn
test -f compare_a1o_berry_r.py
test -f a1o_berry_math.py
if [ -f logs/a1o_berry_r.json ] && [ ! -f logs/a1o_berry_r.32450_polar_bug.json ]; then
  cp -f logs/a1o_berry_r.json logs/a1o_berry_r.32450_polar_bug.json
fi
PY=/public/home/wangjs/software/miniconda3/bin/python
TB=/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/CrI3_window144_tb.dat
echo "start $(date) host=$(hostname)"
$PY compare_a1o_berry_r.py \
  --mmn CrI3_a1o.mmn \
  --eig CrI3_a1o.eig \
  --nnkp CrI3_a1o.nnkp \
  --star-map a1o_star_map.json \
  --tb "$TB" \
  --out logs/a1o_berry_r.json
echo "done $(date)"
