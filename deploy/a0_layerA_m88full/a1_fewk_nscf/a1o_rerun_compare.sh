#!/bin/bash
set -euo pipefail
WD=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
cd "$WD"
test -f CrI3_a1o.mmn
test -f logs/a1o_berry_r.json
cp -f logs/a1o_berry_r.json logs/a1o_berry_r.32450_polar_bug.json
PY=/public/home/wangjs/software/miniconda3/bin/python
TB=/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/CrI3_window144_tb.dat
$PY compare_a1o_berry_r.py \
  --mmn CrI3_a1o.mmn \
  --eig CrI3_a1o.eig \
  --nnkp CrI3_a1o.nnkp \
  --star-map a1o_star_map.json \
  --tb "$TB" \
  --out logs/a1o_berry_r.json
echo COMPARE_DONE
