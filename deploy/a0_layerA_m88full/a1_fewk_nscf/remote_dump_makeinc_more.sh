#!/bin/bash
set -eo pipefail
export PATH=/usr/bin:/bin:$PATH
QE=/public/home/wangjs/software/q-e-qe-7.2
OUT=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/logs
sed -n '1,70p' "$QE/PP/src/Makefile" > "$OUT/qe72_pp_makefile_head.txt"
sed -n '180,230p' "$QE/make.inc" > "$OUT/qe72_make.inc.qelibs.txt"
grep FOX "$QE/make.inc" > "$OUT/qe72_make.inc.fox.txt" || true
echo MORE_OK
