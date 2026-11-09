#!/bin/bash
set -eo pipefail
export PATH=/usr/bin:/bin:$PATH
QE=/public/home/wangjs/software/q-e-qe-7.2
OUT=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/logs
grep -n -E '^(MPIF90|F90FLAGS|FFLAGS|LD |LDFLAGS|QELIBS|BLAS|LAPACK|FFT|DFLAGS|MODFLAGS|MODULES)' "$QE/make.inc" > "$OUT/qe72_make.inc.vars.txt"
grep -n -E 'MODULES|MODFLAGS|QEMODS|libpp' "$QE/PP/src/Makefile" > "$OUT/qe72_pp_makefile_mods.txt"
echo VARS_OK
