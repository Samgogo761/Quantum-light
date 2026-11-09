#!/bin/bash
set -eo pipefail
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH
QE=/public/home/wangjs/software/q-e-qe-7.2
OUT=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/logs
mkdir -p "$OUT"
F="$QE/PP/src/pw2wannier90.f90"
echo "file $F"
wc -l "$F" | tee "$OUT/qe72_pw2wan_wc.txt"
grep -n atom_proj "$F" > "$OUT/qe72_pw2wan_atom_proj.txt" || true
grep -n radialpart "$F" > "$OUT/qe72_pw2wan_radialpart.txt"
grep -n generate_guiding "$F" > "$OUT/qe72_pw2wan_gf.txt"
grep -n scdm_proj "$F" > "$OUT/qe72_pw2wan_scdm.txt" || true
ls "$QE/PP/Doc" > "$OUT/qe72_pp_doc_ls.txt"
echo GREP_DONE
