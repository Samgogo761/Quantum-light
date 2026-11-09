#!/bin/bash
set -eo pipefail
export PATH=/usr/bin:/bin:/public/home/wangjs/software/miniconda3/bin:$PATH
QE=/public/home/wangjs/software/q-e-qe-7.2
OUT=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf/logs
mkdir -p "$OUT"

echo "=== pw2wannier90 symbols ==="
grep -n atom_proj "$QE/PP/src/pw2wannier90.f90" > "$OUT/qe72_pw2wan_atom_proj.txt" || true
grep -n radialpart "$QE/PP/src/pw2wannier90.f90" > "$OUT/qe72_pw2wan_radialpart.txt"
grep -n generate_guiding "$QE/PP/src/pw2wannier90.f90" > "$OUT/qe72_pw2wan_gf.txt"
wc -l "$QE/PP/src/pw2wannier90.f90" > "$OUT/qe72_pw2wan_wc.txt"

echo "=== INPUT_PW2WANNIER ==="
ls "$QE/PP/Doc" > "$OUT/qe72_pp_doc_ls.txt"
if [ -f "$QE/PP/Doc/INPUT_PW2WANNIER.html" ]; then
  grep -n -i atom "$QE/PP/Doc/INPUT_PW2WANNIER.html" | head -50 > "$OUT/qe72_input_pw2wan_atom.txt" || true
fi
if [ -f "$QE/PP/Doc/INPUT_PW2WANNIER.txt" ]; then
  grep -n -i atom "$QE/PP/Doc/INPUT_PW2WANNIER.txt" | head -50 > "$OUT/qe72_input_pw2wan_atom.txt" || true
fi

echo "=== projwfc pawproj ==="
grep -n pawproj "$QE/PP/src/projwfc.f90" > "$OUT/qe72_projwfc_pawproj.txt" || true

echo "=== UPF inspect ==="
PY=/public/home/wangjs/software/miniconda3/bin/python
SAVE=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
$PY "$SAVE/inspect_upf_paw_partial_waves.py" \
  "$SAVE/temp_a1h_scf_clean/CrI3_scf_soc.save/Cr.rel-pbe.upf" \
  "$SAVE/temp_a1h_scf_clean/CrI3_scf_soc.save/I.rel-pbe.upf" \
  --out "$OUT/upf_paw_partial_waves.json"

echo DONE
