#!/bin/bash
# Isolate 112_[-6,13.2] window-only control. Reuse 112_large AMN/MMN/EIG/nnkp.
# Do not cancel 232 job. Do not write temp_scf_soc or temp_a1h_scf_clean.
set -euo pipefail

ROOT=/public/home/wangjs/project/CrI3_TB
A1SAVE="${ROOT}/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
L112="${ROOT}/window144_hilbert/ref112_large"
DST="${ROOT}/window144_hilbert/ref112_13p2"
QL=/public/home/wangjs/project/New_SBEs/Quantum-light
PY=/public/home/wangjs/software/miniconda3/bin/python

count_wfc() {
  find "$1" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]'
}

case "$DST" in
  *temp_scf_soc*|*temp_a1h_scf_clean*) echo "REFUSE: dest path forbidden" >&2; exit 3 ;;
esac
test "$DST" != "$L112"

mkdir -p "$DST/logs"
cd "$DST"

test -f "$L112/CrI3_112large.win"
test -f "$L112/CrI3_112large.amn"
test -f "$L112/CrI3_112large.mmn"
test -f "$L112/CrI3_112large.eig"
test -f "$L112/CrI3_112large.nnkp"
test "$(count_wfc "$A1SAVE")" -eq 52
test "$(count_wfc "$L112/temp_nscf576/CrI3_scf_soc.save")" -eq 576

"$PY" "$QL/tools/analysis/build_112_13p2_inputs.py" \
  --win-112large "$L112/CrI3_112large.win" \
  --out-win "$DST/CrI3_112_13p2.win" \
  --out-json "$DST/logs/build_112_13p2_inputs.json"

grep -q "num_wann = 112" CrI3_112_13p2.win
grep -q "num_bands = 400" CrI3_112_13p2.win
grep -q "dis_win_max  = 13.2" CrI3_112_13p2.win
grep -q "dis_win_min  = -6.0" CrI3_112_13p2.win
grep -q "dis_froz_min = -1.5" CrI3_112_13p2.win
grep -q "I:   p" CrI3_112_13p2.win
! grep -q "num_wann = 232" CrI3_112_13p2.win
! grep -q "I:   d" CrI3_112_13p2.win
! grep -q "dis_win_max  =  8.0" CrI3_112_13p2.win

ln -f "$L112/CrI3_112large.amn" "$DST/CrI3_112_13p2.amn"
ln -f "$L112/CrI3_112large.mmn" "$DST/CrI3_112_13p2.mmn"
ln -f "$L112/CrI3_112large.eig" "$DST/CrI3_112_13p2.eig"
ln -f "$L112/CrI3_112large.nnkp" "$DST/CrI3_112_13p2.nnkp"
test -f CrI3_112_13p2.amn
test -f CrI3_112_13p2.mmn
test -f CrI3_112_13p2.eig
test -f CrI3_112_13p2.nnkp
grep -q "begin spinor_projections" CrI3_112_13p2.nnkp
! grep -q "temp_scf_soc" CrI3_112_13p2.win
! grep -q "temp_a1h_scf_clean" CrI3_112_13p2.win
test "$(count_wfc "$A1SAVE")" -eq 52

if [ -f "$QL/deploy/a0_layerA_m88full/ref112_13p2/sbatch_112_13p2_part2.sh" ]; then
  cp -f "$QL/deploy/a0_layerA_m88full/ref112_13p2/sbatch_112_13p2_part2.sh" "$DST/sbatch_112_13p2_part2.sh"
fi

echo PREP_112_13P2_OK
