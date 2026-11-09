#!/bin/bash
# Isolate 232_I-d reference parent. Reuse 576k nbnd=400 NSCF and 112_large MMN.
# Not production. Not chemical I-d 144. Not WINDOW144.
# Never write temp_scf_soc or temp_a1h_scf_clean.
set -euo pipefail

ROOT=/public/home/wangjs/project/CrI3_TB
A1SAVE="${ROOT}/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
L112="${ROOT}/window144_hilbert/ref112_large"
DST="${ROOT}/window144_hilbert/ref232_id"
QL=/public/home/wangjs/project/New_SBEs/Quantum-light
PY=/public/home/wangjs/software/miniconda3/bin/python
W90=/public/home/wangjs/software/q-e-qe-7.2/wannier90-3.1.0/wannier90.x

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
test -f "$L112/CrI3_112large.eig"
test -f "$L112/CrI3_112large.mmn"
test -f "$L112/CrI3_112large.nnkp"
test -f "$L112/pseudo/Cr.rel-pbe.upf"
test -f "$L112/pseudo/I.rel-pbe.upf"
test "$(count_wfc "$A1SAVE")" -eq 52
test "$(count_wfc "$L112/temp_nscf576/CrI3_scf_soc.save")" -eq 576

if [ ! -e "$DST/pseudo" ]; then
  ln -s "$L112/pseudo" "$DST/pseudo"
fi

"$PY" "$QL/tools/analysis/build_232_id_inputs.py" \
  --win-112large "$L112/CrI3_112large.win" \
  --eig "$L112/CrI3_112large.eig" \
  --cr-upf "$DST/pseudo/Cr.rel-pbe.upf" \
  --i-upf "$DST/pseudo/I.rel-pbe.upf" \
  --out-win "$DST/CrI3_232id.win" \
  --radial-id "$DST/CrI3_232id.radial_id" \
  --radial-R "$DST/CrI3_232id.radial_R" \
  --out-json "$DST/logs/build_232_id_inputs.json"

grep -q "num_wann = 232" CrI3_232id.win
grep -q "num_bands = 400" CrI3_232id.win
grep -q "dis_win_max  = 13.2" CrI3_232id.win
grep -q "dis_win_min  = -6.0" CrI3_232id.win
grep -q "dis_froz_min = -1.5" CrI3_232id.win
grep -q "I:   d" CrI3_232id.win
! grep -q "num_wann = 112" CrI3_232id.win
! grep -q "dis_win_max  =  8.0" CrI3_232id.win
test "$(head -1 CrI3_232id.radial_id)" = "232"

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u
"$W90" -pp CrI3_232id
test -f CrI3_232id.nnkp
grep -q "begin spinor_projections" CrI3_232id.nnkp

"$PY" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "/public/home/wangjs/project/New_SBEs/Quantum-light/tools/analysis")
from build_232_id_inputs import nnkpts_block
old = nnkpts_block(Path("/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large/CrI3_112large.nnkp"))
new = nnkpts_block(Path("CrI3_232id.nnkp"))
if old != new:
    raise SystemExit("nnkpts mismatch; refuse to reuse 112_large.mmn")
print("NNKPTS_MATCH")
PY

ln -f "$L112/CrI3_112large.eig" "$DST/CrI3_232id.eig"
ln -f "$L112/CrI3_112large.mmn" "$DST/CrI3_232id.mmn"
test -f CrI3_232id.eig
test -f CrI3_232id.mmn

# pw2wan is uploaded with the deploy tree; keep a local copy.
if [ -f "$QL/deploy/a0_layerA_m88full/ref232_id/CrI3_232id.pw2wan" ]; then
  cp -f "$QL/deploy/a0_layerA_m88full/ref232_id/CrI3_232id.pw2wan" "$DST/CrI3_232id.pw2wan"
fi
test -f CrI3_232id.pw2wan
grep -q "write_mmn = .false." CrI3_232id.pw2wan
grep -q "write_unk = .false." CrI3_232id.pw2wan
grep -q "ref112_large/temp_nscf576" CrI3_232id.pw2wan
! grep -q "temp_scf_soc" CrI3_232id.pw2wan
! grep -q "temp_a1h_scf_clean" CrI3_232id.pw2wan

test "$(count_wfc "$A1SAVE")" -eq 52
echo PREP_232_OK
