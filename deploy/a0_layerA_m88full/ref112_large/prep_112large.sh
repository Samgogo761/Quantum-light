#!/bin/bash
# Isolate 576-k nbnd=400 NSCF + 112_large Wannier inputs.
# Never write temp_scf_soc or temp_a1h_scf_clean.
set -eo pipefail

A1=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
COUNT=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref232_count
DST=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large
PROD_WIN=/public/home/wangjs/project/CrI3_TB/wannier/CrI3.win
SRC_SAVE="$A1/temp_a1h_scf_clean/CrI3_scf_soc.save"
DST_SAVE="$DST/temp_nscf576/CrI3_scf_soc.save"
PY=/public/home/wangjs/software/miniconda3/bin/python
W90=/public/home/wangjs/software/q-e-qe-7.2/wannier90-3.1.0/wannier90.x
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh 2>/dev/null || true

count_wfc() {
  find "$1" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]'
}

if [ "$DST_SAVE" = "$SRC_SAVE" ]; then
  echo "REFUSE: dest equals A1-O save" >&2
  exit 3
fi
case "$DST" in
  *temp_scf_soc*|*temp_a1h_scf_clean*) echo "REFUSE: dest path forbidden" >&2; exit 3 ;;
esac

mkdir -p "$DST/logs" "$DST/temp_nscf576" "$DST/charge_template"
cd "$DST"
for f in prep_112large.sh sbatch_nscf576_part1.sh sbatch_nscf_chunks_part1.sh \
  sbatch_112large_w90_part1.sh make_nscf576.py merge_nscf_chunks.py CrI3_112large.pw2wan; do
  if [ -f "$f" ]; then
    sed -i 's/\r$//' "$f"
  fi
done
chmod +x prep_112large.sh sbatch_nscf576_part1.sh sbatch_nscf_chunks_part1.sh \
  sbatch_112large_w90_part1.sh
if grep -q $'\r' prep_112large.sh sbatch_nscf_chunks_part1.sh CrI3_112large.pw2wan merge_nscf_chunks.py; then
  echo CRLF_BAD
  exit 9
fi

test -f "$PROD_WIN"
test -f "$COUNT/CrI3_nscf_soc_count.in"
test -f "$SRC_SAVE/charge-density.dat"
n_wfc_src=$(count_wfc "$SRC_SAVE")
echo "src_n_wfc=$n_wfc_src"
test "$n_wfc_src" -eq 52

CHARGE_TPL="$DST/charge_template/CrI3_scf_soc.save"
mkdir -p "$CHARGE_TPL" "$DST_SAVE"
rsync -a \
  --exclude 'wfc*.dat' \
  --exclude 'wfc*.hdf5' \
  --exclude 'wfc*.dat.xml' \
  "$SRC_SAVE"/ "$CHARGE_TPL"/
rsync -a \
  --exclude 'wfc*.dat' \
  --exclude 'wfc*.hdf5' \
  --exclude 'wfc*.dat.xml' \
  "$SRC_SAVE"/ "$DST_SAVE"/
find "$CHARGE_TPL" -maxdepth 1 \( -name 'wfc*.dat' -o -name 'wfc*.hdf5' \) -delete
find "$DST_SAVE" -maxdepth 1 \( -name 'wfc*.dat' -o -name 'wfc*.hdf5' \) -delete
n_wfc_src2=$(count_wfc "$SRC_SAVE")
n_wfc_dst=$(count_wfc "$DST_SAVE")
n_wfc_tpl=$(count_wfc "$CHARGE_TPL")
echo "src_n_wfc_after_copy=$n_wfc_src2 dst_n_wfc=$n_wfc_dst tpl_n_wfc=$n_wfc_tpl"
test "$n_wfc_src2" -eq 52
test "$n_wfc_dst" -eq 0
test "$n_wfc_tpl" -eq 0
test -f "$DST_SAVE/charge-density.dat"
test -f "$CHARGE_TPL/charge-density.dat"

OLD_OUT="$A1/temp_a1h_scf_clean"
NEW_OUT="$DST/temp_nscf576"
patch_xml_outdir() {
  local xml=$1
  local new=$2
  if grep -q "$OLD_OUT" "$xml"; then
    sed -i "s|$OLD_OUT|$new|g" "$xml"
  fi
  if grep -q './temp_a1h_scf_clean' "$xml"; then
    sed -i "s|./temp_a1h_scf_clean|$new|g" "$xml"
  fi
  if grep -q './temp_count' "$xml"; then
    sed -i "s|./temp_count|$new|g" "$xml"
  fi
  if grep -q './temp_nscf576' "$xml" && [ "$new" != "./temp_nscf576" ] && [ "$new" != "$NEW_OUT" ]; then
    sed -i "s|./temp_nscf576|$new|g" "$xml"
  fi
  if grep -q 'temp_a1h_scf_clean\|temp_scf_soc' "$xml"; then
    echo "xml still points at a forbidden outdir: $xml" >&2
    exit 4
  fi
}
patch_xml_outdir "$DST_SAVE/data-file-schema.xml" "./temp_nscf576"
patch_xml_outdir "$CHARGE_TPL/data-file-schema.xml" "./charge_template"
if ! grep -q 'temp_a1h_scf_clean' "$SRC_SAVE/data-file-schema.xml"; then
  echo "WARNING: source xml missing A1-O outdir string" >&2
fi
test "$(count_wfc "$SRC_SAVE")" -eq 52

if [ ! -e "$DST/pseudo" ]; then
  ln -s "$A1/pseudo" "$DST/pseudo"
fi
CR_HASH=be756eda2d6d7f07b1f19f6c4ba216cb2cd37615671bf98c872d4904bfaf2e6a
I_HASH=c7ce679d2426cea1b7508a63104b6a8492529af66860ce7f5afba00f3cc49123
got_cr=$(sha256sum "$DST/pseudo/Cr.rel-pbe.upf" | awk '{print $1}')
got_i=$(sha256sum "$DST/pseudo/I.rel-pbe.upf" | awk '{print $1}')
test "$got_cr" = "$CR_HASH"
test "$got_i" = "$I_HASH"

cp -f "$COUNT/CrI3_nscf_soc_count.in" "$DST/CrI3_nscf_soc_count_template.in"
sed -i 's/\r$//' CrI3_nscf_soc_count_template.in make_nscf576.py
$PY make_nscf576.py \
  --template CrI3_nscf_soc_count_template.in \
  --win "$PROD_WIN" \
  --out CrI3_nscf_soc_576.in \
  --outdir './temp_nscf576/' \
  --chunks \
  --chunk-size 8 \
  --chunk-dir chunks \
  --manifest chunks/manifest.json
grep -q "nbnd = 400" CrI3_nscf_soc_576.in
grep -q "disk_io = 'low'" CrI3_nscf_soc_576.in
grep -q "outdir = './temp_nscf576/'" CrI3_nscf_soc_576.in
grep -q "K_POINTS crystal" CrI3_nscf_soc_576.in
awk 'BEGIN{n=-1} /^K_POINTS/{getline; n=$1} END{exit !(n==576)}' CrI3_nscf_soc_576.in
! grep -q "temp_scf_soc" CrI3_nscf_soc_576.in
! grep -q "temp_a1h_scf_clean" CrI3_nscf_soc_576.in
test -f chunks/chunk_00.in
test -f chunks/chunk_71.in
test ! -f chunks/chunk_72.in
grep -q "outdir = './temp_chunk_00/'" chunks/chunk_00.in
grep -q "outdir = './temp_chunk_71/'" chunks/chunk_71.in
awk 'BEGIN{n=-1} /^K_POINTS/{getline; n=$1} END{exit !(n==8)}' chunks/chunk_00.in
! grep -q "temp_scf_soc" chunks/chunk_00.in
! grep -q "temp_a1h_scf_clean" chunks/chunk_00.in

# 112_large win: same projections and frozen window; only num_bands=400.
# dis_win stays [-6,8] so this test fills the advertised window, not 13.2 eV.
$PY - <<'PY'
from pathlib import Path
src = Path("/public/home/wangjs/project/CrI3_TB/wannier/CrI3.win").read_text()
out = src.replace("num_bands = 200", "num_bands = 400", 1)
if "num_bands = 400" not in out:
    raise SystemExit("failed to set num_bands=400")
if "dis_win_max  =  8.0" not in out and "dis_win_max = 8.0" not in out.replace(" ", ""):
    # keep a readable check
    if "dis_win_max" not in out or "8.0" not in out:
        raise SystemExit("dis_win_max 8.0 missing")
if "dis_froz_min = -1.5" not in out:
    raise SystemExit("frozen window changed")
Path("CrI3_112large.win").write_text(out)
print("wrote CrI3_112large.win")
PY
grep -q "num_wann = 112" CrI3_112large.win
grep -q "num_bands = 400" CrI3_112large.win
grep -q "dis_win_min  = -6.0" CrI3_112large.win
grep -q "dis_win_max  =  8.0" CrI3_112large.win
grep -q "dis_froz_min = -1.5" CrI3_112large.win
grep -q "dis_froz_max =  0.8" CrI3_112large.win
grep -q "Cr1: d" CrI3_112large.win
grep -q "I:   p" CrI3_112large.win
! grep -q "num_bands = 200" CrI3_112large.win

test -x "$W90"
if [ ! -f CrI3_112large.nnkp ]; then
  $W90 -pp CrI3_112large
fi
test -f CrI3_112large.nnkp

echo "=== queue ==="
squeue -u wangjs -p part_1,part_2
sinfo -p part_1,part_2
echo PREP_OK
