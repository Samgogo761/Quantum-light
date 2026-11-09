#!/bin/bash
# Isolate 52-k nbnd=400 counting NSCF. Never touch A1-O 200-band wfc.
set -eo pipefail

count_wfc() {
  find "$1" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]'
}

A1=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
DST=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref232_count
SRC_SAVE="$A1/temp_a1h_scf_clean/CrI3_scf_soc.save"
DST_SAVE="$DST/temp_count/CrI3_scf_soc.save"

if [ "$DST_SAVE" = "$SRC_SAVE" ]; then
  echo "REFUSE: destination save equals A1-O save" >&2
  exit 3
fi
case "$DST_SAVE" in
  *temp_a1h_scf_clean*) echo "REFUSE: dest path contains temp_a1h_scf_clean" >&2; exit 3 ;;
esac

mkdir -p "$DST/logs" "$DST/temp_count"
cd "$DST"
for f in prep_count_nscf.sh sbatch_count_nscf_part1.sh CrI3_nscf_soc_count.in count_outer_window_bands.py; do
  if [ -f "$f" ]; then
    sed -i 's/\r$//' "$f"
  fi
done
chmod +x prep_count_nscf.sh sbatch_count_nscf_part1.sh
if grep -q $'\r' prep_count_nscf.sh sbatch_count_nscf_part1.sh CrI3_nscf_soc_count.in; then
  echo CRLF_BAD
  exit 9
fi
echo CRLF_OK

test -f "$SRC_SAVE/charge-density.dat"
test -f "$SRC_SAVE/data-file-schema.xml"
n_wfc_src=$(count_wfc "$SRC_SAVE")
echo "src_n_wfc=$n_wfc_src"
test "$n_wfc_src" -eq 52

# Copy charge/xml/UPF only. Exclude every wfc. Do not rm anything in SRC_SAVE.
mkdir -p "$DST_SAVE"
rsync -a \
  --exclude 'wfc*.dat' \
  --exclude 'wfc*.hdf5' \
  --exclude 'wfc*.dat.xml' \
  "$SRC_SAVE"/ "$DST_SAVE"/
# Excluded names are not transferred; leftover dest wfc must be removed here only.
find "$DST_SAVE" -maxdepth 1 \( -name 'wfc*.dat' -o -name 'wfc*.hdf5' \) -delete

n_wfc_src2=$(count_wfc "$SRC_SAVE")
n_wfc_dst=$(count_wfc "$DST_SAVE")
echo "src_n_wfc_after_copy=$n_wfc_src2 dst_n_wfc=$n_wfc_dst"
test "$n_wfc_src2" -eq 52
test "$n_wfc_dst" -eq 0
test -f "$DST_SAVE/charge-density.dat"
test -f "$DST_SAVE/data-file-schema.xml"

OLD_OUT="$A1/temp_a1h_scf_clean"
NEW_OUT="$DST/temp_count"
if grep -q "$OLD_OUT" "$DST_SAVE/data-file-schema.xml"; then
  sed -i "s|$OLD_OUT|$NEW_OUT|g" "$DST_SAVE/data-file-schema.xml"
fi
if grep -q './temp_a1h_scf_clean' "$DST_SAVE/data-file-schema.xml"; then
  sed -i "s|./temp_a1h_scf_clean|./temp_count|g" "$DST_SAVE/data-file-schema.xml"
fi
if grep -q "temp_a1h_scf_clean" "$DST_SAVE/data-file-schema.xml"; then
  echo "xml still points at A1-O outdir" >&2
  exit 4
fi
# Source xml must remain relative to A1-O outdir (false alarm if we grep the absolute path).
if ! grep -q 'temp_a1h_scf_clean' "$SRC_SAVE/data-file-schema.xml"; then
  echo "WARNING: source xml no longer has original outdir (unexpected)" >&2
fi
n_wfc_src3=$(count_wfc "$SRC_SAVE")
test "$n_wfc_src3" -eq 52

if [ ! -e "$DST/pseudo" ]; then
  ln -s "$A1/pseudo" "$DST/pseudo"
fi
test -f "$DST/pseudo/Cr.rel-pbe.upf"
test -f "$DST/pseudo/I.rel-pbe.upf"
CR_HASH=be756eda2d6d7f07b1f19f6c4ba216cb2cd37615671bf98c872d4904bfaf2e6a
I_HASH=c7ce679d2426cea1b7508a63104b6a8492529af66860ce7f5afba00f3cc49123
got_cr=$(sha256sum "$DST/pseudo/Cr.rel-pbe.upf" | awk '{print $1}')
got_i=$(sha256sum "$DST/pseudo/I.rel-pbe.upf" | awk '{print $1}')
echo "upf_cr=$got_cr"
echo "upf_i=$got_i"
test "$got_cr" = "$CR_HASH"
test "$got_i" = "$I_HASH"
grep -q "nbnd = 400" CrI3_nscf_soc_count.in
grep -q "outdir = './temp_count/'" CrI3_nscf_soc_count.in
grep -q "disk_io = 'low'" CrI3_nscf_soc_count.in
! grep -q "temp_a1h_scf_clean" CrI3_nscf_soc_count.in
! grep -q "nbnd = 200" CrI3_nscf_soc_count.in

echo "=== queue ==="
squeue -u wangjs -p part_1,part_2
sinfo -p part_1,part_2
echo PREP_OK
