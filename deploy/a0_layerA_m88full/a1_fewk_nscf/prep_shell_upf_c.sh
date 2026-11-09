#!/bin/bash
set -eo pipefail
WD=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
cd "$WD"
PY=/public/home/wangjs/software/miniconda3/bin/python
for f in sbatch_shell_upf_c_part2.sh CrI3_shell_upf.pw2wan Makefile.pw2wannier90_upf \
  extract_upf_defined_radials.py patch_pw2wannier90_upf_radial.py \
  compare_upf_projector_c.py screen_q32_shell_trials.py
do
  if [ -f "$f" ]; then
    sed -i 's/\r$//' "$f"
  fi
done
chmod +x sbatch_shell_upf_c_part2.sh
if grep -q $'\r' sbatch_shell_upf_c_part2.sh CrI3_shell_upf.pw2wan Makefile.pw2wannier90_upf; then
  echo CRLF_BAD
  exit 9
fi
echo CRLF_OK
test -f CrI3_shell.nnkp
cp -f CrI3_shell.nnkp CrI3_shell_upf.nnkp
SAVE=temp_a1h_scf_clean/CrI3_scf_soc.save
test -f "$SAVE/Cr.rel-pbe.upf"
test -f "$SAVE/I.rel-pbe.upf"
n_wfc=$(ls -1 "$SAVE"/wfc*.dat | wc -l)
echo "n_wfc=$n_wfc"
test "$n_wfc" -eq 52
mkdir -p logs
$PY extract_upf_defined_radials.py \
  --cr-upf "$SAVE/Cr.rel-pbe.upf" \
  --i-upf "$SAVE/I.rel-pbe.upf" \
  --out-json logs/upf_defined_radials.json \
  --radial-id CrI3_shell_upf.radial_id \
  --radial-R CrI3_shell_upf.radial_R \
  --shell-map CrI3_shell_upf_map.json \
  --require-known-hash
test -f CrI3_shell_upf.radial_id
test -f CrI3_shell_upf.radial_R
n_id=$(head -1 CrI3_shell_upf.radial_id)
echo "radial_id_n=$n_id"
test "$n_id" = "288"
$PY -c 'import json; d=json.load(open("logs/upf_defined_radials.json")); print(d["definition"]["I_d"]); print("mean_r", d["mean_r_bohr"]); print("I_d channels", d["shells"]["I_d"]["channel_indices"], "rejected", d["shells"]["I_d"]["rejected_exploding_indices"])'
$PY patch_pw2wannier90_upf_radial.py /public/home/wangjs/software/q-e-qe-7.2/PP/src/pw2wannier90.f90 logs/pw2wannier90_upf.f90
grep -q UPF_CUSTOM_RADIAL_LOADED logs/pw2wannier90_upf.f90
echo PATCH_OK
squeue -u wangjs -p part_1,part_2
sinfo -p part_1,part_2
echo PREP_OK
