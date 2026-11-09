#!/bin/bash
set -eo pipefail
WD=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
cd "$WD"
PY=/public/home/wangjs/software/miniconda3/bin/python
for f in sbatch_shell_trial_amn_part2.sh CrI3_shell.pw2wan CrI3_shell.nnkp screen_q32_shell_trials.py inventory_upf_pswfc.py
do
  if [ -f "$f" ]; then
    sed -i 's/\r$//' "$f"
  fi
done
chmod +x sbatch_shell_trial_amn_part2.sh
if grep -q $'\r' sbatch_shell_trial_amn_part2.sh CrI3_shell.pw2wan; then
  echo CRLF_BAD
  exit 9
fi
echo CRLF_OK
test -f CrI3_shell.nnkp
test -f screen_q32_shell_trials.py
test -f a1o_berry_math.py
test -f audit_pi_rq_coupling.py
n_wfc=$(ls -1 temp_a1h_scf_clean/CrI3_scf_soc.save/wfc*.dat | wc -l)
echo "n_wfc=$n_wfc"
test "$n_wfc" -eq 52
echo "nnkp_lines=$(wc -l < CrI3_shell.nnkp)"
grep -A1 'begin spinor_projections' CrI3_shell.nnkp | head -2
mkdir -p logs
$PY inventory_upf_pswfc.py ./pseudo/Cr.rel-pbe.upf ./pseudo/I.rel-pbe.upf --out logs/upf_pswfc_inventory.json
$PY -c 'import json; d=json.load(open("logs/upf_pswfc_inventory.json")); print([(f["file"], [c.get("label") for c in f["atomic_wfc"]], "dchi="+str(f["projwfc_can_see_d"])) for f in d["files"]])'
squeue -u wangjs -p part_1,part_2
sinfo -p part_1,part_2
echo PREP_OK
