#!/bin/bash
set -eo pipefail
WD=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
cd "$WD"
for f in sbatch_shell_radial_amn_part2.sh CrI3_shell_radial.pw2wan CrI3_shell_radial.nnkp \
         screen_q32_shell_trials.py compare_shell_radial_robust.py
do
  if [ -f "$f" ]; then
    sed -i 's/\r$//' "$f"
  fi
done
chmod +x sbatch_shell_radial_amn_part2.sh
if grep -q $'\r' sbatch_shell_radial_amn_part2.sh CrI3_shell_radial.pw2wan; then
  echo CRLF_BAD
  exit 9
fi
echo CRLF_OK
test -f CrI3_shell_radial.nnkp
test -f CrI3_shell_radial_map.json
test -f logs/q32_shell_trial_screen.json
n_wfc=$(ls -1 temp_a1h_scf_clean/CrI3_scf_soc.save/wfc*.dat | wc -l)
echo "n_wfc=$n_wfc nnkp_lines=$(wc -l < CrI3_shell_radial.nnkp)"
test "$n_wfc" -eq 52
grep -A1 'begin spinor_projections' CrI3_shell_radial.nnkp | head -2
squeue -u wangjs -p part_1,part_2
sinfo -p part_1,part_2
echo PREP_OK
