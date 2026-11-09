#!/bin/bash
set -euo pipefail
WD=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
cd "$WD"
for f in sbatch_a1o_mmn_star_part1.sh CrI3_nscf_soc_a1o_star.in CrI3_a1o.nnkp CrI3_a1o.pw2wan a1o_star_map.json compare_a1o_berry_r.py a1o_berry_math.py audit_pi_rq_coupling.py construct_p_conjugate_tb.py; do
  test -f "$f"
done
chmod +x sbatch_a1o_mmn_star_part1.sh
if grep -q $'\r' sbatch_a1o_mmn_star_part1.sh CrI3_nscf_soc_a1o_star.in CrI3_a1o.pw2wan; then
  echo CRLF_BAD
  exit 9
fi
test -f temp_a1h_scf_clean/CrI3_scf_soc.save/charge-density.dat
test -f /public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/CrI3_window144_tb.dat
/public/home/wangjs/software/miniconda3/bin/python -c 'import sys; sys.path.insert(0, "."); import a1o_berry_math; print("pyok")'
echo FILES_OK
squeue -p part_1
sinfo -p part_1,part_2
