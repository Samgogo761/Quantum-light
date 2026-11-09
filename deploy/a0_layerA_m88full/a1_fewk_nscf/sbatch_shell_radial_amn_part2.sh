#!/bin/bash
#SBATCH --job-name=shell_radial_amn
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=56G
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=24:00:00
#
# Radial robustness of Cr-s/p/d + I-s/p/d trials. No new Wannier parent.
# Reuses A1-O 52-k wfc. Never -nk on pw2wannier90. Do not write temp_scf_soc.

set -eo pipefail
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

WORKDIR=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
PW2=/public/home/wangjs/software/q-e-qe-7.2/bin/pw2wannier90.x
NP=8
PY=/public/home/wangjs/software/miniconda3/bin/python

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_2 ntasks=${NP} $(date)"

test -f CrI3_shell_radial.nnkp
test -f CrI3_shell_radial.pw2wan
test -f CrI3_shell_radial_map.json
test -f CrI3_a1o.mmn
test -f logs/q32_shell_trial_screen.json
SAVE=temp_a1h_scf_clean/CrI3_scf_soc.save
n_wfc=$(ls -1 "$SAVE"/wfc*.dat 2>/dev/null | wc -l)
echo "n_wfc=${n_wfc}"
[ "$n_wfc" -eq 52 ]

mkdir -p logs
PW2_LOG=logs/CrI3_shell_radial_pw2wan.out
mpirun -np "$NP" $PW2 -in CrI3_shell_radial.pw2wan > "$PW2_LOG" 2>&1
grep -q 'JOB DONE.' "$PW2_LOG"
test -f CrI3_shell_radial.amn
echo "RADIAL_AMN_DONE $(date)"

set +e
$PY screen_q32_shell_trials.py \
  --amn CrI3_shell_radial.amn \
  --mmn CrI3_a1o.mmn \
  --eig CrI3_a1o.eig \
  --nnkp-a1o CrI3_a1o.nnkp \
  --star-map a1o_star_map.json \
  --shell-map CrI3_shell_radial_map.json \
  --out logs/q32_shell_radial_screen.json
SCR_RC=$?
$PY compare_shell_radial_robust.py \
  --baseline logs/q32_shell_trial_screen.json \
  --radial logs/q32_shell_radial_screen.json \
  --out logs/q32_shell_radial_robust.json
CMP_RC=$?
set -e
echo "RADIAL_SCREEN_RC=${SCR_RC} RADIAL_COMPARE_RC=${CMP_RC} $(date)"
[ "$SCR_RC" -eq 0 ]
[ "$CMP_RC" -eq 0 ]
echo "shell radial robustness complete $(date)"
