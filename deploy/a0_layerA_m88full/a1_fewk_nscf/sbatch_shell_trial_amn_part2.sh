#!/bin/bash
#SBATCH --job-name=shell_trial_amn
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=56G
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=24:00:00
#
# Extended-shell trial AMN on existing A1-O 52-k wavefunctions.
# No new Wannier parent. Do not delete wfc. Do not write temp_scf_soc.
# QE 7.2 pw2wannier90: never pass -nk (pools not implemented).

set -eo pipefail
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/a1_fewk_nscf"
PW2=/public/home/wangjs/software/q-e-qe-7.2/bin/pw2wannier90.x
NP=8
PY=/public/home/wangjs/software/miniconda3/bin/python

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_2 ntasks=${NP} $(date)"

test -f CrI3_shell.nnkp
test -f CrI3_shell.pw2wan
test -f CrI3_shell_map.json
test -f CrI3_a1o.mmn
test -f a1o_star_map.json
SAVE=temp_a1h_scf_clean/CrI3_scf_soc.save
test -f "$SAVE/charge-density.dat"
n_wfc=$(ls -1 "$SAVE"/wfc*.dat 2>/dev/null | wc -l)
echo "n_wfc=${n_wfc}"
[ "$n_wfc" -eq 52 ]

mkdir -p logs
PW2_LOG=logs/CrI3_shell_pw2wan.out
mpirun -np "$NP" $PW2 -in CrI3_shell.pw2wan > "$PW2_LOG" 2>&1
grep -q 'JOB DONE.' "$PW2_LOG"
test -f CrI3_shell.amn
echo "SHELL_AMN_DONE $(date)"

set +e
$PY screen_q32_shell_trials.py \
  --amn CrI3_shell.amn \
  --mmn CrI3_a1o.mmn \
  --eig CrI3_a1o.eig \
  --nnkp-a1o CrI3_a1o.nnkp \
  --star-map a1o_star_map.json \
  --shell-map CrI3_shell_map.json \
  --out logs/q32_shell_trial_screen.json
SCR_RC=$?
set -e
echo "SHELL_SCREEN_RC=${SCR_RC} $(date)"
[ "$SCR_RC" -eq 0 ]
echo "shell trial AMN screen complete $(date)"
