#!/bin/bash
#SBATCH --job-name=r112_13p2
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_13p2
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#
# Window-only control: num_wann=112, dis_win=[-6,13.2], reuse 112_large AMN.
# Serial wannier90. Not I-d, not production, not k10/k20 HHG.

set -euo pipefail

ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/ref112_13p2"
L112="${ROOT}/window144_hilbert/ref112_large"
A1SAVE="${ROOT}/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
W90=/public/home/wangjs/software/q-e-qe-7.2/wannier90-3.1.0/wannier90.x

count_wfc() {
  find "$1" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]'
}

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} 112_13.2 window control part_2 $(date)"
test -f CrI3_112_13p2.win
test -f CrI3_112_13p2.nnkp
test -f CrI3_112_13p2.amn
test -f CrI3_112_13p2.mmn
test -f CrI3_112_13p2.eig
grep -q "num_wann = 112" CrI3_112_13p2.win
grep -q "dis_win_max  = 13.2" CrI3_112_13p2.win
! grep -q "num_wann = 232" CrI3_112_13p2.win
! grep -q "I:   d" CrI3_112_13p2.win
test "$(count_wfc "$A1SAVE")" -eq 52
test "$(count_wfc "$L112/temp_nscf576/CrI3_scf_soc.save")" -eq 576

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs
W90_LOG=logs/CrI3_112_13p2_wannier.out
"$W90" CrI3_112_13p2 > "$W90_LOG" 2>&1
grep -q 'All done' CrI3_112_13p2.wout
test -f CrI3_112_13p2_tb.dat
test -f CrI3_112_13p2_hr.dat
test "$(count_wfc "$A1SAVE")" -eq 52
echo "W90_112_13P2_DONE $(date)"
