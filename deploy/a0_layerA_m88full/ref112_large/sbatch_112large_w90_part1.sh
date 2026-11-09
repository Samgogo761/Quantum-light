#!/bin/bash
#SBATCH --job-name=r112L_w90
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=168:00:00
#
# pw2wannier90 + wannier90 for 112_large only. Not 232.
# pw2wannier90: 16 ranks, no -nk (QE 7.2 pools not implemented).
# wannier90: serial (MPI write of hr/tb has historically raced).

set -eo pipefail
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/ref112_large"
A1SAVE="${ROOT}/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
DSTSAVE="${WORKDIR}/temp_nscf576/CrI3_scf_soc.save"
PW2=/public/home/wangjs/software/q-e-qe-7.2/bin/pw2wannier90.x
W90=/public/home/wangjs/software/q-e-qe-7.2/wannier90-3.1.0/wannier90.x
NP=16

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} 112_large pw2wan+w90 $(date)"
test -f CrI3_112large.nnkp
test -f CrI3_112large.pw2wan
test -f CrI3_112large.win
grep -q "num_wann = 112" CrI3_112large.win
grep -q "num_bands = 400" CrI3_112large.win
grep -q "dis_win_max  =  8.0" CrI3_112large.win
! grep -q "num_bands = 200" CrI3_112large.win
n_src=$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
n_dst=$(find "$DSTSAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
test "$n_src" -eq 52
test "$n_dst" -eq 576
grep -q "outdir = './temp_nscf576/'" CrI3_112large.pw2wan
! grep -q "temp_scf_soc" CrI3_112large.pw2wan

mkdir -p logs
PW2_LOG=logs/CrI3_112large_pw2wan.out
mpirun -np "$NP" $PW2 -in CrI3_112large.pw2wan > "$PW2_LOG" 2>&1
grep -q 'JOB DONE.' "$PW2_LOG"
test -f CrI3_112large.amn
test -f CrI3_112large.mmn
test -f CrI3_112large.eig
echo "PW2WAN_DONE $(date)"

test "$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')" -eq 52

W90_LOG=logs/CrI3_112large_wannier.out
$W90 CrI3_112large > "$W90_LOG" 2>&1
# wannier90 writes the completion banner to seedname.wout, not stdout.
grep -q 'All done' CrI3_112large.wout
test -f CrI3_112large_tb.dat
test -f CrI3_112large_hr.dat
echo "W90_112LARGE_DONE $(date)"
