#!/bin/bash
#SBATCH --job-name=r232id_p2
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=56G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref232_id
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#
# 232_I-d reference parent: AMN with UPF I-d radials, reuse 112_large MMN/eig.
# part_2 because part_1 is full; AMN-only (no MMN rewrite).
# Not production, not GH7, not k40, not chemical I-d 144.

set -euo pipefail

ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/ref232_id"
L112="${ROOT}/window144_hilbert/ref112_large"
A1SAVE="${ROOT}/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
PW2="${ROOT}/window144_hilbert/a1_fewk_nscf/qe_pw2wan_c/pw2wannier90_upf.x"
W90=/public/home/wangjs/software/q-e-qe-7.2/wannier90-3.1.0/wannier90.x
NP=8

count_wfc() {
  find "$1" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]'
}

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} 232_I-d AMN+w90 part_2 $(date)"
test -x "$PW2"
test -f CrI3_232id.nnkp
test -f CrI3_232id.pw2wan
test -f CrI3_232id.win
test -f CrI3_232id.radial_id
test -f CrI3_232id.radial_R
test -f CrI3_232id.mmn
test -f CrI3_232id.eig
grep -q "num_wann = 232" CrI3_232id.win
grep -q "num_bands = 400" CrI3_232id.win
grep -q "dis_win_max  = 13.2" CrI3_232id.win
grep -q "I:   d" CrI3_232id.win
! grep -q "num_wann = 112" CrI3_232id.win
test "$(head -1 CrI3_232id.radial_id)" = "232"
test "$(count_wfc "$A1SAVE")" -eq 52
test "$(count_wfc "$L112/temp_nscf576/CrI3_scf_soc.save")" -eq 576
grep -q "write_mmn = .false." CrI3_232id.pw2wan
! grep -q "temp_scf_soc" CrI3_232id.pw2wan
! grep -q "temp_a1h_scf_clean" CrI3_232id.pw2wan

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

mkdir -p logs
PW2_LOG=logs/CrI3_232id_pw2wan.out
mpirun -np "$NP" "$PW2" -in CrI3_232id.pw2wan > "$PW2_LOG" 2>&1
grep -q 'JOB DONE.' "$PW2_LOG"
grep -q 'UPF_CUSTOM_RADIAL_LOADED' "$PW2_LOG"
test -f CrI3_232id.amn
test -f CrI3_232id.mmn
test "$(count_wfc "$A1SAVE")" -eq 52
echo "PW2WAN_AMN_DONE $(date)"

W90_LOG=logs/CrI3_232id_wannier.out
"$W90" CrI3_232id > "$W90_LOG" 2>&1
grep -q 'All done' CrI3_232id.wout
test -f CrI3_232id_tb.dat
test -f CrI3_232id_hr.dat
test "$(count_wfc "$A1SAVE")" -eq 52
echo "W90_232ID_DONE $(date)"
