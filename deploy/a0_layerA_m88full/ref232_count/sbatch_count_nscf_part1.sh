#!/bin/bash
#SBATCH --job-name=ref232_count
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref232_count
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=48:00:00
#
# Isolated 8-k nbnd=400 NSCF to count N(E) for a 232 outer window.
# 32588: 52-k Berry star x 400 bands was SIGKILL ~239 GB (QE estimate 50 GB).
# Counting does not need the A1-O star; 4 hot k + 4 BZ samples. 576k later.
# Charge cloned WITHOUT wfc. Never write or rm temp_a1h_scf_clean.
# disk_io=low. 16 ranks, -nk 2. Not part_2.

set -eo pipefail
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/ref232_count"
A1SAVE="${ROOT}/window144_hilbert/a1_fewk_nscf/temp_a1h_scf_clean/CrI3_scf_soc.save"
DSTSAVE="${WORKDIR}/temp_count/CrI3_scf_soc.save"
PW=/public/home/wangjs/software/q-e-qe-7.2/bin/pw.x
NP=16
NK=2
PY=""
for cand in \
    /public/home/wangjs/software/miniconda3/bin/python3.9 \
    /public/home/wangjs/software/miniconda3/bin/python \
    "${HOME}/software/miniconda3/bin/python"
do
  if [ -x "$cand" ]; then
    PY=$cand
    break
  fi
done
[ -n "$PY" ] || { echo "no conda python" >&2; exit 2; }

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_1 ntasks=${NP} nk=${NK} $(date)"

test -d "$WORKDIR"
test "$DSTSAVE" != "$A1SAVE"
test -f CrI3_nscf_soc_count.in
test -f count_outer_window_bands.py
test -f "$DSTSAVE/charge-density.dat"
test -f ./pseudo/Cr.rel-pbe.upf
test -f ./pseudo/I.rel-pbe.upf
n_src=$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
n_dst=$(find "$DSTSAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
echo "protect_src_wfc=$n_src dest_wfc=$n_dst"
test "$n_src" -eq 52
test "$n_dst" -eq 0
grep -q "outdir = './temp_count/'" CrI3_nscf_soc_count.in
grep -q "nbnd = 400" CrI3_nscf_soc_count.in
! grep -q "temp_a1h_scf_clean" CrI3_nscf_soc_count.in

mkdir -p logs
NSCF_LOG=logs/CrI3_nscf_soc_count.out
if [ -f "$NSCF_LOG" ]; then
  mv -f "$NSCF_LOG" "logs/CrI3_nscf_soc_count.32588.oom.out"
fi
grep -q "K_POINTS crystal" CrI3_nscf_soc_count.in
awk 'BEGIN{n=-1} /^K_POINTS/{getline; n=$1} END{exit !(n==8)}' CrI3_nscf_soc_count.in
mpirun -np "$NP" $PW -nk "$NK" -in CrI3_nscf_soc_count.in > "$NSCF_LOG" 2>&1
grep -q 'JOB DONE.' "$NSCF_LOG"

n_src2=$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
test "$n_src2" -eq 52
echo "A1O_WFC_STILL_52 $(date)"

$PY count_outer_window_bands.py \
  --nscf "$NSCF_LOG" \
  --out logs/outer_window_count.json \
  --n-k 8
echo "COUNT_DONE $(date)"
cat logs/outer_window_count.json
