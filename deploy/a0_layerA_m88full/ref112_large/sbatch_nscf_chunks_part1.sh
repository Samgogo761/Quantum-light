#!/bin/bash
#SBATCH --job-name=r112L_nscf8
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
# 72 x 8-k nbnd=400 NSCF (same scale as job 32911), then merge to 576-k save.
# 16 ranks, -nk 2, disk_io=low. Isolated dir. Resume skips JOB DONE chunks.
# Never write temp_scf_soc or A1-O save. Not 232 Wannier.

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
CHARGE_TPL="${WORKDIR}/charge_template/CrI3_scf_soc.save"
PRODSAVE="${ROOT}/temp_scf_soc/CrI3_scf_soc.save"
PW=/public/home/wangjs/software/q-e-qe-7.2/bin/pw.x
NP=16
NK=2
N_CHUNKS=72
CHUNK_NK=8
PY=/public/home/wangjs/software/miniconda3/bin/python

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_1 ntasks=${NP} nk=${NK} $(date)"

test "$DSTSAVE" != "$A1SAVE"
test "$DSTSAVE" != "$PRODSAVE"
test -d chunks
test -f chunks/chunk_00.in
test -f chunks/chunk_71.in
test -f merge_nscf_chunks.py
test -f "$CHARGE_TPL/charge-density.dat"
test -f "$DSTSAVE/charge-density.dat"
test "$(find "$CHARGE_TPL" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')" -eq 0

n_src=$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
echo "protect_src_wfc=$n_src"
test "$n_src" -eq 52

grep -q "outdir = './temp_chunk_00/'" chunks/chunk_00.in
grep -q "nbnd = 400" chunks/chunk_00.in
grep -q "disk_io = 'low'" chunks/chunk_00.in
awk 'BEGIN{n=-1} /^K_POINTS/{getline; n=$1} END{exit !(n==8)}' chunks/chunk_00.in
! grep -q "temp_scf_soc" chunks/chunk_00.in
! grep -q "temp_a1h_scf_clean" chunks/chunk_00.in

mkdir -p logs

chunk_done() {
  local tag=$1
  local log="logs/chunk_${tag}.out"
  local save="${WORKDIR}/temp_chunk_${tag}/CrI3_scf_soc.save"
  [ -f "$log" ] || return 1
  grep -q 'JOB DONE.' "$log" || return 1
  local n
  n=$(find "$save" -maxdepth 1 -name 'wfc*.dat' 2>/dev/null | wc -l | tr -d '[:space:]')
  [ "$n" = "$CHUNK_NK" ]
}

seed_chunk() {
  local tag=$1
  local save="${WORKDIR}/temp_chunk_${tag}/CrI3_scf_soc.save"
  mkdir -p "$save"
  rsync -a --exclude 'wfc*.dat' --exclude 'wfc*.hdf5' "$CHARGE_TPL"/ "$save"/
  find "$save" -maxdepth 1 \( -name 'wfc*.dat' -o -name 'wfc*.hdf5' \) -delete
  sed -i "s|<outdir>[^<]*</outdir>|<outdir>./temp_chunk_${tag}/</outdir>|g" "$save/data-file-schema.xml"
  if grep -q 'temp_scf_soc\|temp_a1h_scf_clean' "$save/data-file-schema.xml"; then
    echo "REFUSE: chunk xml points at a forbidden outdir" >&2
    exit 4
  fi
  test -f "$save/charge-density.dat"
}

n_done=0
n_run=0
for i in $(seq 0 $((N_CHUNKS - 1))); do
  tag=$(printf '%02d' "$i")
  if chunk_done "$tag"; then
    echo "SKIP_CHUNK ${tag} $(date)"
    n_done=$((n_done + 1))
    continue
  fi
  echo "RUN_CHUNK ${tag} $(date)"
  seed_chunk "$tag"
  NSCF_LOG="logs/chunk_${tag}.out"
  mpirun -np "$NP" $PW -nk "$NK" -in "chunks/chunk_${tag}.in" > "$NSCF_LOG" 2>&1
  grep -q 'JOB DONE.' "$NSCF_LOG"
  n_wfc=$(find "${WORKDIR}/temp_chunk_${tag}/CrI3_scf_soc.save" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
  test "$n_wfc" -eq "$CHUNK_NK"
  n_src_now=$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
  test "$n_src_now" -eq 52
  n_run=$((n_run + 1))
  echo "CHUNK_DONE ${tag} wfc=${n_wfc} $(date)"
done

echo "CHUNKS_FINISHED done_before=${n_done} ran=${n_run} $(date)"
test "$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')" -eq 52

$PY merge_nscf_chunks.py \
  --root "$WORKDIR" \
  --dest-save "$DSTSAVE" \
  --dest-outdir './temp_nscf576/' \
  --chunk-nk "$CHUNK_NK" \
  --n-chunks "$N_CHUNKS" \
  --expected-nk 576

n_out=$(find "$DSTSAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')
echo "dest_wfc_after=$n_out"
test "$n_out" -eq 576
grep -q '<nks>576</nks>' "$DSTSAVE/data-file-schema.xml"
grep -q '<nbnd>400</nbnd>' "$DSTSAVE/data-file-schema.xml"
! grep -q 'temp_scf_soc\|temp_a1h_scf_clean' "$DSTSAVE/data-file-schema.xml"
test "$(find "$A1SAVE" -maxdepth 1 -name 'wfc*.dat' | wc -l | tr -d '[:space:]')" -eq 52
echo "A1O_WFC_STILL_52 $(date)"

if [ -f count_outer_window_bands.py ]; then
  cat logs/chunk_*.out > logs/nscf_chunks_all.out
  $PY count_outer_window_bands.py \
    --nscf logs/nscf_chunks_all.out \
    --out logs/outer_window_count_576.json \
    --n-k 576 || echo "COUNT_576_PARSE_WARN"
fi
echo "NSCF576_CHUNKS_DONE $(date)"
