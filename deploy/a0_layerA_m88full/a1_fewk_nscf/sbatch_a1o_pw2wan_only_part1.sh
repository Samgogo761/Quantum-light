#!/bin/bash
#SBATCH --job-name=a1o_pw2wan
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=72:00:00
#
# Continuation after 32446: NSCF 52-k and Gamma gate already PASS.
# QE 7.2 pw2wannier90.x does NOT implement pools. Never pass -nk.
# Do not delete wfc. Do not rewrite temp_scf_soc.

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
TB="${ROOT}/wannier/window144_hilbert/CrI3_window144_tb.dat"
NP=16
PY=""
for cand in \
    /public/home/wangjs/software/miniconda3/bin/python3.9 \
    /public/home/wangjs/software/miniconda3/bin/python \
    "${HOME}/software/miniconda3/bin/python" \
    python3 python
do
  if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; assert sys.version_info >= (3, 8)' >/dev/null 2>&1; then
      PY=$cand
      break
    fi
  fi
done
[ -n "$PY" ] || { echo "no Python>=3.8" >&2; exit 2; }

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_1 ntasks=${NP} pw2wan_npool=1 $(date)"

test -f CrI3_a1o.nnkp
test -f CrI3_a1o.pw2wan
test -f a1o_star_map.json
test -f compare_a1o_berry_r.py
test -f a1o_berry_math.py
test -f audit_pi_rq_coupling.py
test -f construct_p_conjugate_tb.py
test -f "$TB"
SAVE=temp_a1h_scf_clean/CrI3_scf_soc.save
test -f "$SAVE/charge-density.dat"
n_wfc=$(ls -1 "$SAVE"/wfc*.dat 2>/dev/null | wc -l)
echo "n_wfc=${n_wfc}"
[ "$n_wfc" -eq 52 ]
grep -q 'JOB DONE.' logs/CrI3_nscf_soc_a1o_star.out
grep -q '"A1H_GAMMA_EIG_GATE": "PASS"' logs/a1o_gamma_eig_gate.json

mkdir -p logs
PW2_LOG=logs/CrI3_a1o_pw2wan.out
mpirun -np "$NP" $PW2 -in CrI3_a1o.pw2wan > "$PW2_LOG" 2>&1
grep -q 'JOB DONE.' "$PW2_LOG"
grep -q 'MMN calculated' "$PW2_LOG"
test -f CrI3_a1o.mmn
test -f CrI3_a1o.eig
echo "A1O_MMN_DONE $(date)"

set +e
$PY compare_a1o_berry_r.py \
  --mmn CrI3_a1o.mmn \
  --eig CrI3_a1o.eig \
  --nnkp CrI3_a1o.nnkp \
  --star-map a1o_star_map.json \
  --tb "$TB" \
  --out logs/a1o_berry_r.json
CMP_RC=$?
set -e
echo "A1O_COMPARE_RC=${CMP_RC} $(date)"
[ "$CMP_RC" -eq 0 ]
echo "A1-O pipeline complete $(date)"
