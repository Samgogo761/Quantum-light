#!/bin/bash
#SBATCH --job-name=a1h_scf_nscf6k
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=72:00:00
#
# Match April SCF: 16 ranks, -nk 2 (npool=2).
# 32092 used -nk 4 and was SIGKILL at SCF iter 185 (MaxRSS ~239 GiB).
# Runtime overlay (not in the April-identity .in): disk_io=high, max_seconds,
# restart_mode so a kill still leaves a .save. NEVER run inside temp_scf_soc.

set -eo pipefail
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

ROOT=/public/home/wangjs/project/CrI3_TB
WORKDIR="${ROOT}/window144_hilbert/a1_fewk_nscf"
PW=/public/home/wangjs/software/q-e-qe-7.2/bin/pw.x
NP=16
NK=2
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
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_1 ntasks=${NP} nk=${NK} $(date)"
test -f CrI3_scf_soc_a1h_clean.in
test -f ./pseudo/Cr.rel-pbe.upf
test -f a1h_prepare_scf_run.py

$PY - "$ROOT/inputs/CrI3_scf_soc.in" CrI3_scf_soc_a1h_clean.in <<'PY'
import sys
from pathlib import Path
def filt(p):
    return [ln.rstrip("\r\n") for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
            if "outdir" not in ln.lower()]
if filt(sys.argv[1]) != filt(sys.argv[2]):
    raise SystemExit("REFUSED: SCF input differs beyond outdir")
print("SCF input matches April file beyond outdir")
PY

SAVE_DIR=temp_a1h_scf_clean/CrI3_scf_soc.save
SCF_LOG=logs/CrI3_scf_soc_a1h_clean.out
MAX_ATTEMPTS=4

mkdir -p logs temp_a1h_scf_clean
: > "$SCF_LOG"

scf_done() {
  grep -q 'JOB DONE.' "$SCF_LOG" && grep -q 'convergence has been achieved' "$SCF_LOG"
}

save_ok() {
  [[ -f "$SAVE_DIR/charge-density.dat" ]] || [[ -f "$SAVE_DIR/charge-density.hdf5" ]] \
    || compgen -G "$SAVE_DIR/wfc*.dat" >/dev/null \
    || compgen -G "$SAVE_DIR/wfc*.hdf5" >/dev/null
}

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  $PY a1h_prepare_scf_run.py CrI3_scf_soc_a1h_clean.in "$SAVE_DIR" CrI3_scf_soc_a1h_run.in --nscf-reserve 9000
  echo "SCF attempt ${attempt}/${MAX_ATTEMPTS} $(date)"
  set +e
  mpirun -np "$NP" $PW -nk "$NK" -in CrI3_scf_soc_a1h_run.in >> "$SCF_LOG" 2>&1
  scf_rc=$?
  set -e
  if scf_done; then
    echo "SCF converged rc=${scf_rc} attempt=${attempt}"
    break
  fi
  if grep -q 'Maximum CPU time exceeded' "$SCF_LOG"; then
    echo "SCF stopped on max_seconds; save kept for a later sbatch" >&2
    exit 5
  fi
  if save_ok; then
    echo "SCF incomplete rc=${scf_rc}; restarting from save $(date)"
    sleep 10
    continue
  fi
  echo "SCF died rc=${scf_rc} with no restartable save" >&2
  exit 4
done

if ! scf_done; then
  echo "SCF did not converge after ${attempt} attempts" >&2
  exit 4
fi

set +e
$PY a1h_scf_provenance_gate.py logs/CrI3_scf_soc_a1h_clean.out a1h_provenance_targets.json logs/a1h_scf_provenance_gate.json
GATE_RC=$?
set -e
if [ "$GATE_RC" -ne 0 ] || ! grep -q 'A1H_SCF_PROVENANCE_GATE=PASS' logs/A1H_SCF_PROVENANCE_GATE.txt; then
  exit 3
fi
grep -q '<calculation>scf</calculation>' temp_a1h_scf_clean/CrI3_scf_soc.save/data-file-schema.xml
mpirun -np "$NP" $PW -nk "$NK" -in CrI3_nscf_soc_a1h_6k.in > logs/CrI3_nscf_soc_a1h_6k.out 2>&1
set +e
$PY a1h_gamma_eig_gate.py logs/CrI3_nscf_soc_a1h_6k.out CrI3.eig a1h_provenance_targets.json logs/a1h_gamma_eig_gate.json
GAMMA_RC=$?
set -e
[ "$GAMMA_RC" -eq 0 ]
echo "A1-H pipeline complete $(date)"
