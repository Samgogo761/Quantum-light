#!/bin/bash
#SBATCH --job-name=a1h_scf_nscf6k
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=56G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=72:00:00
#
# Workspace 2: part_1 is full. This job is NOT the 576-k nscf (208 GB).
# April SCF itself ran with ~50 GB available (2.11 GB/proc).
# 6k NSCF uses nbnd=200; -nk 1 so 8 ranks split G-space (original 576k nscf
# used -nk 8 / 2 ranks per pool and needed 13 GB/proc). 8 G-ranks should fit 56G.
# If OOM: resubmit sbatch_a1h_scf_then_nscf_part1.sh
# MPI: 8 ranks, -nk 1 (divides). NEVER run in temp_scf_soc.
# NSCF starts only if A1H_SCF_PROVENANCE_GATE=PASS.

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
NP=8
NK=1
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
echo "PY=$PY"

cd "$WORKDIR"
mkdir -p logs temp_a1h_scf_clean pseudo

echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} $(date)"
echo "WORKDIR=$WORKDIR partition=part_2 ntasks=${NP} nk=${NK}"

test -f CrI3_scf_soc_a1h_clean.in
test -f CrI3_nscf_soc_a1h_6k.in
test -f a1h_provenance_targets.json
test -f a1h_scf_provenance_gate.py
test -f a1h_gamma_eig_gate.py
test -f ./pseudo/Cr.rel-pbe.upf
test -f ./pseudo/I.rel-pbe.upf
test -f CrI3.eig

$PY - "$ROOT/inputs/CrI3_scf_soc.in" CrI3_scf_soc_a1h_clean.in <<'PY'
import sys
from pathlib import Path
def filt(p):
    return [ln.rstrip("\r\n") for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
            if "outdir" not in ln.lower()]
a, b = sys.argv[1], sys.argv[2]
fa, fb = filt(a), filt(b)
if fa != fb:
    print("REFUSED: SCF input differs from April CrI3_scf_soc.in beyond outdir", file=sys.stderr)
    sys.exit(2)
print("SCF input matches April file beyond outdir")
PY

echo "=== SCF ==="
mpirun -np "$NP" $PW -nk "$NK" -in CrI3_scf_soc_a1h_clean.in > logs/CrI3_scf_soc_a1h_clean.out 2>&1
echo "pw scf wall done $(date)"

echo "=== SCF provenance gate ==="
set +e
$PY a1h_scf_provenance_gate.py \
  logs/CrI3_scf_soc_a1h_clean.out \
  a1h_provenance_targets.json \
  logs/a1h_scf_provenance_gate.json
GATE_RC=$?
set -e
if [ "$GATE_RC" -ne 0 ] || ! grep -q 'A1H_SCF_PROVENANCE_GATE=PASS' logs/A1H_SCF_PROVENANCE_GATE.txt; then
  echo "REFUSED: SCF provenance gate not PASS; NSCF not started" >&2
  cat logs/A1H_SCF_PROVENANCE_GATE.txt >&2 || true
  exit 3
fi
grep -q '<calculation>scf</calculation>' temp_a1h_scf_clean/CrI3_scf_soc.save/data-file-schema.xml

echo "=== NSCF 6k ==="
mpirun -np "$NP" $PW -nk "$NK" -in CrI3_nscf_soc_a1h_6k.in > logs/CrI3_nscf_soc_a1h_6k.out 2>&1
echo "pw nscf wall done $(date)"

echo "=== Gamma vs CrI3.eig gate ==="
set +e
$PY a1h_gamma_eig_gate.py \
  logs/CrI3_nscf_soc_a1h_6k.out \
  CrI3.eig \
  a1h_provenance_targets.json \
  logs/a1h_gamma_eig_gate.json
GAMMA_RC=$?
set -e
echo "gamma gate rc=${GAMMA_RC}"
cat logs/A1H_GAMMA_EIG_GATE.txt || true
if [ "$GAMMA_RC" -ne 0 ]; then
  echo "Gamma gate FAIL: off-mesh A1-H must not be read as interpolation error" >&2
  exit 4
fi
echo "A1-H pipeline complete $(date)"
