#!/bin/bash
#SBATCH --job-name=shell_upf_c
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=56G
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=24:00:00
#
# Scheme C: one UPF-defined I-d projector check. No zona scan.
# No new Wannier parent. Do not delete wfc. Do not write temp_scf_soc.
# Patched pw2wannier90 is local; production QE binary is not overwritten.
# QE 7.2 pw2wannier90: never pass -nk.

set -eo pipefail
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mkl/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export I_MPI_PIN_DOMAIN=core

QE=/public/home/wangjs/software/q-e-qe-7.2
WORKDIR=/public/home/wangjs/project/CrI3_TB/window144_hilbert/a1_fewk_nscf
NP=8
PY=/public/home/wangjs/software/miniconda3/bin/python

cd "$WORKDIR"
echo "Host=$(hostname) Job=${SLURM_JOB_ID:-none} partition=part_2 ntasks=${NP} $(date)"

test -f CrI3_shell_upf.nnkp
test -f CrI3_shell_upf.pw2wan
test -f CrI3_shell_upf_map.json
test -f CrI3_shell_upf.radial_id
test -f CrI3_shell_upf.radial_R
test -f CrI3_a1o.mmn
SAVE=temp_a1h_scf_clean/CrI3_scf_soc.save
n_wfc=$(ls -1 "$SAVE"/wfc*.dat 2>/dev/null | wc -l)
echo "n_wfc=${n_wfc}"
[ "$n_wfc" -eq 52 ]

mkdir -p logs qe_pw2wan_c
test -f "$QE/PP/src/pw2wannier90.f90"
$PY patch_pw2wannier90_upf_radial.py "$QE/PP/src/pw2wannier90.f90" qe_pw2wan_c/pw2wannier90.f90
cp -f Makefile.pw2wannier90_upf qe_pw2wan_c/Makefile
make -C qe_pw2wan_c -f Makefile pw2wannier90_upf.x
PW2="$WORKDIR/qe_pw2wan_c/pw2wannier90_upf.x"
test -x "$PW2"
echo "PATCHED_PW2=$PW2"

PW2_LOG=logs/CrI3_shell_upf_pw2wan.out
mpirun -np "$NP" "$PW2" -in CrI3_shell_upf.pw2wan > "$PW2_LOG" 2>&1
grep -q 'JOB DONE.' "$PW2_LOG"
grep -q 'UPF_CUSTOM_RADIAL_LOADED' "$PW2_LOG"
test -f CrI3_shell_upf.amn
echo "UPF_C_AMN_DONE $(date)"

set +e
$PY screen_q32_shell_trials.py \
  --amn CrI3_shell_upf.amn \
  --mmn CrI3_a1o.mmn \
  --eig CrI3_a1o.eig \
  --nnkp-a1o CrI3_a1o.nnkp \
  --star-map a1o_star_map.json \
  --shell-map CrI3_shell_upf_map.json \
  --out logs/q32_shell_upf_screen.json
SCR_RC=$?
$PY compare_upf_projector_c.py \
  --screen logs/q32_shell_upf_screen.json \
  --radials logs/upf_defined_radials.json \
  --out logs/q32_shell_upf_c.json
CMP_RC=$?
set -e
echo "UPF_C_SCREEN_RC=${SCR_RC} UPF_C_COMPARE_RC=${CMP_RC} $(date)"
[ "$SCR_RC" -eq 0 ]
[ "$CMP_RC" -eq 0 ]
echo "scheme C UPF projector check complete $(date)"
