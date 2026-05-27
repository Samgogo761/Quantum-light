#!/bin/bash
#=============================================================================
# Weak-field scaling test: E0, E0/2, E0/4
# Validates perturbative response:
#   J(omega)  ~ E0     (linear)
#   J(2omega) ~ E0^2   (second-order)
#   J(3omega) ~ E0^3   (third-order)
#
# Usage:
#   sbatch deploy/run_weakfield_scaling.sh
# After completion, use tools/analysis/check_weakfield_scaling.py
#=============================================================================

#--- SLURM directives ---
#SBATCH --job-name=hhg_weakfield
#SBATCH --partition=part_1
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=weakfield_%A_%a.out
#SBATCH --error=weakfield_%A_%a.err

NTHREADS=36
TB_FILE="${WEAKFIELD_TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
WORKDIR="${WEAKFIELD_WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# E0 scales: full, half, quarter (I scales as E0^2)
INTENSITY_FULL=${WEAKFIELD_INTENSITY_FULL:-2.0e11}
OUTROOT="${WEAKFIELD_OUTROOT:-${WORKDIR}/output_weakfield}"
case ${TASK_ID} in
  0) SCALE="E0";    INTENSITY=${INTENSITY_FULL} ;;
  1) SCALE="E0_2";  INTENSITY=$(awk "BEGIN{printf \"%.4e\", ${INTENSITY_FULL}/4.0}") ;;
  2) SCALE="E0_4";  INTENSITY=$(awk "BEGIN{printf \"%.4e\", ${INTENSITY_FULL}/16.0}") ;;
esac

OUTDIR="${OUTROOT}/${SCALE}"

cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  Weak-field scaling: ${SCALE}"
echo "  Intensity: ${INTENSITY} W/cm^2"
echo "============================================="

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

if [ ! -f hhg_sbe ] || [ "$(find src/ -name '*.f90' -newer hhg_sbe 2>/dev/null)" ]; then
    export FC=ifort
    export FFLAGS="-O3 -qopenmp -mkl -fpp"
    make clean && make FC="${FC}" FFLAGS="${FFLAGS}" || { echo "ERROR: Compilation failed!"; exit 1; }
fi

mkdir -p "${OUTDIR}"

sed -e "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
    -e "s|intensity_Wcm2 = .*|intensity_Wcm2 = ${INTENSITY}|" \
    -e "s|bsv_enabled = .*|bsv_enabled = .false.|" \
    deploy/input_production.nml > "${OUTDIR}/input.nml"

cd "${OUTDIR}"

export OMP_NUM_THREADS=${NTHREADS}
export OMP_STACKSIZE=64M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Starting: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee "${OUTDIR}/run.log"
echo "Finished: $(date)"

ls -lh "${OUTDIR}"/*.dat 2>/dev/null
