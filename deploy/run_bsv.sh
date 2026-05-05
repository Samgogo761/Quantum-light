#!/bin/bash
#=============================================================================
# BSV (Bright Squeezed Vacuum) HHG - SLURM Array Job
# CrI3 bilayer AFM with SOC, 120x120 k-grid
#
# Splits N_TOTAL BSV samples across N_JOBS array tasks.
# Each task runs N_PER_JOB samples with a unique seed.
# After all tasks complete, run deploy/combine_bsv.sh to merge results.
#
# Usage:
#   sbatch deploy/run_bsv.sh
#   # wait for all tasks to finish, then:
#   bash deploy/combine_bsv.sh
#=============================================================================

#--- SLURM directives ---
#SBATCH --job-name=hhg_bsv
#SBATCH --partition=part_1
#SBATCH --array=0-9%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=bsv_%A_%a.out
#SBATCH --error=bsv_%A_%a.err

#=============================================================================
# CONFIGURATION
#=============================================================================

NTHREADS=36
TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat"
WORKDIR="/public/home/wangjs/project/Quantum-light"

N_TOTAL=500
N_JOBS=10
N_PER_JOB=$((N_TOTAL / N_JOBS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((42 + TASK_ID * 1000))
OUTDIR="${WORKDIR}/output_bsv/task_${TASK_ID}"

#=============================================================================
# Step 0: Change to workdir and verify
#=============================================================================
cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  BSV HHG - Array Task ${TASK_ID} / $((N_JOBS - 1))"
echo "============================================="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Samples:    ${N_PER_JOB} (of ${N_TOTAL} total)"
echo "  Seed:       ${SEED}"
echo "  Output:     ${OUTDIR}"
echo "  Threads:    ${NTHREADS}"
echo ""

if [ ! -f "${TB_FILE}" ]; then
    echo "  ERROR: CrI3_tb.dat not found at: ${TB_FILE}"
    exit 1
fi

#=============================================================================
# Step 1: Environment setup
#=============================================================================

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh

#=============================================================================
# Step 2: Compile (only task 0 compiles; others wait)
#=============================================================================

cd "${WORKDIR}"

if [ "${TASK_ID}" -eq 0 ]; then
    rm -f "${WORKDIR}/.compile_done"
    if [ ! -f hhg_sbe ] || [ "$(find src/ -name '*.f90' -newer hhg_sbe 2>/dev/null)" ]; then
        export FC=ifort
        export FFLAGS="-O3 -qopenmp -mkl -fpp"
        export LDFLAGS=""
        echo "Task 0: Compiling..."
        make clean
        make FC="${FC}" FFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}"
        if [ $? -ne 0 ]; then
            echo "ERROR: Compilation failed!"
            exit 1
        fi
    fi
    touch "${WORKDIR}/.compile_done"
else
    echo "Task ${TASK_ID}: Waiting for compilation..."
    WAIT_COUNT=0
    while [ ! -f "${WORKDIR}/.compile_done" ]; do
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [ "${WAIT_COUNT}" -gt 120 ]; then
            echo "ERROR: Timed out waiting for compilation (10 min)."
            exit 1
        fi
    done
    sleep 2
fi

#=============================================================================
# Step 3: Prepare task-specific run directory
#=============================================================================

mkdir -p "${OUTDIR}"

sed -e "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
    -e "s|bsv_n_samples = .*|bsv_n_samples = ${N_PER_JOB}|" \
    -e "s|bsv_seed = .*|bsv_seed = ${SEED}|" \
    deploy/input_bsv.nml > "${OUTDIR}/input.nml"

echo "Input: ${OUTDIR}/input.nml"
echo "  bsv_n_samples = ${N_PER_JOB}"
echo "  bsv_seed = ${SEED}"
echo ""

#=============================================================================
# Step 4: Run
#=============================================================================

cd "${OUTDIR}"

export OMP_NUM_THREADS=${NTHREADS}
export OMP_STACKSIZE=64M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================="
echo "  Starting BSV task ${TASK_ID}: $(date)"
echo "============================================="

{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee "${OUTDIR}/run.log"

echo ""
echo "Task ${TASK_ID} finished: $(date)"

ls -lh "${OUTDIR}"/*.dat 2>/dev/null
