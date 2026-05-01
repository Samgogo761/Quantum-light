#!/bin/bash
#=============================================================================
# BSV (Bright Squeezed Vacuum) HHG Calculation Script
# CrI3 bilayer AFM with SOC, 120x120 k-grid, 500 BSV samples
#=============================================================================
#
# Usage:
#   1. Direct:   bash deploy/run_bsv.sh
#   2. SLURM:    sbatch deploy/run_bsv.sh
#
# Before running, edit the CONFIGURATION section below.
#=============================================================================

#--- SLURM directives ---
#SBATCH --job-name=hhg_bsv
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=bsv_%j.out
#SBATCH --error=bsv_%j.err

#=============================================================================
# CONFIGURATION
#=============================================================================

NTHREADS=36
COMPILER="intel"
TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat"
WORKDIR="/public/home/wangjs/project/Quantum-light"
OUTDIR="${WORKDIR}/output_bsv"

#=============================================================================
# Step 0: Change to workdir and verify
#=============================================================================
cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  BSV HHG Calculation - Pre-flight Check"
echo "============================================="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Workdir:    ${WORKDIR}"
echo "  Threads:    ${NTHREADS}"
echo ""

if [ ! -f "${TB_FILE}" ]; then
    echo "  ERROR: CrI3_tb.dat not found at: ${TB_FILE}"
    exit 1
fi
echo "  TB file:    ${TB_FILE} (found)"
echo "============================================="
echo ""

#=============================================================================
# Step 1: Environment setup
#=============================================================================

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh

#=============================================================================
# Step 2: Compile (skip if binary exists and is newer than source)
#=============================================================================

cd "${WORKDIR}"

if [ ! -f hhg_sbe ] || [ "$(find src/ -name '*.f90' -newer hhg_sbe 2>/dev/null)" ]; then
    export FC=ifort
    export FFLAGS="-O3 -qopenmp -mkl -fpp"
    export LDFLAGS=""
    echo "Compiling with Intel ifort + MKL..."
    make clean
    make FC="${FC}" FFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed!"
        exit 1
    fi
    echo "Compilation successful."
else
    echo "Binary hhg_sbe is up to date, skipping compilation."
fi
echo ""

#=============================================================================
# Step 3: Prepare run directory
#=============================================================================

mkdir -p "${OUTDIR}"

sed "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
    deploy/input_bsv.nml > "${OUTDIR}/input.nml"

echo "Input file prepared at: ${OUTDIR}/input.nml"
echo ""

#=============================================================================
# Step 4: Run BSV calculation
#=============================================================================

cd "${OUTDIR}"

export OMP_NUM_THREADS=${NTHREADS}
export OMP_STACKSIZE=64M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================="
echo "  Starting BSV HHG calculation"
echo "  k-grid:      120 x 120"
echo "  BSV samples: 500"
echo "  Ī (scale):   1.0e11 W/cm² (⟨I⟩ = 2Ī = 2e11)"
echo "  Threads:     ${OMP_NUM_THREADS}"
echo "  Start:       $(date)"
echo "============================================="

{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee "${OUTDIR}/run.log"

echo ""
echo "============================================="
echo "  BSV calculation finished: $(date)"
echo "============================================="

#=============================================================================
# Step 5: Quick validation
#=============================================================================

echo ""
echo "--- Output files ---"
ls -lh "${OUTDIR}"/*.dat 2>/dev/null

if [ -f "${OUTDIR}/HHG_bsv.dat" ]; then
    echo ""
    echo "--- BSV HHG (first 10 harmonics) ---"
    awk '$1 ~ /^[0-9]/ && $1+0 >= 0.8 && $1+0 <= 10.2 && ($1+0)%1 < 0.15 {
        printf "  H%-3d  HHG_bsv = %s\n", int($1+0.5), $3
    }' "${OUTDIR}/HHG_bsv.dat"
fi

echo ""
echo "All output in: ${OUTDIR}/"
echo "Done."
