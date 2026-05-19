#!/bin/bash
#=============================================================================
# HHG-SBE Solver Deployment Script
# CrI3 bilayer AFM with SOC, 120x120 k-grid, velocity gauge
#=============================================================================
#
# Usage:
#   1. Direct:   bash run_hhg.sh
#   2. SLURM:    sbatch run_hhg.sh
#   3. PBS:      qsub run_hhg.sh
#
# Before running, edit the CONFIGURATION section below.
#=============================================================================

#--- SLURM directives ---
#SBATCH --job-name=hhg_cri3
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=hhg_%j.out
#SBATCH --error=hhg_%j.err

#=============================================================================
# CONFIGURATION
#=============================================================================

# Number of OpenMP threads (set to number of physical cores)
NTHREADS=36

# Compiler: "gfortran" or "intel"
COMPILER="intel"

# Path to CrI3_tb.dat (absolute path)
TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat"

# Working directory - set this to the actual project root
WORKDIR="/public/home/wangjs/project/Quantum-light"

# Output directory. Keep the current-code validation run separate from older
# 120x120 outputs so spectra from different code versions are not mixed.
OUTDIR="${WORKDIR}/output_120x120_current_dt035"

#=============================================================================
# Step 0: Change to workdir and verify
#=============================================================================
cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }
pwd

echo "============================================="
echo "  HHG-SBE Solver - Pre-flight Check"
echo "============================================="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Workdir:    ${WORKDIR}"
echo "  Threads:    ${NTHREADS}"
echo "  Compiler:   ${COMPILER}"
echo ""

# Check available memory
if command -v free &>/dev/null; then
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
    echo "  Total RAM:  ${TOTAL_MEM_GB} GB"
    echo "  Avail RAM:  ${AVAIL_MEM_GB} GB"
    if [ "${AVAIL_MEM_GB}" -lt 60 ]; then
        echo "  WARNING: Available memory < 60 GB!"
        echo "  HR_proj for 120x120 needs ~54 GB."
        echo "  Consider reducing k-grid or freeing memory."
        echo "  Continue anyway? (Ctrl+C to abort, waiting 10s)"
        sleep 10
    else
        echo "  Memory:     OK"
    fi
fi

# Check CPU info
if [ -f /proc/cpuinfo ]; then
    NCPU=$(grep -c ^processor /proc/cpuinfo)
    CPU_MODEL=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    echo "  CPU cores:  ${NCPU}"
    echo "  CPU model:  ${CPU_MODEL}"
    if [ "${NTHREADS}" -gt "${NCPU}" ]; then
        echo "  WARNING: NTHREADS(${NTHREADS}) > available cores(${NCPU})"
        NTHREADS=${NCPU}
        echo "  Adjusted to: ${NTHREADS}"
    fi
fi

# Check Wannier data file
if [ ! -f "${TB_FILE}" ]; then
    echo "  ERROR: CrI3_tb.dat not found at: ${TB_FILE}"
    echo "  Edit TB_FILE in this script."
    exit 1
fi
echo "  TB file:    ${TB_FILE} (found)"

echo "============================================="
echo ""

#=============================================================================
# Step 1: Environment setup (Intel oneAPI on CentOS 7)
#=============================================================================

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

#=============================================================================
# Step 2: Compile
#=============================================================================

cd "${WORKDIR}"

if [ "${COMPILER}" = "intel" ]; then
    export FC=ifort
    export FFLAGS="-O3 -qopenmp -mkl -fpp"
    export LDFLAGS=""
    echo "Compiling with Intel ifort + MKL..."
    make clean
    make FC="${FC}" FFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}"
else
    export FC=gfortran
    export FFLAGS="-O2 -fopenmp -Wall -std=f2008 -fall-intrinsics"
    export LDFLAGS="-llapack -lblas -lfftw3"
    echo "Compiling with gfortran..."
    make clean
    make
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "Compilation successful."
echo ""

#=============================================================================
# Step 3: Prepare run directory
#=============================================================================

mkdir -p "${OUTDIR}"

# Copy input file and fix tb.dat path
sed "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
    deploy/input_production.nml > "${OUTDIR}/input.nml"

echo "Input file prepared at: ${OUTDIR}/input.nml"
echo ""

#=============================================================================
# Step 4: Run
#=============================================================================

cd "${OUTDIR}"

# Set OpenMP environment
export OMP_NUM_THREADS=${NTHREADS}
export OMP_STACKSIZE=64M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================="
echo "  Starting HHG-SBE calculation"
echo "  k-grid:    120 x 120"
echo "  Bands:     20 (75-94)"
echo "  Threads:   ${OMP_NUM_THREADS}"
echo "  Start:     $(date)"
echo "============================================="

# Use 'time' for accurate wall time
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee "${OUTDIR}/run.log"

echo ""
echo "============================================="
echo "  Calculation finished: $(date)"
echo "============================================="

#=============================================================================
# Step 5: Quick validation
#=============================================================================

echo ""
echo "--- Output files ---"
ls -lh "${OUTDIR}"/*.dat 2>/dev/null

if [ -f "${OUTDIR}/HHG.dat" ]; then
    echo ""
    echo "--- HHG sanity check (first 10 harmonics) ---"
    awk '$1 ~ /^[0-9]/ && $1+0 >= 0.8 && $1+0 <= 10.2 && ($1+0)%1 < 0.15 {
        printf "  H%-3d  HHG_total = %s\n", int($1+0.5), $5
    }' "${OUTDIR}/HHG.dat"
fi

echo ""
echo "All output in: ${OUTDIR}/"
echo "Done."
