#!/bin/bash
#=============================================================================
# HHG-SBE validation run: CrI3, 40x40 k-grid, 30-band window, velocity gauge
#
# Submit on the server with:
#   sbatch deploy/run_hhg_40x40_30bands.sh
#
# If the script was copied from Windows and SLURM reports DOS line breaks, run:
#   sed -i 's/\r$//' deploy/run_hhg_40x40_30bands.sh
#=============================================================================

#--- SLURM directives ---
#SBATCH --job-name=hhg_40x40_30b
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=hhg_40x40_30b_%j.out
#SBATCH --error=hhg_40x40_30b_%j.err

#=============================================================================
# CONFIGURATION
#=============================================================================

NTHREADS=36
COMPILER="intel"

# Absolute path to the server-side Wannier90 tb file.
TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat"

# Project root on the server.
WORKDIR="/public/home/wangjs/project/New_SBEs/Quantum-light"

# Keep this validation run separate from 120x120 or BSV outputs.
OUTDIR="${WORKDIR}/output_40x40_30bands_dt035_T2fs05"

# Input template for this validation run.
INPUT_TEMPLATE="deploy/input_40x40_30bands.nml"

#=============================================================================
# Step 0: Change to workdir and verify
#=============================================================================

cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }
pwd

echo "============================================="
echo "  HHG-SBE Solver - 40x40 / 30-band validation"
echo "============================================="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Workdir:    ${WORKDIR}"
echo "  Threads:    ${NTHREADS}"
echo "  Compiler:   ${COMPILER}"
echo "  Output:     ${OUTDIR}"
echo ""

if command -v free &>/dev/null; then
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
    echo "  Total RAM:  ${TOTAL_MEM_GB} GB"
    echo "  Avail RAM:  ${AVAIL_MEM_GB} GB"
    if [ "${AVAIL_MEM_GB}" -lt 20 ]; then
        echo "  WARNING: Available memory < 20 GB."
        echo "  40x40 / 30-band HR_proj is expected to need roughly 10-15 GB plus overhead."
        echo "  Continue anyway? Ctrl+C to abort; waiting 10s."
        sleep 10
    fi
fi

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

if [ ! -f "${TB_FILE}" ]; then
    echo "  ERROR: CrI3_tb.dat not found at: ${TB_FILE}"
    echo "  Edit TB_FILE in this script."
    exit 1
fi
echo "  TB file:    ${TB_FILE} (found)"

if [ ! -f "${INPUT_TEMPLATE}" ]; then
    echo "  ERROR: input template not found: ${INPUT_TEMPLATE}"
    exit 1
fi

echo "============================================="
echo ""

#=============================================================================
# Step 1: Environment setup (Intel oneAPI on CentOS 7)
#=============================================================================

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh

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

sed "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
    "${INPUT_TEMPLATE}" > "${OUTDIR}/input.nml"

echo "Input file prepared at: ${OUTDIR}/input.nml"
echo "--- Effective run parameters ---"
grep -E "nkx|nky|nb_start|nb_end|T2_fs|dt|wvl_nm|intensity_Wcm2|gauge_method|wannier_tb_file" "${OUTDIR}/input.nml"
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
echo "  Starting HHG-SBE calculation"
echo "  k-grid:    40 x 40"
echo "  Bands:     30 (70-99)"
echo "  T2:        0.5 fs"
echo "  Threads:   ${OMP_NUM_THREADS}"
echo "  Start:     $(date)"
echo "============================================="

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
