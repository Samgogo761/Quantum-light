#!/bin/bash
#=============================================================================
# HHG-SBE validation run: CrI3, 10x10 k-grid, full 112-band Wannier window
#
# Submit on the server with:
#   sbatch deploy/run_hhg_10x10_112bands.sh
#
# If copied from Windows and SLURM reports DOS line breaks, run:
#   sed -i 's/\r$//' deploy/run_hhg_10x10_112bands.sh
#=============================================================================

#SBATCH --job-name=hhg_10x10_112b
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=hhg_10x10_112b_%j.out
#SBATCH --error=hhg_10x10_112b_%j.err

set -eo pipefail

NTHREADS=36
COMPILER="intel"

TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat"
WORKDIR="/public/home/wangjs/project/New_SBEs/Quantum-light"
EXTERNAL_A_FILE="${WORKDIR}/external/old_vg_a_t.txt"
OUTDIR="${WORKDIR}/output_step1_matrixvg_10x10_112bands_externalA_nodeph"
INPUT_TEMPLATE="deploy/input_10x10_112bands.nml"

cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  HHG-SBE Solver - 10x10 / 112-band validation"
echo "============================================="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Workdir:    ${WORKDIR}"
echo "  Threads:    ${NTHREADS}"
echo "  Compiler:   ${COMPILER}"
echo "  Output:     ${OUTDIR}"
echo "  A(t) file:  ${EXTERNAL_A_FILE}"

if command -v free &>/dev/null; then
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
    echo "  Total RAM:  ${TOTAL_MEM_GB} GB"
    echo "  Avail RAM:  ${AVAIL_MEM_GB} GB"
    if [ "${AVAIL_MEM_GB}" -lt 20 ]; then
        echo "  WARNING: Available memory < 20 GB."
        echo "  10x10 / 112-band HR_proj is expected to need roughly 12 GB plus overhead."
        echo "  Continue anyway? Ctrl+C to abort; waiting 10s."
        sleep 10
    fi
fi

if [ -f /proc/cpuinfo ]; then
    NCPU=$(grep -c ^processor /proc/cpuinfo)
    echo "  CPU cores:  ${NCPU}"
    if [ "${NTHREADS}" -gt "${NCPU}" ]; then
        NTHREADS=${NCPU}
        echo "  Adjusted threads to: ${NTHREADS}"
    fi
fi

if [ ! -f "${TB_FILE}" ]; then
    echo "ERROR: CrI3_tb.dat not found at: ${TB_FILE}"
    exit 1
fi

if [ ! -f "${INPUT_TEMPLATE}" ]; then
    echo "ERROR: input template not found: ${INPUT_TEMPLATE}"
    exit 1
fi

if [ ! -f "${EXTERNAL_A_FILE}" ]; then
    echo "ERROR: external A(t) file not found: ${EXTERNAL_A_FILE}"
    echo "Copy the old VG a_t.txt to this path before submitting:"
    echo "  mkdir -p ${WORKDIR}/external"
    echo "  cp /path/to/old/5.14test/a_t.txt ${EXTERNAL_A_FILE}"
    exit 1
fi

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh

if [ "${COMPILER}" = "intel" ]; then
    export FC=ifort
    export FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays"
    export LDFLAGS=""
    make clean
    make FC="${FC}" FFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}"
else
    export FC=gfortran
    export FFLAGS="-O2 -fopenmp -Wall -std=f2008 -fall-intrinsics"
    export LDFLAGS="-llapack -lblas -lfftw3"
    make clean
    make
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

mkdir -p "${OUTDIR}"
sed -e "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
    -e "s|external_A_file = .*|external_A_file = \"${EXTERNAL_A_FILE}\"|" \
    "${INPUT_TEMPLATE}" > "${OUTDIR}/input.nml"

echo "--- Effective run parameters ---"
grep -E "nkx|nky|nb_start|nb_end|ncyc|T2_fs|dt|wvl_nm|intensity_Wcm2|gauge_method|wannier_tb_file|use_external_A|external_A_file" "${OUTDIR}/input.nml"

cd "${OUTDIR}"
export OMP_NUM_THREADS=${NTHREADS}
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================="
echo "  Starting HHG-SBE calculation"
echo "  k-grid:    10 x 10"
echo "  Bands:     112 (1-112)"
echo "  Gauge:     matrix_vg"
echo "  Pulse:     external A(t) copied from old VG a_t.txt"
echo "  T2:        effectively off (1.0e30 fs)"
echo "  Start:     $(date)"
echo "============================================="

{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee "${OUTDIR}/run.log"

echo ""
echo "============================================="
echo "  Calculation finished: $(date)"
echo "============================================="

if [ -f "${OUTDIR}/HHG.dat" ]; then
    echo "--- HHG sanity check (first 10 harmonics) ---"
    awk '$1 ~ /^[0-9]/ && $1+0 >= 0.8 && $1+0 <= 10.2 && ($1+0)%1 < 0.15 {
        printf "  H%-3d  HHG_total = %s\n", int($1+0.5), $5
    }' "${OUTDIR}/HHG.dat"
fi

echo "All output in: ${OUTDIR}/"
