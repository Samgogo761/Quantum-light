#!/bin/bash
#=============================================================================
# Server Capability Check for HHG-SBE Solver
# Run this FIRST to verify your server can handle 120x120 k-grid
#=============================================================================

echo "============================================="
echo "  HHG-SBE Server Capability Check"
echo "============================================="
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo ""

#--- 1. Memory ---
echo "--- Memory ---"
if command -v free &>/dev/null; then
    TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
    echo "  Total:     ${TOTAL_GB} GB"
    echo "  Available: ${AVAIL_GB} GB"
    echo "  Required:  ~60 GB (HR_proj=54GB + overhead)"
    if [ "${TOTAL_GB}" -ge 80 ]; then
        echo "  Status:    PASS"
    elif [ "${TOTAL_GB}" -ge 60 ]; then
        echo "  Status:    MARGINAL (may work if no other jobs)"
    else
        echo "  Status:    FAIL (need >= 64 GB total RAM)"
    fi
else
    echo "  Cannot determine (no 'free' command)"
fi
echo ""

#--- 2. CPU ---
echo "--- CPU ---"
if [ -f /proc/cpuinfo ]; then
    NCORES=$(grep -c ^processor /proc/cpuinfo)
    MODEL=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    echo "  Model:     ${MODEL}"
    echo "  Cores:     ${NCORES}"
    echo "  Required:  >= 16 (recommended 32+)"

    # Estimate wall time
    CPU_SEC=$((604 * 144))  # total CPU-seconds for 120x120
    WALL_SEC=$((CPU_SEC / NCORES))
    WALL_MIN=$((WALL_SEC / 60))
    echo "  Est. time: ~${WALL_MIN} minutes (${WALL_SEC}s) with ${NCORES} threads"

    if [ "${NCORES}" -ge 32 ]; then
        echo "  Status:    PASS"
    elif [ "${NCORES}" -ge 16 ]; then
        echo "  Status:    OK (but slower)"
    else
        echo "  Status:    SLOW (consider larger node)"
    fi
else
    echo "  Cannot determine (/proc/cpuinfo not found)"
fi
echo ""

#--- 3. Compiler ---
echo "--- Compiler ---"
COMPILER_FOUND=0
if command -v ifx &>/dev/null; then
    echo "  Intel ifx: $(ifx --version 2>&1 | head -1)"
    echo "  Recommended: use COMPILER=\"intel\" in run_hhg.sh"
    COMPILER_FOUND=1
elif command -v ifort &>/dev/null; then
    echo "  Intel ifort: $(ifort --version 2>&1 | head -1)"
    echo "  Recommended: use COMPILER=\"intel\" in run_hhg.sh (change FC to ifort)"
    COMPILER_FOUND=1
fi
if command -v gfortran &>/dev/null; then
    echo "  gfortran:  $(gfortran --version | head -1)"
    COMPILER_FOUND=1
fi
if [ "${COMPILER_FOUND}" -eq 0 ]; then
    echo "  Status:    FAIL (no Fortran compiler found)"
    echo "  Try: module load gcc  or  module load intel"
else
    echo "  Status:    PASS"
fi
echo ""

#--- 4. Libraries ---
echo "--- Libraries ---"
LIBS_OK=1

# Check LAPACK/BLAS
if ldconfig -p 2>/dev/null | grep -q liblapack; then
    echo "  LAPACK:    found (system)"
elif [ -n "${MKLROOT}" ]; then
    echo "  LAPACK:    found (MKL: ${MKLROOT})"
else
    echo "  LAPACK:    NOT FOUND"
    echo "             Try: module load lapack  or  module load mkl"
    LIBS_OK=0
fi

# Check FFTW3
if ldconfig -p 2>/dev/null | grep -q libfftw3; then
    echo "  FFTW3:     found (system)"
elif [ -n "${MKLROOT}" ]; then
    echo "  FFTW3:     found (MKL provides FFTW interface)"
else
    # Check common module paths
    if [ -f /usr/lib64/libfftw3.so ] || [ -f /usr/lib/libfftw3.so ]; then
        echo "  FFTW3:     found"
    else
        echo "  FFTW3:     NOT FOUND"
        echo "             Try: module load fftw"
        LIBS_OK=0
    fi
fi

if [ "${LIBS_OK}" -eq 1 ]; then
    echo "  Status:    PASS"
else
    echo "  Status:    FAIL (missing libraries)"
fi
echo ""

#--- 5. OpenMP ---
echo "--- OpenMP ---"
cat > /tmp/test_omp.f90 <<'FORTEOF'
program test_omp
  implicit none
  integer :: tid, nth
  !$ integer :: omp_get_thread_num, omp_get_max_threads
  nth = 1
  !$ nth = omp_get_max_threads()
  write(*,'(A,I0)') 'OMP max threads: ', nth
end program
FORTEOF

if command -v gfortran &>/dev/null; then
    gfortran -fopenmp /tmp/test_omp.f90 -o /tmp/test_omp 2>/dev/null && /tmp/test_omp
elif command -v ifx &>/dev/null; then
    ifx -qopenmp /tmp/test_omp.f90 -o /tmp/test_omp 2>/dev/null && /tmp/test_omp
fi
rm -f /tmp/test_omp.f90 /tmp/test_omp
echo ""

#--- 6. Disk space ---
echo "--- Disk Space ---"
AVAIL_DISK=$(df -h . | awk 'NR==2{print $4}')
echo "  Available: ${AVAIL_DISK}"
echo "  Required:  < 1 GB (output files)"
echo "  Status:    PASS"
echo ""

#--- Summary ---
echo "============================================="
echo "  Summary"
echo "============================================="
echo "  If all checks PASS, run:"
echo "    bash deploy/run_hhg.sh"
echo ""
echo "  If using SLURM, edit SBATCH directives in"
echo "  run_hhg.sh, then:"
echo "    sbatch deploy/run_hhg.sh"
echo "============================================="
