#!/bin/bash
#=============================================================================
# Phase 0 gauge check: matrix_vg vs Peierls vg
# CrI3, 10x10 k-grid, full 112-band Wannier window, dephasing off.
#
# Submit from the project root on the server:
#   sbatch deploy/run_phase0_gauge_10x10_112_nodeph.sh
#=============================================================================

#SBATCH --job-name=phase0_gauge
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=phase0_gauge_%j.out
#SBATCH --error=phase0_gauge_%j.err

set -euo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
WORKDIR="${WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_phase0_gauge_10x10_112_nodeph}"

MATRIX_TEMPLATE="deploy/input_phase0_matrixvg_10x10_112_nodeph.nml"
PEIERLS_TEMPLATE="deploy/input_phase0_peierlsvg_10x10_112_nodeph.nml"

cd "${WORKDIR}"

echo "============================================="
echo "  Phase 0 gauge check"
echo "============================================="
echo "  Date:     $(date)"
echo "  Host:     $(hostname)"
echo "  Workdir:  ${WORKDIR}"
echo "  TB file:  ${TB_FILE}"
echo "  Output:   ${OUTROOT}"
echo "  Threads:  ${NTHREADS}"

if [ ! -f "${TB_FILE}" ]; then
  echo "ERROR: TB file not found: ${TB_FILE}"
  exit 1
fi

if [ ! -f "${MATRIX_TEMPLATE}" ] || [ ! -f "${PEIERLS_TEMPLATE}" ]; then
  echo "ERROR: Missing Phase 0 input templates under deploy/."
  exit 1
fi

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

export FC=ifort
export FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays"
export LDFLAGS=""

echo "Compiling..."
make clean
make FC="${FC}" FFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}"

mkdir -p "${OUTROOT}/matrix_vg" "${OUTROOT}/peierls_vg"

prepare_input() {
  local template="$1"
  local outdir="$2"
  sed -e "s|wannier_tb_file = .*|wannier_tb_file = \"${TB_FILE}\"|" \
      "${template}" > "${outdir}/input.nml"
  sed -i 's/\r$//' "${outdir}/input.nml"
}

run_case() {
  local name="$1"
  local outdir="$2"
  echo ""
  echo "============================================="
  echo "  Running ${name}: $(date)"
  echo "============================================="
  cd "${outdir}"
  export OMP_NUM_THREADS="${NTHREADS}"
  export MKL_NUM_THREADS="${NTHREADS}"
  export OMP_STACKSIZE=256M
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  grep -E "nkx|nky|nb_start|nb_end|T2_fs|T2_cycles|dt|gauge_method|wannier_tb_file|use_external_A" input.nml
  { time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
}

prepare_input "${MATRIX_TEMPLATE}" "${OUTROOT}/matrix_vg"
prepare_input "${PEIERLS_TEMPLATE}" "${OUTROOT}/peierls_vg"

run_case "matrix_vg 10x10 112 bands T2 off" "${OUTROOT}/matrix_vg"
run_case "Peierls vg 10x10 112 bands T2 off" "${OUTROOT}/peierls_vg"

cd "${OUTROOT}"

awk '
function abs(x) { return x < 0 ? -x : x }
BEGIN {
  nt = split("1 3 5 7 9 11 13 15 20 25 30 35", target, " ")
}
FNR == NR {
  if ($1 !~ /^#/ && NF >= 5) {
    n_m++
    h_m[n_m] = $1 + 0.0
    y_m[n_m] = $NF + 0.0
  }
  next
}
{
  if ($1 !~ /^#/ && NF >= 5) {
    n_p++
    h_p[n_p] = $1 + 0.0
    y_p[n_p] = $NF + 0.0
  }
}
END {
  print "H,matrix_order,matrix_HHG,peierls_order,peierls_HHG,matrix_over_peierls,rel_diff"
  for (it = 1; it <= nt; it++) {
    h0 = target[it] + 0.0
    best_m = 1
    best_p = 1
    for (i = 2; i <= n_m; i++) {
      if (abs(h_m[i] - h0) < abs(h_m[best_m] - h0)) best_m = i
    }
    for (i = 2; i <= n_p; i++) {
      if (abs(h_p[i] - h0) < abs(h_p[best_p] - h0)) best_p = i
    }
    ratio = (y_p[best_p] != 0.0) ? y_m[best_m] / y_p[best_p] : "nan"
    denom = abs(y_p[best_p])
    if (denom < 1.0e-300) denom = 1.0e-300
    rel = abs(y_m[best_m] - y_p[best_p]) / denom
    printf "%g,%.8g,%.12e,%.8g,%.12e,%.12e,%.12e\n", \
      h0, h_m[best_m], y_m[best_m], h_p[best_p], y_p[best_p], ratio, rel
  }
}
' matrix_vg/HHG.dat peierls_vg/HHG.dat | tee phase0_key_harmonics.csv

echo ""
echo "Done: $(date)"
echo "Outputs:"
echo "  ${OUTROOT}/matrix_vg"
echo "  ${OUTROOT}/peierls_vg"
echo "  ${OUTROOT}/phase0_key_harmonics.csv"
