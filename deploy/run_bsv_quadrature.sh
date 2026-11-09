#!/bin/bash
#=============================================================================
# BSV deterministic intensity quadrature (replaces Monte-Carlo sampling).
#
#   HHG_bsv(w) = int_0^inf e^{-x} S(w; I0*x) dx,   I0 = 2e11 W/cm^2
#
# 16 deterministic classical runs at the Gauss-Legendre nodes listed in
# deploy/bsv_quad_nodes.dat, then weighted combination with
# tools/analysis/combine_bsv_quadrature.py. Quadrature reproduces the
# exponential moments <x^p>=p! to <=0.3% for p<=7, i.e. zero MC noise at
# ~25-100x less compute than the >=1000-sample MC equivalent.
#
# Usage (from the repo root; <=3 nodes constraint respected by %3):
#   sbatch deploy/run_bsv_quadrature.sh
#   # TB not at the default path:
#   TB_FILE=/path/CrI3_tb.dat sbatch deploy/run_bsv_quadrature.sh
#
# dt-convergence check of the strongest node (optional but recommended):
#   DT_AU=0.175 OUTROOT_SUFFIX=_dt0175 sbatch --array=15 deploy/run_bsv_quadrature.sh
#
# After all tasks finish (locally or on a python3 node):
#   python3 tools/analysis/combine_bsv_quadrature.py \
#     --quad-root output_bsv_quad --nodes deploy/bsv_quad_nodes.dat \
#     --classical <classical HHG.dat> --mc <HHG_bsv_pooled.dat>
#=============================================================================

#SBATCH --job-name=bsv_quad
#SBATCH --partition=part_1
#SBATCH --array=0-15%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=bsvquad_%A_%a.out
#SBATCH --error=bsvquad_%A_%a.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
DT_AU="${DT_AU:-0.35}"
OUTROOT="${WORKDIR}/output_bsv_quad${OUTROOT_SUFFIX:-}"

# Quadrature nodes: x = I/I0, I0 = 2e11 W/cm^2 (see deploy/bsv_quad_nodes.dat).
X_NODES=(0.119130 0.610001 1.423403 2.449696 3.550304 4.576597 5.389999 \
         5.880870 6.238261 7.220001 8.846806 10.899392 13.100608 15.153194 \
         16.779999 17.761739)
I_NODES=(2.382609e+10 1.220001e+11 2.846806e+11 4.899392e+11 7.100608e+11 \
         9.153194e+11 1.078000e+12 1.176174e+12 1.247652e+12 1.444000e+12 \
         1.769361e+12 2.179878e+12 2.620122e+12 3.030639e+12 3.356000e+12 \
         3.552348e+12)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${#I_NODES[@]}" ]; then
  echo "ERROR: array index ${IDX} outside 0..$(( ${#I_NODES[@]} - 1 ))"; exit 1
fi
XVAL="${X_NODES[${IDX}]}"
IVAL="${I_NODES[${IDX}]}"
OUTDIR=$(printf "%s/node_%02d" "${OUTROOT}" "${IDX}")

cd "${WORKDIR}" || { echo "ERROR: cannot cd ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  BSV quadrature node ${IDX}: x=${XVAL}, I=${IVAL} W/cm^2"
echo "  Date: $(date)  Host: $(hostname)"
echo "  Output: ${OUTDIR}   dt=${DT_AU}"
echo "============================================="

if [ ! -f "${TB_FILE}" ]; then
  echo "ERROR: TB file not found: ${TB_FILE}"; exit 1
fi

# vtune lib64 path FIRST is essential: it provides a newer libstdc++.so.6
# (GLIBCXX_3.4.21) that ifort requires; the system /lib64 one is too old.
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

# Order-independent compile: whichever task wins the atomic mkdir lock builds
# (clean, since the copied tree may carry incompatible Windows objects). A
# .compile.done marker means a good Linux binary exists; everyone else waits.
LOCK="${WORKDIR}/.compile_quad.lockdir"
DONE="${WORKDIR}/.compile_quad.done"
if [ ! -f "${DONE}" ] || [ ! -x "${WORKDIR}/hhg_sbe" ]; then
  WAITED=0
  while ! mkdir "${LOCK}" 2>/dev/null; do
    [ -f "${DONE}" ] && [ -x "${WORKDIR}/hhg_sbe" ] && break
    sleep 10; WAITED=$((WAITED + 10))
    if [ "${WAITED}" -gt 2400 ]; then echo "ERROR: compile lock wait timeout"; exit 1; fi
  done
  if [ ! -f "${DONE}" ] || [ ! -x "${WORKDIR}/hhg_sbe" ]; then
    echo "Task ${IDX} won the lock: clean build (ifort + MKL)..."
    rm -f "${DONE}"
    make clean
    if make FC=ifort FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays" LDFLAGS=""; then
      touch "${DONE}"
    else
      rmdir "${LOCK}" 2>/dev/null || true
      echo "ERROR: compilation failed (see this task's .out above)"; exit 1
    fi
  fi
  rmdir "${LOCK}" 2>/dev/null || true
fi
[ -x "${WORKDIR}/hhg_sbe" ] || { echo "ERROR: hhg_sbe missing after compile"; exit 1; }

mkdir -p "${OUTDIR}"
cat > "${OUTDIR}/input.nml" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB_FILE}"
/
&kgrid
  nkx = 40
  nky = 40
/
&bands
  nv_orig = 84
  nb_start = 1
  nb_end = 104
/
&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = ${IVAL}
  theta_deg = 0.0
  phi_cep_deg = 90.0
  ncyc = 4.0
  env_type = 2
/
&timestep
  dt = ${DT_AU}
  n_dt_deph = 5
/
&dephasing
  T2_cycles = 1.0
/
&method
  gauge_method = 'lg_cov'
/
&output
  save_geometry   = .false.
  save_occupation = .false.
/
EOF
sed -i 's/\r$//' "${OUTDIR}/input.nml"

cd "${OUTDIR}" || exit 1
export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export OMP_STACKSIZE=256M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Starting node ${IDX}: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished node ${IDX}: $(date)"
ls -lh HHG.dat Jt.dat 2>/dev/null || true
