#!/bin/bash
#=============================================================================
# Verify the 2026-06 solver changes on the server (single classical run).
#
# Runs lg_cov / 40x40 / nb1-112 / T2_cycles=1.0 / ncyc=4 with the new
# observables on (publication-standard dephasing: T2=10.7 fs, peak FWHM ~0.32 H),
# so one run doubles as a real physics run AND a code check:
#   1. Regression   : HHG.dat / Jt.dat should remain physically consistent
#                     with the corresponding lg_cov / nb1-112 / T2=1 cycle run.
#   2. j_anom / PT   : quantum_geometry.dat at FULL nb1-112 -> max|Omega| floor.
#   3. Spin pipeline : equilibrium <S_z> self-check + nominal spin HHG_spin.dat.
#
# Submit from the repo root so WORKDIR auto-resolves:
#   cd <repo>;  sbatch deploy/run_verify_observables.sh
# Override TB if needed:
#   TB_FILE=/path/CrI3_tb.dat sbatch deploy/run_verify_observables.sh
#=============================================================================

#SBATCH --job-name=verify_obs
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=verify_obs_%j.out
#SBATCH --error=verify_obs_%j.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
OUTDIR="${OUTDIR:-${WORKDIR}/output_verify_obs_nb112_T2_1p0cycle}"

cd "${WORKDIR}" || { echo "ERROR: cannot cd to ${WORKDIR}"; exit 1; }

echo "============================================="
echo "  Verify observables (lg_cov / 40x40 / nb1-112)"
echo "  Date: $(date)   Host: $(hostname)"
echo "  WORKDIR: ${WORKDIR}"
echo "  TB:      ${TB_FILE}"
echo "  OUTDIR:  ${OUTDIR}"
echo "============================================="

if [ ! -f "${TB_FILE}" ]; then
  echo "ERROR: TB file not found: ${TB_FILE}  (set TB_FILE=...)"
  exit 1
fi

# NOTE: the vtune lib64 path FIRST is essential — it provides a libstdc++.so.6
# new enough (GLIBCXX_3.4.21) for ifort; the system /lib64 one is too old.
# (All previously-working compile scripts include this line.)
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

# Always (re)compile here: the copied tree may carry Windows objects/binary.
echo "Compiling (Intel ifort + MKL)..."
make clean
make FC=ifort FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays" LDFLAGS=""
if [ ! -x "${WORKDIR}/hhg_sbe" ]; then echo "ERROR: build failed"; exit 1; fi

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
  nb_end = 112
/
&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = 2.0e11
  theta_deg = 0.0
  phi_cep_deg = 90.0
  ncyc = 4.0
  env_type = 2
/
&timestep
  dt = 0.35
  n_dt_deph = 5
/
&dephasing
  T2_cycles = 1.0
/
&method
  gauge_method = 'lg_cov'
/
&output
  save_geometry   = .true.
  save_occupation = .true.
  occ_stride = 100
  occ_band_resolved = .false.
  save_coherence = .true.
/
&spin
  spin_current = .true.
  spin_order   = 'interleaved'
/
EOF
sed -i 's/\r$//' "${OUTDIR}/input.nml"

cd "${OUTDIR}" || exit 1
export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export OMP_STACKSIZE=256M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Starting verify run: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished: $(date)"

echo ""
echo "=== Quick checks ==="
grep -E "Initial current|max .Omega|Equilibrium spin|sum over valence" run.log || true
echo "Outputs in ${OUTDIR}:"
ls -lh HHG.dat Jt.dat Et.dat quantum_geometry.dat occupation_kt.dat coherence_kt.dat \
  HHG_spin.dat Jt_spin.dat 2>/dev/null || true
echo ""
echo "Next (use python3, NOT python -- server python is 2.x):"
echo "  python3 ${WORKDIR}/tools/analysis/analyze_quantum_geometry.py --file ${OUTDIR}/quantum_geometry.dat"
echo "  (or just run it locally after downloading quantum_geometry.dat)"
