#!/bin/bash
#=============================================================================
# 40-WF opt1: denser k-grid scan (part_2)
# Historical 20/40/60 was for expensive 112-WF; with 40-WF we push denser meshes.
#
# Array:
#   0  k80
#   1  k100
#   2  k120
#
# Fixed: T2_cycles=0.5, dt=0.35, lg_cov, same laser as production.
# Submit:
#   sbatch --array=0-2%2 deploy/run_donly_opt1_kdense_part2.sh
#=============================================================================

#SBATCH --job-name=donly_opt1_kd
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --output=donly_opt1_kd_%A_%a.out
#SBATCH --error=donly_opt1_kd_%A_%a.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf_opt1/CrI3_donly_tb.dat}"
WORKDIR="${WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_donly_opt1_convergence}"
T2_CYCLES=0.5
DT_AU=0.35
ID="${SLURM_ARRAY_TASK_ID:-0}"

case "${ID}" in
  0) NK=80;  TAG="k80_T2_0p5cyc_dt0p35" ;;
  1) NK=100; TAG="k100_T2_0p5cyc_dt0p35" ;;
  2) NK=120; TAG="k120_T2_0p5cyc_dt0p35" ;;
  *) echo "Unknown id ${ID}"; exit 1 ;;
esac

OUTDIR="${OUTROOT}/${TAG}"
cd "${WORKDIR}" || exit 1
echo "=== denser-k id=${ID} ${TAG} threads=${NTHREADS} $(date) host=$(hostname) ==="

test -f "${TB_FILE}" || { echo "ERROR: missing TB"; exit 1; }
test -x "${WORKDIR}/hhg_sbe" || { echo "ERROR: missing hhg_sbe"; exit 1; }
if [[ -f "${OUTDIR}/HHG.dat" && -s "${OUTDIR}/HHG.dat" ]]; then
  echo "SKIP: ${OUTDIR}/HHG.dat exists"; exit 0
fi

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

mkdir -p "${OUTDIR}"
cat > "${OUTDIR}/input.nml" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB_FILE}"
  wannier_hr_file = ""
  wannier_r_file  = ""
/
&kgrid
  nkx = ${NK}
  nky = ${NK}
/
&bands
  nv_orig = 22
  nb_start = 1
  nb_end = 40
/
&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = 2.0e11
  theta_deg = 0.0
  phi_cep_deg = 90.0
  ncyc = 4.0
  env_type = 2
  ellipticity = 0.0
  delta_phase_deg = 90.0
/
&laser2
  wvl_nm_2 = 0.0
  intensity_Wcm2_2 = 0.0
/
&external_field
  use_external_A = .false.
/
&timestep
  dt = ${DT_AU}
  n_dt_deph = 5
/
&dephasing
  T2_fs = 1.0e30
  T2_cycles = ${T2_CYCLES}
/
&bsv
  bsv_enabled = .false.
/
&method
  gauge_method = 'lg_cov'
/
&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .false.
/
EOF

cd "${OUTDIR}"
export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores

grep -E 'nkx|T2_cycles|dt =' input.nml
echo "Starting: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished: $(date)"
ls -lh HHG.dat Jt.dat 2>/dev/null || true
