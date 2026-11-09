#!/bin/bash
#=============================================================================
# 40-WF opt1 classical HHG convergence on part_2 (8 cores / ~64 GB)
# Use when part_1 is congested. Suitable ONLY for light 40-WF TB, not full112/BSV.
#
# Cases: same as run_donly_opt1_convergence.sh (array 0-6)
# Submit:
#   cd /public/home/wangjs/project/New_SBEs/Quantum-light
#   sbatch --array=0-6%3 deploy/run_donly_opt1_convergence_part2.sh
#=============================================================================

#SBATCH --job-name=donly_opt1_p2
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=donly_opt1_p2_%A_%a.out
#SBATCH --error=donly_opt1_p2_%A_%a.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
TB_FILE="${TB_FILE:-/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf_opt1/CrI3_donly_tb.dat}"
WORKDIR="${WORKDIR:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_donly_opt1_convergence}"
NB_START=1
NB_END=40
NV_ORIG=22
ID="${SLURM_ARRAY_TASK_ID:-1}"

case "${ID}" in
  0) NK=20; T2_CYCLES=0.5;  DT_AU=0.35;  TAG="k20_T2_0p5cyc_dt0p35" ;;
  1) NK=40; T2_CYCLES=0.5;  DT_AU=0.35;  TAG="k40_T2_0p5cyc_dt0p35" ;;
  2) NK=60; T2_CYCLES=0.5;  DT_AU=0.35;  TAG="k60_T2_0p5cyc_dt0p35" ;;
  3) NK=40; T2_CYCLES=0.25; DT_AU=0.35;  TAG="k40_T2_0p25cyc_dt0p35" ;;
  4) NK=40; T2_CYCLES=1.0;  DT_AU=0.35;  TAG="k40_T2_1p0cyc_dt0p35" ;;
  5) NK=40; T2_CYCLES=0.5;  DT_AU=0.25;  TAG="k40_T2_0p5cyc_dt0p25" ;;
  6) NK=40; T2_CYCLES=0.5;  DT_AU=0.175; TAG="k40_T2_0p5cyc_dt0p175" ;;
  *) echo "Unknown array id ${ID}"; exit 1 ;;
esac

OUTDIR="${OUTROOT}/${TAG}"

cd "${WORKDIR}" || exit 1
echo "=== donly opt1 PART2 id=${ID} tag=${TAG} threads=${NTHREADS} $(date) host=$(hostname) ==="
echo "TB=${TB_FILE}"

test -f "${TB_FILE}" || { echo "ERROR: missing TB"; exit 1; }
test -x "${WORKDIR}/hhg_sbe" || { echo "ERROR: missing hhg_sbe"; exit 1; }

# skip if already has HHG.dat from a previous successful run
if [[ -f "${OUTDIR}/HHG.dat" && -s "${OUTDIR}/HHG.dat" ]]; then
  echo "SKIP: ${OUTDIR}/HHG.dat already exists"
  exit 0
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
  nv_orig = ${NV_ORIG}
  nb_start = ${NB_START}
  nb_end = ${NB_END}
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

grep -E 'nkx|nv_orig|nb_end|wannier_tb_file|gauge_method|T2_cycles|dt =' input.nml
echo "Starting: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished: $(date)"
ls -lh HHG.dat Jt.dat 2>/dev/null || true
grep -E 'Initial current|Band gap' run.log | tail -10 || true
