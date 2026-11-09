#!/bin/bash
#=============================================================================
# CORRECTED opt1 baseline: nv_orig=12 (opt1 subspace holds 12 occupied spinor
# bands on the full mesh, NOT 22 as in the pilot windows).
# Single case: k40, T2=0.5 cyc, dt=0.35. part_2.
#
# Submit:  sbatch deploy/run_donly_opt1_nv12_baseline.sh
#=============================================================================

#SBATCH --job-name=donly_opt1_nv12
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --output=donly_opt1_nv12_%j.out
#SBATCH --error=donly_opt1_nv12_%j.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/pilot_40wf_opt1/CrI3_donly_tb.dat"
WORKDIR="/public/home/wangjs/project/New_SBEs/Quantum-light"
OUTDIR="${WORKDIR}/output_donly_opt1_nv12/lgcov_k40_b40_nv12"

cd "${WORKDIR}" || exit 1
test -f "${TB_FILE}" || { echo "ERROR: missing TB"; exit 1; }
test -x "${WORKDIR}/hhg_sbe" || { echo "ERROR: missing hhg_sbe"; exit 1; }

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
  nkx = 40
  nky = 40
/
&bands
  nv_orig = 12
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
  dt = 0.35
  n_dt_deph = 5
/
&dephasing
  T2_fs = 1.0e30
  T2_cycles = 0.5
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

grep -E 'nkx|nv_orig|nb_end|T2_cycles' input.nml
echo "Starting: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished: $(date)"
grep -E "Band gap at Gamma|Initial current" run.log || true
