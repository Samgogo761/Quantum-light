#!/bin/bash
#SBATCH --job-name=M64_E0_diag
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --output=M64_E0_%j.out
#SBATCH --error=M64_E0_%j.err
# Zero-field short propagation diagnostic for M64 (P3).
set -eo pipefail
MODEL=M64 NV=36 NB=64 SIGN=plus NKX=40 NKY=40
WORKDIR=/public/home/wangjs/project/New_SBEs/Quantum-light
TB=/public/home/wangjs/project/CrI3_TB/wannier/models/M64_allI_pz/CrI3_M64_tb.dat
OUTDIR=${WORKDIR}/output_magnetic_hhg/M64_E0_diag_k40_nv36_b64
NTHREADS=${SLURM_CPUS_PER_TASK:-8}

export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

mkdir -p "$OUTDIR"
cat > "$OUTDIR/input.nml" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB}"
  wannier_hr_file = ""
  wannier_r_file  = ""
/
&kgrid
  nkx = ${NKX}
  nky = ${NKY}
/
&bands
  nv_orig = ${NV}
  nb_start = 1
  nb_end = ${NB}
/
&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = 0.0
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

cd "$OUTDIR"
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS OMP_STACKSIZE=512M
echo "M64 E=0 diagnostic Host=$(hostname) Job=${SLURM_JOB_ID:-none}"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
grep -E "Initial current|Band gap|Total wall" run.log || true
ls -lh HHG.dat Jt.dat 2>/dev/null || true
