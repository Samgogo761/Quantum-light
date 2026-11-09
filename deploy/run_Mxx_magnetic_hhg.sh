#!/bin/bash
# Flexible parent HHG (+N or -N). Partition/cores chosen by MODEL defaults.
# Env: MODEL M52|M64|M88, NV, NB, SIGN plus|minus, NKX NKY
set -eo pipefail

MODEL="${MODEL:?}"
NV="${NV:?}"
NB="${NB:?}"
SIGN="${SIGN:-plus}"
NKX="${NKX:-40}"
NKY="${NKY:-40}"
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"

case "${MODEL}" in
  M52) TB_PLUS="models/M52_interface/CrI3_M52_tb.dat"
       TB_MINUS="models/M52_interface/CrI3_M52_tb_mN_Pconj.dat" ;;
  M64) TB_PLUS="models/M64_allI_pz/CrI3_M64_tb.dat"
       TB_MINUS="models/M64_allI_pz/CrI3_M64_tb_mN_Pconj.dat" ;;
  M88) TB_PLUS="models/M88_allI_2p/CrI3_M88_tb.dat"
       TB_MINUS="models/M88_allI_2p/CrI3_M88_tb_mN_Pconj.dat" ;;
  *) echo "bad MODEL"; exit 1 ;;
esac

if [[ "${SIGN}" == "minus" ]]; then
  TB_REL="${TB_MINUS}"
  TAG="minusN"
else
  TB_REL="${TB_PLUS}"
  TAG="plusN"
fi

TB_FILE="/public/home/wangjs/project/CrI3_TB/wannier/${TB_REL}"
WORKDIR="/public/home/wangjs/project/New_SBEs/Quantum-light"
OUTDIR="${WORKDIR}/output_magnetic_hhg/${MODEL}_${TAG}_k${NKX}_nv${NV}_b${NB}"

cd "${WORKDIR}"
test -f "${TB_FILE}" || { echo "missing ${TB_FILE}"; exit 1; }
test -x "${WORKDIR}/hhg_sbe" || { echo "missing hhg_sbe"; exit 1; }

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

echo "MODEL=${MODEL} SIGN=${SIGN} NV=${NV} NB=${NB} k=${NKX}x${NKY} Host=$(hostname) Job=${SLURM_JOB_ID:-none}"
echo "TB=${TB_FILE}"
grep -E 'nkx|nv_orig|nb_end|T2_cycles' input.nml
echo "Starting: $(date)"
{ time "${WORKDIR}/hhg_sbe" input.nml ; } 2>&1 | tee run.log
echo "Finished: $(date)"
grep -E "Band gap at Gamma|Initial current" run.log || true
ls -lh HHG.dat Jt.dat 2>/dev/null || true
