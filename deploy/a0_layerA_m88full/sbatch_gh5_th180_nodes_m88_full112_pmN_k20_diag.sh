#!/bin/bash
#=============================================================================
# Targeted classical M88 vs full112 ±N diagnostic (NOT formal GH5).
#
# Source: GH5 squeezed_vacuum r=2.5, θ=180°, nodes 19 / 23 / 24 / 25.
# Drive: classical fixed (I, phi_CEP) with bsv_enabled=.false.
#
# Array 0..15  (INODE*4 + IMODEL*2 + IDOM):
#   nodes   = (19, 23, 24, 25)
#   models  = (M88-full, full112)
#   domains = (plusN, minusN)
#
# Purpose:
#   - id19: same phi as id25, lower I  (~3.68e11)
#   - id23: high I, phi~0° / 360°
#   - id24: high I near B, phi~2.23°
#   - id25: Imax tail (27757 Node B); enables Jo with -N
#
# Does NOT submit formal GH5 or k50.
#=============================================================================
#SBATCH --job-name=a0_hf_pmN_diag
#SBATCH --partition=part_1
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=a0_hf_pmN_diag_%A_%a.out
#SBATCH --error=a0_hf_pmN_diag_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

source_tree_digest() {
  (
    cd "${REPO}"
    sha256sum Makefile
    find src -maxdepth 1 -type f -name '*.f90' -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_gh5_th180_nodes_m88_full112_pmN_k20_diag}"
NK=20

TB_M88_PLUS="${TB_M88_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb.dat}"
TB_M88_MINUS="${TB_M88_MINUS:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb_mN_Pconj.dat}"
TB_F112_PLUS="${TB_F112_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
TB_F112_MINUS="${TB_F112_MINUS:-/public/home/wangjs/project/CrI3_TB_mN/wannier/CrI3_tb_mN_Pconj.dat}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 15 ] || die "bad index ${IDX}"

INODE=$(( IDX / 4 ))
REST=$(( IDX % 4 ))
IMODEL=$(( REST / 2 ))
IDOM=$(( REST % 2 ))

NODE_IDS=(19 23 24 25)
# From nodes_sv_r2p5_th180_gh5.dat (phi_drive rad -> deg)
I_LIST=(3.67544468e11 1.62152975e12 1.62398967e12 1.63245553e12)
PHI_CEP_DEG_LIST=(4.69260335 360.0 2.23049484 4.69260335)
W_LIST=(4.93177151e-02 6.00395271e-03 2.50000000e-03 1.26729310e-04)
RE_ALPHA_LIST=(8.28522631e+00 1.74610401e+01 1.74610401e+01 1.74610401e+01)
IM_ALPHA_LIST=(6.80092791e-01 -1.06197627e-15 6.80092791e-01 1.43328946e+00)

NTAG="${NODE_IDS[${INODE}]}"
I_DRIVE="${I_LIST[${INODE}]}"
PHI_CEP_DEG="${PHI_CEP_DEG_LIST[${INODE}]}"
W_NODE="${W_LIST[${INODE}]}"
RE_A="${RE_ALPHA_LIST[${INODE}]}"
IM_A="${IM_ALPHA_LIST[${INODE}]}"

if [ "${IMODEL}" -eq 0 ]; then
  MODEL="M88full"
  NV=58
  NB_START=1
  NB_END=88
  if [ "${IDOM}" -eq 0 ]; then
    DOM="plusN"
    TB="${TB_M88_PLUS}"
  else
    DOM="minusN"
    TB="${TB_M88_MINUS}"
  fi
else
  MODEL="full112"
  NV=84
  NB_START=1
  NB_END=112
  if [ "${IDOM}" -eq 0 ]; then
    DOM="plusN"
    TB="${TB_F112_PLUS}"
  else
    DOM="minusN"
    TB="${TB_F112_MINUS}"
  fi
fi

CASE=$(printf "hfdiag_r2p5_th180_id%02d_%s_k%02d_%s" "${NTAG}" "${MODEL}" "${NK}" "${DOM}")
OUTDIR="${OUTROOT}/${CASE}"

for c in sha256sum git find xargs awk grep mktemp; do require_cmd "${c}"; done
[ -f "${TB}" ] || die "missing TB ${TB}"
[ -f "${REPO}/Makefile" ] || die "missing Makefile"

if [ -d "${OUTDIR}" ] && [ -n "$(find "${OUTDIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  die "non-empty OUTDIR ${OUTDIR}"
fi
mkdir -p "${OUTDIR}"

LOCK=""
LOCK_HELD=0
RUN_COMPLETED=0
on_exit() {
  rc=$?
  if [ "${LOCK_HELD}" -eq 1 ] && [ -n "${LOCK}" ]; then
    rmdir "${LOCK}" 2>/dev/null || true
  fi
  if [ "${RUN_COMPLETED}" -ne 1 ]; then
    printf 'status=FAILED\nexit_code=%s\n' "${rc}" > "${OUTDIR}/run_status.txt" 2>/dev/null || true
  fi
}
trap on_exit EXIT

echo "============================================="
echo " Classical ±N diagnostic ${CASE}"
echo " node_id=${NTAG} w=${W_NODE} I=${I_DRIVE} phi_CEP=${PHI_CEP_DEG} deg"
echo " alpha=(${RE_A},${IM_A})"
echo " TB=${TB}  nv=${NV} bands=${NB_START}-${NB_END}"
echo " mode=classical fixed field (bsv_enabled=.false.)"
echo " campaign=gh5_th180_nodes_m88_full112_pmN_k20_diag"
echo " NOTE: NOT formal GH5 / NOT k50"
echo " host=$(hostname) date=$(date)"
echo "============================================="

export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

BUILD_FC="ifort"
BUILD_FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays"
BUILD_LDFLAGS=""
require_cmd "${BUILD_FC}"
FC_PATH="$(command -v "${BUILD_FC}")"
COMPILER_VERSION="$(${BUILD_FC} --version 2>&1 | head -n 1)"
SOURCE_SHA256="$(source_tree_digest)"
BUILD_KEY="$({
  printf 'source_sha256=%s\n' "${SOURCE_SHA256}"
  printf 'FC=%s\n' "${BUILD_FC}"
  printf 'FC_path=%s\n' "${FC_PATH}"
  printf 'compiler_version=%s\n' "${COMPILER_VERSION}"
  printf 'FFLAGS=%s\n' "${BUILD_FFLAGS}"
  printf 'LDFLAGS=%s\n' "${BUILD_LDFLAGS}"
} | sha256sum | awk '{print $1}')"

BUILD_ROOT="${REPO}/.a0_build_cache_v2"
LOCK="${BUILD_ROOT}/.compile.lockdir"
CACHE_DIR="${BUILD_ROOT}/${BUILD_KEY}"
CACHED_BIN="${CACHE_DIR}/hhg_sbe"
mkdir -p "${BUILD_ROOT}"

if [ ! -x "${CACHED_BIN}" ]; then
  WAITED=0
  while true; do
    if mkdir "${LOCK}" 2>/dev/null; then LOCK_HELD=1; break; fi
    [ -x "${CACHED_BIN}" ] && break
    sleep 10
    WAITED=$((WAITED+10))
    [ "${WAITED}" -le 2400 ] || die "compile lock timeout"
  done
  if [ "${LOCK_HELD}" -eq 1 ]; then
    (
      cd "${REPO}"
      make clean
      make FC="${BUILD_FC}" FFLAGS="${BUILD_FFLAGS}" LDFLAGS="${BUILD_LDFLAGS}"
    ) 2>&1 | tee "${OUTDIR}/build.log"
    [ -x "${REPO}/hhg_sbe" ] || die "build failed"
    mkdir -p "${CACHE_DIR}"
    cp -f "${REPO}/hhg_sbe" "${CACHED_BIN}"
    chmod 0555 "${CACHED_BIN}"
    {
      echo "build_key=${BUILD_KEY}"
      echo "source_sha256=${SOURCE_SHA256}"
      echo "binary_sha256=$(sha256_file "${CACHED_BIN}")"
      echo "compiler_version=${COMPILER_VERSION}"
    } > "${CACHE_DIR}/build_metadata.txt"
    rmdir "${LOCK}" 2>/dev/null || true
    LOCK_HELD=0
  fi
fi
[ -x "${CACHED_BIN}" ] || die "missing ${CACHED_BIN}"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"

cat > "${OUTDIR}/input.nml" <<EOF
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
  nkx = ${NK}
  nky = ${NK}
/

&bands
  nv_orig = ${NV}
  nb_start = ${NB_START}
  nb_end = ${NB_END}
/

&laser
  wvl_nm = 3200.0
  intensity_Wcm2 = ${I_DRIVE}
  theta_deg = 0.0
  phi_cep_deg = ${PHI_CEP_DEG}
  ncyc = 4.0
  env_type = 2
  ellipticity = 0.0
  delta_phase_deg = 90.0
/

&timestep
  dt = 0.35
  n_dt_deph = 5
/

&dephasing
  T2_cycles = 0.5
/

&bsv
  bsv_enabled = .false.
/

&method
  gauge_method = 'lg_cov'
/

&output
  save_geometry = .false.
  save_occupation = .false.
  save_complex_hhg = .true.
/
EOF

TB_SHA256="$(sha256_file "${TB}")"
{
  echo "case=${CASE}"
  echo "campaign=gh5_th180_nodes_m88_full112_pmN_k20_diag"
  echo "drive_mode=classical_fixed_field"
  echo "bsv_enabled=false"
  echo "node_id=${NTAG}"
  echo "node_weight=${W_NODE}"
  echo "re_alpha=${RE_A}"
  echo "im_alpha=${IM_A}"
  echo "model=${MODEL}"
  echo "domain=${DOM}"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
  echo "source_gh5_r=2.5"
  echo "source_gh5_theta_deg=180"
  echo "source_manifest=nodes_sv_r2p5_th180_gh5.dat"
  echo "nk=${NK}"
  echo "nkx=${NK}"
  echo "nky=${NK}"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "nv_orig=${NV}"
  echo "nb_start=${NB_START}"
  echo "nb_end=${NB_END}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "tb_file=${TB}"
  echo "tb_sha256=${TB_SHA256}"
  echo "note=NOT_formal_GH5_NOT_k50_targeted_material_diag"
  echo "note2=enables_Jo=(Jplus-Jminus)/2_when_pmN_pair_complete"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
} > "${OUTDIR}/run_metadata.txt"

mkdir -p "${OUTDIR}/source_snapshot"
cp -f "${REPO}/Makefile" "${OUTDIR}/source_snapshot/"
cp -f "${REPO}"/src/*.f90 "${OUTDIR}/source_snapshot/"
(
  cd "${OUTDIR}/source_snapshot"
  sha256sum Makefile *.f90 | tee source_files_sha256.txt
)

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

for f in HHG.dat HHG_complex.dat Et.dat run.log; do
  [ -s "${f}" ] || die "missing ${f}"
done
if [ -f nodes_moment_check.txt ]; then
  die "unexpected nodes_moment_check.txt in classical spot run"
fi
if grep -qi 'quantum-light ensemble' run.log; then
  die "run.log suggests quantum-light ensemble path was taken"
fi
grep -q 'Wrote HHG_complex.dat' run.log \
  || die "HHG_complex.dat write confirmation missing in run.log"

sha256sum HHG.dat HHG_complex.dat Et.dat run.log run_metadata.txt input.nml \
  > output_sha256.txt
echo "status=PASS" > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "SUCCESS ${CASE}"
