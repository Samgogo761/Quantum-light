#!/bin/bash
#=============================================================================
# Classical +N only: 112_old vs 112_large at k10 (then a separate k20 script).
#
# Array:
#   0  old    I=2e11    phi_cep=90
#   1  old    I=1.00e12 phi_cep=265.307396651
#   2  large  I=2e11    phi_cep=90
#   3  large  I=1.00e12 phi_cep=265.307396651
#
# Same two field points as the WINDOW144 classical contrast (29458 / 29452_0).
# NOT GH3/GH5/GH7, NOT k40/k50, NOT M88, NOT ±N, NOT Husimi-Q, NOT 232.
# H5/H7 are evidence columns only; this job does not PASS/FAIL a mechanism.
#=============================================================================
#SBATCH --job-name=c112L_k10
#SBATCH --partition=part_2
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

if [ "${COMPARE_112_SUBMIT:-}" != "I_UNDERSTAND_112OLD_LARGE_CLASSICAL_ONLY" ]; then
  die "batch refuses without COMPARE_112_SUBMIT=I_UNDERSTAND_112OLD_LARGE_CLASSICAL_ONLY"
fi

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
WORKDIR="$(pwd)"
ROOT=/public/home/wangjs/project/CrI3_TB
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-d1a1431}"
TB_OLD="${TB_OLD:-${ROOT}/wannier/CrI3_tb.dat}"
TB_LARGE="${TB_LARGE:-${ROOT}/window144_hilbert/ref112_large/CrI3_112large_tb.dat}"
NK="${NK:-10}"
[ "${NK}" = "10" ] || die "this script is k10 only (got NK=${NK})"
PINNED_BINARY_SHA256="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
PINNED_BINARY="${PINNED_BINARY:-${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 3 ] || die "bad array index ${IDX}"

MODEL_LIST=(old old large large)
I_LIST=(2.0e11 1.00e12 2.0e11 1.00e12)
PHI_LIST=(90.0 265.307396651 90.0 265.307396651)
TAG_LIST=(I2p00e11_cep90 I1p00e12_cep265 I2p00e11_cep90 I1p00e12_cep265)

MODEL="${MODEL_LIST[${IDX}]}"
I_DRIVE="${I_LIST[${IDX}]}"
PHI_CEP_DEG="${PHI_LIST[${IDX}]}"
TAG="${TAG_LIST[${IDX}]}"
CASE=$(printf "f112%s_%s_k%02d_plusN" "${MODEL}" "${TAG}" "${NK}")

if [ "${MODEL}" = "old" ]; then
  TB_FILE="${TB_OLD}"
else
  TB_FILE="${TB_LARGE}"
fi

for command_name in sha256sum awk grep sed; do
  require_cmd "${command_name}"
done

OUTROOT="${OUTROOT:-${WORKDIR}/hhg_compare_k10}"
OUTDIR="${OUTROOT}/${CASE}"
[ -f "${TB_FILE}" ] || die "TB not found: ${TB_FILE}"
NWANN_TB="$(sed -n '5p' "${TB_FILE}" | awk '{print $1}')"
[ "${NWANN_TB}" = "112" ] || die "tb nwann=${NWANN_TB}, expected 112"

LOCK="${OUTDIR}.runlock"
LOCK_HELD=0
RUN_COMPLETED=0
on_exit() {
  rc=$?
  if [ "${LOCK_HELD}" -eq 1 ] && [ -n "${LOCK}" ]; then
    rmdir "${LOCK}" 2>/dev/null || true
  fi
  if [ "${RUN_COMPLETED}" -ne 1 ]; then
    if [ -d "${OUTDIR:-}" ]; then
      printf 'status=FAILED\nexit_code=%s\ntime=%s\n' \
        "${rc}" "$(date --iso-8601=seconds 2>/dev/null || date)" \
        > "${OUTDIR}/run_status.txt" 2>/dev/null || true
    fi
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$(dirname "${OUTDIR}")" "${WORKDIR}/logs"
if ! mkdir "${LOCK}"; then
  die "atomic lock exists: ${LOCK}"
fi
LOCK_HELD=1
if [ -e "${OUTDIR}" ]; then
  die "OUTDIR already exists: ${OUTDIR}"
fi
mkdir "${OUTDIR}"

echo "============================================="
echo " 112_${MODEL} classical NK=${NK} case=${CASE}"
echo " I=${I_DRIVE} phi_cep_deg=${PHI_CEP_DEG} plusN_only"
echo " NOT GH, not k40, not 232, not WINDOW144"
echo " partition=part_2 omp=${NTHREADS}; host=$(hostname) date=$(date)"
echo "============================================="

export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

[ -x "${PINNED_BINARY}" ] || die "pinned 29175 binary missing: ${PINNED_BINARY}"
CACHED_BIN="${PINNED_BINARY}"
CACHE_DIR="$(dirname "${CACHED_BIN}")"
[ -f "${CACHE_DIR}/build_metadata.txt" ] || die "pinned binary cache lacks build_metadata.txt"
BINARY_SHA256="$(sha256_file "${CACHED_BIN}")"
[ "${BINARY_SHA256}" = "${PINNED_BINARY_SHA256}" ] || die "pinned binary SHA mismatch"
SOURCE_SHA256="$(awk -F= '$1=="source_sha256"{print $2; exit}' "${CACHE_DIR}/build_metadata.txt")"
BUILD_KEY="$(awk -F= '$1=="build_key"{print $2; exit}' "${CACHE_DIR}/build_metadata.txt")"
TB_SHA_NOW="$(sha256_file "${TB_FILE}")"

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
  nv_orig = 84
  nb_start = 1
  nb_end = 112
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
  dephasing_enabled = .true.
/

&bsv
  bsv_enabled = .false.
/

&method
  gauge_method = 'lg_cov'
/

&output
  save_geometry = .false.
  save_occupation = .true.
  occ_band_resolved = .true.
  occ_stride = 126
  save_complex_hhg = .true.
/
EOF

cp -f "${CACHE_DIR}/build_metadata.txt" "${OUTDIR}/build_metadata.snapshot.txt"

{
  echo "case=${CASE}"
  echo "drive_mode=classical_fixed_field"
  echo "bsv_enabled=false"
  echo "binary_policy=pinned_29175"
  echo "model=112_${MODEL}"
  echo "domain=plusN"
  echo "note=k10_classical_112old_vs_112large; plusN_only; not_GH; not_k40; not_232; not_I-d"
  echo "slurm_partition=part_2"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "binary=${CACHED_BIN}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "tb_file=${TB_FILE}"
  echo "tb_sha256=${TB_SHA_NOW}"
  echo "tb_nwann=${NWANN_TB}"
  echo "input_sha256=$(sha256_file "${OUTDIR}/input.nml")"
  echo "nk=${NK}"
  echo "nb_end=112"
  echo "nv_orig=84"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

if [ -f nodes_moment_check.txt ]; then
  die "unexpected nodes_moment_check.txt in classical compare run"
fi
if grep -qi 'quantum-light ensemble' run.log; then
  die "run.log suggests quantum-light ensemble path was taken"
fi
grep -q 'Starting classical time evolution' run.log \
  || die "classical evolution banner missing"
grep -q 'nwann=112' run.log \
  || die "solver did not report nwann=112"

CRITICAL_OUTPUTS=(
  occupation_kt.dat
  occupation_band_kt.dat
  Jt.dat
  Et.dat
  HHG.dat
  HHG_complex.dat
  run.log
  run_metadata.txt
  input.nml
)
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  [ -s "${output_file}" ] || die "missing or empty critical output: ${OUTDIR}/${output_file}"
done

NONFINITE_RE='(^|[^[:alpha:]])[+-]?(nan|inf(inity)?)([^[:alpha:]]|$)'
for output_file in "${CRITICAL_OUTPUTS[@]}"; do
  if LC_ALL=C grep -Eiq "${NONFINITE_RE}" "${output_file}"; then
    die "NaN/Inf detected in ${OUTDIR}/${output_file}"
  fi
done

sha256sum "${CRITICAL_OUTPUTS[@]}" > output_sha256.txt
{
  echo "status=PASS"
  echo "exit_code=0"
  echo "completed_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "model=112_${MODEL}"
  echo "nk=${NK}"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
  echo "not_a_mechanism_gate=true"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} $(date)"
