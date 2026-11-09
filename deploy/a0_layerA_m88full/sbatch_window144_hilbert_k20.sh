#!/bin/bash
#=============================================================================
# WINDOW144 Hilbert-completion contrast @ k20, plus-N only, CLASSICAL drive.
#
# Two tasks:
#   0  I=2e11      phi_cep=90           vs 29458 baseline
#   1  I=1.00e12   phi_cep=265.307396651 vs 29452_0 (same high-field node)
#
# This is NOT a chemical 144-WF model, NOT GH7, NOT k40/k50, NOT ±N, NOT
# Husimi-Q.  nb_end=144, nv_orig=84, pinned 29175 binary.
#=============================================================================
#SBATCH --job-name=a0_w144_k20
#SBATCH --partition=part_1
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_w144_k20_%A_%a.out
#SBATCH --error=a0_w144_k20_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

if [ "${WINDOW144_SUBMIT:-}" != "I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY" ]; then
  die "batch refuses without WINDOW144_SUBMIT=I_UNDERSTAND_WINDOW144_K20_CLASSICAL_ONLY"
fi

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="$(pwd)"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-d1a1431}"
TB_W144="${TB_W144:-/public/home/wangjs/project/CrI3_TB/wannier/window144_hilbert/CrI3_window144_tb.dat}"
NK="${NK:-20}"
[ "${NK}" = "20" ] || die "WINDOW144 contrast is k20 only"
PINNED_BINARY_SHA256="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
PINNED_BINARY="${PINNED_BINARY:-${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 1 ] || die "bad array index ${IDX}"

I_LIST=(2.0e11 1.00e12)
PHI_LIST=(90.0 265.307396651)
TAG_LIST=(I2p00e11_cep90 I1p00e12_cepGH7c)
REF_LIST=(29458_baseline 29452_0)

I_DRIVE="${I_LIST[${IDX}]}"
PHI_CEP_DEG="${PHI_LIST[${IDX}]}"
TAG="${TAG_LIST[${IDX}]}"
REF_JOB="${REF_LIST[${IDX}]}"
CASE=$(printf "w144_%s_k%02d_plusN" "${TAG}" "${NK}")

for command_name in sha256sum git awk grep sed; do
  require_cmd "${command_name}"
done

git_in_repo() {
  git --git-dir="${REPO}/.git" --work-tree="${REPO}" "$@"
}
GIT_HEAD="$(git_in_repo rev-parse HEAD)"
SHORT="$(git_in_repo rev-parse --short=12 HEAD)"
CAMPAIGN="window144_hilbert_k20_${SHORT}"
if [ -z "${OUTROOT:-}" ]; then
  OUTROOT="${WORKDIR}/output_a0_window144_k${NK}_${SHORT}"
fi
case "${OUTROOT}" in
  *"${SHORT}"*) ;;
  *) die "OUTROOT must include campaign git short ${SHORT}: ${OUTROOT}" ;;
esac

OUTDIR="${OUTROOT}/${CASE}"
[ -f "${TB_W144}" ] || die "WINDOW144 TB not found: ${TB_W144}"
TB_SHA_NOW="$(sha256_file "${TB_W144}")"
# Wannier90 tb header: comment, a1, a2, a3, nwann. No python on compute nodes.
NWANN_TB="$(sed -n '5p' "${TB_W144}" | awk '{print $1}')"
[ "${NWANN_TB}" = "144" ] || die "WINDOW144 tb nwann=${NWANN_TB}, expected 144"

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

mkdir -p "$(dirname "${OUTDIR}")"
if ! mkdir "${LOCK}"; then
  die "atomic lock exists (concurrent submit?): ${LOCK}"
fi
LOCK_HELD=1
if [ -e "${OUTDIR}" ]; then
  die "OUTDIR already exists; use a unique campaign OUTROOT: ${OUTDIR}"
fi
mkdir "${OUTDIR}"

echo "============================================="
echo " A0 WINDOW144 Hilbert completion NK=${NK} case=${CASE}"
echo " drive=classical_fixed_field I=${I_DRIVE} phi_cep_deg=${PHI_CEP_DEG}"
echo " vs_ref=${REF_JOB}; plusN_only; NOT chemical 144; not GH7; not k40"
echo " host=$(hostname) date=$(date)"
echo "============================================="

export OMP_NUM_THREADS=36
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
echo "USING_PINNED_29175_BINARY sha=${BINARY_SHA256} source=${SOURCE_SHA256}"

cat > "${OUTDIR}/input.nml" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB_W144}"
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
  nb_end = 144
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
  echo "campaign=${CAMPAIGN}"
  echo "drive_mode=classical_fixed_field"
  echo "bsv_enabled=false"
  echo "binary_policy=pinned_29175"
  echo "model=window144_hilbert_completion"
  echo "not_chemical_144=true"
  echo "domain=plusN"
  echo "note=k20_classical_contrast_only; plusN_only; not_GH7; not_k40; not_HusimiQ"
  echo "compare_ref=${REF_JOB}"
  echo "cep_pi_tested=false"
  echo "campaign_actual_head=${GIT_HEAD}"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "repo=${REPO}"
  echo "git_head=${GIT_HEAD}"
  echo "source_sha256=${SOURCE_SHA256}"
  echo "build_key=${BUILD_KEY}"
  echo "binary=${CACHED_BIN}"
  echo "binary_sha256=${BINARY_SHA256}"
  echo "tb_file=${TB_W144}"
  echo "tb_sha256=${TB_SHA_NOW}"
  echo "tb_nwann=${NWANN_TB}"
  echo "input_sha256=$(sha256_file "${OUTDIR}/input.nml")"
  echo "nk=${NK}"
  echo "nb_end=144"
  echo "nv_orig=84"
  echo "dt=0.35"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

if [ -f nodes_moment_check.txt ]; then
  die "unexpected nodes_moment_check.txt in classical WINDOW144 run"
fi
if grep -qi 'quantum-light ensemble' run.log; then
  die "run.log suggests quantum-light ensemble path was taken"
fi
grep -q 'Starting classical time evolution' run.log \
  || die "classical evolution banner missing"
grep -q 'nwann=144' run.log \
  || die "solver did not report nwann=144"

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
  echo "binary_sha256=${BINARY_SHA256}"
  echo "tb_sha256=${TB_SHA_NOW}"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
  echo "cep_pi_tested=false"
  echo "nb_end=144"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} I=${I_DRIVE} phi_cep_deg=${PHI_CEP_DEG}"
