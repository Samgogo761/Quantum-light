#!/bin/bash
#=============================================================================
# full112 GH5-node occupation authentication @ k20, plus-N, CLASSICAL drive.
#
# Authenticates actual GH5 quadrature intensities / CEPs, not a uniform I-grid.
# Does not claim "97.75% weight certified". Does not submit GH7 / k40 / k50.
#
# Array 0-7:
#   0 I=2.00e11     phi=90           classical baseline (template CEP)
#   1 I=3.6508e11   phi=270          GH5 th0 id12  w=0.118
#   2 I=3.6508e11   phi=90           GH5 th0 id14  w=0.118  (antiphase of 1)
#   3 I=3.6508e11   phi=180          GH5 th180 id8 w=0.118  (other squeeze axis)
#   4 I=3.6508e11   phi=0            GH5 th180 id18 w=0.118 (antiphase of 3)
#   5 I=3.7601e11   phi=260.185345859  GH5 th0 id2 w=0.0025 (mid-ring max I)
#   6 I=1.0926e10   phi=180          GH5 th0 id3  w=0.006  (inner ring)
#   7 I=1.00e12     phi=265.307396651  dt/2 vs 29453; POSITIVITY_DIAGNOSTIC_ONLY
#
# Pinned 29175 binary, OMP=36, MKL=1, part_1. Does not rebuild.
#=============================================================================
#SBATCH --job-name=a0_node_occ
#SBATCH --partition=part_1
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=a0_node_occ_%A_%a.out
#SBATCH --error=a0_node_occ_%A_%a.err

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

if [ "${NODE_AUTH_SUBMIT:-}" != "I_UNDERSTAND_GH5_NODE_OCC_K20_ONLY" ]; then
  die "batch refuses without NODE_AUTH_SUBMIT=I_UNDERSTAND_GH5_NODE_OCC_K20_ONLY"
fi

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="$(pwd)"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light-wt-d1a1431}"
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/CrI3_tb.dat}"
NK="${NK:-20}"
[ "${NK}" = "20" ] || die "node occupation scan is k20 only"
OCC_STRIDE="${OCC_STRIDE:-126}"
PINNED_BINARY_SHA256="34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
PINNED_TB_PLUS_SHA256="66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18"
PINNED_BINARY="${PINNED_BINARY:-${REPO}/.a0_build_cache_v2/8aaba35732b005dbd39f2639ba24d68e3db0a114c2924841184c4556c9a8d8fe/hhg_sbe}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "${IDX}" -ge 0 ] && [ "${IDX}" -le 7 ] || die "bad array index ${IDX}"

I_LIST=(2.00e11 3.65084547634179932e11 3.65084547634179932e11 3.65084547634179932e11 3.65084547634179932e11 3.76010329150605530e11 1.09257815164255180e10 1.00e12)
PHI_LIST=(90.0 270.0 90.0 180.0 0.0 260.185345859 180.0 265.307396651)
DT_LIST=(0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.175)
TAG_LIST=(
  I2p00e11_cep90_baseline
  I3p65e11_cep270_th0id12
  I3p65e11_cep90_th0id14
  I3p65e11_cep180_th180id8
  I3p65e11_cep0_th180id18
  I3p76e11_cep260_th0id2
  I1p09e10_cep180_th0id3
  I1p00e12_cepGH7c_dthalf
)
NOTE_LIST=(
  "classical_baseline_template_cep90"
  "GH5_th0_id12_midring"
  "GH5_th0_id14_antiphase_of_id12"
  "GH5_th180_id8_other_squeeze_axis"
  "GH5_th180_id18_antiphase_of_id8"
  "GH5_th0_id2_midring_maxI"
  "GH5_th0_id3_inner_ring"
  "POSITIVITY_DIAGNOSTIC_ONLY_dt_half_vs_29453"
)

I_DRIVE="${I_LIST[${IDX}]}"
PHI_CEP_DEG="${PHI_LIST[${IDX}]}"
DT="${DT_LIST[${IDX}]}"
TAG="${TAG_LIST[${IDX}]}"
NOTE="${NOTE_LIST[${IDX}]}"
CASE=$(printf "nodeocc_%s_k%02d_plusN" "${TAG}" "${NK}")

for command_name in sha256sum git awk grep; do
  require_cmd "${command_name}"
done

git_in_repo() {
  git --git-dir="${REPO}/.git" --work-tree="${REPO}" "$@"
}
GIT_HEAD="$(git_in_repo rev-parse HEAD)"
SHORT="$(git_in_repo rev-parse --short=12 HEAD)"
CAMPAIGN="full112_gh5_node_occ_k20_${SHORT}"
if [ -z "${OUTROOT:-}" ]; then
  OUTROOT="${WORKDIR}/output_a0_full112_k${NK}_node_occ_${SHORT}"
fi
case "${OUTROOT}" in
  *"${SHORT}"*) ;;
  *) die "OUTROOT must include campaign git short ${SHORT}: ${OUTROOT}" ;;
esac

OUTDIR="${OUTROOT}/${CASE}"
[ -f "${TB_PLUS}" ] || die "TB not found: ${TB_PLUS}"
TB_SHA_NOW="$(sha256_file "${TB_PLUS}")"
[ "${TB_SHA_NOW}" = "${PINNED_TB_PLUS_SHA256}" ] || die "plus TB SHA mismatch: ${TB_SHA_NOW}"

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
echo " A0 full112 GH5-node occupation NK=${NK} case=${CASE}"
echo " drive=classical_fixed_field I=${I_DRIVE} phi_cep_deg=${PHI_CEP_DEG} dt=${DT}"
echo " note=${NOTE}; plusN_only; not GH7; not HusimiQ"
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
  wannier_tb_file = "${TB_PLUS}"
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
  dt = ${DT}
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
  occ_stride = ${OCC_STRIDE}
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
  echo "model=full112"
  echo "domain=plusN"
  echo "note=GH5_node_occupation_auth; plusN_only_no_cep_pi; not_GH7; not_HusimiQ"
  echo "note2=${NOTE}"
  echo "label=FULL112_K20_SAFE_INTENSITY_BRACKET_BY_CEP"
  echo "cep_pi_tested=false"
  echo "approve_k40=false"
  echo "approve_k50=false"
  echo "approve_full_gh7=false"
  echo "predecessor_fail_job=29452"
  echo "predecessor_fail_I_Wcm2=1.00e12"
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
  echo "tb_file=${TB_PLUS}"
  echo "tb_sha256=${TB_SHA_NOW}"
  echo "input_sha256=$(sha256_file "${OUTDIR}/input.nml")"
  echo "nk=${NK}"
  echo "dt=${DT}"
  echo "T2_cycles=0.5"
  echo "wvl_nm=3200.0"
  echo "intensity_Wcm2=${I_DRIVE}"
  echo "phi_cep_deg=${PHI_CEP_DEG}"
  echo "occ_stride=${OCC_STRIDE}"
  echo "save_occupation=true"
  echo "occ_band_resolved=true"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
} > "${OUTDIR}/run_metadata.txt"

printf 'status=RUNNING\ntime=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "${OUTDIR}/run_status.txt"

cd "${OUTDIR}"
"${CACHED_BIN}" input.nml | tee run.log

if [ -f nodes_moment_check.txt ]; then
  die "unexpected nodes_moment_check.txt in classical node-occ run"
fi
if grep -qi 'quantum-light ensemble' run.log; then
  die "run.log suggests quantum-light ensemble path was taken"
fi
grep -q 'Starting classical time evolution' run.log \
  || die "classical evolution banner missing"

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
  echo "dt=${DT}"
  echo "cep_pi_tested=false"
} > run_status.txt
touch SUCCESS
RUN_COMPLETED=1
echo "DONE ${CASE} I=${I_DRIVE} phi_cep_deg=${PHI_CEP_DEG} dt=${DT}"
