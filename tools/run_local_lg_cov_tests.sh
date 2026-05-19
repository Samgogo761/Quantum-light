#!/usr/bin/env bash
set -euo pipefail

# Local classical-light LG-covariant tests.
# Run from the repository root through MSYS2 bash, for example:
#   export PATH=/mingw64/bin:/usr/bin:$PATH
#   export HOME=.tmp TMPDIR=.tmp TMP=.tmp TEMP=.tmp
#   export OMP_NUM_THREADS=8
#   bash tools/run_local_lg_cov_tests.sh k10_b30

REPO="$(pwd)"
EXE_FROM_CASE="../../../hhg_sbe.exe"
TB_FILE_FROM_CASE="../../CrI3_tb.dat"
ROOT_REL="local_runs/lg_cov_window_20260517_nodeph"

export PATH="/mingw64/bin:/usr/bin:${PATH}"
export HOME=".tmp"
export TMPDIR=".tmp"
export TMP="${REPO}/.tmp"
export TEMP="${REPO}/.tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-256M}"

mkdir -p "${ROOT_REL}" "${TMPDIR}"
ROOT="${ROOT_REL}"

if [[ ! -x "hhg_sbe.exe" ]]; then
  echo "ERROR: executable not found: ${REPO}/hhg_sbe.exe" >&2
  exit 1
fi

if [[ ! -f "local_runs/CrI3_tb.dat" ]]; then
  echo "ERROR: local ASCII-path TB file not found: ${REPO}/local_runs/CrI3_tb.dat" >&2
  echo "Copy C:/Users/26507/Documents/量子光研究/新SBEs/data/CrI3_tb.dat to local_runs/CrI3_tb.dat first." >&2
  exit 1
fi

write_input() {
  local out="$1"
  local nk="$2"
  local nb_start="$3"
  local nb_end="$4"
  cat > "${out}" <<EOF
&crystal
  a1_ang = 6.998304941, -0.001225784, 0.000
  a2_ang = -3.500214030, 6.062548544, 0.000
  a3_ang = 0.000, 0.000, 25.000
  E_fermi_eV = 0.0843
  SOC = 1
  wannier_tb_file = "${TB_FILE_FROM_CASE}"
  wannier_hr_file = ""
  wannier_r_file  = ""
/

&kgrid
  nkx = ${nk}
  nky = ${nk}
/

&bands
  nv_orig = 84
  nb_start = ${nb_start}
  nb_end = ${nb_end}
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
  external_A_file = ""
/

&timestep
  dt = 0.35
  n_dt_deph = 5
/

&dephasing
  T2_fs = 1.0e30
/

&bsv
  bsv_enabled = .false.
  bsv_n_samples = 100
  bsv_mean_intensity = 2.0e11
  bsv_seed = 42
/

&method
  gauge_method = 'lg_cov'
/

&diagnostics
  run_pcenter_check = .false.
  stop_after_diagnostics = .true.
/
EOF
}

run_case() {
  local name="$1"
  local nk="$2"
  local nb_start="$3"
  local nb_end="$4"
  local dir="${ROOT}/${name}"

  mkdir -p "${dir}"
  write_input "${dir}/input.nml" "${nk}" "${nb_start}" "${nb_end}"

  echo "=== RUN ${name} ==="
  cd "${dir}"
  local start end elapsed status
  start=$(date +%s)
  set +e
  "${EXE_FROM_CASE}" input.nml > run.log 2>&1
  status=$?
  set -e
  end=$(date +%s)
  elapsed=$((end - start))
  printf "%s,%s,%s,%s,%s,%s\n" "${name}" "${nk}" "${nb_start}" "${nb_end}" "${status}" "${elapsed}" >> "../run_summary.csv"
  if [[ "${status}" -ne 0 ]]; then
    echo "ERROR: ${name} failed with exit code ${status}; see ${dir}/run.log" >&2
    exit "${status}"
  fi
}

if [[ ! -f "${ROOT}/run_summary.csv" || "${RESET_SUMMARY:-0}" == "1" ]]; then
  echo "case,nk,nb_start,nb_end,exit_code,elapsed_seconds" > "${ROOT}/run_summary.csv"
fi

case_requested() {
  local name="$1"
  local requested
  shift
  if [[ "${FILTER_COUNT:-0}" == "0" ]]; then
    return 0
  fi
  for requested in "$@"; do
    if [[ "${name}" == "${requested}" ]]; then
      return 0
    fi
  done
  return 1
}

FILTER_COUNT="$#"

if case_requested k10_b30 "$@"; then run_case k10_b30 10 70 99; fi
if case_requested k10_b40 "$@"; then run_case k10_b40 10 65 104; fi
if case_requested k10_b60 "$@"; then run_case k10_b60 10 53 112; fi
if case_requested k10_b112 "$@"; then run_case k10_b112 10 1 112; fi
if case_requested k20_b30 "$@"; then run_case k20_b30 20 70 99; fi
if case_requested k20_b40 "$@"; then run_case k20_b40 20 65 104; fi

echo "All cases finished: ${ROOT}"
