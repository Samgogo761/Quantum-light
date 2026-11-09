#!/usr/bin/env bash
set -euo pipefail

# Usage on the server:
#   cd /public/home/wangjs/project/hBN_TB
#   bash run_all.sh
#
# Recommended command overrides on the current server, if modules are not already loaded:
#   PW_COMMAND="mpirun -np 4 /public/home/wangjs/software/q-e-qe-7.2/bin/pw.x" \
#   BANDS_COMMAND="mpirun -np 4 /public/home/wangjs/software/q-e-qe-7.2/bin/bands.x" \
#   W90_COMMAND="/public/home/wangjs/software/q-e-qe-7.2/wannier90-3.1.0/wannier90.x" \
#   PW2WAN_COMMAND="mpirun -np 4 /public/home/wangjs/software/q-e-qe-7.2/bin/pw2wannier90.x" \
#   bash run_all.sh

PW_COMMAND=${PW_COMMAND:-pw.x}
BANDS_COMMAND=${BANDS_COMMAND:-bands.x}
W90_COMMAND=${W90_COMMAND:-wannier90.x}
PW2WAN_COMMAND=${PW2WAN_COMMAND:-pw2wannier90.x}

mkdir -p tmp logs

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 1
  fi
}

need_file pseudo/B.pbe-n-kjpaw_psl.1.0.0.UPF
need_file pseudo/N.pbe-n-kjpaw_psl.1.0.0.UPF

run_pw() {
  local input=$1
  local output=$2
  echo "[$(date '+%F %T')] ${PW_COMMAND} < ${input} > ${output}"
  bash -lc "${PW_COMMAND} < '${input}' > '${output}'"
}

run_tool_redirect() {
  local command=$1
  local args=$2
  local output=$3
  echo "[$(date '+%F %T')] ${command} ${args} > ${output}"
  bash -lc "${command} ${args} > '${output}'"
}

run_pw hBN_scf.in logs/hBN_scf.out
run_pw hBN_bands.in logs/hBN_bands.out
run_tool_redirect "${BANDS_COMMAND}" "< hBN_bands_pp.in" logs/hBN_bands_pp.out
run_pw hBN_nscf.in logs/hBN_nscf.out
run_tool_redirect "${W90_COMMAND}" "-pp hBN" logs/hBN_wannier90_pp.out
run_tool_redirect "${PW2WAN_COMMAND}" "< hBN.pw2wan" logs/hBN_pw2wan.out

cat <<'MSG'

First-pass data generated: hBN.eig, hBN.mmn, hBN.amn.
Before the final wannier90 run, inspect hBN.eig and set dis_win_* / dis_froz_* in hBN.win.
A useful check is:
  for emax in 8 12 16 20; do
    count=$(awk -v e1="-20" -v e2="$emax" '$3>e1 && $3<e2' hBN.eig | awk '{print $2}' | sort -n | uniq -c | awk '{print $1}' | sort -n | head -1)
    echo "dis_win [-20, $emax] eV: min bands/kpt = $count"
  done

Then run:
  ${W90_COMMAND} hBN > logs/hBN_wannier90.out

MSG
