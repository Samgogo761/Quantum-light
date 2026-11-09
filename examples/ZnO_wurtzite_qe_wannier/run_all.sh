#!/usr/bin/env bash
set -euo pipefail

# Usage on the server:
#   cd /public/home/wangjs/project/ZnO_TB
#   bash run_all.sh
#
# Override executable names if your module environment uses different wrappers:
#   PW_COMMAND="mpirun -np 36 pw.x" W90_COMMAND="wannier90.x" PW2WAN_COMMAND="pw2wannier90.x" bash run_all.sh

PW_COMMAND=${PW_COMMAND:-pw.x}
W90_COMMAND=${W90_COMMAND:-wannier90.x}
PW2WAN_COMMAND=${PW2WAN_COMMAND:-pw2wannier90.x}

mkdir -p tmp logs

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 1
  fi
}

need_file pseudo/Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF
need_file pseudo/O.pbe-n-kjpaw_psl.1.0.0.UPF

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

run_pw ZnO_scf.in logs/ZnO_scf.out
run_pw ZnO_bands.in logs/ZnO_bands.out
run_pw ZnO_nscf.in logs/ZnO_nscf.out
run_tool_redirect "${W90_COMMAND}" "-pp ZnO" logs/ZnO_wannier90_pp.out
run_tool_redirect "${PW2WAN_COMMAND}" "< ZnO.pw2wan" logs/ZnO_pw2wan.out
run_tool_redirect "${W90_COMMAND}" "ZnO" logs/ZnO_wannier90.out

echo "[$(date '+%F %T')] Done. Expected main outputs: ZnO.eig, ZnO.mmn, ZnO.amn, ZnO_hr.dat, ZnO_tb.dat, ZnO_band.dat, ZnO.wout"
