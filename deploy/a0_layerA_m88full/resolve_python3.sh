#!/bin/bash
# Resolve a usable Python >=3.8 with numpy for A0 production gates.
# Preference order:
#   1) A0_PYTHON3 if set and working
#   2) python3 / python on PATH if >=3.8 and has numpy
#   3) $HOME/software/miniconda3 or micromamba env "a0tools"
#
# Usage:
#   source resolve_python3.sh   # exports A0_PYTHON3
#   ./resolve_python3.sh --print
set -euo pipefail

_a0_python_ok() {
  local py="$1"
  command -v "$py" >/dev/null 2>&1 || return 1
  "$py" - <<'PY' >/dev/null 2>&1
import sys
assert sys.version_info >= (3, 8)
import numpy
PY
}

_a0_pick() {
  if [ -n "${A0_PYTHON3:-}" ] && _a0_python_ok "${A0_PYTHON3}"; then
    echo "${A0_PYTHON3}"
    return 0
  fi
  local cand
  for cand in python3 python; do
    if _a0_python_ok "${cand}"; then
      command -v "${cand}"
      return 0
    fi
  done
  for cand in \
      "${HOME}/software/miniconda3/bin/python" \
      "${HOME}/software/micromamba/envs/a0tools/bin/python" \
      "${HOME}/miniconda3/bin/python" \
      "${REPO:-}/.a0_python/bin/python"
  do
    if [ -x "${cand}" ] && _a0_python_ok "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

if [ "${1:-}" = "--print" ]; then
  _a0_pick || {
    echo "ERROR: no usable Python>=3.8+numpy. Install with:" >&2
    echo "  bash deploy/a0_layerA_m88full/bootstrap_python3_user.sh" >&2
    exit 1
  }
else
  A0_PYTHON3="$(_a0_pick)" || {
    echo "ERROR: no usable Python>=3.8+numpy. Install with:" >&2
    echo "  bash deploy/a0_layerA_m88full/bootstrap_python3_user.sh" >&2
    return 1 2>/dev/null || exit 1
  }
  export A0_PYTHON3
  echo "A0_PYTHON3=${A0_PYTHON3}"
fi
