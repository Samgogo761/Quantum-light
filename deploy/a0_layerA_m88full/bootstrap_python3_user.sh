#!/bin/bash
# Install a user-local Miniconda + numpy for A0 production validators.
# Login nodes here are often glibc 2.17 (CentOS 7-class); the "latest"
# Miniconda installer requires glibc >= 2.28 and will fail.  Use a pinned
# py39 installer known to run on glibc 2.17 unless overridden.
set -euo pipefail

PREFIX="${A0_CONDA_PREFIX:-${HOME}/software/miniconda3}"
INSTALLER="${TMPDIR:-/tmp}/Miniconda3-a0-bootstrap.sh"
# Pinned for glibc 2.17 clusters; override with A0_MINICONDA_URL if needed.
URL="${A0_MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh}"

mkdir -p "$(dirname "${PREFIX}")"

if [ -x "${PREFIX}/bin/python" ]; then
  echo "Existing python: ${PREFIX}/bin/python"
else
  echo "Downloading Miniconda from ${URL} ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${URL}" -o "${INSTALLER}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${INSTALLER}" "${URL}"
  else
    echo "ERROR: need curl or wget to fetch Miniconda" >&2
    exit 1
  fi
  bash "${INSTALLER}" -b -p "${PREFIX}"
fi

PY="${PREFIX}/bin/python"
if ! "${PY}" -c "import numpy" >/dev/null 2>&1; then
  if [ -x "${PREFIX}/bin/conda" ]; then
    "${PREFIX}/bin/conda" install -y numpy || true
  fi
  if ! "${PY}" -c "import numpy" >/dev/null 2>&1; then
    "${PY}" -m pip install numpy
  fi
fi

"${PY}" - <<'PY'
import sys
import numpy
print(sys.version)
print("numpy", numpy.__version__)
assert sys.version_info >= (3, 8)
PY

echo "OK: export A0_PYTHON3=${PY}"
echo "    or: source deploy/a0_layerA_m88full/resolve_python3.sh"
