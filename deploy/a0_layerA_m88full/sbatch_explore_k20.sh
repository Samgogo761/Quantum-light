#!/bin/bash
#=============================================================================
# A0 explore: M88-full @ k20, squeezed_vacuum GH3, shared ±N nodes
# Partition: part_1
#
# Array index encoding:
#   idx = ((ir * NTH + ith) * 2 + idom)
#   ir  in 0..NR-1   (r = 0.5,1.5,2.5)
#   ith in 0..NTH-1  (theta = 0,90,180,270)
#   idom 0=+N, 1=-N
# Total: 3*4*2 = 24 tasks
#=============================================================================
#SBATCH --job-name=a0_m88_k20
#SBATCH --partition=part_1
#SBATCH --array=0-23%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=a0_m88_k20_%A_%a.out
#SBATCH --error=a0_m88_k20_%A_%a.err

set -eo pipefail

NTHREADS="${SLURM_CPUS_PER_TASK:-36}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO="${REPO:-/public/home/wangjs/project/New_SBEs/Quantum-light}"
TB_PLUS="${TB_PLUS:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb.dat}"
TB_MINUS="${TB_MINUS:-/public/home/wangjs/project/CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb_mN_Pconj.dat}"
OUTROOT="${OUTROOT:-${WORKDIR}/output_a0_m88full_k20}"
NODES_DIR="${NODES_DIR:-${WORKDIR}/nodes_gh3}"
NK=20

R_LIST=(0.5 1.5 2.5)
TH_LIST=(0 90 180 270)
NR=${#R_LIST[@]}
NTH=${#TH_LIST[@]}

IDX="${SLURM_ARRAY_TASK_ID:-0}"
IDOM=$(( IDX % 2 ))
TMP=$(( IDX / 2 ))
ITH=$(( TMP % NTH ))
IR=$(( TMP / NTH ))
if [ "${IR}" -ge "${NR}" ]; then
  echo "ERROR: bad array index ${IDX}"; exit 1
fi

RVAL="${R_LIST[${IR}]}"
THVAL="${TH_LIST[${ITH}]}"
if [ "${IDOM}" -eq 0 ]; then
  DOM="plusN"; TB="${TB_PLUS}"
else
  DOM="minusN"; TB="${TB_MINUS}"
fi

RTAG=$(echo "${RVAL}" | tr '.' 'p')
NODES_FILE="${NODES_DIR}/nodes_sv_r${RTAG}_th${THVAL}_gh3.dat"
CASE=$(printf "sv_r%s_th%03d_%s" "${RTAG}" "${THVAL}" "${DOM}")
OUTDIR="${OUTROOT}/${CASE}"

cd "${WORKDIR}"
mkdir -p "${OUTDIR}"
echo "============================================="
echo " A0 explore M88-full k${NK}  case=${CASE}"
echo " nodes=${NODES_FILE}"
echo " TB=${TB}"
echo " host=$(hostname) date=$(date)"
echo "============================================="

if [ ! -f "${NODES_FILE}" ]; then
  echo "ERROR: missing shared nodes file ${NODES_FILE}"
  echo "Run: python3 deploy/a0_layerA_m88full/make_gh_nodes.py --outdir ${NODES_DIR} --gh-order 3"
  exit 1
fi
if [ ! -f "${TB}" ]; then
  echo "ERROR: TB not found: ${TB}"; exit 1
fi

export OMP_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/vtune/2023.2.0/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/public/software/compiler/intel/oneapi/mkl/2023.2.0/lib/intel64:${LD_LIBRARY_PATH:-}
set +u
source /public/software/compiler/intel/oneapi/compiler/2023.2.0/env/vars.sh
source /public/software/compiler/intel/oneapi/mpi/2021.10.0/env/vars.sh
set -u

# compile once
LOCK="${REPO}/.compile_a0.lockdir"
DONE="${REPO}/.compile_a0.done"
if [ ! -f "${DONE}" ] || [ ! -x "${REPO}/hhg_sbe" ]; then
  WAITED=0
  while ! mkdir "${LOCK}" 2>/dev/null; do
    [ -f "${DONE}" ] && [ -x "${REPO}/hhg_sbe" ] && break
    sleep 10; WAITED=$((WAITED+10))
    if [ "${WAITED}" -gt 2400 ]; then echo "ERROR: compile lock timeout"; exit 1; fi
  done
  if [ ! -f "${DONE}" ] || [ ! -x "${REPO}/hhg_sbe" ]; then
    echo "Compiling hhg_sbe with ifort..."
    (
      cd "${REPO}"
      make clean
      make FC=ifort FFLAGS="-O3 -qopenmp -mkl -fpp -heap-arrays" LDFLAGS=""
      touch "${DONE}"
    )
  fi
  rmdir "${LOCK}" 2>/dev/null || true
fi

# build case input from template
TMPL="${REPO}/deploy/a0_layerA_m88full/input_template.nml"
sed -e "s|TB_PLACEHOLDER|${TB}|g" \
    -e "s|NK_PLACEHOLDER|${NK}|g" \
    -e "s|R_PLACEHOLDER|${RVAL}|g" \
    -e "s|TH_PLACEHOLDER|${THVAL}|g" \
    -e "s|NODES_PLACEHOLDER|${NODES_FILE}|g" \
    "${TMPL}" > "${OUTDIR}/input.nml"
cp -f "${NODES_FILE}" "${OUTDIR}/"

cd "${OUTDIR}"
"${REPO}/hhg_sbe" input.nml | tee run.log
echo "DONE ${CASE} $(date)"
