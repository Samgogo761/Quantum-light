#!/bin/bash
# Prepare / submit k60 for up to two S2 winners (P2).
# Edit WINNERS below after P1 gates, then run on login node.
# Example:
#   WINNERS="M88:58:68 M52:24:30"
set -eo pipefail
DEP=/public/home/wangjs/project/New_SBEs/Quantum-light/deploy
WINNERS="${WINNERS:-}"
if [[ -z "$WINNERS" ]]; then
  echo "Set WINNERS='MODEL:NV:NB ...' e.g. WINNERS='M88:58:68 M52:24:30'"
  exit 1
fi
cd "$DEP"
for spec in $WINNERS; do
  IFS=: read -r model nv nb <<<"$spec"
  for sign in plus minus; do
    script=$(mktemp "$DEP/sbatch_k60_${model}_b${nb}_${sign}_XXXXXX.sh")
    cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_k60_b${nb}_${sign:0:1}N
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=36:00:00
#SBATCH --output=k60_${model}_b${nb}_${sign}_%j.out
#SBATCH --error=k60_${model}_b${nb}_${sign}_%j.err
export MODEL=${model} NV=${nv} NB=${nb} SIGN=${sign} NKX=60 NKY=60
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
EOF
    chmod +x "$script"
    jid=$(sbatch --parsable "$script")
    echo "$spec $sign -> $jid"
  done
done
squeue -u "$USER"
