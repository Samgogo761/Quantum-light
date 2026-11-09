#!/bin/bash
# Submit S2 targeted Bloch-truncation +/-N array on part_2 (no M64).
# Usage (on login node):
#   bash deploy/submit_S2_targeted_pmN_part2.sh
set -eo pipefail
DEP=/public/home/wangjs/project/New_SBEs/Quantum-light/deploy
cd "$DEP"

submit_one() {
  local model="$1" nv="$2" nb="$3" sign="$4" time="$5" mem="$6" name="$7"
  local script
  script=$(mktemp "$DEP/sbatch_S2_${name}_${sign}_XXXX.sh")
  cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}_${sign:0:1}N
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=S2_${name}_${sign}_%j.out
#SBATCH --error=S2_${name}_${sign}_%j.err
export MODEL=${model} NV=${nv} NB=${nb} SIGN=${sign} NKX=40 NKY=40
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
EOF
  chmod +x "$script"
  local jid
  jid=$(sbatch --parsable "$script")
  echo "${name} ${sign}N -> ${jid}  (nv=${nv} nb=${nb})"
}

echo "=== S2 targeted +/-N @k40 (part_2) ==="
# A M88-v58c10=68
submit_one M88 58 68 plus  24:00:00 56G M88v58c10
submit_one M88 58 68 minus 24:00:00 56G M88v58c10
# B M88-v58c20=78
submit_one M88 58 78 plus  30:00:00 56G M88v58c20
submit_one M88 58 78 minus 30:00:00 56G M88v58c20
# C M52-v24c6=30
submit_one M52 24 30 plus  12:00:00 56G M52v24c6
submit_one M52 24 30 minus 12:00:00 56G M52v24c6
# D M52-v24c10=34
submit_one M52 24 34 plus  12:00:00 56G M52v24c10
submit_one M52 24 34 minus 12:00:00 56G M52v24c10

squeue -u "$USER"
echo "NOTE: no M64 truncations submitted."
