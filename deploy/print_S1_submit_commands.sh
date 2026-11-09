#!/bin/bash
# Prepare (do NOT auto-submit) S1 +N jobs for M52/M64/M88.
# After user confirms P4_SBE执行前确认.md, run the echoed sbatch lines.
set -euo pipefail
cd "$(dirname "$0")"

echo "# Suggested submits when part_2 has free nodes:"
echo "MODEL=M64 NV=36 NB=64 sbatch --job-name=M64_k40_pN run_Mxx_parent_hhg_k40.sh"
echo "MODEL=M52 NV=24 NB=52 sbatch --job-name=M52_k40_pN run_Mxx_parent_hhg_k40.sh"
echo "MODEL=M88 NV=58 NB=88 sbatch --job-name=M88_k40_pN --mem=60G --time=18:00:00 run_Mxx_parent_hhg_k40.sh"
echo
echo "# -N requires P-conjugate TB construction (same as historical AFM workflow)."
echo "# Do not submit -N until +N baselines pass |J(0)| and gap checks."
