#!/bin/bash
# Paste on cri3-login (MobaXterm). Uploads assumed already done, OR run scp from Windows first.
set -eo pipefail
QL=/public/home/wangjs/project/New_SBEs/Quantum-light
TB=/public/home/wangjs/project/CrI3_TB/wannier
cd "$QL/deploy"
chmod +x run_Mxx_magnetic_hhg.sh sbatch_M64_plusN_part1.sh sbatch_M64_minusN_part1.sh \
         sbatch_M52_plusN_part2.sh sbatch_M88_plusN_part1.sh 2>/dev/null || true

echo "=== TB presence ==="
ls -lh "$TB/models/M52_interface/CrI3_M52_tb.dat" \
       "$TB/models/M64_allI_pz/CrI3_M64_tb.dat" \
       "$TB/models/M88_allI_2p/CrI3_M88_tb.dat" \
       "$TB/models/M64_allI_pz/CrI3_M64_tb_mN_Pconj.dat" 2>&1 || true

echo "=== Submit +N queue ==="
J64=$(sbatch --parsable sbatch_M64_plusN_part1.sh)
J52=$(sbatch --parsable sbatch_M52_plusN_part2.sh)
J88=$(sbatch --parsable sbatch_M88_plusN_part1.sh)
echo "M64+N=$J64  M52+N=$J52  M88+N=$J88"

if [[ -f "$TB/models/M64_allI_pz/CrI3_M64_tb_mN_Pconj.dat" ]]; then
  J64m=$(sbatch --parsable sbatch_M64_minusN_part1.sh)
  echo "M64-N=$J64m"
else
  echo "SKIP M64-N: missing CrI3_M64_tb_mN_Pconj.dat — upload then: sbatch sbatch_M64_minusN_part1.sh"
fi

squeue -u "$USER"
