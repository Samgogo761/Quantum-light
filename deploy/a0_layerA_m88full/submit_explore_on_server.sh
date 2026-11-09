#!/bin/bash
set -eo pipefail
REPO=/public/home/wangjs/project/New_SBEs/Quantum-light
cd "$REPO"
rm -f .compile_a0.done
mkdir -p runs/a0_m88full_k20
cd runs/a0_m88full_k20
export WORKDIR="$PWD"
export NODES_DIR="$REPO/deploy/a0_layerA_m88full/nodes_gh3"
export REPO
chmod +x "$REPO/deploy/a0_layerA_m88full/sbatch_explore_k20.sh"
ls "$NODES_DIR" | head
JOB=$(sbatch --parsable "$REPO/deploy/a0_layerA_m88full/sbatch_explore_k20.sh")
echo "SUBMITTED_JOB=$JOB"
squeue -u wangjs | head -40
