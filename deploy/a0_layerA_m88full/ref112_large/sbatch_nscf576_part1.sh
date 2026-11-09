#!/bin/bash
#SBATCH --job-name=r112L_nscf576
#SBATCH --partition=part_1
#SBATCH --chdir=/public/home/wangjs/project/CrI3_TB/window144_hilbert/ref112_large
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --time=00:05:00
#
# Job 32972: one-shot 576-k x nbnd=400 OOM SIGKILL ~240 GB. Do not resubmit.

set -eo pipefail
echo "REFUSE: one-shot 576-k x 400 NSCF OOM'd as job 32972 (~240 GB SIGKILL)." >&2
echo "Use sbatch_nscf_chunks_part1.sh (72 x 8 k, same scale as 32911)." >&2
exit 1
