#!/bin/bash
#SBATCH --job-name=S2_M52c10_p
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=S2_M52v24c10_plus_%j.out
#SBATCH --error=S2_M52v24c10_plus_%j.err
export MODEL=M52 NV=24 NB=34 SIGN=plus NKX=40 NKY=40
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
