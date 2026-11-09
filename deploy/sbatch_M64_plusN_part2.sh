#!/bin/bash
#SBATCH --job-name=M64_k40_pN
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=18:00:00
#SBATCH --output=M64_plusN_%j.out
#SBATCH --error=M64_plusN_%j.err
export MODEL=M64 NV=36 NB=64 SIGN=plus
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
