#!/bin/bash
#SBATCH --job-name=M52_k40_pN
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=18:00:00
#SBATCH --output=M52_plusN_%j.out
#SBATCH --error=M52_plusN_%j.err
export MODEL=M52 NV=24 NB=52 SIGN=plus
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
