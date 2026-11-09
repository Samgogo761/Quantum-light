#!/bin/bash
#SBATCH --job-name=M64_k40_pN
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=M64_plusN_%j.out
#SBATCH --error=M64_plusN_%j.err
export MODEL=M64 NV=36 NB=64 SIGN=plus
exec bash "$(dirname "$0")/run_Mxx_magnetic_hhg.sh"
