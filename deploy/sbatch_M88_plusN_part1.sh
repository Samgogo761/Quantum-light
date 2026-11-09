#!/bin/bash
#SBATCH --job-name=M88_k40_pN
#SBATCH --partition=part_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=M88_plusN_%j.out
#SBATCH --error=M88_plusN_%j.err
export MODEL=M88 NV=58 NB=88 SIGN=plus
exec bash "$(dirname "$0")/run_Mxx_magnetic_hhg.sh"
