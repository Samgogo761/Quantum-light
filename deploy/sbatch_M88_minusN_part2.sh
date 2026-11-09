#!/bin/bash
#SBATCH --job-name=M88_k40_mN
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=36:00:00
#SBATCH --output=M88_minusN_%j.out
#SBATCH --error=M88_minusN_%j.err
export MODEL=M88 NV=58 NB=88 SIGN=minus
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
