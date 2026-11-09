#!/bin/bash
#SBATCH --job-name=M88c20_k60_m
#SBATCH --partition=part_1
#SBATCH --nodelist=node05,node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=P2_M88c20_k60_minus_%j.out
#SBATCH --error=P2_M88c20_k60_minus_%j.err
export MODEL=M88 NV=58 NB=78 SIGN=minus NKX=60 NKY=60
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
