#!/bin/bash
#SBATCH --job-name=S2_M88c20_m
#SBATCH --partition=part_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=30:00:00
#SBATCH --output=S2_M88v58c20_minus_%j.out
#SBATCH --error=S2_M88v58c20_minus_%j.err
export MODEL=M88 NV=58 NB=78 SIGN=minus NKX=40 NKY=40
exec bash /public/home/wangjs/project/New_SBEs/Quantum-light/deploy/run_Mxx_magnetic_hhg.sh
