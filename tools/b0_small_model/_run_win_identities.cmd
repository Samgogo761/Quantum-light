@echo off
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
cd /d C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model
py -3 q8_m0_g1_win_identities_20261024.py > C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\g1_win_identities_20261024\stdout.txt 2>&1
echo exit=%ERRORLEVEL% >> C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\g1_win_identities_20261024\stdout.txt
