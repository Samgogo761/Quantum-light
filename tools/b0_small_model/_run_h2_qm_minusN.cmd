@echo off
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
cd /d C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model
py -3 q8_m0_h2_qm_minusN_20261025.py > C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\h2_qm_minusN_20261025\stdout.txt 2>&1
echo exit=%ERRORLEVEL% >> C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\h2_qm_minusN_20261025\stdout.txt
