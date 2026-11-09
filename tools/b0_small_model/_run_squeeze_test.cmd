@echo off
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model
cd /d C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model
set LOG=C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\_tmp_squeeze_test_20261031.txt
echo start> %LOG%
"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe" -u -c "import sys; print('pyok', sys.version); import numpy; print('np', numpy.__version__); import test_q8_m0_qinput_shared_kk_squeeze_diag_20261031 as t; print('imported')" >> %LOG% 2>&1
echo import_exit=%ERRORLEVEL% >> %LOG%
"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe" -u -m unittest -v test_q8_m0_qinput_shared_kk_squeeze_diag_20261031 >> %LOG% 2>&1
echo unittest_exit=%ERRORLEVEL% >> %LOG%
