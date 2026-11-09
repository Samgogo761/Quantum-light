import os
import subprocess
import time
from pathlib import Path

py = Path(r"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe")
script = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model\_run_squeeze_test.py")
out = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line")
env = dict(os.environ)
env["PYTHONPATH"] = str(script.parent)
env["PYTHONUNBUFFERED"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "1"
p = subprocess.Popen(
    [str(py), "-u", str(script)],
    cwd=str(script.parent),
    env=env,
    creationflags=0x00000008 | 0x00000200 | 0x08000000,
)
(out / "_tmp_squeeze_test_spawned.txt").write_text(
    f"pid={p.pid} t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8"
)
print(p.pid)
