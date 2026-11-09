import os
import subprocess
import time
from pathlib import Path

py = Path(r"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe")
script = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model\q8_m0_qinput_shared_kk_squeeze_diag_20261031.py")
root = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
out = root / "results" / "q_line" / "qinput_a_shared_kk_squeeze_diag_20261031"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env["PYTHONPATH"] = str(script.parent)
env["PYTHONUNBUFFERED"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "1"
p = subprocess.Popen(
    [str(py), "-u", str(script),
     "--report", str(out / "diag.json")],
    cwd=str(root),
    env=env,
    creationflags=0x00000008 | 0x00000200 | 0x08000000,
)
(out / "spawned.txt").write_text(
    f"pid={p.pid} t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8"
)
print(p.pid)
