import os
import subprocess
import time
from pathlib import Path

py = Path(r"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe")
root = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
out = root / "results" / "q_line" / "qinput_a_shared_kk_ds_nmax10_20261106"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env["PYTHONPATH"] = str(root / "tools" / "b0_small_model")
env["PYTHONUNBUFFERED"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "1"
p = subprocess.Popen(
    [str(py), "-u", "-m", "unittest",
     "test_q8_m0_qinput_shared_kk_ds_nmax10_20261106"],
    cwd=str(root / "tools" / "b0_small_model"),
    env=env,
    stdout=open(out / "unittest.txt", "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    creationflags=0x00000008 | 0x00000200 | 0x08000000,
)
(out / "spawned_test.txt").write_text(
    f"pid={p.pid} t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
print(p.pid)
