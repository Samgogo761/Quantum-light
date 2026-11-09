import os
import subprocess
import time
from pathlib import Path

py = Path(r"C:\Users\26507\AppData\Local\Python\pythoncore-3.14-64\python.exe")
root = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light")
log = root / "results" / "q_line" / "_tmp_vacuum_switch_test.txt"
log.parent.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env["PYTHONPATH"] = str(root / "tools" / "b0_small_model")
env["PYTHONUNBUFFERED"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "1"
cmd = [
    str(py), "-u", "-m", "unittest", "-v",
    "test_q8_m0_qinput_shared_kk_vacuum_switch_20261101",
]
with log.open("w", encoding="utf-8") as fh:
    fh.write(f"start {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    fh.flush()
    p = subprocess.Popen(
        cmd,
        cwd=str(root / "tools" / "b0_small_model"),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        creationflags=0x00000008 | 0x00000200 | 0x08000000,
    )
print(p.pid)
print(log)
