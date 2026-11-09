import subprocess
import time
from pathlib import Path

cmd = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\tools\b0_small_model\_run_win_identities.cmd")
out = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\g1_win_identities_20261024")
out.mkdir(parents=True, exist_ok=True)
p = subprocess.Popen(
    ["cmd.exe", "/c", str(cmd)],
    cwd=str(cmd.parent),
    creationflags=0x00000008 | 0x00000200 | 0x08000000,
)
(out / "spawned.txt").write_text(
    f"pid={p.pid} t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8"
)
print(p.pid)
