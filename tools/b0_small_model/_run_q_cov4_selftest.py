from pathlib import Path
import sys

log = Path(
    r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\g1_a6_fix_20261024\q_cov4_selftest.txt"
)
log.parent.mkdir(parents=True, exist_ok=True)
stream = open(log, "w", encoding="utf-8")
sys.stdout = stream
sys.stderr = stream
import q_cov4_reference_check as m

ok = m.selftest()
print("EXIT", 0 if ok else 1)
stream.flush()
raise SystemExit(0 if ok else 1)
