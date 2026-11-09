#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_conj_prop_20261006"
HERE = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["OPENBLAS_NUM_THREADS"] = "1"
env["PYTHONUNBUFFERED"] = "1"
log = (OUT / "_run_stdout.txt").open("w", encoding="utf-8")
p = subprocess.Popen(
    [sys.executable, "-u", str(HERE / "q8_m0_p1_h2_finite_kcv_gamma12_conj_prop_20261006.py")],
    cwd=str(REPO), env=env, stdout=log, stderr=log,
    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
)
(OUT / "_pid.txt").write_text(str(p.pid), encoding="utf-8")
print(json.dumps({"pid": p.pid, "log": str(OUT / "_run_stdout.txt")}), flush=True)
