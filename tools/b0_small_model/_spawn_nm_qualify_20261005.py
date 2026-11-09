#!/usr/bin/env python3
"""Spawn the nm-qualify runner as a detached subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005"
PID = OUT / "_pid.txt"
HERE = Path(__file__).resolve().parent


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.setdefault("Q8_STAGE", "probe")
    p = subprocess.Popen(
        [sys.executable, str(HERE / "_run_nm_qualify_20261005.py")],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    PID.write_text(str(p.pid), encoding="utf-8")
    print(
        json.dumps(
            {
                "pid": p.pid,
                "stage": env.get("Q8_STAGE"),
                "log": str(OUT / "_run_stdout.txt"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
