#!/usr/bin/env python3
"""Launch nm-qualify.main with a log file. Does not overwrite old JSON."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("Q8_STAGE", "probe")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005"
LOG = OUT / "_run_stdout.txt"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    log = LOG.open("w", encoding="utf-8")
    sys.stdout = log
    sys.stderr = log
    print("runner start", flush=True)
    print("Q8_STAGE", os.environ.get("Q8_STAGE"), flush=True)
    import q8_m0_p1_h2_finite_kcv_input_close_20261005 as m

    print("imported", flush=True)
    rc = int(m.main())
    print("runner done", rc, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
