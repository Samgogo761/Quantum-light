#!/usr/bin/env python3
"""Launch gamma12 CCR-fix main with a log file. Does not overwrite old JSON."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_gamma12_ccr_fix_20261006"
LOG = OUT / "_run_stdout.txt"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    log = LOG.open("w", encoding="utf-8")
    sys.stdout = log
    sys.stderr = log
    print("runner start", flush=True)
    import q8_m0_p1_h2_finite_kcv_gamma12_ccr_fix_20261006 as m

    print("imported", flush=True)
    rc = int(m.main())
    print("runner done", rc, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
