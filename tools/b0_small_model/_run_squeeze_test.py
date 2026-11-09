#!/usr/bin/env python3
"""File-redirected unittest runner. Avoids cmd stdout capture issues."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

LOG = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\q_line\_tmp_squeeze_test_20261031.txt")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_q8_m0_qinput_shared_kk_squeeze_diag_20261031 as mod

def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromModule(mod)
    with LOG.open("w", encoding="utf-8") as fh:
        fh.write("runner start\n")
        fh.flush()
        result = unittest.TextTestRunner(stream=fh, verbosity=2).run(suite)
        fh.write(f"tests={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}\n")
        ok = result.wasSuccessful()
        fh.write("PASS\n" if ok else "FAIL\n")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
