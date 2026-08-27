#!/usr/bin/env python3
"""Deprecated cluster wrapper. Canonical generator: diagnose_full112_top.py."""
from __future__ import annotations

import sys
from pathlib import Path

from diagnose_full112_top import DEFAULT_DATA, main as generate_main

CLUSTER_ROOT = Path("/public/home/wangjs/project/CrI3_TB")


def main() -> int:
    argv = list(sys.argv[1:])
    if not any(a.startswith("--data-root") for a in argv) and (CLUSTER_ROOT / "wannier").is_dir():
        argv = ["--data-root", str(CLUSTER_ROOT)] + argv
    elif not any(a.startswith("--data-root") for a in argv):
        argv = ["--data-root", str(DEFAULT_DATA)] + argv
    return generate_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
