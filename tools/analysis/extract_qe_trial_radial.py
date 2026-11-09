#!/usr/bin/env python3
"""Extract the radial definition of Wannier90 trial orbitals from QE 7.2 pw2wannier90.f90.

Does not compute physics. Prints the source snippets that implement R(r).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

KEYS = (
    "zona",
    "radial",
    "hydrogen",
    "rvalue",
    "alpha",
    "x_w",
    "compute_amn",
    "guiding",
    "ylm",
    "bspline",
    "r_w",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = args.src.read_text(errors="replace")
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines, 1):
        low = line.lower()
        if any(k in low for k in KEYS):
            hits.append(f"{i}:{line.rstrip()}")
    # Also grab subroutines
    subs = [f"{i}:{line.rstrip()}" for i, line in enumerate(lines, 1) if re.match(r"\s*subroutine\s+", line, re.I)]
    report = "\n".join(
        ["# subroutines"] + subs + ["", "# keyword hits"] + hits[:200]
    )
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
