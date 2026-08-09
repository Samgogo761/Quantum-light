#!/usr/bin/env python3
"""Negative tests for bsv_propagate_ids parsing (Fortran CLI helper).

CLI contract for --expect-fail <list>:
  - parser rejects  -> nonzero exit (error stop) => PASS for this case
  - parser accepts  -> exit 0 and prints PARSE_ACCEPTED => FAIL for this case
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BAD_CASES = [
    "",
    ",",
    "1,",
    ",1",
    "1,,2",
    "1, ,2",
    "1 2,3",
    "1,2 3",
    "0",
    "-1",
    "1,-2",
    "a",
    "1.5",
    "1,2,",
    "1e2",
    "+3",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bin",
        type=Path,
        default=Path("test_parse_ids"),
        help="Path to test_parse_propagate_ids executable",
    )
    args = ap.parse_args()
    bin_path = args.bin
    if not bin_path.is_file() and Path(str(bin_path) + ".exe").is_file():
        bin_path = Path(str(bin_path) + ".exe")
    if not bin_path.is_file():
        print(f"FAIL: missing binary {args.bin}", file=sys.stderr)
        return 2

    r = subprocess.run(
        [str(bin_path), "--expect-ok", "1,2,3"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("FAIL: --expect-ok 1,2,3 returned non-zero", file=sys.stderr)
        print(r.stdout, r.stderr, file=sys.stderr)
        return 1

    # Sanity: the false-gate case must be rejected by the parser.
    r_space = subprocess.run(
        [str(bin_path), "--expect-fail", "1 2,3"],
        capture_output=True,
        text=True,
    )
    if r_space.returncode == 0 or "PARSE_ACCEPTED" in (r_space.stdout + r_space.stderr):
        print("FAIL: '1 2,3' was accepted (false negative gate)", file=sys.stderr)
        print(r_space.stdout, r_space.stderr, file=sys.stderr)
        return 1

    failed: list[str] = []
    for case in BAD_CASES:
        r = subprocess.run(
            [str(bin_path), "--expect-fail", case],
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        # Accepted illegal input => exit 0 / PARSE_ACCEPTED.
        if r.returncode == 0 or "PARSE_ACCEPTED" in out:
            failed.append(repr(case))

    if failed:
        print("FAIL: these invalid lists were accepted:", file=sys.stderr)
        for c in failed:
            print(f"  {c}", file=sys.stderr)
        return 1

    print(f"RESULT[parse_ids_neg]: PASS ({len(BAD_CASES)} cases rejected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
