#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import compare_112old_large_hhg as H


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_peak_at() -> None:
    rows = [(4.6, 0.1), (5.0, 2.0), (5.3, 0.4), (7.0, 0.5)]
    p5 = H.peak_at(rows, 5)
    require(not p5["empty"], p5)
    require(abs(p5["HHG_total"] - 2.0) < 1e-12, p5)
    p9 = H.peak_at(rows, 9)
    require(p9["empty"], p9)


def test_ratio() -> None:
    require(abs(H.ratio(4.0, 2.0) - 2.0) < 1e-12, H.ratio(4.0, 2.0))
    require(H.ratio(1.0, 0.0) is None, "zero")


def main() -> int:
    test_peak_at()
    test_ratio()
    print("test_compare_112old_large_hhg: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
