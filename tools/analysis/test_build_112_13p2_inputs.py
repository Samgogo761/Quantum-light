#!/usr/bin/env python3
"""Tests for the 112_[-6,13.2] window-only control."""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import build_112_13p2_inputs as B


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


STUB = """num_wann = 112
num_bands = 400
dis_win_min  = -6.0
dis_win_max  =  8.0
dis_froz_min = -1.5
dis_froz_max =  0.8
begin projections
Cr1: d
Cr2: d
I:   p
end projections
"""


def test_patch_win() -> None:
    out = B.patch_win(STUB)
    require("num_wann = 112" in out, out)
    require("num_wann = 232" not in out, out)
    require("dis_win_max  = 13.2" in out, out)
    require("dis_win_max  =  8.0" not in out, out)
    require("I:   p" in out, out)
    require("I:   d" not in out, out)
    require("dis_froz_min = -1.5" in out, out)


def test_refuse_id() -> None:
    bad = STUB.replace("I:   p\n", "I:   p\nI:   d\n")
    try:
        B.patch_win(bad)
    except RuntimeError as exc:
        require("I:d" in str(exc), exc)
        return
    raise RuntimeError("should refuse I:d")


def test_refuse_wrong_wann() -> None:
    bad = STUB.replace("num_wann = 112", "num_wann = 232")
    try:
        B.patch_win(bad)
    except RuntimeError:
        return
    raise RuntimeError("should refuse num_wann=232")


def main() -> int:
    test_patch_win()
    test_refuse_id()
    test_refuse_wrong_wann()
    print("test_build_112_13p2_inputs: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
