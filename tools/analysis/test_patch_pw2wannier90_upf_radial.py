#!/usr/bin/env python3
"""Tests for the QE 7.2 pw2wannier90 UPF-radial patcher."""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from patch_pw2wannier90_upf_radial import (
    BODY_OLD,
    CALL_OLD,
    DECL_OLD,
    SUB_OLD,
    patch_text,
)


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_patch_stub() -> None:
    stub = (
        "      gf(:,iw) = (0.d0,0.d0)\n"
        + CALL_OLD
        + "\n"
        + SUB_OLD
        + "\n"
        + "  ! local variables\n"
        + "  real(DP), PARAMETER :: xmin=-6.d0, dx=0.025d0, rmax=10.d0\n"
        + "  real(DP) :: rad_int, pref, x\n"
        + DECL_OLD
        + "\n"
        + "  mesh_r = nint ( ( log ( rmax ) - xmin ) / dx + 1 )\n"
        + BODY_OLD
        + "  RETURN\nEND SUBROUTINE radialpart\n"
    )
    out = patch_text(stub)
    require("UPF_CUSTOM_RADIAL_LOADED" in out, "marker")
    require("radial, iw)" in out, "new signature")
    require(CALL_OLD not in out, "old call remains")
    require("load_upf_custom_radials" in out, "loader")
    again = patch_text(out)
    require(again == out, "idempotent")


def main() -> int:
    test_patch_stub()
    print("test_patch_pw2wannier90_upf_radial: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
