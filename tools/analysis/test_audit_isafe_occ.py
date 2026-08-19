#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_isafe_occ import FAIL_ANCHOR_29403, bracket_for_family


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def test_valid_bracket() -> None:
    rows = [
        {"cep_family": "GH7c", "I_Wcm2": 1.0e12, "overall": "PASS"},
        {"cep_family": "GH7c", "I_Wcm2": 1.63e12, "overall": "PASS"},
        {"cep_family": "GH7c", "I_Wcm2": 2.5e12, "overall": "FAIL"},
    ]
    out = bracket_for_family(rows, "GH7c", FAIL_ANCHOR_29403)
    require(out["valid_bracket"] is True, "expected valid bracket")
    require(abs(out["I_P"] - 1.63e12) < 1.0, "I_P")
    require(abs(out["I_F"] - 2.5e12) < 1.0, "I_F")


def test_non_monotonic_is_not_a_bracket() -> None:
    rows = [
        {"cep_family": "GH7c", "I_Wcm2": 1.0e12, "overall": "PASS"},
        {"cep_family": "GH7c", "I_Wcm2": 1.63e12, "overall": "FAIL"},
        {"cep_family": "GH7c", "I_Wcm2": 1.90e12, "overall": "PASS"},
    ]
    out = bracket_for_family(rows, "GH7c", None)
    require(out["valid_bracket"] is False, "non-monotonic must not bracket")
    require(out["reason"] == "non_monotonic_PASS_FAIL", out["reason"])


def test_single_antiphase_point_is_not_a_bracket() -> None:
    rows = [{"cep_family": "GH5th0", "I_Wcm2": 1.63e12, "overall": "PASS"}]
    out = bracket_for_family(rows, "GH5th0", None)
    require(out["valid_bracket"] is False, "single point is not a bracket")


def main() -> int:
    test_valid_bracket()
    test_non_monotonic_is_not_a_bracket()
    test_single_antiphase_point_is_not_a_bracket()
    print("PASS test_audit_isafe_occ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
