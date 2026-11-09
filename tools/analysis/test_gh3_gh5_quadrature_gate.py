#!/usr/bin/env python3
"""Offline unit tests for the frozen GH3↔GH5 quadrature thresholds."""
from __future__ import annotations

from gh3_gh5_quadrature_gate import (
    ICS_CS_RTOL,
    JE_RTOL,
    JO_RTOL,
    PHASE_ATOL_RAD,
    jones_phase_diff,
    mixed_rel,
)


def test_mixed_rel_strong_pass_fail() -> None:
    scale = 1.0e-3
    ok = mixed_rel(0.05 * 8.0e-4, 8.0e-4, scale, JO_RTOL, 1.0e-4)
    assert ok["strong"] and ok["pass"]
    bad = mixed_rel(0.20 * 8.0e-4, 8.0e-4, scale, JO_RTOL, 1.0e-4)
    assert bad["strong"] and not bad["pass"]


def test_mixed_rel_floor_uses_abs() -> None:
    scale = 1.0e-3
    tiny = 1.0e-10
    ok = mixed_rel(1.0e-8, tiny, scale, JE_RTOL, 1.0e-4)
    assert not ok["strong"] and ok["pass"]
    bad = mixed_rel(2.0e-4, tiny, scale, JE_RTOL, 1.0e-4)
    assert not bad["strong"] and not bad["pass"]


def test_phase_and_thresholds() -> None:
    assert abs(ICS_CS_RTOL - 0.05) < 1e-15
    assert abs(JO_RTOL - 0.10) < 1e-15
    assert abs(PHASE_ATOL_RAD - 0.1) < 1e-15
    dphi = jones_phase_diff(1 + 0j, 0j, 1 * 1j, 0j)
    assert abs(dphi - 0.5 * 3.141592653589793) < 1e-12
    aligned = jones_phase_diff(1 + 0j, 0.2j, 1.01 + 0j, 0.202j)
    assert aligned < 0.05


if __name__ == "__main__":
    test_mixed_rel_strong_pass_fail()
    test_mixed_rel_floor_uses_abs()
    test_phase_and_thresholds()
    print("test_gh3_gh5_quadrature_gate: PASS")
