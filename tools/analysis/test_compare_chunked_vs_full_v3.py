#!/usr/bin/env python3
"""Frozen-norm unit tests for Jones-vector V3 gate.

Requirements (must hold before trusting V3 on production C8/U8):
  1. Unitary x/y polarization-basis rotation → identical PASS/FAIL verdict
  2. Amplitude error on a strong Jones vector → FAIL
  3. Global phase error → FAIL
  4. Polarization / ellipticity error → FAIL
  5. Weak orthogonal-component change with vector abs under gate → PASS
"""
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

# Allow running from repo root or tools/analysis/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_chunked_vs_full_v3 import (
    V3_ABS_ATOL,
    V3_REL_ATOL,
    apply_unitary_to_modes,
    evaluate_modes_pair,
    jones_norm,
    verdict_pass,
)


def _make_modes(
    vectors: dict[tuple[int, int], tuple[complex, complex]],
    *,
    weight: float = 1.0,
) -> dict[tuple[int, int], dict]:
    out = {}
    for key, (jx, jy) in vectors.items():
        out[key] = {
            "weight": weight,
            "jx": jx,
            "jy": jy,
            "power": abs(jx) ** 2 + abs(jy) ** 2,
        }
    return out


def _unitary_rot(theta: float) -> tuple[complex, complex, complex, complex]:
    c, s = math.cos(theta), math.sin(theta)
    return c + 0j, -s + 0j, s + 0j, c + 0j


def test_unitary_rotation_invariance() -> None:
    # Strong elliptical Jones + one weaker node
    ref = _make_modes(
        {
            (1, 2): (1.0e-2 + 2.0e-3j, 3.0e-3 - 1.0e-3j),
            (2, 2): (1.0e-3, 2.0e-3j),
            (1, 5): (2.0e-2, 1.0e-2j),
            (2, 5): (5.0e-4 - 1.0e-4j, 2.0e-4),
        }
    )
    # Tiny numerical noise well under gate
    chk = _make_modes(
        {
            (1, 2): (1.0e-2 + 2.0e-3j + 1.0e-16, 3.0e-3 - 1.0e-3j),
            (2, 2): (1.0e-3 + 1.0e-17j, 2.0e-3j),
            (1, 5): (2.0e-2, 1.0e-2j + 1.0e-16),
            (2, 5): (5.0e-4 - 1.0e-4j, 2.0e-4),
        }
    )
    base = evaluate_modes_pair(chk, ref, harmonics=(2, 5))
    assert verdict_pass(base), base

    for theta in (0.0, 0.3, math.pi / 4, 1.7, math.pi):
        u = _unitary_rot(theta)
        chk_r = apply_unitary_to_modes(chk, *u)
        ref_r = apply_unitary_to_modes(ref, *u)
        rot = evaluate_modes_pair(chk_r, ref_r, harmonics=(2, 5))
        assert verdict_pass(rot) == verdict_pass(base), (theta, rot, base)
        # Both sides remain under the frozen gate (verdict invariance is the requirement).
        assert rot["jones_modes"]["max_e_abs_vec"] <= V3_ABS_ATOL
        assert rot["jones_modes"]["max_e_rel_vec"] <= V3_REL_ATOL


def test_amplitude_error_fails() -> None:
    ref = _make_modes({(1, 2): (1.0e-2, 0.0j), (2, 2): (1.0e-3, 0.0j)})
    # Relative amplitude +1e-8 on strong vector >> 1e-10 / 1e-12 gates
    bad = _make_modes({(1, 2): (1.0e-2 * (1.0 + 1.0e-8), 0.0j), (2, 2): (1.0e-3, 0.0j)})
    rep = evaluate_modes_pair(bad, ref, harmonics=(2,))
    assert not verdict_pass(rep), rep
    assert rep["jones_modes"]["n_vec_rel_fail"] >= 1 or rep["jones_modes"]["n_vec_abs_fail"] >= 1


def test_global_phase_error_fails() -> None:
    jx, jy = 1.0e-2 + 2.0e-3j, 4.0e-3 - 1.0e-3j
    ref = _make_modes({(1, 2): (jx, jy), (2, 2): (1.0e-3, 2.0e-4j)})
    phase = cmath.exp(1.0e-8j)  # small but exceeds 1e-10 relative on strong vector
    bad = _make_modes(
        {
            (1, 2): (jx * phase, jy * phase),
            (2, 2): (1.0e-3, 2.0e-4j),
        }
    )
    rep = evaluate_modes_pair(bad, ref, harmonics=(2,))
    # d ≈ |e^{iφ}-1| * ||J|| ≈ |φ| * ||J|| for small φ
    assert not verdict_pass(rep), rep


def test_polarization_ellipticity_error_fails() -> None:
    # Linear-x reference; inject y component that changes polarization
    ref = _make_modes({(1, 2): (1.0e-2, 0.0j), (2, 2): (2.0e-3, 0.0j)})
    # Rotate by angle whose sin(θ)*||J|| / ||J|| = sin(θ) > 1e-10
    theta = 1.0e-8
    c, s = math.cos(theta), math.sin(theta)
    bad = _make_modes(
        {
            (1, 2): (1.0e-2 * c, 1.0e-2 * s),
            (2, 2): (2.0e-3, 0.0j),
        }
    )
    rep = evaluate_modes_pair(bad, ref, harmonics=(2,))
    assert not verdict_pass(rep), rep


def test_weak_orthogonal_under_abs_passes() -> None:
    """Weak orthogonal change: scary per-component rel, vector gate PASS."""
    A_scale = 1.0e-2
    # Node1 sets A_n; node2 has moderate ||J|| mostly in x, near-zero y.
    ref = _make_modes(
        {
            (1, 2): (A_scale, 0.0j),
            (2, 2): (1.0e-3, 1.0e-25),
        }
    )
    d_weak = 1.0e-17  # e_abs_vec = 1e-15; e_rel_vec = 1e-14; |ΔJy|/A = 1e-15
    chk = _make_modes(
        {
            (1, 2): (A_scale, 0.0j),
            (2, 2): (1.0e-3, 1.0e-25 + d_weak),
        }
    )
    rep = evaluate_modes_pair(chk, ref, harmonics=(2,))
    assert verdict_pass(rep), rep
    # Pure per-component relative on y (no floor) is huge — V2-style false alarm.
    pure_comp_rel = d_weak / 1.0e-25
    assert pure_comp_rel > 1.0e6


def test_node4_like_weak_jy_passes_vector_gate() -> None:
    """Reproduce V2 near-miss geometry: tiny ΔJy / small Jy, tiny e_abs_vec."""
    # A_n ~ 1.6e-3, |Jy|~5e-9, e_rel_comp~1.8e-10, e_abs_comp~5e-16
    A = 1.6605082011343978e-3
    jy = 4.993370889178376e-9
    jx = math.sqrt(max(A * A - jy * jy, 0.0))  # strong mostly-x Jones
    e_rel_comp = 1.8708431797659455e-10
    djy = e_rel_comp * jy  # ~9e-19
    ref = _make_modes({(4, 9): (jx, jy), (1, 9): (A, 0.0j)})
    chk = _make_modes({(4, 9): (jx, jy + djy), (1, 9): (A, 0.0j)})
    rep = evaluate_modes_pair(chk, ref, harmonics=(9,))
    assert verdict_pass(rep), rep
    assert rep["jones_modes"]["max_e_abs_vec"] < V3_ABS_ATOL
    assert rep["jones_modes"]["max_e_rel_vec"] < V3_REL_ATOL


def main() -> int:
    tests = [
        test_unitary_rotation_invariance,
        test_amplitude_error_fails,
        test_global_phase_error_fails,
        test_polarization_ellipticity_error_fails,
        test_weak_orthogonal_under_abs_passes,
        test_node4_like_weak_jy_passes_vector_gate,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 — report each test
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        print(f"SUMMARY FAIL ({failed}/{len(tests)})")
        return 1
    print(f"SUMMARY PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
