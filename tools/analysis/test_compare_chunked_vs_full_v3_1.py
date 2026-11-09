#!/usr/bin/env python3
"""Unit tests for Jones-vector V3.1 mixed-tolerance logic fix.

Required cases:
  1. Weak + abs PASS  → BELOW_SIGNAL_FLOOR / PASS
  2. Weak + abs FAIL  → FAIL
  3. Strong rel FAIL  → FAIL
  4. Strong abs FAIL  → FAIL
  5. Unitary x/y rotation → verdict unchanged
  6. node5-like near-zero cannot evade absolute gate by being weak
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_chunked_vs_full_v3 import apply_unitary_to_modes, jones_norm
from compare_chunked_vs_full_v3_1 import (
    V31_ABS_ATOL,
    V31_ETA,
    evaluate_jones_modes_v31,
    evaluate_modes_pair_v31,
    verdict_pass,
)


def _modes(vectors, *, weight: float = 1.0):
    out = {}
    for key, (jx, jy) in vectors.items():
        out[key] = {
            "weight": weight,
            "jx": jx,
            "jy": jy,
            "power": abs(jx) ** 2 + abs(jy) ** 2,
        }
    return out


def _rot(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return c + 0j, -s + 0j, s + 0j, c + 0j


def test_weak_abs_pass_below_floor() -> None:
    A = 1.0e-2
    ref = _modes({(1, 2): (A, 0j), (5, 2): (0j, 1.0e-20)})  # weak node5-like
    # tiny d: e_abs = 1e-17/A = 1e-15 < 1e-12
    chk = _modes({(1, 2): (A, 0j), (5, 2): (0j, 1.0e-20 + 1.0e-17)})
    j = evaluate_jones_modes_v31(chk, ref, harmonics=(2,))
    assert j["status"] == "PASS", j
    assert j["n_below_signal_floor"] >= 1
    assert j["n_vec_rel_fail"] == 0
    weak = [r for r in j["below_signal_floor_examples"] if r["node"] == 5]
    assert weak and weak[0]["label"] == "BELOW_SIGNAL_FLOOR"


def test_weak_abs_fail() -> None:
    A = 1.0e-2
    ref = _modes({(1, 2): (A, 0j), (5, 2): (0j, 1.0e-20)})
    # d = 2e-13 → e_abs = 2e-11 > 1e-12 even though weak
    chk = _modes({(1, 2): (A, 0j), (5, 2): (0j, 1.0e-20 + 2.0e-13)})
    j = evaluate_jones_modes_v31(chk, ref, harmonics=(2,))
    assert j["status"] == "FAIL", j
    assert j["n_vec_abs_fail"] >= 1
    assert any(f["node"] == 5 for f in j["vec_abs_fail"])


def test_strong_rel_fail() -> None:
    ref = _modes({(1, 2): (1.0e-2, 0j), (2, 2): (1.0e-3, 0j)})
    bad = _modes({(1, 2): (1.0e-2 * (1.0 + 1.0e-8), 0j), (2, 2): (1.0e-3, 0j)})
    j = evaluate_jones_modes_v31(bad, ref, harmonics=(2,))
    assert j["status"] == "FAIL", j
    assert j["n_vec_rel_fail"] >= 1 or j["n_vec_abs_fail"] >= 1


def test_strong_abs_fail() -> None:
    ref = _modes({(1, 2): (1.0e-2, 0j)})
    # Large absolute change
    bad = _modes({(1, 2): (1.0e-2 + 2.0e-13, 0j)})  # e_abs = 2e-11 > 1e-12
    j = evaluate_jones_modes_v31(bad, ref, harmonics=(2,))
    assert j["status"] == "FAIL", j
    assert j["n_vec_abs_fail"] >= 1


def test_unitary_rotation_invariance() -> None:
    ref = _modes(
        {
            (1, 2): (1.0e-2 + 2e-3j, 3e-3 - 1e-3j),
            (5, 2): (0j, 1.0e-20),
            (1, 5): (2.0e-2, 1.0e-2j),
            (5, 5): (1.0e-21, 0j),
        }
    )
    chk = _modes(
        {
            (1, 2): (1.0e-2 + 2e-3j + 1e-16, 3e-3 - 1e-3j),
            (5, 2): (0j, 1.0e-20 + 1e-17),
            (1, 5): (2.0e-2, 1.0e-2j + 1e-16),
            (5, 5): (1.0e-21, 0j),
        }
    )
    base = evaluate_modes_pair_v31(chk, ref, harmonics=(2, 5))
    assert verdict_pass(base), base
    for theta in (0.0, 0.3, math.pi / 4, 1.7, math.pi):
        u = _rot(theta)
        rot = evaluate_modes_pair_v31(
            apply_unitary_to_modes(chk, *u),
            apply_unitary_to_modes(ref, *u),
            harmonics=(2, 5),
        )
        assert verdict_pass(rot) == verdict_pass(base), (theta, rot)


def test_node5_like_cannot_evade_abs() -> None:
    """Being BELOW_SIGNAL_FLOOR does not waive absolute gate."""
    A = 4.0e-3
    ref = _modes(
        {
            (1, 9): (A, 0j),
            (5, 9): (1.0e-15, 1.0e-15),  # ||J|| << η A
        }
    )
    assert jones_is_weak(ref, 5, 9, A)
    # Inject absolute error above τ_abs while remaining weak
    d = 5.0e-14  # e_abs = d/A = 1.25e-11 > 1e-12
    chk = _modes(
        {
            (1, 9): (A, 0j),
            (5, 9): (1.0e-15 + d, 1.0e-15),
        }
    )
    j = evaluate_jones_modes_v31(chk, ref, harmonics=(9,))
    assert j["status"] == "FAIL", j
    assert j["n_below_signal_floor"] >= 1
    assert j["n_vec_abs_fail"] >= 1


def jones_is_weak(modes, nid, h, A_hint=None) -> bool:
    j = evaluate_jones_modes_v31(modes, modes, harmonics=(h,))
    # Reconstruct classification
    from compare_chunked_vs_full_v3 import jones_norm

    n = jones_norm(modes[(nid, h)]["jx"], modes[(nid, h)]["jy"])
    A = A_hint if A_hint is not None else n
    # Better: use scales from same-harmonic max
    A = max(
        jones_norm(modes[k]["jx"], modes[k]["jy"])
        for k in modes
        if k[1] == h
    )
    return n < V31_ETA * A


def test_node4_like_strongish_comp_rel_passes_vector() -> None:
    """V2 near-miss geometry: tiny ΔJy; V3.1 vector gates PASS."""
    A = 1.6605082011343978e-3
    jy = 4.993370889178376e-9
    jx = math.sqrt(max(A * A - jy * jy, 0.0))
    djy = 1.8708431797659455e-10 * jy
    ref = _modes({(4, 9): (jx, jy), (1, 9): (A, 0j)})
    chk = _modes({(4, 9): (jx, jy + djy), (1, 9): (A, 0j)})
    j = evaluate_jones_modes_v31(chk, ref, harmonics=(9,))
    assert j["status"] == "PASS", j
    assert j["max_e_abs_vec"] < V31_ABS_ATOL


def main() -> int:
    tests = [
        test_weak_abs_pass_below_floor,
        test_weak_abs_fail,
        test_strong_rel_fail,
        test_strong_abs_fail,
        test_unitary_rotation_invariance,
        test_node5_like_cannot_evade_abs,
        test_node4_like_strongish_comp_rel_passes_vector,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        print(f"SUMMARY FAIL ({failed}/{len(tests)})")
        return 1
    print(f"SUMMARY PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
