#!/usr/bin/env python3
"""Unit tests for CEP-π V2 gate (Jones + mixed-tolerance)."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_full112_gh3_candidates import pair_nodes_by_neg_alpha
from cep_pi_gate_v2 import (
    CEP_V2_ABS_ATOL,
    CEP_V2_ETA,
    CEP_V2_REL_ATOL,
    evaluate_pair_harmonic,
    phase_equals_plus_pi,
    scale_A_n,
    validate_antipode_phases,
)
from compare_chunked_vs_full_v3 import apply_unitary_to_modes


def test_phase_0_and_2pi_equivalent() -> None:
    assert phase_equals_plus_pi(0.0, math.pi)
    assert phase_equals_plus_pi(0.0, math.pi + 2.0 * math.pi)
    assert phase_equals_plus_pi(2.0 * math.pi, math.pi)
    assert not phase_equals_plus_pi(0.0, 0.0)


def test_self_alpha0_pairing() -> None:
    man = [
        {"id": 1, "weight": 0.25, "re": 1.0, "im": 0.0, "intensity": 1.0, "phase": 0.0},
        {"id": 2, "weight": 0.25, "re": -1.0, "im": 0.0, "intensity": 1.0, "phase": math.pi},
        {"id": 5, "weight": 0.5, "re": 0.0, "im": 0.0, "intensity": 1.0, "phase": 0.0},
    ]
    pairs = pair_nodes_by_neg_alpha(man)
    kinds = {p["kind"] for p in pairs}
    assert "self_alpha0" in kinds
    assert "neg_alpha" in kinds
    self_p = next(p for p in pairs if p["kind"] == "self_alpha0")
    assert self_p["id_alpha"] == self_p["id_neg"] == 5
    errs = validate_antipode_phases(man, pairs)
    assert errs == []


def test_missing_antipode_fails() -> None:
    man = [
        {"id": 1, "weight": 0.5, "re": 1.0, "im": 0.0, "intensity": 1.0, "phase": 0.0},
        {"id": 2, "weight": 0.5, "re": 0.5, "im": 0.0, "intensity": 1.0, "phase": 0.0},
    ]
    try:
        pair_nodes_by_neg_alpha(man)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_weight_mismatch_fails() -> None:
    man = [
        {"id": 1, "weight": 0.6, "re": 1.0, "im": 0.0, "intensity": 1.0, "phase": 0.0},
        {"id": 2, "weight": 0.4, "re": -1.0, "im": 0.0, "intensity": 1.0, "phase": math.pi},
    ]
    try:
        pair_nodes_by_neg_alpha(man)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "weight" in str(exc).lower()


def test_weak_center_abs_pass() -> None:
    A = 1.0e-2
    # Nearly canceling weak center: d tiny
    r = evaluate_pair_harmonic(
        jx_p=1e-15,
        jy_p=0j,
        jx_m=-1e-15 + 1e-17,
        jy_m=0j,
        A_n=A,
        abs_atol=CEP_V2_ABS_ATOL,
        rel_atol=CEP_V2_REL_ATOL,
        eta=CEP_V2_ETA,
    )
    assert not r["strong"]
    assert r["label"] == "BELOW_SIGNAL_FLOOR"
    assert r["pass"]


def test_weak_center_abs_fail() -> None:
    A = 1.0e-2
    r = evaluate_pair_harmonic(
        jx_p=1e-15,
        jy_p=0j,
        jx_m=5e-14,  # does not cancel; d ~ 5e-14; e_abs ~ 5e-12 > 1e-12
        jy_m=0j,
        A_n=A,
        abs_atol=CEP_V2_ABS_ATOL,
        rel_atol=CEP_V2_REL_ATOL,
        eta=CEP_V2_ETA,
    )
    assert not r["strong"]
    assert r["abs_fail"]
    assert not r["pass"]


def test_strong_rel_fail() -> None:
    A = 1.0e-2
    j = 1.0e-2
    # Perfect cancel would be -j; inject relative residual ~1e-2
    r = evaluate_pair_harmonic(
        jx_p=j,
        jy_p=0j,
        jx_m=-j * (1.0 - 2.0e-2),
        jy_m=0j,
        A_n=A,
        abs_atol=CEP_V2_ABS_ATOL,
        rel_atol=CEP_V2_REL_ATOL,
        eta=CEP_V2_ETA,
    )
    assert r["strong"]
    assert r["rel_fail"]
    assert not r["pass"]


def test_strong_abs_fail() -> None:
    A = 1.0e-2
    j = 1.0e-2
    # Large absolute residual
    r = evaluate_pair_harmonic(
        jx_p=j,
        jy_p=0j,
        jx_m=0j,  # d = |j| = 1e-2; e_abs = 1
        jy_m=0j,
        A_n=A,
        abs_atol=CEP_V2_ABS_ATOL,
        rel_atol=CEP_V2_REL_ATOL,
        eta=CEP_V2_ETA,
    )
    assert r["abs_fail"]
    assert not r["pass"]


def test_unitary_rotation_invariance_of_residual() -> None:
    from compare_chunked_vs_full_v3 import jones_norm

    jxp, jyp = 1.0e-2 + 2e-3j, 3e-3 - 1e-3j
    jxm, jym = -jxp + 1e-16, -jyp
    A = max(jones_norm(jxp, jyp), jones_norm(jxm, jym))
    base = evaluate_pair_harmonic(
        jx_p=jxp,
        jy_p=jyp,
        jx_m=jxm,
        jy_m=jym,
        A_n=A,
        abs_atol=CEP_V2_ABS_ATOL,
        rel_atol=CEP_V2_REL_ATOL,
        eta=CEP_V2_ETA,
    )
    for theta in (0.3, math.pi / 4, 1.1):
        c, s = math.cos(theta), math.sin(theta)
        modes = {
            (1, 2): {"jx": jxp, "jy": jyp, "weight": 1.0, "power": 0.0},
            (2, 2): {"jx": jxm, "jy": jym, "weight": 1.0, "power": 0.0},
        }
        rot = apply_unitary_to_modes(modes, c, -s, s, c)
        # Recompute A after rotation (should match within roundoff)
        A_r = max(
            jones_norm(rot[(1, 2)]["jx"], rot[(1, 2)]["jy"]),
            jones_norm(rot[(2, 2)]["jx"], rot[(2, 2)]["jy"]),
        )
        r = evaluate_pair_harmonic(
            jx_p=rot[(1, 2)]["jx"],
            jy_p=rot[(1, 2)]["jy"],
            jx_m=rot[(2, 2)]["jx"],
            jy_m=rot[(2, 2)]["jy"],
            A_n=A_r,
            abs_atol=CEP_V2_ABS_ATOL,
            rel_atol=CEP_V2_REL_ATOL,
            eta=CEP_V2_ETA,
        )
        assert r["pass"] == base["pass"], (theta, r, base)
        assert abs(r["e_abs"] - base["e_abs"]) <= 1.0e-14


def test_both_directions_concept() -> None:
    """A and B directions can differ; both must be evaluated by audit."""
    # Construct asymmetric: A cancels, B does not
    ja = 1.0e-2
    # Dir A: +N(α)=ja, -N(-α)=-ja → cancel
    # Dir B: +N(-α)=0, -N(α)=-ja → residual |ja|
    r_a = evaluate_pair_harmonic(
        jx_p=ja, jy_p=0j, jx_m=-ja, jy_m=0j,
        A_n=ja, abs_atol=CEP_V2_ABS_ATOL, rel_atol=CEP_V2_REL_ATOL, eta=CEP_V2_ETA,
    )
    r_b = evaluate_pair_harmonic(
        jx_p=0j, jy_p=0j, jx_m=-ja, jy_m=0j,
        A_n=ja, abs_atol=CEP_V2_ABS_ATOL, rel_atol=CEP_V2_REL_ATOL, eta=CEP_V2_ETA,
    )
    assert r_a["pass"]
    assert not r_b["pass"]


def main() -> int:
    tests = [
        test_phase_0_and_2pi_equivalent,
        test_self_alpha0_pairing,
        test_missing_antipode_fails,
        test_weight_mismatch_fails,
        test_weak_center_abs_pass,
        test_weak_center_abs_fail,
        test_strong_rel_fail,
        test_strong_abs_fail,
        test_unitary_rotation_invariance_of_residual,
        test_both_directions_concept,
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
