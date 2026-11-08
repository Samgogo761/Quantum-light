"""Targeted v4 tests for the statistics interface fixes (no TDSE required)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import analyze_statistics_v2 as stats
from ql_tip_1d.statistics_v2 import linear_weights, tail_bound

AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
CATEGORIES = stats.ERROR_CATEGORIES


def test_error_map_counts_each_category_exactly_once() -> None:
    """No numerical error class may be charged twice in one region."""
    with (AUDIT / "error_component_map.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for F in ["0", "1"]:
        for region in ["low", "mid", "high"]:
            components = [row["component"] for row in rows
                          if row["F"] == F and row["region"] == region]
            assert sorted(components) == sorted(CATEGORIES)
            assert len(components) == len(CATEGORIES)


def test_high_intensity_envelope_does_not_add_a_full_r1_total() -> None:
    """The 32 mu envelope = seven direct R2 controls + four R1-only categories."""
    active = json.loads((PROJECT / stats.ACTIVE_EVIDENCE.relative_to(PROJECT)).read_text(encoding="utf-8"))
    r1 = json.loads((PROJECT / active["r1_gate"]).read_text(encoding="utf-8"))
    r2 = json.loads((PROJECT / active["r2_gate"]).read_text(encoding="utf-8"))
    for F in [0.0, 1.0]:
        label = f"F{F:g}"
        direct = sum(abs(float(r2["controls"][f"{label}_{key}"]["relative_change"]))
                     for key in ["dx048", "dt005", "box900", "caphalf"])
        direct += sum(abs(float(r2[name][label]))
                      for name in ["eta32_relative", "grid32_relative", "range32_relative"])
        r1_only = sum(max(float(r1["components"][f"{label}_I1"]["components"][name]),
                          float(r1["components"][f"{label}_I4"]["components"][name]))
                      for name in ["preparation", "mask", "time"])
        # External review D: high-intensity CEP uses the saved 8/12/20 mu
        # envelope, so the expected value must be read from the audited map
        # instead of hard-coding the old R1 4 mu CEP.
        cep_row = next(row for row in stats.error_map_rows(F, "high")
                       if row["component"] == "cep")
        r1_only += abs(float(cep_row["relative_estimate"]))
        value, origin = stats.response_error_envelope(F, 32.0)
        assert value == pytest.approx(direct + r1_only, rel=1e-12)
        assert "per_category_once" in origin
        # No full R1 total may be added on top: the audited envelope must stay
        # strictly below "seven R2 categories + one R1 total + CEP".  The CEP
        # envelope itself can legitimately exceed the R1 total (measured 20 mu
        # 2->4 change is ~7%), so the check must not assume CEP < R1 total.
        wrong_double = (direct
                        + max(float(r1["components"][f"{label}_I1"]["total"]),
                              float(r1["components"][f"{label}_I4"]["total"]))
                        + abs(float(cep_row["relative_estimate"])))
        assert value < wrong_double


def test_tail_uses_positive_lower_bound_and_total_5pct_includes_tail() -> None:
    """A tail judged only against the centre value, or excluded from the 5%, must fail."""
    # 0.0095 <= 1% of the centre (0.01) but > 1% of the lower bound (0.008).
    gate = stats.evaluate_yield_gate(1.0, 0.20, 0.0095, 0.0, 0.0)
    assert gate["Y_lo"] == pytest.approx(0.80)
    assert gate["lower_positive"] is True
    assert gate["tail_ok"] is False
    # tail must be part of the conservative 5% total, not only a separate check.
    gate2 = stats.evaluate_yield_gate(1.0, 0.040, 0.020, 0.0, 0.0)
    assert gate2["total_relative_error"] == pytest.approx(0.060)
    assert gate2["total_5pct_ok"] is False
    assert gate2["status"] == "INCOMPLETE"
    ok = stats.evaluate_yield_gate(1.0, 0.01, 0.001, 0.0001, 0.001)
    assert ok["status"] == "PASS"


def test_interval_containing_one_is_a_valid_delivery() -> None:
    """D≈1 with reliable errors is VALID_UNRESOLVED, not a failure."""
    assert stats.classify_result(True, 0.95, 1.05) == ("PASS", "VALID_UNRESOLVED")
    assert stats.classify_result(True, 1.02, 1.30) == ("PASS", "VALID_DIFFERENCE")
    assert stats.classify_result(False, 1.02, 1.30) == ("INCOMPLETE", "INCOMPLETE")
    assert stats.classify_result(True, 0.9, 1.1, preliminary=True) == ("PRELIMINARY", "PRELIMINARY")
    assert stats.final_validity(True, True, ["phase_validation.json"], 0.9, False) is False
    assert stats.final_validity(True, True, [], 0.0, False) is False


def test_linear_weights_accept_truncation_at_last_node() -> None:
    """Weights must integrate to 1-tail_bound when Imax equals the last node."""
    nodes = stats.INITIAL_NODES
    for state in ["bsv", "thermal"]:
        weights = linear_weights(nodes, state, nodes[-1])
        assert all(value >= 0.0 for value in weights)
        assert sum(weights) == pytest.approx(1.0 - tail_bound(nodes[-1], state), rel=1e-8, abs=1e-10)


def test_missing_validation_artifacts_are_reported() -> None:
    """A missing phase or tail audit can never be silently treated as zero."""
    validation, missing = stats.validation_artifacts()
    assert isinstance(validation, dict)
    assert "phase" in validation and "tail" in validation
    if not (AUDIT / "phase_validation.json").exists():
        assert any("phase_validation" in item for item in missing)
