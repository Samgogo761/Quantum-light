"""Targeted success/failure tests for the 2026-11-07 external-review fixes.

These cover the scientific branches the 30 pre-existing smoke tests do not:
mismatched R1 references, readout preflight failures, the real S4 gate,
single-channel phase accounting, stale tail bindings, event timeouts and
resume invalidation.  No TDSE, spectrum or project state is modified.
"""
from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import analyze_statistics_v2 as stats
import build_error_component_map_v4 as emap
import run_v3_full_v4 as orch
import v3_stage_v4 as stage

R1_V5 = PROJECT / "results/bias_v3_audit/r1_20261107_v5"


def test_r1_v5_controls_are_paired_with_their_own_reference() -> None:
    """Review A: no cross-condition reference may be used as a pair."""
    with (R1_V5 / "R1_components_v5.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row["missing_reason"] == ""
        if row["coverage"] == "not_applicable_DC_zero":
            continue
        assert row["reference_Y"] and row["control_Y"], row
        assert row["signed_change"] and row["state_config_summary_source_SHA"], row
        if row["component"] in ("box", "cap"):
            ref = row["reference_readout_ids"].split("_F")[1][:5]
            ctrl = row["control_readout_ids"].split("_F")[1][:5]
            assert ref == ctrl, f"cross-condition pair: {row}"
    gate = json.loads((R1_V5 / "R1_gate_r1_20261107_v5.json").read_text(encoding="utf-8"))
    assert gate["status"] == "PASS" and gate["missing"] == []
    assert gate["numerical_evidence_complete"] is True
    assert "own matched reference" in gate["pairing_rule"]


def test_readout_preflight_failure_is_never_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review 4.3: a failed or unverifiable preflight must raise."""
    failed = SimpleNamespace(returncode=1, stdout="", stderr="preflight exploded")
    monkeypatch.setattr(orch, "subprocess", SimpleNamespace(
        run=lambda *a, **k: failed, TimeoutExpired=subprocess.TimeoutExpired))
    with pytest.raises(RuntimeError, match="preflight failed"):
        orch.pending_readouts()
    empty = SimpleNamespace(returncode=0, stdout="{}", stderr="")
    monkeypatch.setattr(orch, "subprocess", SimpleNamespace(
        run=lambda *a, **k: empty, TimeoutExpired=subprocess.TimeoutExpired))
    with pytest.raises(RuntimeError, match="no verifiable output"):
        orch.pending_readouts()


def test_s4_gate_other_than_proceed_blocks_s5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review B: computing dx/dt is not a pass; S5 must be blocked."""
    monkeypatch.setattr(orch, "call", lambda label, args, **kw: 0)
    monkeypatch.setattr(orch, "done", lambda run_id: True)
    monkeypatch.setattr(orch, "drain_readouts", lambda label: None)
    monkeypatch.setattr(orch, "checkpoint", lambda step: None)
    monkeypatch.setattr(stage, "jload", lambda path: (
        {"J_over_mu": 8.0} if "s4_selection" in str(path)
        else {"status": "LIMITED_REPAIR", "blockers": ["unstable control"]}
        if "s4_gate" in str(path) else {}))
    with pytest.raises(RuntimeError, match="S5 must not start"):
        orch.do_s4()


def test_s4_selection_does_not_rank_by_response_difference() -> None:
    """Review C: |q-1| must not be added to a weighted contribution."""
    source = inspect.getsource(stage.step_contributions)
    assert "0.05 * bias" not in source
    assert "min_channel_share" in source
    assert "channel_shares" in source


def test_phase_error_has_a_single_channel_and_direct_evidence(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Review D: no outer phase delta, and direct 4->8 keys must match F labels."""
    source = inspect.getsource(stats.final_statistics)
    assert "delta = y * rel_err + quad_err + phase_change" not in source
    assert "delta = y * rel_err + quad_err" in source
    assert "phase_change = 0.0" in source
    validation = tmp_path / "phase_validation.json"
    validation.write_text(json.dumps({"rows": [
        {"status": "PASS", "F": 1.0, "I_over_mu": 32.0,
         "relative_4to8_change": 0.013}]}), encoding="utf-8")
    monkeypatch.setattr(emap, "PHASE_VALIDATION", validation)
    _envelope, direct, _missing = emap.phase_envelopes()
    assert "1" in direct and direct["1"] == pytest.approx(0.013)
    assert "UNRESOLVED_phase_coverage" in inspect.getsource(emap)


def test_stale_tail_binding_is_rejected(monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    """Review 4.7: complete=true with a stale input SHA is not accepted."""
    tail = tmp_path / "tail_audit.json"
    tail.write_text(json.dumps({
        "complete": True,
        "input_binding": {"library_sha256": "deadbeef",
                          "representatives": {"phase_conditions": [[1.0, 32.0]]}}}),
        encoding="utf-8")
    phase = tmp_path / "phase_validation.json"
    phase.write_text(json.dumps({
        "complete": True,
        "input_binding": {"library_sha256": "cafebabe",
                          "representatives": {"phase_conditions": [[1.0, 32.0]]}}}),
        encoding="utf-8")
    monkeypatch.setattr(stats, "TAIL_AUDIT", tail)
    monkeypatch.setattr(stats, "PHASE_VALIDATION", phase)
    monkeypatch.setattr(stats, "library_identity",
                        lambda response=None: {"status": "PASS", "library_sha256": "cafebabe"})
    monkeypatch.setattr(stats, "expected_representatives",
                        lambda: {"phase_conditions": [[1.0, 32.0]],
                                 "midpoints": [2.0, 6.0], "tail_node_I_over_mu": 32.0})
    _validation, missing = stats.validation_artifacts()
    assert any("stale" in item and "tail_audit" in item for item in missing)
    assert not any("stale" in item and "phase_validation" in item for item in missing)


def test_tail_precheck_never_certifies_the_final_gate(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    """Review E: precheck writes its own file and cannot set complete=true."""
    snapshot = {"status": "SUFFICIENT", "complete": True, "Imax_over_mu": 32.0,
                "rows": [{"yield_key": "bsv_F0", "F": 0.0, "state": "bsv",
                          "Imax_over_mu": 32.0, "Y": 0.02, "delta": 0.001,
                          "positive_lower_Y": 0.019, "tail_upper": 1.5e-8,
                          "one_percent_of_lower": 1.9e-4, "tail_ok": True,
                          "total_relative_error": 0.05}],
                "input_binding": {"status": "PASS"}, "representatives": {}}
    monkeypatch.setattr(stage.stats, "tail_snapshot", lambda include_midpoints=True: snapshot)
    monkeypatch.setattr(stage, "AUDIT", tmp_path)
    monkeypatch.setattr(stage, "tail_extension_plan",
                        lambda value: {"status": "NOT_REQUIRED"})
    assert stage.step_tail("precheck") == 2
    precheck = json.loads((tmp_path / "tail_precheck.json").read_text(encoding="utf-8"))
    assert precheck["complete"] is False
    assert not (tmp_path / "tail_audit.json").exists()


def test_event_timeout_is_triggered_and_recorded(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    """Review 4.1: an outer event timeout is raised and its real time is logged."""
    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="fake", timeout=1.0)

    monkeypatch.setattr(orch, "subprocess", SimpleNamespace(
        run=raise_timeout, TimeoutExpired=subprocess.TimeoutExpired,
        STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(orch, "LOG", tmp_path / "orchestrator.log")
    monkeypatch.setattr(orch, "AUDIT", tmp_path)
    monkeypatch.setattr(orch, "V2_RESULTS", tmp_path)
    monkeypatch.setattr(stage, "jload", lambda path: {"budget": {"event_hard_timeout_s": 5400}})
    with pytest.raises(RuntimeError, match="outer limit"):
        orch.call("fake event", ["fake.py"])
    journal = tmp_path / "v3_event_timeouts.jsonl"
    assert journal.exists()
    entry = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["status"] == "TIMEOUT" and entry["elapsed_seconds"] >= 0.0
    with (tmp_path / "compute_ledger_v2.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[-1]["status"] == "TIMEOUT" and rows[-1]["kind"] == "timeout"


def test_resume_is_invalidated_when_a_dependency_changes(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Review 4.4: a checkpoint is bound to its inputs, not just a step name."""
    dependency = tmp_path / "dependency.json"
    dependency.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(orch, "PROGRESS", tmp_path / "v3_progress.json")
    monkeypatch.setattr(orch, "step_signature",
                        lambda step: {"dependency": dependency.read_text(encoding="utf-8")})
    orch.checkpoint("probe")
    assert orch.completed("probe") is True
    dependency.write_text('{"changed": 1}', encoding="utf-8")
    assert orch.completed("probe") is False


def test_s3_cap_uses_explicit_task_ids(monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    """Review 4.2: S3 hours are summed from task ids, not from phase labels."""
    ledger = tmp_path / "compute_ledger_v2.csv"
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "event_id", "parent_event", "kind", "phase", "run_id", "host",
            "started_utc", "finished_utc", "actual_seconds", "counted_seconds",
            "status", "output_sha256", "note"])
        writer.writeheader()
        writer.writerow({"run_id": "bias_v2_lib_F0_I20_phi000_r01", "kind": "tdse",
                         "phase": "R3", "counted_seconds": "3600", "status": "PASS"})
        writer.writerow({"run_id": "bias_v2_lib_F0_I20_phi000_r01", "kind": "readout",
                         "phase": "R2/R3", "counted_seconds": "1800", "status": "PASS"})
        writer.writerow({"run_id": "bias_v2_lib_F0_I32_phi000_r01", "kind": "tdse",
                         "phase": "R3", "counted_seconds": "99999", "status": "PASS"})
    monkeypatch.setattr(orch, "V2_RESULTS", tmp_path)
    assert orch.s3_hours_used() == pytest.approx(1.5)
