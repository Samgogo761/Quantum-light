"""Build planning metadata only; never import or execute research code.

Run from the repository root with Python. Existing different metadata is not
overwritten; a changed workspace requires a new plan revision.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "QL-TIP-1D"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_new_or_identical(path: Path, obj: object) -> None:
    raw = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"Existing metadata differs; use a new revision: {path}")
    else:
        path.write_bytes(raw)


def record_source(name: str) -> dict:
    path = ROOT / name
    if not path.is_file():
        raise FileNotFoundError(name)
    return {"path": name, "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> None:
    sources = [
        "AGENTS.md",
        "Q-MATH/README.md",
        "Q-MATH/SYNC.md",
        "Q-MATH/manifest.json",
        "tools/qmath/manage.py",
        "QL-TIP-1D/plans/20261105_continuation_v2/EXECUTION_TASKBOOK.md",
        "QL-TIP-1D/plans/20261105_evening_decision/DECISION_AND_AGENT_INSTRUCTIONS.md",
        "QL-TIP-1D/scripts/run_continuation_v2.py",
        "QL-TIP-1D/scripts/analyze_statistics_v2.py",
        "QL-TIP-1D/scripts/analyze_high_intensity_v2.py",
        "QL-TIP-1D/scripts/readout_continuation_v2.py",
        "QL-TIP-1D/src/ql_tip_1d/continuation_v2.py",
        "QL-TIP-1D/tests/test_continuation_v2.py",
        "QL-TIP-1D/results/bias_v2/compute_ledger_v2.csv",
        "QL-TIP-1D/results/bias_v2/reports/r1_20261105_v1/R1_gate_r1_20261105_v1.json",
        "QL-TIP-1D/results/bias_v2/reports/r1_20261105_v1/R1_gate_r1_20261106_v2.json",
        "QL-TIP-1D/results/bias_v2/reports/r1_20261105_v1/R2_gate_r1_20261105_v1.json",
        "QL-TIP-1D/results/bias_v2/reports/r1_20261105_v1/R2_gate_r2_20261106_v2.json",
        "QL-TIP-1D/reports/COST_PLAN_20261106.md",
    ]
    source_records = [record_source(s) for s in sources]
    with (PROJECT / "results/bias_v2/compute_ledger_v2.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        ledger = list(csv.DictReader(stream))
    event_ids = [r["event_id"] for r in ledger]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("Ledger contains duplicate event IDs; do not freeze a plan.")
    counts = Counter(r["kind"] for r in ledger)
    seconds = sum(float(r["counted_seconds"]) for r in ledger)
    # This assertion detects changes to the reviewed checkpoint; it is not a
    # scientific acceptance test and must not be weakened to ignore new events.
    assert counts == {"tdse": 16, "readout": 90}, counts
    tdse_ids = {r["run_id"] for r in ledger if r["kind"] == "tdse"}

    phases = (0, 90, 180, 270)
    batches = [
        ("S3_A", "S3", 8.0, (90, 180, 270)),
        ("S3_B", "S3", 12.0, phases),
        ("S3_C", "S3", 20.0, phases),
        ("S5_D", "S5", 32.0, (90, 180, 270)),
        ("S5_E", "S5", 0.25, phases),
        ("S5_F", "S5", 0.5, phases),
        ("S5_G", "S5", 2.0, phases),
    ]
    pending = []
    for batch, stage, intensity, angles in batches:
        for F in (0, 1):
            for deg in angles:
                run_id = f"bias_v2_lib_F{F:g}_I{intensity:g}_phi{deg:03d}_r01"
                assert run_id not in tdse_ids, f"Already in ledger: {run_id}"
                pending.append({
                    "task_id": run_id,
                    "batch": batch,
                    "stage": stage,
                    "F_dc_V_per_nm": F,
                    "I_over_mu": intensity,
                    "CEP_deg": deg,
                    "status": "PLANNED_NOT_RUN",
                    "new_laser_TDSE": 1,
                    "required_readout": "common_eta005_dE00125_three_frozen_thresholds",
                    "numerical_regime": "original_dt_0.1au_24wellcells",
                    "preparation": "verify existing matched preparation; no implicit new prep",
                })
    reused = []
    for F in (0, 1):
        for intensity in (1.0, 4.0, 8.0, 32.0):
            angles = phases if intensity in (1.0, 4.0) else (0,)
            for deg in angles:
                if intensity in (8.0, 32.0):
                    run_id = f"bias_v2_probe_F{F:g}_I{intensity:g}_phi000_r01"
                    state = PROJECT / "data/raw/bias_v2" / run_id / "state_100fs.npz"
                else:
                    prefix = "bias_p1" if intensity == 1 and deg == 0 else "bias_p2"
                    run_id = f"{prefix}_F{F:.3f}_I{intensity:g}_phi{deg:03d}_r02"
                    state = PROJECT / "data/raw/bias_v1" / run_id / "state_100fs.npz"
                reused.append({
                    "run_id": run_id,
                    "F_dc_V_per_nm": F,
                    "I_over_mu": intensity,
                    "CEP_deg": deg,
                    "status": "REUSE_AFTER_STATE_CONFIG_READOUT_SHA_CHECK",
                    "state_path": relative(state),
                    "state_present_at_plan_creation": state.is_file(),
                    "new_laser_TDSE": 0,
                })

    assert len(pending) == 52 and len(reused) == 20
    assert sum(t["stage"] == "S3" for t in pending) == 22
    assert sum(t["stage"] == "S5" for t in pending) == 30
    assert len({t["task_id"] for t in pending}) == 52
    assert all(n <= 8 for n in Counter(t["batch"] for t in pending).values())
    actual_cells = {(t["F_dc_V_per_nm"], t["I_over_mu"], t["CEP_deg"])
                    for t in pending + reused}
    expected_cells = {(f, i, p) for f in (0, 1)
                      for i in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 32.0)
                      for p in phases}
    assert len(actual_cells) == 72 and actual_cells == expected_cells

    manifest = {
        "schema": "ql-tip-1d-finish-plan-v3",
        "document_date": "2026-11-06",
        "kind": "PLANNING_ONLY_NO_SCIENTIFIC_EXECUTION",
        "git_head_at_review": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_snapshot": source_records,
        "snapshot_warning": "Source hashes identify the reviewed state, not approval gates. Reconcile subsequent changes without overwriting them.",
        "existing_ledger": {
            "counts_by_kind": dict(counts),
            "unique_event_count": len(event_ids),
            "counted_seconds": seconds,
            "counted_hours": seconds / 3600,
            "values_are_ledger_observations_not_science_validation": True,
        },
        "limits": {
            "global_TDSE": 240, "global_science_hours": 72,
            "continuation_TDSE": 112, "continuation_science_hours": 40,
            "R1_repair_TDSE": 12, "R1_repair_science_hours": 8,
            "nominal_remaining_continuation_TDSE": 112 - counts["tdse"],
            "nominal_remaining_continuation_hours": 40 - seconds / 3600,
            "S3_new_TDSE": 22, "S3_science_hours": 8,
            "batch_max_TDSE": 8, "event_max_minutes": 90,
            "max_nonzero_intensities_per_F": 12, "Imax_over_mu": 64,
            "all_preparations_failures_controls_readouts_count": True,
            "no_budget_reset_between_stages": True,
        },
        "default_sequence": [
            "S0_preserve_old_bytes_and_checkpoint_repairs",
            "S1_complete_R1_category_evidence_and_shared_gate_statistics_interfaces",
            "S2_cost_all_mandatory_tasks_and_enforce_timeouts",
            "S3_bounded_22_run_scout_library",
            "S4_direct_controls_in_important_intensity_region",
            "S5_remaining_30_runs_and_preliminary_statistics",
            "S6_tail_real_midpoints_phase_checks_and_final_audit",
        ],
        "default_dt0025_first": False,
        "no_new_physics_or_material_variant": True,
        "runner_note": "Adapt explicit task selection and gate checks first. This JSON is not an executable runner config; do not use a bare offset as authorization.",
        "initial_library_new_tasks": pending,
        "initial_library_existing_tasks": reused,
        "initial_library_DC_endpoints": [
            {"F_dc_V_per_nm": f, "run_id": f"bias_p1_dc_F{f:.3f}_r02",
             "required": "real common readout; never substitute exact zero"}
            for f in (0, 1)
        ],
        "adaptive_reservations_not_run": [
            {"stage": "S4", "purpose": "important-region dx/dt controls",
             "typical_TDSE": 4, "preparations_if_missing_extra": True,
             "choose_after": "S3 contribution evidence, before seeing control outcomes"},
            {"stage": "S6", "purpose": "two distinct real interpolation midpoints",
             "typical_TDSE": 16, "both_F_four_CEP": True},
            {"stage": "S6", "purpose": "highest and important-region 4-to-8 CEP",
             "max_TDSE_without_further_phase_refinement": 8,
             "reuse_if_same_condition": True},
            {"stage": "CONDITIONAL", "purpose": "one dt=.025 diagnostic",
             "typical_TDSE": 2, "default": "DEFERRED_UNTIL_WEIGHTED_ERROR_JUSTIFIES",
             "fine_production_other_phases_not_included": True},
            {"stage": "CONDITIONAL", "purpose": "one sufficient tail node, 48 or 64",
             "minimum_production_TDSE": 8,
             "new_max_controls_prep_phase_and_readout_extra": True},
        ],
        "typical_mandatory_remaining_TDSE_before_optional_or_extra_preparations": 80,
        "acceptance": {
            "final_weighted_Y_total_relative_error_max": 0.05,
            "tail_bound_over_positive_Y_lower_max": 0.01,
            "node_relative_error_max": 0.01,
            "quadrature_relative_error_max": 0.001,
            "phase_4_to_8_change_target": 0.02,
            "phase_error_included_once_in_total": True,
            "final_classifications": ["VALID_DIFFERENCE", "VALID_UNRESOLVED", "INCOMPLETE"],
            "gate_PASS_alone_is_insufficient": True,
            "intervals_are_empirical_audit_intervals_not_confidence_intervals": True,
        },
        "checks_performed_by_this_script": [
            "required small source files exist and SHA256 captured",
            "ledger event identities unique; 16 TDSE and 90 readout observed",
            "52 unique planned new library tasks split 22 plus 30",
            "20 reusable phase identities plus 52 planned tasks cover 72 cells",
            "all fixed library batches <=8 tasks",
            "no scientific module imported; no research solver/test executed",
        ],
    }
    write_new_or_identical(HERE / "PLAN_MANIFEST.json", manifest)
    print(json.dumps({"manifest": relative(HERE / "PLAN_MANIFEST.json"),
                      "sources": len(source_records), "new_tasks": len(pending),
                      "reused_tasks": len(reused), "S3": 22, "S5": 30,
                      "reused_states_present": sum(t["state_present_at_plan_creation"] for t in reused),
                      "ledger_hours": seconds / 3600}, ensure_ascii=False))


if __name__ == "__main__":
    main()
