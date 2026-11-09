"""Apply the rebuilt R1 v5 gate to the shared evidence pointer.

Also removes the stale top-level budget_tdse/budget_hours keys so that the
nested `budget` block is the single authoritative cap, and records a source
freeze (hashes of every executing module) for restart validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"
POINTER = PROJECT / "results/bias_v3_audit/r1_20261107_v5/r1_v5_pointer.json"
SOURCE_FILES = [
    "scripts/run_continuation_v2.py", "scripts/readout_continuation_v2.py",
    "scripts/analyze_statistics_v2.py", "scripts/build_error_component_map_v4.py",
    "scripts/rebuild_r1_pairs_v5.py", "scripts/v3_stage_v4.py",
    "scripts/run_v3_full_v4.py", "scripts/build_budget_reallocation_v4.py",
    "scripts/build_tail_extension_v5.py", "scripts/apply_r1_v5.py",
    "scripts/backfill_ledger_v5.py", "scripts/build_node_error_table.py",
    "src/ql_tip_1d/continuation_v2.py", "src/ql_tip_1d/core.py",
    "src/ql_tip_1d/bias_v1.py", "src/ql_tip_1d/spectrum_v1.py",
    "src/ql_tip_1d/statistics_v2.py", "configs/continuation_v2/protocol_v2.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_freeze() -> dict[str, str]:
    return {name: sha256(PROJECT / name) for name in SOURCE_FILES if (PROJECT / name).exists()}


def main() -> int:
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    gate_path = PROJECT / pointer["r1_gate"]
    if sha256(gate_path) != pointer["r1_sha256"]:
        raise RuntimeError("R1 v5 gate hash mismatch")
    active = json.loads(ACTIVE.read_text(encoding="utf-8"))
    previous = {"r1_gate": active.get("r1_gate"), "r1_sha256": active.get("r1_sha256")}
    active["r1_gate"] = pointer["r1_gate"]
    active["r1_sha256"] = pointer["r1_sha256"]
    active["r1_components_csv"] = "results/bias_v3_audit/r1_20261107_v5/R1_components_v5.csv"
    active["r1_components_csv_sha256"] = sha256(
        PROJECT / "results/bias_v3_audit/r1_20261107_v5/R1_components_v5.csv")
    active["r1_rebuild"] = {
        "revision": "r1_20261107_v5", "status": pointer["status"],
        "numerical_evidence_complete": pointer["numerical_evidence_complete"],
        "pairing": "each control paired with its own matched reference before transfer",
        "previous": previous,
    }
    # One authoritative budget cap; remove stale duplicated top-level keys.
    active.pop("budget_tdse", None)
    active.pop("budget_hours", None)
    if "budget" not in active:
        raise RuntimeError("nested budget block missing")
    active["budget"]["stage_s3_hours_limit"] = 10.0
    active["budget"]["authoritative"] = True
    active["source_freeze"] = source_freeze()
    ACTIVE.write_text(json.dumps(active, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"r1_gate": active["r1_gate"], "r1_status": pointer["status"],
                      "r1_sha256": active["r1_sha256"],
                      "removed_duplicate_budget_keys": True,
                      "frozen_sources": len(active["source_freeze"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
