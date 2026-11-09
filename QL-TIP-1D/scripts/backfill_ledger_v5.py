"""Backfill the two ledger omissions found by the 2026-11-08 review.

No TDSE, spectrum or science is run here.  Two gaps are repaired:

1. Six completed S4 saved-state readouts (two biases x eta / fixed-eta grid /
   extended range at J=8 mu) were written by `v3_stage_v4.step_s4_readouts`
   through `run_readout`, which does not append the compute ledger.  Their
   measured `wall_seconds` are recorded once, keyed by readout id, so a rerun
   cannot duplicate them.
2. One propagation attempt (`bias_v2_lib_F1_I20_phi180_r01`) was interrupted
   by the host blue screen.  Its start time is known from the preserved
   provenance, but it has no end time.  It is recorded as a separate
   `tdse_attempt` with `INTERRUPTED_POWER_LOSS_ESTIMATED`: elapsed is unknown,
   so a 0.5 h conservative reserve is booked and explicitly labelled as an
   estimate, never as a proven upper bound.  The successful rerun remains the
   authoritative `tdse:` event.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
V2 = PROJECT / "results/bias_v2"
LEDGER = V2 / "compute_ledger_v2.csv"
AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
CONTROLS = AUDIT / "s4_readout_controls.json"
PROVENANCE = V2 / "incomplete/bluescreen_20261107/runs_F1_I20_phi180/provenance.json"
FIELDS = ["event_id", "parent_event", "kind", "phase", "run_id", "host",
          "started_utc", "finished_utc", "actual_seconds", "counted_seconds",
          "status", "output_sha256", "note"]
ATTEMPT_ID = "attempt:tdse:bias_v2_lib_F1_I20_phi180_r01:bluescreen_20261107"
RESERVE_SECONDS = 1800.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def main() -> int:
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8", newline="")))
    seen = {row["event_id"] for row in rows}
    additions: list[dict[str, Any]] = []
    readout_seconds = 0.0
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    for row in controls["rows"]:
        readout_id = row["readout_id"]
        summary_path = V2 / "readouts" / readout_id / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        wall = float(summary["wall_seconds"])
        readout_seconds += wall
        event_id = f"readout:{readout_id}"
        if event_id in seen:
            continue
        finished = summary_path.stat().st_mtime
        additions.append({
            "event_id": event_id,
            "parent_event": f"tdse:{summary['run_id']}",
            "kind": "readout", "phase": "R3", "run_id": summary["run_id"],
            "host": "local", "started_utc": iso(finished - wall),
            "finished_utc": iso(finished), "actual_seconds": wall,
            "counted_seconds": wall, "status": "PASS",
            "output_sha256": sha256(summary_path),
            "note": "backfilled S4 saved-state control readout; v3_stage_v4 called "
                    "run_readout directly and wrote no ledger row",
        })
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if ATTEMPT_ID not in seen:
        additions.append({
            "event_id": ATTEMPT_ID,
            "parent_event": "", "kind": "tdse_attempt", "phase": "R3",
            "run_id": provenance["run_id"], "host": "local",
            "started_utc": provenance["started_utc"], "finished_utc": "",
            "actual_seconds": "", "counted_seconds": RESERVE_SECONDS,
            "status": "INTERRUPTED_POWER_LOSS_ESTIMATED", "output_sha256": "",
            "note": ("host blue screen; no end timestamp and status stayed RUNNING; "
                     f"elapsed unknown, a {RESERVE_SECONDS / 3600.0:.1f} h conservative "
                     "reserve is booked from the 18:13:45Z start to the ~19:03Z host "
                     "restart window and is NOT a proven upper bound; successful rerun "
                     "is tdse:bias_v2_lib_F1_I20_phi180_r01"),
        })
    if additions:
        exists = LEDGER.exists()
        with LEDGER.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(additions)
    total_seconds = sum(float(row["counted_seconds"] or 0.0) for row in rows + additions)
    print(json.dumps({
        "appended": len(additions),
        "backfilled_readouts": sum(1 for row in additions if row["kind"] == "readout"),
        "readout_seconds": readout_seconds,
        "readout_hours": readout_seconds / 3600.0,
        "interrupted_attempt_recorded": any(row["event_id"] == ATTEMPT_ID for row in additions),
        "interruption_reserve_hours": RESERVE_SECONDS / 3600.0,
        "confirmed_continuation_hours_excluding_reserve": (total_seconds - RESERVE_SECONDS) / 3600.0,
        "continuation_hours_including_reserve": total_seconds / 3600.0,
        "events_after": len(rows) + len(additions),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
