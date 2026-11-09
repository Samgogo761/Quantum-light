"""Resumable, serial orchestrator for the remaining v3-r2 stages S4 -> S6.

External review 2026-11-07 fixes implemented here:

* an outer hard limit per SCIENTIFIC EVENT (not 90 min for a whole 8-run batch),
  with the actual elapsed time recorded on timeout;
* readout preflight failures are never parsed as "zero pending";
* checkpoints bind real inputs (file hashes, gate hashes, frozen source
  revision) so a changed dependency invalidates the downstream resume;
* the 10 h S3 stage cap is enforced from an explicit task-id ledger sum;
* S4 has a real gate: S5 cannot start unless its status is PROCEED;
* the tail is audited right after preliminary statistics, before midpoint and
  phase costs are committed, with a conditional 48/64 mu extension branch;
* the final tail audit uses the current library, never the preliminary file.

Every propagation still goes through run_continuation_v2.py with explicit
allowed_task_ids.  The process refuses to start unless the frozen source
revision matches the files on disk, so a running process can never silently
execute an older in-memory orchestration logic after a source edit.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import v3_stage_v4 as stage
from ql_tip_1d.continuation_v2 import library_tasks

PY = PROJECT / ".venv/Scripts/python.exe"
RUNNER = PROJECT / "scripts/run_continuation_v2.py"
STATS = PROJECT / "scripts/analyze_statistics_v2.py"
STAGE = PROJECT / "scripts/v3_stage_v4.py"
TAIL_GENERATOR = PROJECT / "scripts/build_tail_extension_v5.py"
AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ACTIVE = stage.ACTIVE
REPORT = stage.REPORT
V2_RUNS = PROJECT / "results/bias_v2/runs"
V2_READOUTS = PROJECT / "results/bias_v2/readouts"
V2_RESULTS = PROJECT / "results/bias_v2"
LOG = PROJECT / "logs/v3_full_20261106.log"
PROGRESS = AUDIT / "v3_progress.json"
TIMEOUTS = AUDIT / "v3_event_timeouts.jsonl"

LEDGER_FIELDS = ["event_id", "parent_event", "kind", "phase", "run_id", "host",
                 "started_utc", "finished_utc", "actual_seconds", "counted_seconds",
                 "status", "output_sha256", "note"]

STEP_FILES: dict[str, list[Path]] = {
    "s3c": [],
    "s4": [AUDIT / "s4_selection.json", AUDIT / "s4_controls.json",
           AUDIT / "s4_gate.json", AUDIT / "s4_readout_controls.json"],
    "s5": [REPORT / "R3_statistics_preliminary.json", REPORT / "classical_response_prelim.csv"],
    "tail_precheck": [AUDIT / "tail_precheck.json"],
    "tail_extension": [AUDIT / "tail_extension_status.json"],
    "midpoints": [REPORT / "midpoint_plan.json"],
    "phase8": [AUDIT / "phase8_selection.json"],
    "phase_validation": [AUDIT / "phase_validation.json"],
    "error_map": [AUDIT / "error_component_map.csv", AUDIT / "phase_validation.json"],
    "tail_final": [AUDIT / "tail_audit.json"],
    "final": [REPORT / "R3_statistics.json", AUDIT / "decision.md"],
}


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}\n"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def verify_source_freeze() -> None:
    """Refuse to run against sources that differ from the frozen revision."""
    active = stage.jload(ACTIVE)
    frozen = active.get("source_freeze")
    if not frozen:
        raise RuntimeError("No source_freeze in active_evidence.json; freeze at a batch boundary first")
    mismatch = [name for name, expected in frozen.items()
                if stage.file_sha(PROJECT / name) != expected]
    if mismatch:
        raise RuntimeError(f"Frozen source mismatch; restart a fresh process: {mismatch}")
    log(f"source_freeze verified: {len(frozen)} modules match the running revision")


def record_timeout(label: str, args: list[str], elapsed: float, hard: float) -> None:
    """Persist the failed event and its real elapsed time (review section 4.1)."""
    utc = datetime.now(timezone.utc).isoformat()
    AUDIT.mkdir(parents=True, exist_ok=True)
    with (AUDIT / "v3_event_timeouts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc, "label": label, "args": args,
                                 "elapsed_seconds": elapsed, "hard_timeout_s": hard,
                                 "status": "TIMEOUT"}) + "\n")
    ledger = V2_RESULTS / "compute_ledger_v2.csv"
    exists = ledger.exists()
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"event_id": f"timeout:{label}:{utc}", "parent_event": "",
                         "kind": "timeout", "phase": "R3", "run_id": label, "host": "local",
                         "started_utc": utc, "finished_utc": utc,
                         "actual_seconds": elapsed, "counted_seconds": elapsed,
                         "status": "TIMEOUT", "output_sha256": "",
                         "note": f"outer event limit hard={hard}s"})


def call(label: str, args: list[str], allow_incomplete: bool = False,
         events: int = 1, timeout_s: float | None = None) -> int:
    """Run one command with an outer limit scaled by its scientific event count."""
    cmd = [str(PY)] + args
    hard = float(stage.jload(ACTIVE).get("budget", {}).get("event_hard_timeout_s", 5400))
    timeout = timeout_s if timeout_s is not None else events * (hard + 300.0)
    log(f"START {label}: {' '.join(args)} (events={events}, outer_timeout={timeout:.0f}s)")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(cmd)}\n")
        handle.flush()
        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            handle.write(f"EVENT_TIMEOUT label={label} elapsed_s={elapsed:.3f} hard_s={hard}\n")
            record_timeout(label, args, elapsed, hard)
            log(f"FAILED {label}: outer event timeout after {elapsed:.1f}s")
            raise RuntimeError(f"{label} exceeded its outer limit of {timeout:.0f}s")
        elapsed = time.monotonic() - start
    if returncode != 0 and not (allow_incomplete and returncode == 2):
        log(f"FAILED {label}: exit={returncode}")
        raise RuntimeError(f"{label} failed with exit code {returncode}; see {LOG}")
    log(f"DONE {label}: exit={returncode} elapsed={elapsed:.1f}s")
    return returncode


def done(run_id: str) -> bool:
    path = V2_RUNS / run_id / "summary.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("execution_status") == "PASS"


def pending_readouts() -> int:
    """A failed preflight is never silently interpreted as zero pending work."""
    proc = subprocess.run([str(PY), str(RUNNER), "--stage", "library-readouts",
                           "--max-events", "8", "--dry-run"], cwd=PROJECT,
                          capture_output=True, text=True, timeout=600.0)
    if proc.returncode != 0:
        raise RuntimeError(f"Readout preflight failed (exit={proc.returncode}): "
                           f"{proc.stdout[-1200:]} {proc.stderr[-600:]}")
    if "v2_readouts" not in proc.stdout or "runnable=" not in proc.stdout:
        raise RuntimeError(f"Readout preflight produced no verifiable output: {proc.stdout[-1200:]}")
    match = re.search(r"runnable=(\d+)", proc.stdout)
    if match is None:
        raise RuntimeError("Readout preflight did not report runnable=")
    return int(match.group(1))


def drain_readouts(label: str) -> None:
    previous: int | None = None
    for index in range(8):
        pending = pending_readouts()
        if pending == 0:
            log(f"{label}: no pending readouts")
            return
        if previous is not None and pending >= previous:
            raise RuntimeError(f"{label}: readout queue did not shrink ({previous} -> {pending})")
        previous = pending
        call(f"{label} readouts {index + 1}",
             [str(RUNNER), "--stage", "library-readouts", "--max-events", "8"], events=8)
    raise RuntimeError(f"{label}: readouts still pending after 8 rounds")


def batches(ids: list[str], size: int = 8) -> list[list[str]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def step_signature(step: str) -> dict[str, Any]:
    """Bind a checkpoint to real outputs, gate hashes and the frozen sources."""
    active = stage.jload(ACTIVE)
    return {
        "source_freeze": stage.stats.stable_digest(active.get("source_freeze", {})),
        "r1_sha256": active.get("r1_sha256", ""),
        "r2_sha256": active.get("r2_sha256", ""),
        "files": {str(path.relative_to(PROJECT)).replace("\\", "/"): stage.file_sha(path)
                  for path in STEP_FILES.get(step, [])},
    }


def checkpoint(step: str) -> None:
    data = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    data.setdefault("completed", [])
    data.setdefault("signatures", {})
    if step not in data["completed"]:
        data["completed"].append(step)
    data["signatures"][step] = step_signature(step)
    data["last_utc"] = datetime.now(timezone.utc).isoformat()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def completed(step: str) -> bool:
    data = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    if step not in data.get("completed", []):
        return False
    if data.get("signatures", {}).get(step) != step_signature(step):
        log(f"INVALIDATED {step}: dependency signature changed; the step will be rerun")
        return False
    return True


S3C_IDS = [f"bias_v2_lib_F{F}_I20_phi{deg:03d}_r01"
           for F in ["0", "1"] for deg in [0, 90, 180, 270]]
S3_IDS = set(S3C_IDS)
for _I in [8.0, 12.0]:
    for _F in ["0", "1"]:
        for _deg in [0, 90, 180, 270]:
            S3_IDS.add(f"bias_v2_probe_F{_F}_I{_I:g}_phi000_r01" if (_I == 8.0 and _deg == 0)
                       else f"bias_v2_lib_F{_F}_I{_I:g}_phi{_deg:03d}_r01")


def s3_hours_used() -> float:
    """Explicit task-id ledger sum, including readouts and failed events."""
    path = V2_RESULTS / "compute_ledger_v2.csv"
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sum(float(row["counted_seconds"] or 0.0) for row in rows
               if row["run_id"] in S3_IDS) / 3600.0


def enforce_s3_cap(reserve_hours: float = 0.0) -> None:
    """Settle the S3 stage from an explicit task-id sum; refuse NEW S3 events.

    A batch already in flight is allowed to finish and be recorded honestly.
    The global 72 h / 240 TDSE ceiling is enforced separately by the runner.
    """
    limit = float(stage.jload(ACTIVE).get("budget", {}).get("stage_s3_hours_limit", 10.0))
    used = s3_hours_used()
    if reserve_hours > 0.0 and used + reserve_hours > limit:
        raise RuntimeError(f"S3 stage cap {limit} h would be exceeded by a new event: "
                           f"used {used:.4f} h + reserve {reserve_hours:.4f} h")
    status = "WITHIN_CAP" if used <= limit else "OVER_CAP_SETTLED_NO_NEW_S3_EVENTS"
    stage.jwrite(AUDIT / "s3_stage_budget.json", {
        "schema": "ql-tip-1d-s3-stage-budget-v5", "explicit_task_ids": sorted(S3_IDS),
        "hours_used": used, "hours_limit": limit, "reserve_hours": reserve_hours,
        "status": status,
        "note": "recorded from explicit S3 task ids including readouts and failures; "
                "global ceiling unchanged"})
    log(f"S3 stage budget: {used:.4f} h used of {limit:.1f} h -> {status}")


def s3c_complete() -> bool:
    """Both the propagations and the common readouts of batch C must exist."""
    for run_id in S3C_IDS:
        if not done(run_id):
            return False
        readout_id = f"ro2_{run_id}_t100_eta0p05_dE0p0125_m2-4"
        path = V2_READOUTS / readout_id / "summary.json"
        if not path.exists():
            return False
        if json.loads(path.read_text(encoding="utf-8")).get("execution_status") != "PASS":
            return False
    return True


def wait_for_s3c(max_hours: float = 8.0) -> None:
    """Serial discipline: do not start another TDSE while batch C is in flight."""
    deadline = time.monotonic() + max_hours * 3600.0
    while not s3c_complete():
        if time.monotonic() > deadline:
            raise RuntimeError("Timed out waiting for S3-C propagations and readouts")
        log("waiting for S3-C (propagations + common readouts) ...")
        time.sleep(120.0)
    log("S3-C is complete; starting S4")


def verify_s3c() -> None:
    missing = [run_id for run_id in S3C_IDS if not done(run_id)]
    if missing:
        raise RuntimeError(f"S3-C is not complete; missing {missing}")
    readout_missing = [run_id for run_id in S3C_IDS
                       if not (V2_READOUTS / f"ro2_{run_id}_t100_eta0p05_dE0p0125_m2-4"
                               / "summary.json").exists()]
    if readout_missing:
        raise RuntimeError(f"S3-C readouts are missing: {readout_missing}")
    drain_readouts("verify_s3c")
    enforce_s3_cap()
    stage.set_evidence("INITIAL_LIBRARY", [], v3_stage="S4_PREP")
    checkpoint("s3c")


def do_s4() -> None:
    call("contributions", [str(STAGE), "--step", "contributions"])
    selection = stage.jload(AUDIT / "s4_selection.json")
    J = float(selection["J_over_mu"])
    ids = [f"bias_v2_s4_F{F}_I{J:g}_phi000_{label}_r01"
           for F in ["0", "1"] for label in ["dx048", "dt005"]]
    todo = [run_id for run_id in ids if not done(run_id)]
    if todo:
        stage.set_evidence("INITIAL_LIBRARY", todo, v3_stage="S4")
        call("S4 controls", [str(RUNNER), "--stage", "s4-controls",
                             "--max-new-runs", str(min(8, len(todo)))], events=len(todo))
        drain_readouts("S4")
    # Direct eta / energy-grid / energy-range checks at the representative J.
    call("S4 direct readouts", [str(STAGE), "--step", "s4-readouts"], events=6)
    call("S4 gate", [str(STAGE), "--step", "s4"], allow_incomplete=True)
    gate = stage.jload(AUDIT / "s4_gate.json")
    if gate.get("status") != "PROCEED":
        raise RuntimeError(f"S4 gate is {gate.get('status')}; S5 must not start. "
                           f"blockers={gate.get('blockers')}")
    checkpoint("s4")


def remaining_library_ids() -> list[str]:
    ids = []
    for task in library_tasks():
        run_id = task["run_id"]
        if task.get("existing") or done(run_id):
            continue
        ids.append(run_id)
    return ids


def do_s5() -> None:
    remaining = remaining_library_ids()
    log(f"S5 remaining library tasks: {len(remaining)}")
    for index, batch in enumerate(batches(remaining), start=1):
        todo = [run_id for run_id in batch if not done(run_id)]
        if not todo:
            continue
        stage.set_evidence("INITIAL_LIBRARY", todo, v3_stage="S5", s5_batch=index)
        call(f"S5 batch {index} ({len(todo)})",
             [str(RUNNER), "--stage", "library", "--max-new-runs", str(len(todo))],
             events=len(todo))
        drain_readouts(f"S5 batch {index}")
    leftover = remaining_library_ids()
    if leftover:
        raise RuntimeError(f"S5 still has unrun tasks: {leftover}")
    stage.set_evidence("INITIAL_LIBRARY", [], v3_stage="PRELIMINARY")
    call("preliminary statistics", [str(STATS), "--preliminary"], allow_incomplete=True)
    checkpoint("s5")


def do_tail_extension(plan: dict[str, Any]) -> None:
    """Conditional 48/64 mu branch; never unconditional, never invented."""
    reason = ""
    if plan.get("status") != "REQUIRED_WITHIN_BUDGET":
        reason = f"extension status={plan.get('status')}; no admissible in-budget extension"
    elif not TAIL_GENERATOR.exists():
        reason = "tail-extension task generator is not available in this revision"
    if reason:
        stage.jwrite(AUDIT / "tail_extension_status.json", {
            "schema": "ql-tip-1d-tail-extension-status-v5", "status": "INCOMPLETE",
            "reason": reason, "plan": plan,
            "rule": "deliver INCOMPLETE; do not invent tail certification"})
        log(f"tail extension NOT executed: {reason}")
        checkpoint("tail_extension")
        return
    nodes = ",".join(f"{value:g}" for value in plan.get("new_nodes", []))
    call("tail extension plan", [str(TAIL_GENERATOR), "--nodes", nodes])
    tasks = stage.jload(AUDIT / "tail_extension_tasks.json")
    ids = [item["run_id"] for item in tasks.get("tasks", [])]
    if not ids:
        raise RuntimeError("tail extension plan produced no tasks")
    for index, batch in enumerate(batches(ids), start=1):
        todo = [run_id for run_id in batch if not done(run_id)]
        if not todo:
            continue
        stage.set_evidence("TAIL_EXTENSION", todo, v3_stage="TAIL_EXTENSION",
                           tail_extension_batch=index)
        call(f"tail extension batch {index} ({len(todo)})",
             [str(RUNNER), "--stage", "tail-extension", "--max-new-runs", str(len(todo))],
             events=len(todo))
        drain_readouts(f"tail extension batch {index}")
    stage.jwrite(AUDIT / "tail_extension_status.json", {
        "schema": "ql-tip-1d-tail-extension-status-v5", "status": "EXECUTED",
        "nodes": plan.get("new_nodes", []), "task_ids": ids,
        "rule": "smallest sufficient extension; the final tail audit is rerun afterwards"})
    checkpoint("tail_extension")


def do_tail_precheck() -> None:
    """Tail audit BEFORE midpoint/phase costs; branch on the shortfall."""
    call("tail precheck", [str(STAGE), "--step", "tail", "--tail-scope", "precheck"],
         allow_incomplete=True)
    audit = stage.jload(AUDIT / "tail_precheck.json")
    checkpoint("tail_precheck")
    if audit.get("sufficient_at_32mu"):
        log("32 mu tail satisfies the positive lower bound; no extension required")
        stage.jwrite(AUDIT / "tail_extension_status.json", {
            "schema": "ql-tip-1d-tail-extension-status-v5", "status": "NOT_REQUIRED",
            "plan": audit.get("extension", {})})
        checkpoint("tail_extension")
        return
    log(f"32 mu tail shortfall; extension={audit.get('extension', {}).get('status')}")
    do_tail_extension(audit.get("extension", {}))


def do_midpoints() -> None:
    stage.set_evidence("FINAL_STATISTICS", [], v3_stage="MIDPOINT_PLAN")
    call("midpoint plan", [str(STATS), "--plan-midpoints"])
    plan = stage.jload(REPORT / "midpoint_plan.json")
    intervals = plan.get("intervals", [])
    if len(intervals) != 2:
        raise RuntimeError("Two real midpoint intervals are required")
    ids = []
    for item in intervals:
        I = float(item["midpoint_I_over_mu"])
        for F in ["0", "1"]:
            for deg in [0, 90, 180, 270]:
                ids.append(f"bias_v2_mid_F{F}_I{I:g}_phi{deg:03d}_r01")
    for index, batch in enumerate(batches(ids), start=1):
        todo = [run_id for run_id in batch if not done(run_id)]
        if not todo:
            continue
        stage.set_evidence("FINAL_STATISTICS", todo, v3_stage="MIDPOINTS", midpoint_batch=index)
        call(f"midpoints batch {index}", [str(RUNNER), "--stage", "midpoints",
                                          "--max-new-runs", str(len(todo))], events=len(todo))
        drain_readouts(f"midpoints batch {index}")
    checkpoint("midpoints")


def do_phase8() -> None:
    call("contributions (full library)", [str(STAGE), "--step", "contributions"])
    call("phase8 selection", [str(STAGE), "--step", "phase8"])
    selection = stage.jload(AUDIT / "phase8_selection.json")
    ids = []
    for F, I in selection.get("conditions", []):
        for deg in [45, 135, 225, 315]:
            ids.append(f"bias_v2_ph8_F{float(F):g}_I{float(I):g}_phi{deg:03d}_r01")
    todo = [run_id for run_id in ids if not done(run_id)]
    for index, batch in enumerate(batches(todo), start=1):
        stage.set_evidence("FINAL_STATISTICS", batch, v3_stage="PHASE8", phase8_batch=index)
        call(f"phase8 batch {index}", [str(RUNNER), "--stage", "phase8",
                                       "--max-new-runs", str(len(batch))], events=len(batch))
        drain_readouts(f"phase8 batch {index}")
    checkpoint("phase8")


def do_final() -> None:
    # Phase evidence first, then fold direct 4->8 into the single CEP category.
    call("phase validation", [str(STAGE), "--step", "phase-validation"],
         allow_incomplete=True)
    checkpoint("phase_validation")
    call("error map refresh", [str(STAGE), "--step", "error-map"])
    checkpoint("error_map")
    # Final tail audit uses the CURRENT library, not the preliminary file.
    call("tail final", [str(STAGE), "--step", "tail", "--tail-scope", "final"],
         allow_incomplete=True)
    checkpoint("tail_final")
    call("final statistics", [str(STATS), "--final"], allow_incomplete=True)
    call("decision", [str(STAGE), "--step", "decision"])
    stage.set_evidence("NONE", [], v3_stage="COMPLETE")
    checkpoint("final")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="skip checkpointed steps")
    parser.add_argument("--wait-s3c", action="store_true",
                        help="wait until the already-running 20 mu batch finishes first")
    args = parser.parse_args()
    log("=== v3-r2 full run started (review v5 fixes) ===")
    verify_source_freeze()
    if args.wait_s3c and not s3c_complete():
        wait_for_s3c()
    steps = [("s3c", verify_s3c), ("s4", do_s4), ("s5", do_s5),
             ("tail_precheck", do_tail_precheck), ("midpoints", do_midpoints),
             ("phase8", do_phase8), ("final", do_final)]
    for name, function in steps:
        if args.resume and completed(name):
            log(f"SKIP {name} (checkpointed against unchanged inputs)")
            continue
        log(f"=== STAGE {name} ===")
        function()
    log("=== v3-r2 full run complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
