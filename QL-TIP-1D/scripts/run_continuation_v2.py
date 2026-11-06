"""Explicit continuation-v2 stages with real budget checks and safe reuse.

The legacy bias_v1 solver modules remain byte-frozen.  This runner only builds
new configs, validates preparation bindings, calls the existing observation
function, and records new provenance in results/bias_v2.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from ql_tip_1d.bias_v1 import cap_arrays, make_grid, sha256, write_json_new
from ql_tip_1d.continuation_v2 import (BASE, CONTROL_STEP, GRID_STEP_ETA0025, GRID_STEP_ETA005,
                                       MU_APEX, READOUT_ETA, READOUT_STEP, THRESHOLDS_EV,
                                       V2_CONFIGS, V2_RAW, V2_RESULTS, load_manifest, project_path,
                                       readout_plan, repair_tasks)
from ql_tip_1d.core import A0_NM
import run_bias_v1 as legacy
from readout_continuation_v2 import run_readout

REVISION = "r1_20261105_v1"
REPORT_DIR = V2_RESULTS / "reports" / REVISION


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes() -> dict[str, str]:
    """Hash the new wrapper, common plan and frozen numerical modules."""
    paths = [
        PROJECT / "scripts/run_continuation_v2.py",
        PROJECT / "scripts/readout_continuation_v2.py",
        PROJECT / "src/ql_tip_1d/continuation_v2.py",
        PROJECT / "src/ql_tip_1d/core.py",
        PROJECT / "src/ql_tip_1d/bias_v1.py",
        PROJECT / "src/ql_tip_1d/spectrum_v1.py",
        PROJECT / "configs/continuation_v2/protocol_v2.json",
    ]
    return {str(path.relative_to(PROJECT)).replace("\\", "/"): sha256(path) for path in paths}


LEDGER_FIELDS = ["event_id", "parent_event", "kind", "phase", "run_id", "host", "started_utc",
                 "finished_utc", "actual_seconds", "counted_seconds", "status", "output_sha256", "note"]


def ledger_path() -> Path:
    """Return the continuation compute ledger."""
    return V2_RESULTS / "compute_ledger_v2.csv"


def read_ledger() -> list[dict[str, str]]:
    """Read all recorded events without creating or modifying the ledger."""
    path = ledger_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_ledger(event: dict[str, Any]) -> None:
    """Append one unique scientific event to the compute ledger."""
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: event.get(key, "") for key in LEDGER_FIELDS})


def budget_snapshot() -> dict[str, Any]:
    """Summarise new and globally inherited compute use."""
    rows = read_ledger()
    new_tdse = sum(1 for row in rows if row["kind"] == "tdse" and row["status"] != "preflight_failed")
    new_seconds = sum(float(row["counted_seconds"] or 0.0) for row in rows)
    repair_tdse = sum(1 for row in rows if row["kind"] == "tdse" and row["phase"] == "R1")
    repair_seconds = sum(float(row["counted_seconds"] or 0.0) for row in rows if row["phase"] == "R1")
    return {
        "new_tdse_attempts": new_tdse,
        "new_science_hours": new_seconds / 3600.0,
        "repair_tdse_attempts": repair_tdse,
        "repair_science_hours": repair_seconds / 3600.0,
        "rows": len(rows),
    }


def check_budget(add_tdse: int = 0, add_seconds: float = 0.0, phase: str = "R1") -> dict[str, Any]:
    """Refuse a task when either continuation or repair hard limits would be exceeded."""
    snap = budget_snapshot()
    if snap["new_tdse_attempts"] + add_tdse > 112:
        raise RuntimeError("Continuation TDSE limit 112 would be exceeded")
    if snap["new_science_hours"] + add_seconds / 3600.0 > 40.0:
        raise RuntimeError("Continuation science limit 40 h would be exceeded")
    if phase == "R1":
        if snap["repair_tdse_attempts"] + add_tdse > 12:
            raise RuntimeError("Repair TDSE limit 12 would be exceeded")
        if snap["repair_science_hours"] + add_seconds / 3600.0 > 8.0:
            raise RuntimeError("Repair science limit 8 h would be exceeded")
    return snap


def validate_inputs() -> dict[str, Any]:
    """Verify every frozen manifest input and every saved-state readout label."""
    manifest = load_manifest()
    records = []
    for item in manifest["input_files"]:
        path = project_path(item["path"])
        exists = path.is_file()
        actual = sha256(path) if exists else None
        records.append({"path": item["path"], "exists": exists, "expected_sha256": item["sha256"],
                        "actual_sha256": actual, "match": bool(exists and actual == item["sha256"]),
                        "bytes": path.stat().st_size if exists else None})
    bad = [record for record in records if not record["match"]]
    state_times = []
    for group in ["existing_primary_states", "existing_half_phase_states",
                  "existing_numerical_control_states", "existing_dc_states", "existing_late_states"]:
        for item in manifest[group]:
            path = project_path(item["state"])
            time_value = None
            if path.exists():
                with np.load(path) as state:
                    time_value = float(state["readout_fs"])
            state_times.append({"run_id": item["run_id"], "state": item["state"],
                                "exists": path.exists(), "actual_readout_fs": time_value,
                                "expected_readout_fs": item["expected_readout_fs"],
                                "match": bool(path.exists() and abs(time_value - item["expected_readout_fs"]) < 1e-9)})
    bad_times = [item for item in state_times if not item["match"]]
    return {"schema": "continuation-v2-inventory", "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256(PROJECT / "plans/20261105_continuation_v2/TASK_MANIFEST.json"),
            "input_count": len(records), "inputs": records, "bad_inputs": bad,
            "state_time_count": len(state_times), "state_times": state_times, "bad_state_times": bad_times,
            "budget": budget_snapshot(), "readout_count": len(readout_plan()),
            "repair_tasks": [task["run_id"] for task in repair_tasks()],
            "pass": not bad and not bad_times}


def validate_prep_binding(task: dict[str, Any]) -> dict[str, Any]:
    """Reject a preparation cache that is not the exact intended state."""
    config = task["config"]
    parent = read_json(project_path(task["parent_config"]))
    prep_config = task["prep_config"]
    prep_summary = task["prep_summary"]
    prep_path = project_path(task["prep_state_path"])
    if prep_summary.get("status") != "PASS":
        raise ValueError(f"Preparation summary is not PASS: {task['run_id']}")
    if abs(float(prep_config["F_dc_V_per_nm"]) - float(config["F_dc_V_per_nm"])) > 1e-12:
        raise ValueError("Preparation F does not match task F")
    for key in ["well_cells", "dt_au", "xmin_nm", "xmax_nm", "left_cap_interval_nm",
                "right_cap_interval_nm", "cap_strength_au", "cap_power"]:
        if prep_config["numerics"][key] != config["numerics"][key]:
            raise ValueError(f"Preparation numerics mismatch: {key}")
    if sha256(prep_path) != config["prep_state_sha256"]:
        raise ValueError("Preparation state SHA mismatch")
    if sha256(project_path(task["prep_config_path"])) != config["prep_config_sha256"]:
        raise ValueError("Preparation config SHA mismatch")
    with np.load(prep_path) as state:
        x_au = state["x_au"]
        cap_left = state["left_CAP_au"]
        cap_right = state["right_CAP_au"]
        dx_au = float(state["dx_au"])
    grid = make_grid(config)
    if not np.allclose(grid.x_au, x_au, rtol=0.0, atol=1e-13):
        raise ValueError("Preparation x grid mismatch")
    if abs(grid.dx_au - dx_au) > 1e-13:
        raise ValueError("Preparation dx mismatch")
    expected_left, expected_right = cap_arrays(grid, config)
    if not np.allclose(expected_left, cap_left, rtol=0.0, atol=1e-14):
        raise ValueError("Preparation left CAP mismatch")
    if not np.allclose(expected_right, cap_right, rtol=0.0, atol=1e-14):
        raise ValueError("Preparation right CAP mismatch")
    actual = task["config"]["continuation"]["actual_preparation_fs"]
    for key in ["ramp_start_fs", "ramp_end_fs", "prep_end_fs"]:
        stored = prep_summary.get(key)
        if stored is None or abs(float(stored) - float(actual[key])) > 1e-9:
            raise ValueError(f"Preparation time mismatch: {key}")
    return {"prep_state_sha256": sha256(prep_path),
            "prep_config_sha256": sha256(project_path(task["prep_config_path"])),
            "prep_summary_sha256": sha256(project_path(task["prep_config_path"]).parent / "summary.json"),
            "readout_fs": None}


def stage_inventory(dry_run: bool) -> int:
    """Validate old inputs and print the exact R1 task/readout surface."""
    inventory = validate_inputs()
    print(json.dumps({key: inventory[key] for key in ["pass", "input_count", "bad_inputs",
                                                      "state_time_count", "bad_state_times",
                                                      "budget", "readout_count", "repair_tasks"]},
                     indent=2, ensure_ascii=False)[:12000])
    if dry_run:
        return 0 if inventory["pass"] else 2
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "inventory.json"
    if path.exists():
        raise FileExistsError(path)
    write_json_new(path, inventory)
    return 0 if inventory["pass"] else 2


def stage_repair_tdse(dry_run: bool, max_new_runs: int) -> int:
    """Run the four mandatory repair propagations, never overwriting old data."""
    tasks = repair_tasks()
    selected = tasks[:max_new_runs]
    print(f"repair_tdse listed={len(tasks)} selected={len(selected)}")
    for task in selected:
        print(f"{task['run_id']}: F={task['config']['F_dc_V_per_nm']} "
              f"I/mu={task['config']['I_over_mu']} prep={task['prep_state_path']} "
              f"baseline={task['baseline_run_id']}")
    if dry_run:
        print(json.dumps(budget_snapshot(), indent=2))
        return 0
    for task in selected:
        run_id = task["run_id"]
        result_dir = V2_RESULTS / "runs" / run_id
        raw_dir = V2_RAW / run_id
        if result_dir.exists():
            summary_path = result_dir / "summary.json"
            if summary_path.exists() and read_json(summary_path).get("execution_status") == "PASS":
                print(f"REUSE {run_id}")
                continue
            raise RuntimeError(f"Incomplete v2 run refuses overwrite: {run_id}")
        validate_prep_binding(task)
        check_budget(add_tdse=1, add_seconds=5400.0, phase="R1")
        V2_CONFIGS.mkdir(parents=True, exist_ok=True)
        config_path = V2_CONFIGS / f"{run_id}.json"
        config = task["config"]
        write_json_new(config_path, config)
        result_dir.mkdir(parents=True, exist_ok=False)
        raw_dir.mkdir(parents=True, exist_ok=False)
        # Keep a self-contained copy beside provenance; the generated source
        # config remains the immutable input of record.
        (result_dir / "config.json").write_bytes(config_path.read_bytes())
        started = datetime.now(timezone.utc).isoformat()
        start_clock = time.perf_counter()
        provenance = {
            "schema": "continuation-v2-run", "run_id": run_id, "started_utc": started,
            "config_sha256": sha256(config_path), "parent_config_sha256": task["parent_config_sha256"],
            "prep_state_sha256": config["prep_state_sha256"],
            "prep_config_sha256": config["prep_config_sha256"],
            "prep_summary_sha256": config["prep_summary_sha256"],
            "source_sha256": source_hashes(),
            "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
            "host": "local", "execution_status": "RUNNING", "scientific_gate_status": "NOT_EVALUATED",
        }
        write_json_new(result_dir / "provenance.json", provenance)
        try:
            summary = legacy.run_observation(config, run_id, result_dir, raw_dir, start_clock)
            elapsed = time.perf_counter() - start_clock
            summary.update({
                "schema": "continuation-v2-run-summary", "run_id": run_id,
                "I_over_mu": float(config["I_over_mu"]),
                "I_inc_W_per_cm2": float(config["I_apex_W_per_cm2"]) / 9.0,
                "wall_seconds": elapsed, "execution_status": "PASS",
                "scientific_gate_status": "NOT_EVALUATED",
                "prep_state_sha256": config["prep_state_sha256"],
                "prep_config_sha256": config["prep_config_sha256"],
                "prep_summary_sha256": config["prep_summary_sha256"],
                "state_100fs_sha256": sha256(raw_dir / "state_100fs.npz"),
                "peak_working_set_bytes": legacy.peak_working_set_bytes(),
            })
            if summary["max_probability_budget_residual"] > 1e-8:
                summary["execution_status"] = "FAIL"
            write_json_new(result_dir / "summary.json", summary)
            provenance["execution_status"] = summary["execution_status"]
            provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (result_dir / "provenance.json").write_text(
                json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            append_ledger({"event_id": f"tdse:{run_id}", "parent_event": "", "kind": "tdse", "phase": "R1",
                           "run_id": run_id, "host": "local", "started_utc": started,
                           "finished_utc": provenance["finished_utc"], "actual_seconds": elapsed,
                           "counted_seconds": elapsed, "status": summary["execution_status"],
                           "output_sha256": summary["state_100fs_sha256"], "note": ""})
            print(json.dumps({"run_id": run_id, "execution_status": summary["execution_status"],
                              "wall_seconds": elapsed, "Y_diagnostic": summary["Y_H"]}, indent=2))
            if summary["execution_status"] != "PASS":
                return 2
        except Exception as exc:
            elapsed = time.perf_counter() - start_clock
            failure = {"schema": "continuation-v2-failure", "run_id": run_id, "error_type": type(exc).__name__,
                       "error": str(exc), "started_utc": started,
                       "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": elapsed}
            write_json_new(result_dir / "failure.json", failure)
            append_ledger({"event_id": f"tdse:{run_id}:failed", "parent_event": "", "kind": "tdse", "phase": "R1",
                           "run_id": run_id, "host": "local", "started_utc": started,
                           "finished_utc": failure["finished_utc"], "actual_seconds": elapsed,
                           "counted_seconds": elapsed, "status": "failed", "output_sha256": "", "note": str(exc)})
            raise
    return 0


def stage_repair_readouts(dry_run: bool, max_events: int, include_planned: bool) -> int:
    """Read old saved states first, then new repair states on a later --resume."""
    specs = readout_plan(include_new=True)
    pending = []
    runnable = []
    for spec in specs:
        state = project_path(spec["state_path"])
        out = V2_RESULTS / "readouts" / spec["readout_id"]
        if out.exists() and (out / "summary.json").exists():
            continue
        if not state.exists():
            pending.append(spec)
            continue
        runnable.append(spec)
    selected = runnable[:max(1, min(int(max_events), 8))]
    print(f"repair_readouts listed={len(specs)} runnable={len(runnable)} selected={len(selected)}")
    for spec in selected:
        print(f"{spec['readout_id']} eta={spec['eta_eV']} step={spec['step_eV']} "
              f"mask={spec['mask_start_nm']}-{spec['mask_end_nm']} emax={spec['energy_max_mode']}")
    if dry_run:
        print(json.dumps({"planned_readouts": len(specs), "missing_dependencies": len(pending),
                          "missing_ids": [spec["readout_id"] for spec in pending][:8],
                          "include_planned": include_planned, "budget": budget_snapshot()}, indent=2))
        return 0
    for spec in selected:
        run_spec_readout(spec)
    print(json.dumps({"completed_now": len(selected), "runnable_remaining": max(0, len(runnable) - len(selected)),
                      "planned_dependencies": len(pending)}, indent=2))
    return 0


def run_spec_readout(spec: dict[str, Any]) -> None:
    """Run and ledger one planned readout event."""
    validate_readout_spec(spec)
    check_budget(add_seconds=1800.0, phase="R1")
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    summary = run_readout(spec)
    elapsed = time.perf_counter() - start
    append_ledger({"event_id": "readout:" + spec["readout_id"], "parent_event": "tdse:" + spec["run_id"],
                   "kind": "readout", "phase": "R1", "run_id": spec["run_id"], "host": "local",
                   "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "actual_seconds": elapsed, "counted_seconds": elapsed,
                   "status": summary["execution_status"],
                   "output_sha256": sha256(V2_RESULTS / "readouts" / spec["readout_id"] / "summary.json"),
                   "note": f"eta={spec['eta_eV']},step={spec['step_eV']}"})


def validate_readout_spec(spec: dict[str, Any]) -> None:
    """Check the fixed thresholds and state/config hashes before post-processing."""
    if [float(x) for x in spec["thresholds_eV"]] != list(THRESHOLDS_EV):
        raise ValueError("Readout thresholds were changed")
    state = project_path(spec["state_path"])
    config = read_json(project_path(spec["config_path"]))
    if not state.is_file():
        raise FileNotFoundError(state)
    if abs(float(spec["F_dc_V_per_nm"]) - float(config["F_dc_V_per_nm"])) > 1e-12:
        raise ValueError("Readout F mismatch")
    if abs(float(spec["I_over_mu"]) - float(config["I_over_mu"])) > 1e-12:
        raise ValueError("Readout I mismatch")
    with np.load(state) as data:
        if abs(float(data["readout_fs"]) - float(spec["expected_readout_fs"])) > 1e-9:
            raise ValueError("Readout time mismatch")


def stage_high_intensity(dry_run: bool, max_new_runs: int, offset: int) -> int:
    """Run 8mu probes, then 32mu probes and controls, with a hard dependency gate."""
    from ql_tip_1d.continuation_v2 import high_intensity_tasks, v2_readout_specs
    if not (REPORT_DIR / "R1_gate_r1_20261105_v1.json").exists():
        raise RuntimeError("R1 gate file is missing")
    gate = read_json(REPORT_DIR / "R1_gate_r1_20261105_v1.json")
    if gate.get("status") != "PASS":
        raise RuntimeError("R1 gate is not PASS")
    tasks = high_intensity_tasks()
    selected = tasks[offset:offset + max_new_runs]
    print(f"high_intensity listed={len(tasks)} selected={len(selected)}")
    for task in selected:
        print(f"{task['run_id']}: F={task['config']['F_dc_V_per_nm']} I/mu={task['config']['I_over_mu']} "
              f"numerics={task['config']['numerics']['well_cells']}/{task['config']['numerics']['dt_au']}")
    if dry_run:
        print(json.dumps(budget_snapshot(), indent=2))
        return 0
    if offset >= 2:
        for F in [0.0, 1.0]:
            path = V2_RESULTS / "runs" / f"bias_v2_probe_F{F:g}_I8_phi000_r01/summary.json"
            if not path.exists() or read_json(path).get("execution_status") != "PASS":
                raise RuntimeError("32mu probes require both completed 8mu probes")
    for task in selected:
        run_id = task["run_id"]
        result_dir = V2_RESULTS / "runs" / run_id
        raw_dir = V2_RAW / run_id
        if result_dir.exists():
            summary_path = result_dir / "summary.json"
            if summary_path.exists() and read_json(summary_path).get("execution_status") == "PASS":
                print(f"REUSE {run_id}")
                continue
            raise RuntimeError(f"Incomplete v2 run refuses overwrite: {run_id}")
        check_budget(add_tdse=1, add_seconds=5400.0, phase="R2")
        V2_CONFIGS.mkdir(parents=True, exist_ok=True)
        config_path = V2_CONFIGS / f"{run_id}.json"
        config = task["config"]
        write_json_new(config_path, config)
        result_dir.mkdir(parents=True, exist_ok=False)
        raw_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "config.json").write_bytes(config_path.read_bytes())
        started = datetime.now(timezone.utc).isoformat()
        start_clock = time.perf_counter()
        provenance = {"schema": "continuation-v2-run", "run_id": run_id, "started_utc": started,
                      "config_sha256": sha256(config_path), "parent_config_sha256": task["parent_config_sha256"],
                      "prep_state_sha256": config.get("prep_state_sha256"),
                      "prep_config_sha256": config.get("prep_config_sha256"),
                      "prep_summary_sha256": config.get("prep_summary_sha256"),
                      "source_sha256": source_hashes(),
                      "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                      "host": "local", "execution_status": "RUNNING", "scientific_gate_status": "NOT_EVALUATED"}
        write_json_new(result_dir / "provenance.json", provenance)
        try:
            summary = legacy.run_observation(config, run_id, result_dir, raw_dir, start_clock)
            elapsed = time.perf_counter() - start_clock
            summary.update({"schema": "continuation-v2-run-summary", "run_id": run_id,
                            "I_over_mu": float(config["I_over_mu"]),
                            "I_inc_W_per_cm2": float(config["I_apex_W_per_cm2"]) / 9.0,
                            "wall_seconds": elapsed, "execution_status": "PASS",
                            "scientific_gate_status": "NOT_EVALUATED",
                            "state_100fs_sha256": sha256(raw_dir / "state_100fs.npz"),
                            "peak_working_set_bytes": legacy.peak_working_set_bytes()})
            if summary["max_probability_budget_residual"] > 1e-8:
                summary["execution_status"] = "FAIL"
            write_json_new(result_dir / "summary.json", summary)
            provenance["execution_status"] = summary["execution_status"]
            provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (result_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
            append_ledger({"event_id": f"tdse:{run_id}", "parent_event": "", "kind": "tdse", "phase": "R2",
                           "run_id": run_id, "host": "local", "started_utc": started,
                           "finished_utc": provenance["finished_utc"], "actual_seconds": elapsed,
                           "counted_seconds": elapsed, "status": summary["execution_status"],
                           "output_sha256": summary["state_100fs_sha256"], "note": task["config"]["continuation"]["phase"]})
            print(json.dumps({"run_id": run_id, "status": summary["execution_status"],
                              "wall_seconds": elapsed, "Y": summary["Y_H"]}))
            if summary["execution_status"] != "PASS":
                return 2
        except Exception as exc:
            elapsed = time.perf_counter() - start_clock
            write_json_new(result_dir / "failure.json", {"run_id": run_id, "error": str(exc),
                                                         "wall_seconds": elapsed})
            append_ledger({"event_id": f"tdse:{run_id}:failed", "parent_event": "", "kind": "tdse", "phase": "R2",
                           "run_id": run_id, "host": "local", "started_utc": started,
                           "finished_utc": datetime.now(timezone.utc).isoformat(), "actual_seconds": elapsed,
                           "counted_seconds": elapsed, "status": "failed", "output_sha256": "", "note": str(exc)})
            raise
    return 0


def stage_v2_readouts(dry_run: bool, max_events: int) -> int:
    """Read out all available v2 probe/library states without crossing stages."""
    from ql_tip_1d.continuation_v2 import v2_readout_specs
    specs = readout_plan(include_new=True) + v2_readout_specs()
    unique = {}
    for spec in specs:
        unique[spec["readout_id"]] = spec
    runnable = []
    for spec in unique.values():
        out = V2_RESULTS / "readouts" / spec["readout_id"]
        if out.exists() and (out / "summary.json").exists():
            continue
        if project_path(spec["state_path"]).exists():
            runnable.append(spec)
    selected = runnable[:max(1, min(max_events, 8))]
    print(f"v2_readouts listed={len(unique)} runnable={len(runnable)} selected={len(selected)}")
    for spec in selected:
        print(f"{spec['readout_id']} eta={spec['eta_eV']} step={spec['step_eV']} emax={spec['energy_max_mode']}")
    if dry_run:
        return 0
    for spec in selected:
        validate_readout_spec(spec)
        started = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        summary = run_readout(spec)
        elapsed = time.perf_counter() - start
        append_ledger({"event_id": "readout:" + spec["readout_id"], "parent_event": "tdse:" + spec["run_id"],
                       "kind": "readout", "phase": "R2/R3", "run_id": spec["run_id"], "host": "local",
                       "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                       "actual_seconds": elapsed, "counted_seconds": elapsed,
                       "status": summary["execution_status"],
                       "output_sha256": sha256(V2_RESULTS / "readouts" / spec["readout_id"] / "summary.json"),
                       "note": f"eta={spec['eta_eV']},step={spec['step_eV']}"})
    print(json.dumps({"completed_now": len(selected), "remaining": len(runnable) - len(selected)}, indent=2))
    return 0


def execute_legacy_observation(config: dict[str, Any], run_id: str, result_dir: Path,
                               raw_dir: Path, start_clock: float) -> dict[str, Any]:
    """Call the frozen observation function, optionally omitting redundant .1 spectra."""
    if not config.get("skip_legacy_spectrum", False):
        return legacy.run_observation(config, run_id, result_dir, raw_dir, start_clock)
    original = legacy.spectrum_for_state

    def diagnostic_placeholder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        label = args[1]
        eta = float(args[8])
        step = float(args[9])
        mask = args[10]
        return {"label": label, "eta_eV": eta, "energy_step_eV": step,
                "mask_bounds_nm": list(mask), "energy_range_eV": [0.0, 0.0],
                "energy_points": 0, "masked_norm": 0.0, "spectrum_integral": 0.0,
                "spectrum_norm_absolute_difference": 0.0, "Y_H_lower": 0.0,
                "Y_H_main": 0.0, "Y_H_upper": 0.0,
                "note": "legacy eta=.1 diagnostic skipped; continuation readout is authoritative"}

    try:
        legacy.spectrum_for_state = diagnostic_placeholder
        return legacy.run_observation(config, run_id, result_dir, raw_dir, start_clock)
    finally:
        legacy.spectrum_for_state = original


def stage_library(dry_run: bool, max_new_runs: int, offset: int) -> int:
    """Run the remaining two-bias production phases in explicit batches."""
    from ql_tip_1d.continuation_v2 import library_tasks
    if not (REPORT_DIR / "R1_gate_r1_20261105_v1.json").exists():
        raise RuntimeError("R1 gate file is missing")
    if read_json(REPORT_DIR / "R1_gate_r1_20261105_v1.json").get("status") != "PASS":
        raise RuntimeError("R1 gate is not PASS")
    if midpoint:
        plan = read_json(REPORT_DIR / "midpoint_plan.json")
        tasks = [task for task in midpoint_tasks(plan)]
    else:
        tasks = [task for task in library_tasks() if not task.get("existing")]
    selected = tasks[offset:offset + max_new_runs]
    print(f"library new={len(tasks)} selected={len(selected)}")
    for task in selected:
        print(f"{task['run_id']}: F={task['F_dc_V_per_nm']} I/mu={task['I_over_mu']} CEP={task['CEP_rad']}")
    if dry_run:
        print(json.dumps(budget_snapshot(), indent=2))
        return 0
    for task in selected:
        run_id = task["run_id"]
        result_dir = V2_RESULTS / "runs" / run_id
        raw_dir = V2_RAW / run_id
        if result_dir.exists():
            summary_path = result_dir / "summary.json"
            if summary_path.exists() and read_json(summary_path).get("execution_status") == "PASS":
                print(f"REUSE {run_id}")
                continue
            raise RuntimeError(f"Incomplete v2 library run refuses overwrite: {run_id}")
        check_budget(add_tdse=1, add_seconds=5400.0, phase="R3")
        config_path = V2_CONFIGS / f"{run_id}.json"
        write_json_new(config_path, task["config"])
        result_dir.mkdir(parents=True, exist_ok=False)
        raw_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "config.json").write_bytes(config_path.read_bytes())
        start_clock = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        provenance = {"schema": "continuation-v2-run", "run_id": run_id, "started_utc": started,
                      "config_sha256": sha256(config_path), "parent_config_sha256": task["parent_config_sha256"],
                      "prep_state_sha256": task["config"].get("prep_state_sha256"),
                      "prep_config_sha256": task["config"].get("prep_config_sha256"),
                      "prep_summary_sha256": task["config"].get("prep_summary_sha256"),
                      "source_sha256": source_hashes(),
                      "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                      "host": "local", "execution_status": "RUNNING", "scientific_gate_status": "NOT_EVALUATED"}
        write_json_new(result_dir / "provenance.json", provenance)
        try:
            summary = execute_legacy_observation(task["config"], run_id, result_dir, raw_dir, start_clock)
            elapsed = time.perf_counter() - start_clock
            summary.update({"schema": "continuation-v2-run-summary", "run_id": run_id,
                            "I_over_mu": float(task["config"]["I_over_mu"]),
                            "I_inc_W_per_cm2": float(task["config"]["I_apex_W_per_cm2"]) / 9.0,
                            "wall_seconds": elapsed, "execution_status": "PASS",
                            "scientific_gate_status": "NOT_EVALUATED",
                            "state_100fs_sha256": sha256(raw_dir / "state_100fs.npz"),
                            "peak_working_set_bytes": legacy.peak_working_set_bytes()})
            if summary["max_probability_budget_residual"] > 1e-8:
                summary["execution_status"] = "FAIL"
            write_json_new(result_dir / "summary.json", summary)
            provenance["execution_status"] = summary["execution_status"]
            provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (result_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
            append_ledger({"event_id": f"tdse:{run_id}", "parent_event": "", "kind": "tdse", "phase": "R3",
                           "run_id": run_id, "host": "local", "started_utc": started,
                           "finished_utc": provenance["finished_utc"], "actual_seconds": elapsed,
                           "counted_seconds": elapsed, "status": summary["execution_status"],
                           "output_sha256": summary["state_100fs_sha256"], "note": "library"})
            print(json.dumps({"run_id": run_id, "status": summary["execution_status"],
                              "wall_seconds": elapsed, "Y": summary["Y_H"]}))
            if summary["execution_status"] != "PASS":
                return 2
        except Exception as exc:
            elapsed = time.perf_counter() - start_clock
            write_json_new(result_dir / "failure.json", {"run_id": run_id, "error": str(exc),
                                                         "wall_seconds": elapsed})
            append_ledger({"event_id": f"tdse:{run_id}:failed", "parent_event": "", "kind": "tdse", "phase": "R3",
                           "run_id": run_id, "host": "local", "started_utc": started,
                           "finished_utc": datetime.now(timezone.utc).isoformat(), "actual_seconds": elapsed,
                           "counted_seconds": elapsed, "status": "failed", "output_sha256": "", "note": str(exc)})
            raise
    return 0


def main() -> int:
    """Parse a single explicit stage and execute no implicit matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["inventory", "repair-readouts", "repair-tdse", "repair-gate",
                                 "high-intensity-check", "high-intensity-readouts", "library",
                                 "library-midpoints", "library-readouts", "statistics"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-runs", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-planned", action="store_true")
    args = parser.parse_args()
    if args.stage == "inventory":
        return stage_inventory(args.dry_run)
    if args.stage == "repair-tdse":
        return stage_repair_tdse(args.dry_run, args.max_new_runs)
    if args.stage == "repair-readouts":
        return stage_repair_readouts(args.dry_run, args.max_events, args.include_planned)
    if args.stage == "high-intensity-check":
        return stage_high_intensity(args.dry_run, args.max_new_runs, args.offset)
    if args.stage in {"high-intensity-readouts", "library-readouts"}:
        return stage_v2_readouts(args.dry_run, args.max_events)
    if args.stage == "library":
        return stage_library(args.dry_run, args.max_new_runs, args.offset)
    if args.stage == "library-midpoints":
        return stage_library(args.dry_run, args.max_new_runs, args.offset, midpoint=True)
    if args.stage == "statistics":
        import analyze_statistics_v2 as statistics
        return statistics.final_statistics()
    raise RuntimeError(f"Stage not yet implemented in this revision: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
