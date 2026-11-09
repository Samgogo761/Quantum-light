"""Explicit staged task list for the P0-P2 solo-v1 execution.

This runner never submits all stages by default.  Use --dry-run to inspect the
exact TDSE tasks.  P2 is blocked until a separate P1_gate.json says PASS.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = BASE / "configs/protocol_v1.json"
RESULT_ROOT = BASE / "results/bias_v1"
RAW_ROOT = BASE / "data/raw/bias_v1"
GENERATED = BASE / "configs/generated_bias_v1"


@dataclass(frozen=True)
class Task:
    """One uniquely named preparation or observation propagation."""

    run_id: str
    kind: str
    F_dc: float = 0.0
    I_over_mu: float = 0.0
    cep: float = 0.0
    prep_run_id: str | None = None
    numerics_override: dict[str, Any] | None = None
    preparation_override: dict[str, Any] | None = None
    full_spectrum_controls: bool = False
    late_readout: bool = True


def numerical_override(**changes: Any) -> dict[str, Any]:
    """Create a single-factor numerical control override."""
    return changes


def tasks_for_stage(stage: str) -> list[Task]:
    """Return only the user-authorised task list for the selected gate."""
    tasks: list[Task] = []
    if stage == "p1-prep":
        for F in [0.0, 1.0]:
            tasks.append(Task(f"bias_p1_prep_F{F:.3f}_r02", "preparation", F_dc=F))
    elif stage == "p1-dc":
        for F in [0.0, 1.0]:
            prep = f"bias_p1_prep_F{F:.3f}_r02"
            tasks.append(Task(f"bias_p1_dc_F{F:.3f}_r02", "dc-only", F_dc=F, prep_run_id=prep,
                              full_spectrum_controls=True))
    elif stage == "p1-laser":
        for F in [0.0, 1.0]:
            prep = f"bias_p1_prep_F{F:.3f}_r02"
            tasks.append(Task(f"bias_p1_F{F:.3f}_I1_phi000_r02", "laser", F_dc=F, I_over_mu=1.0,
                              prep_run_id=prep, full_spectrum_controls=True))
    elif stage == "p1-preparation-controls":
        for label, ramp, wait in [("ramp080", 80.0, 20.0), ("wait040", 40.0, 40.0)]:
            prep = f"bias_p1_prep_F1.000_{label}_r02"
            override = {"ramp_duration_fs": ramp, "wait_duration_fs": wait}
            tasks.append(Task(prep, "preparation", F_dc=1.0, preparation_override=override))
            tasks.append(Task(f"bias_p1_dc_F1.000_{label}_r02", "dc-only", F_dc=1.0, prep_run_id=prep,
                              preparation_override=override))
    elif stage == "p1-numerical-controls":
        controls = [
            ("dx048", {"well_cells": 48}),
            ("dt005", {"dt_au": 0.05}),
            ("box900", {"xmax_nm": 900.0, "right_cap_interval_nm": [750.0, 900.0]}),
            ("caphalf", {"cap_strength_au": 0.5}),
        ]
        for F in [0.0, 1.0]:
            for label, override in controls:
                prep = f"bias_p1_prep_F{F:.3f}_{label}_r02"
                tasks.append(Task(prep, "preparation", F_dc=F, numerics_override=override))
                tasks.append(Task(f"bias_p1_F{F:.3f}_I1_phi000_{label}_r02", "laser", F_dc=F,
                                  I_over_mu=1.0, prep_run_id=prep, numerics_override=override))
    elif stage == "p2-main":
        base_cep = [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469]
        for F in [0.0, 1.0]:
            prep = f"bias_p1_prep_F{F:.3f}_r02"
            for intensity in [1.0, 4.0]:
                for index, cep in enumerate(base_cep):
                    run_id = f"bias_p2_F{F:.3f}_I{intensity:g}_phi{index * 90:03d}_r02"
                    if intensity == 1.0 and index == 0:
                        # The P1 CEP=0 run is reused and never repeated.
                        continue
                    tasks.append(Task(run_id, "laser", F_dc=F, I_over_mu=intensity, cep=cep,
                                      prep_run_id=prep, full_spectrum_controls=False))
    elif stage == "p2-two-phase":
        # Complete the {0,pi} subset first.  Existing P1 CEP=0 points are reused.
        for F in [0.0, 1.0]:
            prep = f"bias_p1_prep_F{F:.3f}_r02"
            for intensity in [1.0, 4.0]:
                for index, cep in [(0, 0.0), (2, 3.141592653589793)]:
                    if intensity == 1.0 and index == 0:
                        continue
                    tasks.append(Task(f"bias_p2_F{F:.3f}_I{intensity:g}_phi{index * 90:03d}_r02", "laser",
                                      F_dc=F, I_over_mu=intensity, cep=cep, prep_run_id=prep,
                                      full_spectrum_controls=False))
    elif stage == "p2-highest-intensity-controls":
        for label, override in [("dx048", {"well_cells": 48}), ("dt005", {"dt_au": 0.05}),
                                ("box900", {"xmax_nm": 900.0, "right_cap_interval_nm": [750.0, 900.0]}),
                                ("caphalf", {"cap_strength_au": 0.5})]:
            prep = f"bias_p1_prep_F1.000_{label}_r02"
            tasks.append(Task(f"bias_p2_F1.000_I4_phi000_{label}_r02", "laser", F_dc=1.0,
                              I_over_mu=4.0, prep_run_id=prep, numerics_override=override))
    elif stage == "p2-eight-phase":
        choice_path = RESULT_ROOT / "P2_phase_choice.json"
        if not choice_path.exists():
            raise FileNotFoundError("P2_phase_choice.json is required before the eight-phase check")
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        F, intensity = float(choice["F_dc_V_per_nm"]), float(choice["I_over_mu"])
        prep = f"bias_p1_prep_F{F:.3f}_r02"
        for degrees in [45, 135, 225, 315]:
            tasks.append(Task(f"bias_p2_F{F:.3f}_I{intensity:g}_phi{degrees:03d}_r02", "laser",
                              F_dc=F, I_over_mu=intensity, cep=math.radians(degrees), prep_run_id=prep,
                              full_spectrum_controls=False))
    else:
        raise ValueError(f"Unknown stage {stage}")
    # The +125 fs readout is a representative check, not a per-run physical
    # requirement.  It is kept for the first DC/laser points and omitted from
    # control and production phase runs to stay inside the 8 h gate budget.
    if stage in {"p1-preparation-controls", "p1-numerical-controls", "p2-main", "p2-two-phase",
                 "p2-highest-intensity-controls", "p2-eight-phase"}:
        tasks = [replace(task, late_readout=False) for task in tasks]
    return tasks


def make_config(task: Task, protocol: dict[str, Any]) -> dict[str, Any]:
    """Embed the frozen protocol and all per-run choices in one configuration."""
    numerics = json.loads(json.dumps(protocol["numerics"]))
    if task.numerics_override:
        numerics.update(task.numerics_override)
    preparation = json.loads(json.dumps(protocol["preparation"]))
    if task.preparation_override:
        preparation.update(task.preparation_override)
    config = {
        "run_kind": task.kind,
        "F_dc_V_per_nm": task.F_dc,
        "I_over_mu": task.I_over_mu,
        "I_apex_W_per_cm2": task.I_over_mu * protocol["optics"]["mu_I_apex_W_per_cm2"],
        "CEP_rad": task.cep,
        "protocol": protocol,
        "protocol_sha256": __import__("hashlib").sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "numerics": numerics,
        "preparation": preparation,
        "full_spectrum_controls": task.full_spectrum_controls,
        "late_readout": task.late_readout,
        "detector_nm": 10.0,
    }
    if task.kind != "preparation":
        if not task.prep_run_id:
            raise ValueError("Observation requires a preparation run")
        prep_results = RESULT_ROOT / task.prep_run_id
        prep_raw = RAW_ROOT / task.prep_run_id / "state_prep.npz"
        summary_path = prep_results / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Required preparation is incomplete: {task.prep_run_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise ValueError(f"Required preparation did not PASS: {task.prep_run_id}")
        config["prep_state_path"] = str(prep_raw.relative_to(BASE)).replace("\\", "/")
        config["prep_summary_path"] = str(summary_path.relative_to(BASE)).replace("\\", "/")
        config["prep_state_sha256"] = __import__("hashlib").sha256(prep_raw.read_bytes()).hexdigest()
    return config


def append_manifest(task: Task, status: str, wall_seconds: float) -> None:
    """Append one completed run to the shared, auditable CSV manifest."""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULT_ROOT / "run_manifest.csv"
    summary_path = RESULT_ROOT / task.run_id / "summary.json"
    provenance_path = RESULT_ROOT / task.run_id / "provenance.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    row = {
        "run_id": task.run_id, "F_dc_V_per_nm": task.F_dc,
        "I_apex_W_per_cm2": task.I_over_mu * 1.5e12, "I_inc_W_per_cm2": task.I_over_mu * 1.5e12 / 9.0,
        "CEP_rad": task.cep, "grid_id": f"well{summary['well_cells']}_xmax{summary['xmax_nm']:g}",
        "dt_au": summary.get("actual_dt_au", summary.get("actual_dt_au_main")),
        "readout_fs": "100;125" if task.kind != "preparation" else "-100",
        "source_hash": ";".join(f"{key}={value}" for key, value in provenance["source_sha256"].items()),
        "status": status, "wall_seconds": wall_seconds,
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    """List or execute one stage sequentially, preserving existing results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["p1-prep", "p1-dc", "p1-laser", "p1-preparation-controls",
                                 "p1-numerical-controls", "p2-main", "p2-two-phase",
                                 "p2-highest-intensity-controls", "p2-eight-phase"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-wall-hours", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tasks = tasks_for_stage(args.stage)
    if args.stage.startswith("p2"):
        gate_path = RESULT_ROOT / "P1_gate.json"
        if not gate_path.exists() or json.loads(gate_path.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("P2 is blocked: results/bias_v1/P1_gate.json is not PASS")
    selected = tasks[args.offset : args.offset + args.max_runs]
    print(f"stage={args.stage} listed={len(tasks)} selected={len(selected)} single_worker=true")
    for task in selected:
        print(f"{task.run_id}: kind={task.kind}, F={task.F_dc}, I/mu={task.I_over_mu}, CEP={task.cep}, "
              f"prep={task.prep_run_id}, numerics={task.numerics_override}, preparation={task.preparation_override}")
    if args.dry_run:
        print(f"reuse_rule=existing PASS summaries are skipped; P1 CEP0 laser runs are reused in P2")
        print(f"budget=max_new_TDSE=240, max_wall_hours={args.max_wall_hours}, no_BSV_no_server_no_push")
        return 0
    completed = 0
    for task in selected:
        result_dir = RESULT_ROOT / task.run_id
        summary_path = result_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "PASS":
                print(f"REUSE {task.run_id}")
                continue
            raise RuntimeError(f"Existing non-PASS run refuses overwrite: {task.run_id}")
        if result_dir.exists():
            raise RuntimeError(f"Existing incomplete run refuses overwrite: {task.run_id}")
        config = make_config(task, protocol)
        GENERATED.mkdir(parents=True, exist_ok=True)
        config_path = GENERATED / f"{task.run_id}.json"
        with config_path.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        command = [sys.executable, str(BASE / "scripts/run_bias_v1.py"), "--config", str(config_path),
                   "--run-id", task.run_id]
        result = subprocess.run(command, cwd=BASE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Run failed: {task.run_id}; see saved failure/log")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        append_manifest(task, summary["status"], float(summary["wall_seconds"]))
        completed += 1
    print(f"STAGE_DONE {args.stage}; new_runs={completed}; reused={len(selected)-completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
