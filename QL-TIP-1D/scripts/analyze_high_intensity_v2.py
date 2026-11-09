"""R2 high-intensity cost/risk gate for continuation-v2."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.continuation_v2 import (CONTROL_ETA, CONTROL_STEP, READOUT_ETA, READOUT_STEP,
                                       V2_RESULTS, v2_readout_specs)
from ql_tip_1d.bias_v1 import sha256

REPORT_DIR = V2_RESULTS / "reports" / "r1_20261105_v1"


def load(path: Path) -> dict[str, Any]:
    """Read JSON or return an empty mapping."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def find_readout(run_id: str, eta: float = READOUT_ETA, step: float = READOUT_STEP,
                 emax: str = "base") -> dict[str, Any] | None:
    """Load one v2 readout by matching its immutable identity fields."""
    for spec in v2_readout_specs():
        if (spec["run_id"] == run_id and abs(spec["eta_eV"] - eta) < 1e-12
                and abs(spec["step_eV"] - step) < 1e-12 and spec["energy_max_mode"] == emax):
            path = V2_RESULTS / "readouts" / spec["readout_id"] / "summary.json"
            return load(path) or None
    return None


def rel(reference: float, test: float) -> float | None:
    """Return a relative change unless the reference is zero."""
    return None if reference == 0.0 else (test - reference) / reference


def compute() -> dict[str, Any]:
    """Check probe usability, numerical controls, CAP warning and spectrum range."""
    result: dict[str, Any] = {"schema": "continuation-v2-R2-gate", "probes": {}, "controls": {}}
    probe_ids = [f"bias_v2_probe_F{F:g}_I{intensity:g}_phi000_r01"
                 for intensity in [8.0, 32.0] for F in [0.0, 1.0]]
    for run_id in probe_ids:
        summary = load(V2_RESULTS / "runs" / run_id / "summary.json")
        readout = find_readout(run_id)
        if not summary or not readout:
            result["probes"][run_id] = {"status": "INCOMPLETE"}
            continue
        cap_ratio = summary["right_CAP_loss_observation_100fs"] / summary["Y_H"] if summary["Y_H"] else None
        result["probes"][run_id] = {
            "wall_seconds": summary["wall_seconds"], "peak_working_set_bytes": summary.get("peak_working_set_bytes"),
            "Y_diagnostic_eta01": summary["Y_H"], "Y_main_eta005": readout["Y_at_E_main"],
            "right_CAP_fraction": cap_ratio, "budget_residual": summary["max_probability_budget_residual"],
            "execution_status": summary["execution_status"],
            "cap_warning": bool(cap_ratio is not None and cap_ratio > 1e-4),
        }
    base_F = {}
    for F in [0.0, 1.0]:
        run_id = f"bias_v2_probe_F{F:g}_I32_phi000_r01"
        base = find_readout(run_id)
        if not base:
            continue
        base_F[F] = base
        for label in ["dx048", "dt005", "box900", "caphalf"]:
            control_id = f"bias_v2_probe_F{F:g}_I32_phi000_{label}_r01"
            control = find_readout(control_id)
            if control:
                change = rel(float(base["Y_at_E_main"]), float(control["Y_at_E_main"]))
                result["controls"][f"F{F:g}_{label}"] = {
                    "run_id": control_id, "Y": control["Y_at_E_main"], "relative_change": change,
                    "pass": change is not None and abs(change) < 0.05}
            else:
                result["controls"][f"F{F:g}_{label}"] = {"status": "missing"}
    if not base_F:
        result["status"] = "INCOMPLETE"
        return result
    eta32 = {}
    grid32 = {}
    for F in [0.0, 1.0]:
        run_id = f"bias_v2_probe_F{F:g}_I32_phi000_r01"
        base = base_F.get(F)
        eta = find_readout(run_id, eta=CONTROL_ETA, step=CONTROL_STEP)
        if base and eta:
            eta32[f"F{F:g}"] = rel(float(base["Y_at_E_main"]), float(eta["Y_at_E_main"]))
        grid = find_readout(run_id, eta=READOUT_ETA, step=0.00625)
        if base and grid:
            grid32[f"F{F:g}"] = rel(float(base["Y_at_E_main"]), float(grid["Y_at_E_main"]))
    range32 = {}
    for F in [0.0, 1.0]:
        run_id = f"bias_v2_probe_F{F:g}_I32_phi000_r01"
        base = base_F.get(F)
        extended = find_readout(run_id, eta=READOUT_ETA, step=READOUT_STEP, emax="extended")
        if base and extended:
            range32[f"F{F:g}"] = rel(float(base["Y_at_E_main"]), float(extended["Y_at_E_main"]))
    result["eta32_relative"] = eta32
    result["grid32_relative"] = grid32
    result["range32_relative"] = range32
    result["controls_pass"] = all(item.get("pass", False) for item in result["controls"].values())
    cap_warning = any(item.get("cap_warning", False) for item in result["probes"].values())
    box_mitigation = {}
    for F in [0.0, 1.0]:
        item = result["controls"].get(f"F{F:g}_box900", {})
        box_mitigation[f"F{F:g}"] = item.get("pass", False)
    result["cap_warning_present"] = cap_warning
    result["cap_mitigated_by_box"] = (not cap_warning) or all(box_mitigation.values())
    total_error = {}
    for F in [0.0, 1.0]:
        values = []
        for label in ["dx048", "dt005", "box900", "caphalf"]:
            item = result["controls"].get(f"F{F:g}_{label}", {})
            if isinstance(item.get("relative_change"), (int, float)):
                values.append(abs(float(item["relative_change"])))
        for mapping in [eta32, grid32, range32]:
            if f"F{F:g}" in mapping:
                values.append(abs(float(mapping[f"F{F:g}"])))
        total_error[f"F{F:g}"] = sum(values)
    result["total_high_intensity_error"] = total_error
    probe_runs = [load(V2_RESULTS / "runs" / f"bias_v2_probe_F{F:g}_I{intensity:g}_phi000_r01" / "summary.json")
                  for intensity in [8.0, 32.0] for F in [0.0, 1.0]]
    propagation_seconds = sorted(float(item["wall_seconds"]) for item in probe_runs if item)
    median_propagation = propagation_seconds[len(propagation_seconds) // 2] if propagation_seconds else 960.0
    planned_propagations = 52 + 16 + 8
    planned_readouts = planned_propagations
    estimated_seconds = planned_propagations * median_propagation + planned_readouts * 120.0 + 3600.0
    result["estimated_library_hours"] = estimated_seconds / 3600.0
    result["estimated_library_note"] = (
        f"{planned_propagations} propagations at measured median {median_propagation/60:.2f} min "
        f"+ {planned_readouts} readouts at 2 min + 1 h reserve; scheduling estimate, not guarantee")
    required_ok = (result["controls_pass"] and result["cap_mitigated_by_box"]
                   and all(item.get("execution_status") == "PASS" for item in result["probes"].values())
                   and len(eta32) == 2 and len(grid32) == 2 and len(range32) == 2
                   and all(abs(value) < 0.05 for value in eta32.values())
                   and all(abs(value) < 0.05 for value in grid32.values())
                   and all(abs(value) < 0.05 for value in range32.values()))
    result["high_risk_flag"] = any(value > 0.05 for value in total_error.values())
    result["status"] = "PASS" if required_ok else "INCOMPLETE"
    return result


def main() -> int:
    """Write and print the R2 gate snapshot."""
    result = compute()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "R2_gate_r2_20261106_v2.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
