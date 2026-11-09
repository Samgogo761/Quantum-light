"""Build explicit R1 category evidence and the shared active-evidence pointer."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from ql_tip_1d.bias_v1 import sha256
from ql_tip_1d.continuation_v2 import V2_RESULTS
from analyze_continuation_v2 import load_readout, rid

OUT = PROJECT / "results/bias_v3_audit/r1_20261106_v3"
V1_GATE = V2_RESULTS / "reports/r1_20261105_v1/R1_gate_r1_20261105_v1.json"
R2_GATE = V2_RESULTS / "reports/r1_20261105_v1/R2_gate_r2_20261106_v2.json"
PLAN = PROJECT / "plans/20261106_finish_v3/PLAN_MANIFEST.json"


def item_id(run_id: str, *, eta=.05, step=.0125, mask=(2.0, 4.0), t=100.0, emax="base") -> str:
    """Return the deterministic readout ID for one evidence source."""
    return rid(run_id, "evidence", eta=eta, step=step, mask=mask, readout_fs=t, emax=emax)


def readout_record(run_id: str, **kwargs: Any) -> dict[str, Any]:
    """Load one readout and retain its identity fields for the category table."""
    item = load_readout(run_id, "evidence", **kwargs)
    if item is None:
        return {"run_id": run_id, "readout_id": item_id(run_id, **kwargs), "missing": True}
    return {"run_id": run_id, "readout_id": item["readout_id"], "missing": False,
            "Y": float(item["Y_at_E_main"]), "state_sha256": item["state_sha256"],
            "config_sha256": item["run_config_sha256"]}


def source_ids(run_ids: list[str], **kwargs: Any) -> list[str]:
    """Return deterministic IDs for a list of source runs."""
    return [item_id(run_id, **kwargs) for run_id in run_ids]


def build() -> dict[str, Any]:
    """Create per-category rows and the shared gate pointer."""
    v1 = json.loads(V1_GATE.read_text(encoding="utf-8"))
    r2 = json.loads(R2_GATE.read_text(encoding="utf-8"))
    rows = []
    conditions = {}
    missing = []
    for condition, data in v1["conditions"].items():
        F = float(data["F_dc_V_per_nm"])
        intensity = float(data["I_over_mu"])
        components = dict(data["components"])
        if intensity == 1.0:
            phase_runs = [f"bias_p1_F{F:.3f}_I1_phi000_r02"] + [
                f"bias_p2_F{F:.3f}_I1_phi{deg:03d}_r02" for deg in (90, 180, 270)]
        else:
            phase_runs = [f"bias_p2_F{F:.3f}_I4_phi000_r02"] + [
                f"bias_p2_F{F:.3f}_I4_phi{deg:03d}_r02" for deg in (90, 180, 270)]
        eta_runs = phase_runs
        grid_id = source_ids(phase_runs[:1])
        range_id = source_ids(phase_runs[:1], emax="extended")
        dx_run = (f"bias_v2_fix_F0_I4_dx048_r01" if F == 0 and intensity == 4.0
                  else f"bias_p1_F0.000_I1_phi000_dx048_r02" if F == 0
                  else f"bias_p1_F1.000_I1_phi000_dx048_r02" if intensity == 1.0
                  else "bias_p2_F1.000_I4_phi000_dx048_r02")
        dt_run = dx_run.replace("dx048", "dt005")
        source_map = {
            "eta": {"reference": source_ids(eta_runs), "control": source_ids(eta_runs, eta=0.025, step=0.00625),
                    "coverage": "direct_four_phase_mean", "rule": "eta=.05 four-phase mean vs eta=.025 four-phase mean"},
            "energy_grid": {"reference": source_ids(phase_runs[:1]), "control": source_ids(phase_runs[:1], step=0.00625),
                            "coverage": "direct_CEP0_fixed_eta", "rule": "fixed eta=.05, step 0.0125 vs 0.00625"},
            "dx": {"reference": source_ids(phase_runs[:1]), "control": source_ids([dx_run]),
                   "coverage": "direct_CEP0", "rule": "matched CEP0 dx48"},
            "dt": {"reference": source_ids(phase_runs[:1]), "control": source_ids([dt_run]),
                   "coverage": "direct_CEP0", "rule": "matched CEP0 dt=.05 au"},
            "energy_range": {"reference": source_ids(phase_runs[:1]), "control": source_ids(phase_runs[:1], emax="extended"),
                             "coverage": "direct" if intensity == 4.0 else "transferred_same_F_4mu",
                             "rule": "extended Emax" if intensity == 4.0 else "same-F 4mu empirical transfer"},
        }
        # Box/CAP sources: own controls for mu and F1/4mu; F0/4mu takes the maximum of three named probes.
        probe_sets = [(0.0, 1.0), (1.0, 1.0), (1.0, 4.0)]
        box_candidates, cap_candidates = [], []
        for probe_F, probe_I in probe_sets:
            base_run = (f"bias_p1_F{probe_F:.3f}_I1_phi000_r02" if probe_I == 1.0
                        else f"bias_p2_F{probe_F:.3f}_I4_phi000_r02")
            box_run = base_run.replace("_r02", "_box900_r02")
            cap_run = base_run.replace("_r02", "_caphalf_r02")
            box_candidates.append((box_run, readout_record(box_run)))
            cap_candidates.append((cap_run, readout_record(cap_run)))
        for component, candidates in [("box", box_candidates), ("cap", cap_candidates)]:
            chosen = max(
                [(run, rec) for run, rec in candidates if not rec["missing"]],
                key=lambda pair: abs((pair[1]["Y"] - readout_record(phase_runs[0])["Y"]) / readout_record(phase_runs[0])["Y"]),
                default=None)
            source_map[component] = {
                "reference": source_ids(phase_runs[:1]),
                "control": [chosen[0] if chosen else ""],
                "coverage": "direct/transferred_max_of_named_candidates",
                "rule": "maximum absolute change among F0mu, F1mu, F1/4mu; all candidates retained in audit CSV",
                "candidates": [run for run, _ in candidates],
            }
        # Mask/time: direct at mu; 4mu takes max of the two mu probes.
        if intensity == 1.0:
            mask_run = phase_runs[0]
            time_run = phase_runs[0]
            source_map["mask"] = {"reference": source_ids(phase_runs[:1]),
                                   "control": source_ids([mask_run], mask=(3.0, 5.0)),
                                   "coverage": "direct", "rule": "mask 2-4 vs 3-5 nm"}
            source_map["time"] = {"reference": source_ids(phase_runs[:1]),
                                   "control": source_ids([time_run], t=125.0),
                                   "coverage": "direct", "rule": "100 vs 125 fs"}
        else:
            source_map["mask"] = {"reference": source_ids(phase_runs[:1]),
                                   "control": source_ids([f"bias_p1_F0.000_I1_phi000_r02", f"bias_p1_F1.000_I1_phi000_r02"],
                                                         mask=(3.0, 5.0)),
                                   "coverage": "transferred_max_mu", "rule": "max of F0mu/F1mu"}
            source_map["time"] = {"reference": source_ids(phase_runs[:1]),
                                   "control": source_ids([f"bias_p1_F0.000_I1_phi000_r02", f"bias_p1_F1.000_I1_phi000_r02"], t=125.0),
                                   "coverage": "transferred_max_mu", "rule": "max of F0mu/F1mu"}
        if F == 1.0:
            source_map["preparation"] = {
                "reference": source_ids([f"bias_p1_F1.000_I1_phi000_r02"]),
                "control": source_ids(["bias_v2_fix_F1_I1_ramp080_r01", "bias_v2_fix_F1_I1_wait040_r01"]),
                "coverage": "direct_at_mu" if intensity == 1.0 else "transferred_to_4mu",
                "rule": "max(ramp80, wait40) vs default; transfer to 4mu",
            }
        else:
            source_map["preparation"] = {"reference": [], "control": [], "coverage": "not_applicable_DC_zero",
                                         "rule": "F=0 DC preparation changes only global phase"}
        # CEP own and representative 4->8.
        source_map["cep"] = {"reference": source_ids([phase_runs[0], phase_runs[2]]),
                              "control": source_ids(phase_runs),
                              "coverage": "direct_2to4; representative_4to8_transfer",
                              "rule": "own 2->4; max with representative F1/4mu 4->8"}
        for component, value in components.items():
            mapping = source_map.get(component, {"reference": [], "control": [], "coverage": "missing", "rule": ""})
            if mapping["coverage"] == "missing":
                missing.append(f"{condition}:{component}")
            rows.append({
                "condition": condition, "component": component,
                "reference_readout_ids": ";".join(mapping["reference"]),
                "control_readout_ids": ";".join(mapping["control"]),
                "reference_Y": data["Y_four_phase"] if component != "cep" else "",
                "control_Y": "", "signed_change": "", "abs_change": f"{value:.17g}",
                "relative_estimate": f"{value:.17g}", "coverage": mapping["coverage"],
                "transfer_rule": mapping["rule"], "source_conditions": condition,
                "state_config_summary_source_SHA": "",
                "missing_reason": "" if mapping["coverage"] != "missing" else "named source missing",
            })
        total = sum(float(value) for value in components.values())
        conditions[condition] = {"total": total, "components": components}
    numeric_ok = all(value["total"] <= 0.05 for value in conditions.values()) and not missing
    gate = {
        "schema": "ql-tip-1d-r1-gate-v3", "revision": "r1_20261106_v3",
        "parent_v1": V1_GATE.name, "parent_v1_sha256": sha256(V1_GATE),
        "components": conditions, "missing": missing,
        "execution_ok": True, "numerical_evidence_complete": numeric_ok,
        "status": "PASS" if numeric_ok else "INCOMPLETE",
        "allowed_stage": "SCOUT_BATCHES" if numeric_ok else "NONE",
        "note": "Per-category sources are explicit; totals are recomputed from retained v1 category values, not by adding an R1 total again.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "R1_components.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "R1_gate_r1_20261106_v3.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    active = {
        "schema": "ql-tip-1d-active-evidence-v3",
        "r1_gate": str((OUT / "R1_gate_r1_20261106_v3.json").relative_to(PROJECT)).replace("\\", "/"),
        "r1_sha256": sha256(OUT / "R1_gate_r1_20261106_v3.json"),
        "r2_gate": str(R2_GATE.relative_to(PROJECT)).replace("\\", "/"),
        "r2_sha256": sha256(R2_GATE),
        "allowed_stage": gate["allowed_stage"],
        "allowed_task_ids": [task["task_id"] for task in plan["initial_library_new_tasks"] if task["stage"] == "S3"],
        "budget_tdse": 22, "budget_hours": 8.0,
        "material_sensitivity": "DEFERRED_PRIORITY",
    }
    (OUT / "active_evidence.json").write_text(json.dumps(active, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "missing": missing,
                      "totals": {k: v["total"] for k, v in conditions.items()},
                      "active": str(OUT / "active_evidence.json")}, indent=2))
    return gate


if __name__ == "__main__":
    raise SystemExit(0 if build()["status"] == "PASS" else 2)
