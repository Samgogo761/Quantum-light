"""Build a per-node, per-category error table (no TDSE, no new spectrum).

External review 2026-11-08 section 3 asked for one auditable table that, for
every response node, states whether a numerical error value is

* a direct control measured AT that node,
* a direct control measured at another condition and transferred (with scope),
* a latest coarse-to-fine correction (historical 4 mu value),
* an empirical 2->4 phase envelope, or
* a direct 4->8 phase check at that exact node.

The table also carries the weighted absolute contribution w_i * Y_i * e_ic so
that the dominant category/node can be identified without re-integrating.
No value is deleted to make a gate pass; a category is only narrowed where a
direct measurement actually exists for that condition.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import analyze_statistics_v2 as stats
from ql_tip_1d.statistics_v2 import linear_weights

AUDIT = stats.AUDIT_DIR
REPORT = stats.REPORT_DIR
OUT_CSV = AUDIT / "node_error_table.csv"
OUT_JSON = AUDIT / "node_error_summary.json"
CATEGORIES = list(stats.ERROR_CATEGORIES)
FIELDS = ["F", "I_over_mu", "component", "value", "evidence_type", "source",
          "transfer_scope", "reason", "weight_bsv", "weighted_bsv",
          "weight_thermal", "weighted_thermal"]


def jload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def r1_tables() -> dict[str, dict[str, float]]:
    gate = jload(PROJECT / stats.load_active()["r1_gate"])
    return {key: {name: abs(float(value)) for name, value in item["components"].items()}
            for key, item in gate["components"].items()}


def r2_direct() -> dict[str, dict[str, float]]:
    gate = jload(PROJECT / stats.load_active()["r2_gate"])
    result: dict[str, dict[str, float]] = {}
    for label in ["F0", "F1"]:
        values = {key: abs(float(gate["controls"][f"{label}_{key}"]["relative_change"]))
                  for key in ["dx048", "dt005", "box900", "caphalf"]}
        values.update({name: abs(float(gate[name][label]))
                       for name in ["eta32_relative", "grid32_relative", "range32_relative"]})
        result[label] = {
            "dx": values["dx048"], "dt": values["dt005"],
            "box": values["box900"], "cap": values["caphalf"],
            "eta": values["eta32_relative"], "energy_grid": values["grid32_relative"],
            "energy_range": values["range32_relative"],
        }
    return result


def s4_direct() -> dict[tuple[float, str], tuple[float, str]]:
    """Direct eta / grid / range and dx / dt values measured at J=8 mu."""
    result: dict[tuple[float, str], tuple[float, str]] = {}
    path = AUDIT / "s4_controls.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                F = float(row["F_dc_V_per_nm"])
                component = {"dx048": "dx", "dt005": "dt"}[row["control"]]
                result[(F, component)] = (abs(float(row["relative_change"])),
                                          "s4_controls.csv")
    readouts = jload(AUDIT / "s4_readout_controls.json")
    for row in readouts.get("rows", []):
        F = float(row["F"])
        component = {"eta025": "eta", "grid005": "energy_grid",
                     "emaxplus": "energy_range"}[row["control"]]
        result[(F, component)] = (abs(float(row["relative_change"])),
                                  row["readout_id"])
    return result


def _readout_ok(readout_id: str) -> bool:
    """Check protocol values and that provenance fields are PRESENT.

    It does not re-hash the state/config files.  Content hashing and
    original-byte verification are performed by `manage.py check` and the
    frozen-source manifest, which have different scopes; this helper must not
    be described as a full content re-hash.
    """
    path = PROJECT / "results/bias_v2/readouts" / readout_id / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    thresholds = [float(value) for value in summary.get("thresholds_eV", [])]
    mask = [float(value) for value in summary.get("mask_bounds_nm", [])]
    return (summary.get("execution_status") == "PASS"
            and abs(float(summary.get("eta_eV", -1.0)) - 0.05) < 1e-12
            and abs(float(summary.get("step_eV", -1.0)) - 0.0125) < 1e-12
            and thresholds == [0.5170816, 0.7170816, 0.9170816]
            and mask == [2.0, 4.0]
            and bool(summary.get("state_sha256"))
            and bool(summary.get("run_config_sha256")))


def phase_direct() -> dict[tuple[float, float], tuple[float, str]]:
    """Direct 4->8 change for the exact (F, node) conditions measured.

    Two audited sources are merged (external review 2026-11-08 section 4):
    `phase_validation.json` for the representative F1/8 and F1/32 checks, and
    `R1_pair_audit_v5.csv`, which already contains the F1/4 mu 4->8 pair
    (2.9918675812e-11) under the same eta/dE/mask/threshold protocol.  A row is
    accepted only when the frozen protocol tag is present, the 4+4 phase sets
    are complete, and a provenance SHA is recorded.  Nothing is invented: a
    value is used only at its own (F, node).
    """
    result: dict[tuple[float, float], tuple[float, str]] = {}
    validation = jload(AUDIT / "phase_validation.json")
    for row in validation.get("rows", []):
        if row.get("status") != "PASS" or row.get("relative_4to8_change") in (None, ""):
            continue
        result[(float(row["F"]), float(row["I_over_mu"]))] = (
            abs(float(row["relative_4to8_change"])), "phase_validation.json")
    r1_audit = PROJECT / "results/bias_v3_audit/r1_20261107_v5/R1_pair_audit_v5.csv"
    if r1_audit.exists():
        four = ["phi000", "phi090", "phi180", "phi270"]
        extra = ["phi045", "phi135", "phi225", "phi315"]
        with r1_audit.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                # Actual audit columns: pair_label, source_condition,
                # reference_readout_ids, control_readout_ids, reference_Y,
                # control_Y, signed_change, relative_estimate, coverage, rule.
                if not str(row.get("pair_label", "")).endswith(":cep:4to8"):
                    continue
                if row.get("coverage") != "direct_4to8":
                    continue
                reference = row.get("reference_readout_ids", "")
                control = row.get("control_readout_ids", "")
                if not all(tag in reference for tag in four):
                    continue
                if not all(tag in control for tag in extra):
                    continue
                ref_ids = [item for item in reference.split(";") if item]
                ctl_ids = [item for item in control.split(";") if item]
                if len(ref_ids) != 4 or len(ctl_ids) != 4:
                    continue
                if not all(_readout_ok(readout_id) for readout_id in ref_ids + ctl_ids):
                    continue
                if not row.get("reference_Y") or not row.get("control_Y"):
                    continue
                condition = row["source_condition"]
                F = float(condition.split("_I")[0][1:])
                I = float(condition.split("_I")[1])
                result[(F, I)] = (abs(float(row["relative_estimate"])),
                                  f"R1_pair_audit_v5.csv:{condition}:cep:4to8")
    return result


def phase_envelope() -> tuple[float, str]:
    coverage = jload(stats.ACTIVE_EVIDENCE).get("phase_coverage", {})
    value = coverage.get("saved_2to4_envelope")
    if value is None:
        return 0.0, "missing"
    return float(value), "saved 8/12/20 mu 2->4 envelope (max over conditions)"


def main() -> int:
    r1 = r1_tables()
    r2 = r2_direct()
    s4 = s4_direct()
    direct_phase = phase_direct()
    envelope, envelope_source = phase_envelope()
    response, _ = stats.build_response(include_midpoints=True)
    if response.get("status") != "PASS":
        raise RuntimeError("Response library is incomplete")
    nodes = [float(node) for node in response["nodes"]]
    Imax = max(nodes)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema": "ql-tip-1d-node-error-summary-v5",
                               "nodes": nodes, "envelope": envelope,
                               "envelope_source": envelope_source,
                               "direct_phase_conditions": sorted(
                                   [list(key) for key in direct_phase]),
                               "per_channel": {}}
    for F in [0.0, 1.0]:
        label = f"F{F:g}"
        values = np.asarray(response["by_F"][label]["values"], dtype=float)
        weights_all = {state: linear_weights(nodes, state, Imax)
                       for state in ["bsv", "thermal"]}
        for state in ["bsv", "thermal"]:
            weights = weights_all[state]
            for index, I in enumerate(nodes):
                y = float(values[index])
                row_specs: list[tuple[str, float, str, str, str, str]] = []

                def add(component: str, value: float, evidence: str, source: str,
                        scope: str, reason: str) -> None:
                    row_specs.append((component, value, evidence, source, scope, reason))

                if I == 0.0:
                    for component in CATEGORIES:
                        add(component, r1[f"{label}_I1"][component], "dc_endpoint_mu_transfer",
                            "R1_gate_r1_20261107_v5", "zero-light endpoint only",
                            "DC endpoint has no light-intensity control; mu categories used")
                elif I <= 2.0:
                    for component in CATEGORIES:
                        add(component, r1[f"{label}_I1"][component],
                            "direct_or_transferred_at_mu", "R1_gate_r1_20261107_v5",
                            f"{label} low-intensity nodes", "matched mu categories")
                elif I == 4.0:
                    for component in CATEGORIES:
                        if component == "cep":
                            if (F, I) in direct_phase:
                                add(component, direct_phase[(F, I)][0], "direct_4to8",
                                    direct_phase[(F, I)][1], f"{label} 4 mu node",
                                    "direct 4->8 pair already saved under the same protocol")
                            else:
                                add(component, envelope, "envelope_2to4",
                                    "8/12/20 mu saved readouts", f"{label} mid nodes",
                                    "no direct 4->8 at this node; envelope transferred")
                        elif component in {"mask", "time", "preparation"}:
                            add(component, max(r1[f"{label}_I1"][component],
                                               r1[f"{label}_I4"][component]),
                                "historical_4mu_max_mu", "R1_gate_r1_20261107_v5",
                                f"{label} mid nodes", "preparation/readout categories at 4 mu")
                        else:
                            add(component, r1[f"{label}_I4"][component],
                                "direct_at_4mu", "R1_gate_r1_20261107_v5",
                                f"{label} 4 mu node", "matched 4 mu control")
                elif I == 32.0:
                    for component in CATEGORIES:
                        if component in r2[label]:
                            add(component, r2[label][component], "direct_32mu",
                                "R2_gate_r2_20261106_v2", f"{label} 32 mu node",
                                "direct 32 mu control")
                        elif component == "cep":
                            if (F, I) in direct_phase:
                                add(component, direct_phase[(F, I)][0], "direct_4to8",
                                    direct_phase[(F, I)][1], f"{label} 32 mu node",
                                    "direct 4->8 check at this exact node")
                            else:
                                add(component, envelope, "envelope_2to4",
                                    "8/12/20 mu saved readouts", f"{label} 32 mu node",
                                    "no direct 4->8 at this node; envelope transferred")
                        else:
                            add(component, max(r1[f"{label}_I1"][component],
                                               r1[f"{label}_I4"][component]),
                                "historical_4mu_max_mu", "R1_gate_r1_20261107_v5",
                                f"{label} 32 mu node", "categories not directly controlled at 32 mu")
                else:  # mid nodes 6, 8, 10, 12, 20
                    for component in CATEGORIES:
                        if component == "cep":
                            if (F, I) in direct_phase:
                                add(component, direct_phase[(F, I)][0], "direct_4to8",
                                    direct_phase[(F, I)][1], f"{label} I={I:g} node",
                                    "direct 4->8 check at this exact node")
                            else:
                                add(component, envelope, "envelope_2to4",
                                    "8/12/20 mu saved readouts", f"{label} mid nodes",
                                    "no direct 4->8 at this node; envelope transferred")
                        elif I == 8.0 and (F, component) in s4:
                            add(component, s4[(F, component)][0], "direct_at_J",
                                s4[(F, component)][1], f"{label} J=8 mu node",
                                "direct S4 saved-state control at J")
                        elif component in {"mask", "time", "preparation"}:
                            add(component, max(r1[f"{label}_I1"][component],
                                               r1[f"{label}_I4"][component]),
                                "historical_4mu_max_mu", "R1_gate_r1_20261107_v5",
                                f"{label} mid nodes", "preparation/readout categories")
                        else:
                            value = max(r1[f"{label}_I4"][component], r2[label][component])
                            add(component, value, "transferred_max_4mu_32mu",
                                "R1_gate_r1_20261107_v5 + R2_gate_r2_20261106_v2",
                                f"{label} mid nodes except J=8 mu",
                                "no direct control at this node; conservative per-category max")
                if state != "bsv":
                    continue
                wb = float(weights_all["bsv"][index])
                wt = float(weights_all["thermal"][index])
                for component, value, evidence, source, scope, reason in row_specs:
                    rows.append({
                        "F": f"{F:g}", "I_over_mu": I, "component": component,
                        "value": f"{value:.17g}", "evidence_type": evidence,
                        "source": source, "transfer_scope": scope, "reason": reason,
                        "weight_bsv": f"{wb:.17g}",
                        "weighted_bsv": f"{wb * y * value:.17g}",
                        "weight_thermal": f"{wt:.17g}",
                        "weighted_thermal": f"{wt * y * value:.17g}",
                    })

    # Per-channel sums from the table itself (independent of the region map).
    for F in [0.0, 1.0]:
        label = f"F{F:g}"
        values = np.asarray(response["by_F"][label]["values"], dtype=float)
        for state in ["bsv", "thermal"]:
            weights = linear_weights(nodes, state, Imax)
            y = float(np.dot(weights, values))
            total = 0.0
            no_cep = 0.0
            per_component: dict[str, float] = {}
            for index, I in enumerate(nodes):
                w = float(weights[index])
                for row in rows:
                    if row["F"] != f"{F:g}" or row["I_over_mu"] != I:
                        continue
                    value = float(row["value"])
                    total += w * float(values[index]) * value
                    per_component[row["component"]] = (
                        per_component.get(row["component"], 0.0) + w * float(values[index]) * value)
                    if row["component"] != "cep":
                        no_cep += w * float(values[index]) * value
            summary["per_channel"][f"{state}_F{F:g}"] = {
                "Y": y, "response_error_absolute": total,
                "response_relative": total / y if y else None,
                "relative_if_cep_zero": no_cep / y if y else None,
                "per_component_absolute": per_component,
            }
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    # Register the table in the shared evidence pointer so `load_active()`
    # rejects a stale table.  Read the JSON directly: `load_active()` would
    # validate the old SHA against the just-replaced CSV.
    active = json.loads(stats.ACTIVE_EVIDENCE.read_text(encoding="utf-8"))
    active["node_error_table"] = str(OUT_CSV.relative_to(PROJECT)).replace("\\", "/")
    active["node_error_table_sha256"] = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    active["node_error_table_note"] = (
        "direct evidence used only at the node where it was measured; historical 4 mu "
        "values and the 2->4 envelope transferred to nodes without direct evidence; "
        "no value deleted to pass a gate")
    stats.ACTIVE_EVIDENCE.write_text(
        json.dumps(active, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(OUT_CSV),
                      "per_channel": {key: {"response_relative": round(value["response_relative"], 6),
                                            "if_cep_zero": round(value["relative_if_cep_zero"], 6)}
                                      for key, value in summary["per_channel"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
