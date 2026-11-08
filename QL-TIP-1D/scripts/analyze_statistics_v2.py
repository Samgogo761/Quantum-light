"""Two-bias statistics, midpoint planning and final error propagation.

Inputs are absolute continuation-v2 readouts.  No TDSE, interpolation fit or
new physical model is introduced here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.integrate import quad

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.bias_v1 import sha256
from ql_tip_1d.continuation_v2 import (READOUT_ETA, READOUT_STEP, THRESHOLDS_EV, V2_RESULTS,
                                       project_path, readout_plan, v2_readout_specs)
from ql_tip_1d.statistics_v2 import (bsv_pdf, linear_average, linear_weights, tail_bound,
                                     thermal_pdf, tight_quadrature_error)

REVISION = "r1_20261105_v1"
REPORT_DIR = V2_RESULTS / "reports" / "v3_r2_20261106"
ACTIVE_EVIDENCE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"
AUDIT_DIR = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ERROR_MAP = AUDIT_DIR / "error_component_map.csv"
PHASE_VALIDATION = AUDIT_DIR / "phase_validation.json"
TAIL_AUDIT = AUDIT_DIR / "tail_audit.json"
ERROR_CATEGORIES = ["dx", "dt", "eta", "energy_grid", "energy_range", "box", "cap",
                    "mask", "time", "preparation", "cep"]
INITIAL_NODES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 32.0]
PHASES = [0, 90, 180, 270]


def response_nodes() -> list[float]:
    """Production nodes plus audited conditional tail-extension nodes."""
    active = load_json(ACTIVE_EVIDENCE)
    extra = [float(value) for value in active.get("tail_extension_nodes", [])]
    return sorted(set(INITIAL_NODES + extra))


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON or an empty mapping."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stable_digest(value: Any) -> str:
    """Deterministic SHA256 of a JSON-serialisable audit object."""
    import hashlib
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_active() -> dict[str, Any]:
    """Load and verify the shared active-evidence pointer and both gate hashes."""
    active = load_json(ACTIVE_EVIDENCE)
    if not active:
        raise RuntimeError("active_evidence.json is missing")
    for key, path_key in [("r1_sha256", "r1_gate"), ("r2_sha256", "r2_gate")]:
        path = project_path(active[path_key])
        if not path.exists() or sha256(path) != active[key]:
            raise RuntimeError(f"Active evidence hash mismatch: {path_key}")
    if active.get("error_component_map_sha256"):
        if not ERROR_MAP.exists() or sha256(ERROR_MAP) != active["error_component_map_sha256"]:
            raise RuntimeError("Active evidence hash mismatch: error_component_map")
    return active


def find_readout(run_id: str) -> dict[str, Any] | None:
    """Find the common eta=.05/dE=.0125 readout for a v2 run."""
    candidates = readout_plan(include_new=True) + v2_readout_specs()
    for spec in candidates:
        if (spec["run_id"] == run_id and abs(spec["eta_eV"] - READOUT_ETA) < 1e-12
                and abs(spec["step_eV"] - READOUT_STEP) < 1e-12
                and spec.get("energy_max_mode", "base") == "base"
                and abs(spec.get("mask_start_nm", 2.0) - 2.0) < 1e-12
                and abs(spec.get("mask_end_nm", 4.0) - 4.0) < 1e-12):
            path = V2_RESULTS / "readouts" / spec["readout_id"] / "summary.json"
            return load_json(path) or None
    return None


def run_ids_for_node(F: float, intensity: float, *, midpoint: bool = False) -> list[str]:
    """Return the four production run IDs for one response node."""
    if midpoint:
        return [f"bias_v2_mid_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01" for degrees in PHASES]
    if intensity == 1.0:
        first = f"bias_p1_F{F:.3f}_I1_phi000_r02"
        return [first] + [f"bias_p2_F{F:.3f}_I1_phi{degrees:03d}_r02" for degrees in PHASES[1:]]
    if intensity == 4.0:
        first = f"bias_p2_F{F:.3f}_I4_phi000_r02"
        return [first] + [f"bias_p2_F{F:.3f}_I4_phi{degrees:03d}_r02" for degrees in PHASES[1:]]
    if intensity in {8.0, 32.0}:
        first = f"bias_v2_probe_F{F:g}_I{intensity:g}_phi000_r01"
        return [first] + [f"bias_v2_lib_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01" for degrees in PHASES[1:]]
    return [f"bias_v2_lib_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01" for degrees in PHASES]


def node_values(F: float, intensity: float, *, midpoint: bool = False) -> dict[str, Any] | None:
    """Load four phases and return the absolute main-threshold node yield."""
    run_ids = run_ids_for_node(F, intensity, midpoint=midpoint)
    items = [find_readout(run_id) for run_id in run_ids]
    if any(item is None for item in items):
        return {"F_dc_V_per_nm": F, "I_over_mu": intensity, "status": "INCOMPLETE",
                "missing": [run_id for run_id, item in zip(run_ids, items) if item is None]}
    return {"F_dc_V_per_nm": F, "I_over_mu": intensity, "status": "PASS", "run_ids": run_ids,
            "Y_main": float(np.mean([item["Y_at_E_main"] for item in items])),
            "Y_low": float(np.mean([item["Y_at_E_low"] for item in items])),
            "Y_high": float(np.mean([item["Y_at_E_high"] for item in items])),
            "phase_yields": {str(degrees): float(item["Y_at_E_main"])
                             for degrees, item in zip(PHASES, items)}}


def error_map_rows(F: float, region: str) -> list[dict[str, str]]:
    """Load the audited per-category map; every category must appear exactly once."""
    load_active()
    with ERROR_MAP.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["F"] == f"{F:g}" and row["region"] == region]
    components = [row["component"] for row in rows]
    missing = [name for name in ERROR_CATEGORIES if name not in components]
    if missing or len(components) != len(ERROR_CATEGORIES):
        raise RuntimeError(f"Incomplete/duplicate error map for F{F:g}/{region}: missing={missing}")
    return rows


def response_error_envelope(F: float, intensity: float) -> tuple[float, str]:
    """Sum each numerical error category once, with an explicit region and source."""
    if intensity == 0.0:
        return 0.0, "zero_light_endpoint"
    region = "low" if intensity <= 2.0 else "high" if intensity >= 32.0 else "mid"
    rows = error_map_rows(F, region)
    total = float(sum(abs(float(row["relative_estimate"])) for row in rows))
    sources = "+".join(sorted({row["source_gate"] for row in rows}))
    return total, f"{region}_per_category_once[{sources}]"


def interval_contributions(nodes: list[float], yields: list[float], state: str) -> list[dict[str, Any]]:
    """Integrate the linear response within each adjacent node interval."""
    rows = []
    pdf = bsv_pdf if state == "bsv" else thermal_pdf
    for left, right, yl, yr in zip(nodes[:-1], nodes[1:], yields[:-1], yields[1:]):
        def integrand(x: float) -> float:
            y = yl + (yr - yl) * (x - left) / (right - left)
            return pdf(x) * y
        value, _ = quad(integrand, left, right, epsabs=1e-12, epsrel=1e-10, limit=200)
        rows.append({"left": left, "right": right, "contribution": value})
    return rows


def build_response(include_midpoints: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build response nodes and optional midpoint plan."""
    midpoint_plan = load_json(REPORT_DIR / "midpoint_plan.json")
    mid_nodes = [float(item["midpoint_I_over_mu"]) for item in midpoint_plan.get("intervals", [])]
    nodes = sorted(set(response_nodes() + (mid_nodes if include_midpoints else [])))
    result: dict[str, Any] = {"schema": "continuation-v2-classical-response", "nodes": nodes,
                              "midpoint_I_over_mu": mid_nodes, "by_F": {}, "status": "PASS"}
    csv_rows = []
    for F in [0.0, 1.0]:
        values = []
        errors = []
        for intensity in nodes:
            if intensity == 0.0:
                dc = find_readout(f"bias_p1_dc_F{F:.3f}_r02")
                if dc is None:
                    result["status"] = "INCOMPLETE"
                    values.append(None)
                    errors.append(None)
                    csv_rows.append([F, 0.0, "", "", "", "", "missing_DC_only"])
                    continue
                values.append(float(dc["Y_at_E_main"]))
                dc_error, _origin = response_error_envelope(F, 1.0)
                errors.append(dc_error if dc_error is not None else 0.05)
                csv_rows.append([F, 0.0, float(dc["Y_at_E_main"]), float(dc["Y_at_E_low"]),
                                 float(dc["Y_at_E_high"]), errors[-1], "direct_DC_only_readout"])
                continue
            item = node_values(F, intensity, midpoint=intensity in mid_nodes)
            if item is None or item.get("status") != "PASS":
                result["status"] = "INCOMPLETE"
                values.append(None)
                errors.append(None)
                csv_rows.append([F, intensity, "", "", "", "", "missing"])
                continue
            e_rel, origin = response_error_envelope(F, intensity)
            values.append(item["Y_main"])
            errors.append(e_rel)
            csv_rows.append([F, intensity, item["Y_main"], item["Y_low"], item["Y_high"],
                             e_rel, origin])
        result["by_F"][f"F{F:g}"] = {"values": values, "relative_errors": errors}
    return result, csv_rows


def plan_midpoints() -> int:
    """Choose the two largest weighted intervals and write a real-TDSE midpoint plan."""
    active = load_active()
    if active.get("allowed_stage") not in {"INITIAL_LIBRARY", "FINAL_STATISTICS"}:
        raise RuntimeError(f"Midpoint planning requires INITIAL_LIBRARY, got {active.get('allowed_stage')}")
    response, rows = build_response(include_midpoints=False)
    if response["status"] != "PASS":
        print(json.dumps(response, indent=2))
        return 2
    nodes = response["nodes"]
    # External review 2026-11-07 section C: select by contribution COVERAGE of
    # BSV and thermal for BOTH biases, not by a BSV-only ranking and never by
    # adding a constant times |q-1| to a tiny weighted contribution.
    channels = {(state, F): interval_contributions(
        nodes, response["by_F"][f"F{F:g}"]["values"], state)
        for state in ["bsv", "thermal"] for F in [0.0, 1.0]}
    totals = {key: sum(item["contribution"] for item in rows) or 1.0
              for key, rows in channels.items()}
    keys = [f"{state}_F{F:g}" for state in ["bsv", "thermal"] for F in [0.0, 1.0]]
    scored, coverage_table = [], []
    for index in range(len(nodes) - 1):
        shares = {(state, F): channels[(state, F)][index]["contribution"] / totals[(state, F)]
                  for state in ["bsv", "thermal"] for F in [0.0, 1.0]}
        flat = {f"{state}_F{F:g}": shares[(state, F)]
                for state in ["bsv", "thermal"] for F in [0.0, 1.0]}
        item = {"left": nodes[index], "right": nodes[index + 1],
                "min_channel_share": min(flat.values()),
                "sum_channel_shares": sum(flat.values()), "shares": flat}
        scored.append(item)
        coverage_table.append({"interval_left": item["left"], "interval_right": item["right"],
                               **flat, "min_channel_share": item["min_channel_share"]})
    eligible = [item for item in scored if item["left"] != 0.0]
    eligible.sort(key=lambda item: (item["min_channel_share"],
                                    item["sum_channel_shares"], -item["left"]), reverse=True)
    chosen = []
    for item in eligible:
        if any(not (item["right"] <= chosen_item["left"]
                    or item["left"] >= chosen_item["right"]) for chosen_item in chosen):
            continue
        chosen.append({"left": item["left"], "right": item["right"],
                       "midpoint_I_over_mu": 0.5 * (item["left"] + item["right"]),
                       "min_channel_share": item["min_channel_share"],
                       "coverage_shares": item["shares"]})
        if len(chosen) == 2:
            break
    covered = {key: sum(entry["coverage_shares"][key] for entry in chosen) for key in keys}
    gaps = [key for key, value in covered.items() if value < 0.5]
    plan = {"schema": "continuation-v2-midpoint-plan-v5", "intervals": chosen,
            "selection": ("maximin normalised contribution coverage across "
                          "BSV/thermal x F0/F1; two disjoint real intervals; "
                          "second sort key = sum of channel shares"),
            "nodes_before": nodes, "coverage_table": coverage_table,
            "covered_channel_share": covered, "coverage_gap": gaps,
            "transfer_rule": ("any channel with covered share < 0.5 is bounded for all "
                              "four channels by the node_change / node_ok gate, never "
                              "assumed zero"),
            "source_response_status": response["status"]}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "midpoint_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    with (REPORT_DIR / "classical_response_prelim.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "I_over_mu", "Y_main", "Y_low", "Y_high",
                         "relative_response_error", "error_origin"])
        writer.writerows(rows)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def evaluate_yield_gate(y: float, delta: float, tail: float, quad_err: float,
                        node_change: float, *, coherent: bool = False) -> dict[str, Any]:
    """One deterministic gate shared by JSON, every CSV and the process return code."""
    lo = max(0.0, y - delta)
    hi = min(1.0, y + delta + tail)
    error_bar = delta + tail
    total_relative = error_bar / y if y > 0.0 else math.inf
    lower_positive = lo > 0.0
    tail_ok = True if coherent else (lower_positive and tail <= 0.01 * lo)
    total_ok = total_relative <= 0.05
    node_ok = (node_change <= 0.01 * y) if y > 0.0 else False
    quad_ok = (quad_err <= 0.001 * y) if y > 0.0 else False
    signal_ok = (y > 10.0 * error_bar) if y > 0.0 else False
    ok = bool(lower_positive and tail_ok and total_ok and node_ok and quad_ok and signal_ok)
    return {"Y_lo": lo, "Y_hi": hi, "tail_ok": tail_ok, "lower_positive": lower_positive,
            "total_relative_error": total_relative, "total_5pct_ok": total_ok,
            "node_ok": node_ok, "quadrature_ok": quad_ok, "signal_ok": signal_ok,
            "status": "PASS" if ok else "INCOMPLETE"}


def final_validity(denominator_ok: bool, gate_ok: bool, missing_validation: list[str],
                   d_lo: float | None, preliminary: bool) -> bool:
    """Pure predicate shared by JSON, CSV and the process return code."""
    return bool(denominator_ok and gate_ok and not missing_validation
                and d_lo is not None and d_lo > 0.0 and not preliminary)


def classify_result(valid: bool, d_lo: float | None, d_hi: float | None,
                    preliminary: bool = False) -> tuple[str, str]:
    """Return (program status, scientific label); D≈1 is a complete delivery."""
    if preliminary:
        return "PRELIMINARY", "PRELIMINARY"
    if not valid:
        return "INCOMPLETE", "INCOMPLETE"
    if d_lo is not None and d_hi is not None and (d_lo > 1.0 or d_hi < 1.0):
        return "PASS", "VALID_DIFFERENCE"
    return "PASS", "VALID_UNRESOLVED"


def library_identity(response: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind a validation artifact to the actual current response library.

    External review 2026-11-07 section 4.7: `complete=true` alone is not
    sufficient.  The node set, every common readout summary, the frozen
    protocol and the two gate hashes must all match the library actually used
    by the final statistics.  A stale artifact is rejected, never zeroed.
    """
    response = response if response is not None else build_response(include_midpoints=True)[0]
    nodes = sorted(float(node) for node in response.get("nodes", []))
    plan = load_json(REPORT_DIR / "midpoint_plan.json")
    mids = sorted(float(item["midpoint_I_over_mu"]) for item in plan.get("intervals", []))
    common: dict[str, str] = {}
    for spec in readout_plan(include_new=True) + v2_readout_specs():
        if (abs(spec["eta_eV"] - READOUT_ETA) < 1e-12
                and abs(spec["step_eV"] - READOUT_STEP) < 1e-12
                and spec.get("energy_max_mode", "base") == "base"
                and abs(spec.get("mask_start_nm", 2.0) - 2.0) < 1e-12
                and abs(spec.get("mask_end_nm", 4.0) - 4.0) < 1e-12):
            common.setdefault(spec["run_id"], spec["readout_id"])
    digests: list[str] = []
    missing: list[str] = []
    for F in [0.0, 1.0]:
        for intensity in nodes:
            ids = ([f"bias_p1_dc_F{F:.3f}_r02"] if intensity == 0.0
                   else run_ids_for_node(F, intensity, midpoint=intensity in mids))
            for run_id in ids:
                readout_id = common.get(run_id)
                path = (V2_RESULTS / "readouts" / readout_id / "summary.json"
                        if readout_id else None)
                if path and path.exists():
                    digests.append(f"{run_id}:{readout_id}:{sha256(path)}")
                else:
                    missing.append(run_id)
    if missing or response.get("status") != "PASS":
        return {"status": "INCOMPLETE", "missing_readouts": missing,
                "library_sha256": "", "nodes": nodes, "midpoints": mids}
    protocol = project_path("configs/continuation_v2/protocol_v2.json")
    active = load_active()
    payload = {
        "nodes": nodes, "midpoints": mids, "readout_digests": sorted(digests),
        "protocol_sha256": sha256(protocol) if protocol.exists() else "",
        "r1_sha256": active.get("r1_sha256", ""),
        "r2_sha256": active.get("r2_sha256", ""),
        "readout_eta_eV": READOUT_ETA, "readout_step_eV": READOUT_STEP,
        "thresholds_eV": [float(value) for value in THRESHOLDS_EV],
    }
    return {**payload, "status": "PASS", "library_sha256": stable_digest(payload)}


def expected_representatives() -> dict[str, Any]:
    """Non-empty representative sets required by the final validation gate."""
    phase_selection = load_json(AUDIT_DIR / "phase8_selection.json")
    midpoint_plan = load_json(REPORT_DIR / "midpoint_plan.json")
    return {
        "phase_conditions": [[float(F), float(I)]
                             for F, I in phase_selection.get("conditions", [])],
        "midpoints": [float(item["midpoint_I_over_mu"])
                      for item in midpoint_plan.get("intervals", [])],
        "tail_node_I_over_mu": 32.0,
    }


def tail_snapshot(include_midpoints: bool = True) -> dict[str, Any]:
    """Audit the omitted tail against the CURRENT response library and deltas.

    This deliberately does not read R3_statistics_preliminary.json: that file
    describes the library before midpoints/phase controls and would certify a
    stale lower bound (external review 2026-11-07 section E).
    """
    response, _ = build_response(include_midpoints=include_midpoints)
    if response.get("status") != "PASS":
        return {"status": "INCOMPLETE", "complete": False, "rows": [],
                "reason": "current response library is incomplete"}
    nodes = response["nodes"]
    Imax = max(nodes)
    plan = load_json(REPORT_DIR / "midpoint_plan.json")
    mids = ([float(item["midpoint_I_over_mu"]) for item in plan.get("intervals", [])]
            if include_midpoints else [])
    rows: list[dict[str, Any]] = []
    for F in [0.0, 1.0]:
        values = np.asarray(response["by_F"][f"F{F:g}"]["values"], dtype=float)
        errors = np.asarray(response["by_F"][f"F{F:g}"]["relative_errors"], dtype=float)
        for state in ["bsv", "thermal"]:
            weights = linear_weights(nodes, state, Imax)
            y = linear_average(nodes, values, weights)
            tail = tail_bound(Imax, state)
            quad_err = tight_quadrature_error(nodes, values, state, Imax)
            response_err = float(np.dot(weights, values * errors))
            node_change = 0.0
            for midpoint in mids:
                reduced = [node for node in nodes if node != midpoint]
                reduced_values = [values[nodes.index(node)] for node in reduced]
                w_reduced = linear_weights(reduced, state, Imax)
                y_reduced = linear_average(reduced, reduced_values, w_reduced)
                node_change += abs(y - y_reduced)
            delta = response_err + node_change + quad_err
            gate = evaluate_yield_gate(y, delta, tail, quad_err, node_change)
            lower = gate["Y_lo"]
            allowed = 0.01 * lower
            rows.append({"yield_key": f"{state}_F{F:g}", "F": F, "state": state,
                         "Imax_over_mu": Imax, "Y": y, "delta": delta,
                         "positive_lower_Y": lower, "tail_upper": tail,
                         "one_percent_of_lower": allowed,
                         "tail_ok": bool(gate["lower_positive"] and gate["tail_ok"]),
                         "total_relative_error": gate["total_relative_error"]})
    sufficient = bool(rows) and all(row["tail_ok"] for row in rows)
    return {"schema": "ql-tip-1d-tail-snapshot-v5", "status": "SUFFICIENT" if sufficient
            else "INSUFFICIENT", "complete": sufficient, "Imax_over_mu": Imax,
            "rows": rows, "input_binding": library_identity(response),
            "representatives": expected_representatives()}


def validation_artifacts(response: dict[str, Any] | None = None
                         ) -> tuple[dict[str, Any], list[str]]:
    """Return phase/tail validation plus missing/stale final-gate inputs."""
    phase = load_json(PHASE_VALIDATION)
    tail = load_json(TAIL_AUDIT)
    binding = library_identity(response)
    reps = expected_representatives()
    missing: list[str] = []
    if not phase.get("complete"):
        missing.append("phase_validation.json (4->8 phase check)")
    if not tail.get("complete"):
        missing.append("tail_audit.json (positive lower bound audit)")
    if not reps["phase_conditions"]:
        missing.append("phase8_selection.json (empty representative set)")
    if len(reps["midpoints"]) != 2:
        missing.append("midpoint_plan.json (two real midpoint intervals required)")
    if binding["status"] != "PASS":
        missing.append("current response library is incomplete; validation cannot be bound")
    for name, artifact in [("phase_validation.json", phase), ("tail_audit.json", tail)]:
        if not artifact.get("complete"):
            continue
        recorded = artifact.get("input_binding") or {}
        if not recorded:
            missing.append(f"{name} has no input binding to the current library")
            continue
        if binding["status"] == "PASS" and recorded.get("library_sha256") != binding["library_sha256"]:
            missing.append(f"{name} is stale: response library SHA mismatch")
        # The representative set is stored at the artifact top level (written by
        # v3_stage_v4.step_phase_validation/step_tail); older revisions may also
        # carry it inside input_binding.  Check both, never only the binding.
        reps_recorded = artifact.get("representatives") or recorded.get("representatives") or {}
        if not reps_recorded.get("phase_conditions"):
            missing.append(f"{name} has no representative condition set")
    return {"phase": phase, "tail": tail, "input_binding": binding,
            "representatives": reps}, missing


def final_statistics(preliminary: bool = False) -> int:
    """Integrate BSV/thermal responses, apply midpoint/tail/quadrature errors and D."""
    active = load_active()
    if active.get("allowed_stage") not in {"INITIAL_LIBRARY", "FINAL_STATISTICS"}:
        raise RuntimeError(f"Final statistics require INITIAL_LIBRARY/FINAL_STATISTICS, got {active.get('allowed_stage')}")
    response, rows = build_response(include_midpoints=not preliminary)
    if response["status"] != "PASS":
        print(json.dumps(response, indent=2))
        return 2
    plan = load_json(REPORT_DIR / "midpoint_plan.json")
    mid_nodes = [float(item["midpoint_I_over_mu"]) for item in plan.get("intervals", [])]
    if preliminary:
        mid_nodes = []
    elif len(mid_nodes) != 2:
        print("INCOMPLETE: two real midpoint intervals are required")
        return 2
    validation, missing_validation = ({}, []) if preliminary else validation_artifacts(response)
    nodes = response["nodes"]
    Imax = max(nodes)
    results = {}
    weights_rows = []
    audit_rows = []
    contribution_rows = []
    for F in [0.0, 1.0]:
        values = np.asarray(response["by_F"][f"F{F:g}"]["values"], dtype=float)
        errors = np.asarray(response["by_F"][f"F{F:g}"]["relative_errors"], dtype=float)
        for state in ["coherent", "bsv", "thermal"]:
            if state == "coherent":
                idx = nodes.index(1.0)
                y = float(values[idx])
                rel_err = float(errors[idx])
                tail = 0.0
                quad_err = 0.0
                weights = [1.0 if i == idx else 0.0 for i in range(len(nodes))]
                node_change = 0.0
            else:
                weights = linear_weights(nodes, state, Imax)
                y = linear_average(nodes, values, weights)
                tail = tail_bound(Imax, state)
                quad_err = tight_quadrature_error(nodes, values, state, Imax)
                response_err = float(np.dot(weights, values * errors))
                node_change = 0.0
                for midpoint in mid_nodes:
                    reduced = [node for node in nodes if node != midpoint]
                    reduced_values = [values[nodes.index(node)] for node in reduced]
                    w_reduced = linear_weights(reduced, state, Imax)
                    y_reduced = linear_average(reduced, reduced_values, w_reduced)
                    node_change += abs(y - y_reduced)
                rel_err = (response_err + node_change) / y if y else math.inf
            # External review 2026-11-07 section D: CEP/phase error is charged
            # exactly ONCE, through the `cep` category of the per-category
            # error map.  The old outer `phase_change` term double-counted it.
            # phase_validation.json supplies direct 4->8 evidence only to that
            # CEP category; it contributes no separate outer delta term.
            phase_change = 0.0
            delta = y * rel_err + quad_err
            gate = evaluate_yield_gate(y, delta, tail, quad_err, node_change,
                                       coherent=(state == "coherent"))
            lo, hi = gate["Y_lo"], gate["Y_hi"]
            results[f"{state}_F{F:g}"] = {
                "state": state, "F": F, "Y": y, "Y_lo": lo, "Y_hi": hi,
                "relative_response_error": rel_err, "delta": delta,
                "tail_upper": tail, "tail_ok": gate["tail_ok"],
                "quadrature_error": quad_err, "node_change": node_change,
                "phase_change": phase_change,
                "phase_accounting": ("single CEP category in error_component_map.csv; "
                                     "no outer phase delta; kept at 0 for schema stability"),
                "total_relative_error": gate["total_relative_error"],
                "gate": gate, "weights": [float(value) for value in weights]}
            audit_rows.append([F, state, Imax, y, tail, gate["tail_ok"], node_change, quad_err,
                               phase_change, delta, gate["total_relative_error"],
                               gate["status"], lo, hi])
            for node, weight in zip(nodes, weights):
                weights_rows.append([state, F, node, weight, Imax, tail])
        for state in ["bsv", "thermal"]:
            for interval in interval_contributions(nodes, list(values), state):
                contribution_rows.append([state, F, interval["left"], interval["right"],
                                          interval["contribution"], ""])

    def q_and_d(state: str) -> dict[str, Any]:
        zero = results[f"{state}_F0"]
        one = results[f"{state}_F1"]
        q = one["Y"] / zero["Y"] if zero["Y"] else None
        qlo = one["Y_lo"] / zero["Y_hi"] if zero["Y_hi"] else None
        qhi = one["Y_hi"] / zero["Y_lo"] if zero["Y_lo"] else None
        return {"Y_F0": zero["Y"], "Y_F1": one["Y"], "Q": q, "Q_lo": qlo, "Q_hi": qhi,
                "F0_interval": [zero["Y_lo"], zero["Y_hi"]], "F1_interval": [one["Y_lo"], one["Y_hi"]]}

    q = {state: q_and_d(state) for state in ["coherent", "bsv", "thermal"]}
    b, c = q["bsv"], q["coherent"]
    denominator_ok = all(value > 0.0 for value in [b["F0_interval"][0], c["F0_interval"][0], c["F1_interval"][0]])
    if denominator_ok and None not in [b["Q"], c["Q"]]:
        d = b["Q"] / c["Q"]
        d_lo = (b["F1_interval"][0] * c["F0_interval"][0]) / (b["F0_interval"][1] * c["F1_interval"][1])
        d_hi = (b["F1_interval"][1] * c["F0_interval"][1]) / (b["F0_interval"][0] * c["F1_interval"][0])
    else:
        d = d_lo = d_hi = None
    gate_ok = all(item["gate"]["status"] == "PASS" for item in results.values())
    valid = final_validity(denominator_ok, gate_ok, missing_validation, d_lo, preliminary)
    status, label = classify_result(valid, d_lo, d_hi, preliminary)
    output = {"schema": "continuation-v2-statistics", "revision": REVISION,
              "mode": "PRELIMINARY" if preliminary else "FINAL", "Imax_over_mu": Imax,
              "nodes": nodes, "midpoints": mid_nodes, "yields": results, "Q": q,
              "D": {"value": d, "lo": d_lo, "hi": d_hi, "valid": valid},
              "status": status, "scientific_label": label,
              "gate_ok": gate_ok, "denominator_ok": denominator_ok,
              "missing_validation": missing_validation,
              "validation_binding": validation.get("input_binding", {}),
              "midpoint_coverage_gaps": (load_json(REPORT_DIR / "midpoint_plan.json")
                                         .get("coverage_gap", []) if not preliminary else []),
              "phase_error_policy": ("single CEP category in error_component_map.csv; "
                                     "phase_validation.json supplies 4->8 evidence only; "
                                     "no outer phase delta is charged"),
              "error_component_map": str(ERROR_MAP.relative_to(PROJECT)).replace("\\", "/"),
              "interpretation": ("D interval is a conservative audit interval, not a "
                                 "statistical confidence interval. VALID_UNRESOLVED is a "
                                 "complete delivery when the interval contains 1.")}
    suffix = "_preliminary" if preliminary else ""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"R3_statistics{suffix}.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (REPORT_DIR / f"classical_response{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "I_over_mu", "Y_main", "Y_low", "Y_high",
                         "relative_response_error", "error_origin"])
        writer.writerows(rows)
    with (REPORT_DIR / f"integration_weights{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "F_dc_V_per_nm", "I_over_mu", "weight", "Imax_over_mu", "tail_bound"])
        writer.writerows(weights_rows)
    with (REPORT_DIR / f"statistical_yields{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "F_dc_V_per_nm", "Y_trunc", "Y_error_lo", "Y_error_hi",
                         "tail_upper", "tail_ok", "node_change", "phase_change",
                         "quadrature_error", "delta", "total_relative_error", "status"])
        for item in results.values():
            writer.writerow([item["state"], item["F"], item["Y"], item["Y_lo"], item["Y_hi"],
                             item["tail_upper"], item["tail_ok"], item["node_change"],
                             item["phase_change"], item["quadrature_error"], item["delta"],
                             item["total_relative_error"], item["gate"]["status"]])
    with (REPORT_DIR / f"integration_audit{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F", "state", "Imax_over_mu", "Y_trunc", "tail_upper", "tail_ok",
                         "node_change", "quadrature_error", "phase_change", "delta",
                         "total_relative_error", "status", "Y_lo", "Y_hi"])
        writer.writerows(audit_rows)
    with (REPORT_DIR / f"intensity_contributions{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "F", "interval_left", "interval_right", "weighted_contribution", "cumulative_fraction"])
        writer.writerows(contribution_rows)
    print(json.dumps(output, indent=2, ensure_ascii=False)[:12000])
    return 0 if status == "PASS" else 2


def main() -> int:
    """CLI for midpoint planning or final statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-midpoints", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--preliminary", action="store_true",
                        help="initial-node-only integration, always labelled PRELIMINARY")
    args = parser.parse_args()
    if args.plan_midpoints:
        return plan_midpoints()
    if args.final:
        return final_statistics()
    if args.preliminary:
        return final_statistics(preliminary=True)
    parser.error("choose --plan-midpoints, --final or --preliminary")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
