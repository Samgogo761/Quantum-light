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
REPORT_DIR = V2_RESULTS / "reports" / REVISION
INITIAL_NODES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 32.0]
PHASES = [0, 90, 180, 270]


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON or an empty mapping."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def response_error_envelope(F: float, intensity: float) -> tuple[float, str]:
    """Return a labelled relative response-error envelope for one node."""
    r1 = load_json(REPORT_DIR / "R1_gate_r1_20261105_v1.json")
    conditions = r1.get("conditions", {})
    mu = conditions.get(f"F{F:g}_I1", {}).get("estimated_total_rel_error")
    four = conditions.get(f"F{F:g}_I4", {}).get("estimated_total_rel_error")
    r2 = load_json(REPORT_DIR / "R2_gate_r1_20261105_v1.json")
    high_components = []
    for key in [f"F{F:g}_dx048", f"F{F:g}_dt005", f"F{F:g}_box900", f"F{F:g}_caphalf"]:
        item = r2.get("controls", {}).get(key, {})
        if isinstance(item.get("relative_change"), (int, float)):
            high_components.append(abs(float(item["relative_change"])))
    for mapping_name in ["eta32_relative", "grid32_relative", "range32_relative"]:
        value = r2.get(mapping_name, {}).get(f"F{F:g}")
        if isinstance(value, (int, float)):
            high_components.append(abs(float(value)))
    # Use one representative R1 envelope, not both μ and 4μ totals: their
    # components overlap and adding both double-counts the same numerical classes.
    high = float(sum(high_components) + max(mu or 0.0, four or 0.0))
    if intensity in {0.0}:
        return 0.0, "zero_light_endpoint"
    if intensity in {0.25, 0.5, 1.0, 2.0} and mu is not None:
        return float(mu), "transferred_from_R1_mu"
    if intensity in {4.0} and four is not None:
        return float(four), "direct_R1_F1I4" if F == 1.0 else "direct_R1_F0I4"
    return high, "direct_32mu_controls_plus_one_R1_envelope"


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
    nodes = sorted(set(INITIAL_NODES + (mid_nodes if include_midpoints else [])))
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
    response, rows = build_response(include_midpoints=False)
    if response["status"] != "PASS":
        print(json.dumps(response, indent=2))
        return 2
    nodes = response["nodes"]
    bsv_contrib = {f"F{F:g}": interval_contributions(nodes, response["by_F"][f"F{F:g}"]["values"], "bsv")
                   for F in [0.0, 1.0]}
    scored = []
    for index in range(len(nodes) - 1):
        score = bsv_contrib["F0"][index]["contribution"] + bsv_contrib["F1"][index]["contribution"]
        scored.append((score, nodes[index], nodes[index + 1]))
    scored.sort(reverse=True)
    chosen = []
    for score, left, right in scored:
        if left == 0.0:
            continue
        if any(not (right <= item["left"] or left >= item["right"]) for item in chosen):
            continue
        chosen.append({"left": left, "right": right,
                       "midpoint_I_over_mu": 0.5 * (left + right),
                       "bsv_contribution_both_F": score})
        if len(chosen) == 2:
            break
    plan = {"schema": "continuation-v2-midpoint-plan", "intervals": chosen,
            "selection": "two largest disjoint BSV weighted intervals before midpoint TDSE",
            "nodes_before": nodes}
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


def final_statistics() -> int:
    """Integrate BSV/thermal responses, apply midpoint/tail/quadrature errors and D."""
    response, rows = build_response(include_midpoints=True)
    if response["status"] != "PASS":
        print(json.dumps(response, indent=2))
        return 2
    plan = load_json(REPORT_DIR / "midpoint_plan.json")
    mid_nodes = [float(item["midpoint_I_over_mu"]) for item in plan.get("intervals", [])]
    if len(mid_nodes) != 2:
        print("INCOMPLETE: two real midpoint intervals are required")
        return 2
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
            tail_ok = tail <= 0.01 * y if state != "coherent" else True
            delta = y * rel_err + quad_err
            lo = max(0.0, y - delta)
            hi = min(1.0, y + delta + tail)
            results[f"{state}_F{F:g}"] = {"state": state, "F": F, "Y": y, "Y_lo": lo, "Y_hi": hi,
                                          "relative_response_error": rel_err,
                                          "tail_upper": tail, "tail_ok": tail_ok,
                                          "quadrature_error": quad_err, "node_change": node_change,
                                          "weights": [float(value) for value in weights]}
            audit_rows.append([F, state, Imax, y, tail, tail_ok, node_change, quad_err,
                               y * rel_err, lo, hi])
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
    signal_ok = all(item["Y"] > 10.0 * (item["Y_hi"] - item["Y_lo"]) for item in results.values())
    statistical_ok = all(item["relative_response_error"] <= 0.05 and item["tail_ok"]
                         and item["node_change"] <= 0.01 * item["Y"]
                         and item["quadrature_error"] <= 0.001 * item["Y"]
                         for key, item in results.items() if key.startswith(("bsv_", "thermal_")))
    valid = denominator_ok and signal_ok and statistical_ok and d_lo is not None and d_lo > 0.0
    status = "PASS" if valid else "INCOMPLETE"
    output = {"schema": "continuation-v2-statistics", "revision": REVISION, "Imax_over_mu": Imax,
              "nodes": nodes, "midpoints": mid_nodes, "yields": results, "Q": q,
              "D": {"value": d, "lo": d_lo, "hi": d_hi, "valid": valid},
              "status": status, "signal_ok": signal_ok, "statistical_gate_ok": statistical_ok,
              "denominator_ok": denominator_ok,
              "interpretation": "D interval is a conservative audit interval, not a statistical confidence interval."}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "R3_statistics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    with (REPORT_DIR / "classical_response.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "I_over_mu", "Y_main", "Y_low", "Y_high",
                         "relative_response_error", "error_origin"])
        writer.writerows(rows)
    with (REPORT_DIR / "integration_weights.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "F_dc_V_per_nm", "I_over_mu", "weight", "Imax_over_mu", "tail_bound"])
        writer.writerows(weights_rows)
    with (REPORT_DIR / "statistical_yields.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "F_dc_V_per_nm", "Y_trunc", "Y_error_lo", "Y_error_hi",
                         "tail_upper", "tail_ok", "node_change", "quadrature_error", "status"])
        for key, item in results.items():
            writer.writerow([item["state"], item["F"], item["Y"], item["Y_lo"], item["Y_hi"],
                             item["tail_upper"], item["tail_ok"], item["node_change"],
                             item["quadrature_error"], "PASS" if item["tail_ok"] else "FAIL"])
    with (REPORT_DIR / "integration_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F", "state", "Imax_over_mu", "Y_trunc", "tail_upper", "tail_ok",
                         "node_change", "quadrature_error", "response_error_abs", "Y_lo", "Y_hi"])
        writer.writerows(audit_rows)
    with (REPORT_DIR / "intensity_contributions.csv").open("w", encoding="utf-8", newline="") as handle:
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
    args = parser.parse_args()
    if args.plan_midpoints:
        return plan_midpoints()
    if args.final:
        return final_statistics()
    parser.error("choose --plan-midpoints or --final")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
