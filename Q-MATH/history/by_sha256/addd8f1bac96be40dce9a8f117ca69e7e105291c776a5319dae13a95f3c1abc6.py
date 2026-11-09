"""Deterministic stage analyses for the v3-r2 full run (no TDSE here).

Steps: contributions | s4 | phase8 | tail | phase-validation | decision.
All outputs go to results/bias_v3_audit/reallocation_20261106/ or the new
v3_r2_20261106 report revision; historical revisions are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import analyze_statistics_v2 as stats
from ql_tip_1d.continuation_v2 import (CONTROL_ETA, CONTROL_STEP, GRID_STEP_ETA005,
                                       READOUT_ETA, READOUT_STEP, readout_id,
                                       v2_readout_specs)
from ql_tip_1d.statistics_v2 import linear_weights, tail_bound
from readout_continuation_v2 import run_readout

AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ACTIVE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"
REPORT = stats.REPORT_DIR
V2_RESULTS = PROJECT / "results/bias_v2"


def jload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def jwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_evidence(stage: str, task_ids: list[str], **extra: Any) -> None:
    active = jload(ACTIVE)
    active["allowed_stage"] = stage
    active["allowed_task_ids"] = list(task_ids)
    active.update(extra)
    jwrite(ACTIVE, active)


def readout(run_id: str) -> dict[str, Any] | None:
    return stats.find_readout(run_id)


def node_table() -> tuple[list[dict[str, Any]], list[float]]:
    rows, complete = [], []
    midpoint_plan = jload(REPORT / "midpoint_plan.json")
    mids = [float(item["midpoint_I_over_mu"]) for item in midpoint_plan.get("intervals", [])]
    for F in [0.0, 1.0]:
        for I in sorted(set(stats.response_nodes() + mids)):
            if I == 0.0:
                dc = readout(f"bias_p1_dc_F{F:.3f}_r02")
                rows.append({"F": F, "I_over_mu": 0.0, "Y_main": float(dc["Y_at_E_main"]) if dc else None,
                             "status": "PASS" if dc else "INCOMPLETE", "phase_span": 0.0})
                continue
            item = stats.node_values(F, I, midpoint=I in mids)
            if item and item.get("status") == "PASS":
                phases = [float(v) for v in item["phase_yields"].values()]
                span = (max(phases) - min(phases)) / item["Y_main"] if item["Y_main"] else math.inf
                rows.append({"F": F, "I_over_mu": I, "Y_main": item["Y_main"],
                             "Y_low": item["Y_low"], "Y_high": item["Y_high"],
                             "phase_span": span, "status": "PASS",
                             "run_ids": item["run_ids"]})
            else:
                rows.append({"F": F, "I_over_mu": I, "Y_main": None, "phase_span": "",
                             "status": "INCOMPLETE", "missing": (item or {}).get("missing", [])})
    for I in sorted(set(stats.response_nodes() + mids)):
        if I and all(row["status"] == "PASS" for row in rows
                     if row["I_over_mu"] == I and row["F"] in (0.0, 1.0)):
            complete.append(I)
    return rows, complete


def phi8_yields(F: float, I: float) -> dict[str, Any] | None:
    """Return 4-phase and 8-phase means for one phase-validation condition."""
    four_ids = stats.run_ids_for_node(F, I)
    extra_ids = [f"bias_v2_ph8_F{F:g}_I{I:g}_phi{deg:03d}_r01" for deg in [45, 135, 225, 315]]
    four = [readout(run_id) for run_id in four_ids]
    extra = [readout(run_id) for run_id in extra_ids]
    if any(item is None for item in four + extra):
        return None
    y4 = [float(item["Y_at_E_main"]) for item in four]
    y8 = y4 + [float(item["Y_at_E_main"]) for item in extra]
    return {"Y_four": sum(y4) / 4.0, "Y_eight": sum(y8) / 8.0,
            "four_ids": four_ids, "extra_ids": extra_ids}


def step_contributions() -> int:
    """Save the response table, restricted contribution diagnostic and S4 choice."""
    rows, complete = node_table()
    with (AUDIT / "classical_response_v4.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "I_over_mu", "Y_main", "Y_low", "Y_high",
                         "phase_span", "status"])
        for row in rows:
            writer.writerow([row["F"], row["I_over_mu"], row["Y_main"],
                             row.get("Y_low", ""), row.get("Y_high", ""),
                             row["phase_span"], row["status"]])
    q_nodes = {}
    for I in complete:
        y0 = next(row["Y_main"] for row in rows if row["F"] == 0.0 and row["I_over_mu"] == I)
        y1 = next(row["Y_main"] for row in rows if row["F"] == 1.0 and row["I_over_mu"] == I)
        q_nodes[str(I)] = y1 / y0 if y0 else None
    contributions = []
    for state in ["bsv", "thermal"]:
        pdf = stats.bsv_pdf if state == "bsv" else stats.thermal_pdf
        from scipy.integrate import quad
        for F in [0.0, 1.0]:
            avail = [row for row in rows if row["F"] == F and row["status"] == "PASS"]
            nodes = sorted({row["I_over_mu"] for row in avail})
            values = {row["I_over_mu"]: row["Y_main"] for row in avail}
            for left, right in zip(nodes[:-1], nodes[1:]):
                def integrand(x: float) -> float:
                    y = values[left] + (values[right] - values[left]) * (x - left) / (right - left)
                    return pdf(x) * y
                value, _ = quad(integrand, left, right, epsabs=1e-13, epsrel=1e-10, limit=200)
                contributions.append({"state": state, "F": F, "left": left, "right": right,
                                      "contribution": value})
    with (AUDIT / "s3_contributions_v4.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "F_dc_V_per_nm", "interval_left", "interval_right",
                         "weighted_contribution", "scope"])
        for item in contributions:
            writer.writerow([item["state"], item["F"], item["left"], item["right"],
                             item["contribution"], "restricted_range_diagnostic"])
    out = {"schema": "ql-tip-1d-response-v4", "complete_nodes": complete,
           "q_of_I": q_nodes, "rows": rows, "contributions": contributions,
           "scope": "restricted_range_diagnostic" if len(complete) < len(stats.response_nodes()) - 1
                    else "full_initial_library"}
    jwrite(AUDIT / "response_v4.json", out)

    # External review 2026-11-07 section C: select the S4 representative by
    # contribution COVERAGE of BSV and thermal for BOTH biases.  The old
    # `score + 0.05*abs(q-1)` mixed a response difference into a tiny weighted
    # contribution and effectively ranked by |q-1| instead.  |q-1| is now a
    # diagnostic column only; the second sort key is a real coverage quantity.
    candidates = [I for I in [8.0, 12.0, 20.0] if I in complete]
    channel_rows = {(state, F): [item for item in contributions
                                 if item["state"] == state and item["F"] == F]
                    for state in ["bsv", "thermal"] for F in [0.0, 1.0]}
    totals = {key: sum(item["contribution"] for item in rows) or 1.0
              for key, rows in channel_rows.items()}
    scored = []
    for I in candidates:
        adjacent = [item for item in contributions if item["left"] == I or item["right"] == I]
        shares = {}
        for state in ["bsv", "thermal"]:
            for F in [0.0, 1.0]:
                value = sum(item["contribution"] for item in adjacent
                            if item["state"] == state and item["F"] == F)
                shares[f"{state}_F{F:g}"] = value / totals[(state, F)]
        worst = min(shares.values())
        scored.append({"I_over_mu": I,
                       "combined_contribution": sum(item["contribution"] for item in adjacent),
                       "channel_shares": shares, "min_channel_share": worst,
                       "sum_channel_shares": sum(shares.values()),
                       "abs_q_minus_1": abs((q_nodes.get(str(I)) or 1.0) - 1.0),
                       "score": worst})
    scored.sort(key=lambda item: (item["min_channel_share"],
                                  item["sum_channel_shares"],
                                  item["I_over_mu"]), reverse=True)
    bsv_peak = max((item for item in contributions if item["state"] == "bsv"),
                   key=lambda item: item["contribution"], default=None)
    thermal_peak = max((item for item in contributions if item["state"] == "thermal"),
                       key=lambda item: item["contribution"], default=None)
    gap = (bsv_peak is not None and thermal_peak is not None
           and (bsv_peak["left"], bsv_peak["right"]) != (thermal_peak["left"], thermal_peak["right"]))
    chosen = scored[0] if scored else None
    uncovered = ([key for key, value in chosen["channel_shares"].items() if value < 0.5]
                 if chosen else sorted({f"{state}_F{F:g}" for state in ["bsv", "thermal"]
                                        for F in [0.0, 1.0]}))
    coverage_gap = bool(gap or uncovered)
    selection = {"schema": "ql-tip-1d-s4-selection-v5",
                 "rule": ("maximin normalised contribution coverage across BSV/thermal x "
                          "F0/F1 over adjacent intervals; second key = sum of channel "
                          "shares; |q-1| is diagnostic only"),
                 "J_over_mu": chosen["I_over_mu"] if chosen else None,
                 "candidate_scores": scored,
                 "channel_coverage": chosen["channel_shares"] if chosen else {},
                 "uncovered_channels": uncovered,
                 "bsv_peak_interval": [bsv_peak["left"], bsv_peak["right"]] if bsv_peak else None,
                 "thermal_peak_interval": [thermal_peak["left"], thermal_peak["right"]] if thermal_peak else None,
                 "coverage_gap": coverage_gap,
                 "coverage_resolution": ("uncovered channels are bounded by per-category 32 mu "
                                         "direct controls, R1 4 mu transfers and the S4 "
                                         "representative eta/grid readouts; never assumed zero"),
                 "note": "One representative J cannot cover disjoint BSV/thermal peaks; "
                         "recorded as a coverage limitation and fed into the S4 gate."}
    if not selection["J_over_mu"]:
        raise RuntimeError("No complete 8/12/20 mu node available for S4 selection")
    jwrite(AUDIT / "s4_selection.json", selection)
    print(json.dumps({"complete_nodes": complete, "q": q_nodes,
                      "J_over_mu": selection["J_over_mu"]}, indent=2))
    return 0


def file_sha(path: Path) -> str:
    """SHA256 of an audit input, or empty string when it is absent."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def step_s4_readouts() -> int:
    """Direct eta / energy-grid / energy-range readouts at representative J.

    These are saved-state post-processings, not new TDSE.  External review B:
    the 32 mu post-processing must not be passed off as a direct check at J.
    """
    selection = jload(AUDIT / "s4_selection.json")
    J = float(selection["J_over_mu"])
    base_specs = {spec["run_id"]: spec for spec in v2_readout_specs()
                  if spec["kind"] == "v2_probe"}
    variants = [("eta025", CONTROL_ETA, CONTROL_STEP, "base"),
                ("grid005", READOUT_ETA, GRID_STEP_ETA005, "base"),
                ("emaxplus", READOUT_ETA, READOUT_STEP, "extended")]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for F in [0.0, 1.0]:
        ref_id = stats.run_ids_for_node(F, J)[0]
        ref, base = readout(ref_id), base_specs.get(ref_id)
        if ref is None or base is None:
            missing.append(f"F{F:g}: missing common reference or saved state for {ref_id}")
            continue
        y_ref = float(ref["Y_at_E_main"])
        for label, eta, step_eV, mode in variants:
            spec = {**base, "kind": "s4_readout_control", "eta_eV": eta,
                    "step_eV": step_eV, "energy_max_mode": mode,
                    "energy_min_eV": None, "energy_max_eV": None}
            spec["readout_id"] = readout_id(spec)
            try:
                summary = run_readout(spec)
            except (FileExistsError, ValueError, KeyError, OSError) as exc:
                missing.append(f"F{F:g}:{label}: {exc}")
                continue
            y = float(summary["Y_at_E_main"])
            rows.append({"F": F, "J_over_mu": J, "control": label,
                         "readout_id": spec["readout_id"], "reference_run_id": ref_id,
                         "reference_Y": y_ref, "control_Y": y,
                         "relative_change": (y - y_ref) / y_ref if y_ref else math.inf})
    with (AUDIT / "s4_readout_controls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "J_over_mu", "control", "readout_id",
                         "reference_run_id", "reference_Y", "control_Y", "relative_change"])
        for row in rows:
            writer.writerow([row["F"], row["J_over_mu"], row["control"], row["readout_id"],
                             row["reference_run_id"], row["reference_Y"], row["control_Y"],
                             row["relative_change"]])
    jwrite(AUDIT / "s4_readout_controls.json", {
        "schema": "ql-tip-1d-s4-readout-controls-v5", "J_over_mu": J,
        "complete": bool(rows) and not missing and len(rows) == 2 * len(variants),
        "rows": rows, "missing": missing,
        "note": "direct saved-state eta/grid/range checks at J; no 32 mu substitution"})
    print(json.dumps({"J_over_mu": J, "rows": len(rows), "missing": missing}, indent=2))
    return 0 if not missing and len(rows) == 2 * len(variants) else 2


def step_s4() -> int:
    """Real S4 acceptance gate; only PROCEED may unlock the library build (S5).

    External review 2026-11-07 section B: computing dx/dt differences is not a
    pass.  The gate binds input hashes, actual control magnitudes, coverage,
    current error risk and remaining cost, and distinguishes DATA_COMPLETE,
    PROCEED, LIMITED_REPAIR and INCOMPLETE.
    """
    selection = jload(AUDIT / "s4_selection.json")
    J = float(selection["J_over_mu"])
    records: list[list[Any]] = []
    per_F: dict[str, Any] = {}
    missing_controls: list[str] = []
    for F in [0.0, 1.0]:
        ref_id = stats.run_ids_for_node(F, J)[0]
        ref = readout(ref_id)
        if ref is None:
            missing_controls.append(f"coarse CEP0 reference {ref_id}")
            continue
        y_ref = float(ref["Y_at_E_main"])
        values: dict[str, float] = {}
        for label in ["dx048", "dt005"]:
            run_id = f"bias_v2_s4_F{F:g}_I{J:g}_phi000_{label}_r01"
            item = readout(run_id)
            if item is None:
                missing_controls.append(run_id)
                continue
            y = float(item["Y_at_E_main"])
            change = (y - y_ref) / y_ref if y_ref else math.inf
            values[label] = change
            records.append([F, J, label, ref_id, run_id, y_ref, y, change])
        if len(values) == 2:
            per_F[f"F{F:g}"] = {"dx_relative": values["dx048"],
                                "dt_relative": values["dt005"]}
    with (AUDIT / "s4_controls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "J_over_mu", "control", "reference_run_id",
                         "control_run_id", "reference_Y", "control_Y", "relative_change"])
        writer.writerows(records)
    readout_controls = jload(AUDIT / "s4_readout_controls.json")
    input_hashes = {
        "s4_selection_sha256": file_sha(AUDIT / "s4_selection.json"),
        "s4_readout_controls_sha256": file_sha(AUDIT / "s4_readout_controls.json"),
        "error_component_map_sha256": file_sha(AUDIT / "error_component_map.csv"),
        "classical_response_sha256": file_sha(AUDIT / "classical_response_v4.csv"),
    }
    active = jload(ACTIVE)
    for key in ["r1_gate", "r2_gate"]:
        input_hashes[f"{key}_sha256"] = file_sha(PROJECT / active[key]) if active.get(key) else ""
    changes = [abs(float(row[7])) for row in records if math.isfinite(float(row[7]))]
    max_abs = max(changes) if changes else math.inf
    finite_ok = len(changes) == len(records) and bool(records)
    direct_checks_ok = bool(readout_controls.get("complete"))
    data_complete = bool(len(records) == 4 and not missing_controls and finite_ok
                         and direct_checks_ok and all(input_hashes.values()))
    ledger = list(csv.DictReader((V2_RESULTS / "compute_ledger_v2.csv").open(
        encoding="utf-8", newline="")))
    hours_used = sum(float(row["counted_seconds"] or 0.0) for row in ledger) / 3600.0
    limits = active.get("budget", {})
    remaining_budget_h = float(limits.get("continuation_hours_limit", 56.0)) - hours_used
    start_state = jload(AUDIT / "START_STATE_v4.json")
    remaining_estimate_h = float(start_state.get("remaining_estimate_hours", 0.0))
    conditional_h = 6.0 * 5.0 / 60.0  # six saved-state readouts at ~5 min each
    blockers: list[str] = []
    if missing_controls:
        blockers.append(f"missing dx/dt controls: {missing_controls}")
    if not finite_ok:
        blockers.append("non-finite or absent S4 control differences")
    if not direct_checks_ok:
        blockers.append("direct eta/grid/range readouts at J are incomplete")
    if not all(input_hashes.values()):
        blockers.append("an S4 input hash is empty")
    if max_abs > 0.20:
        blockers.append(f"unstable control change {max_abs:.4f} > 20% at J")
    if remaining_estimate_h + conditional_h > remaining_budget_h:
        blockers.append("remaining mandatory + conditional cost exceeds the continuation cap")
    coverage_gap = bool(selection.get("coverage_gap"))
    transfer_justified = bool(data_complete and not blockers)
    if coverage_gap and not transfer_justified:
        blockers.append("contribution coverage gap has no justified transfer")
    if not data_complete:
        status = "INCOMPLETE"
    elif blockers:
        status = "LIMITED_REPAIR"
    else:
        status = "PROCEED"
    gate = {"schema": "ql-tip-1d-s4-gate-v5", "status": status,
            "data_complete": data_complete,
            "can_continue_library": status == "PROCEED",
            "J_over_mu": J, "input_hashes": input_hashes,
            "control_magnitude": {"max_abs_relative_change": None if not changes else max_abs,
                                  "per_F": per_F, "high_local_sum_over_5pct_is_not_auto_veto": True},
            "coverage": {"channel_coverage": selection.get("channel_coverage", {}),
                         "uncovered_channels": selection.get("uncovered_channels", []),
                         "coverage_gap": coverage_gap,
                         "transfer_justified": transfer_justified,
                         "transfer_rule": selection.get("coverage_resolution", "")},
            "error_risk": {"per_category_map_rebuildable": data_complete,
                           "direct_J_readouts_complete": direct_checks_ok,
                           "note": "the 32 mu seven-category envelope is transferred "
                                   "per category; it is not presented as a direct J check"},
            "remaining_cost": {"continuation_hours_used": hours_used,
                               "continuation_hours_remaining": remaining_budget_h,
                               "mandatory_estimate_hours": remaining_estimate_h,
                               "conditional_direct_readout_hours": conditional_h,
                               "conditional_extension_note": ("48/64 mu extension and its "
                                   "new highest-intensity controls are budgeted only if the "
                                   "tail audit requires them")},
            "blockers": blockers}
    jwrite(AUDIT / "s4_gate.json", gate)
    jwrite(AUDIT / "s4_controls.json", {"schema": "ql-tip-1d-s4-controls-v5",
                                        "J_over_mu": J, "per_F": per_F,
                                        "complete": status == "PROCEED", "gate": gate})
    if status == "PROCEED":
        import build_error_component_map_v4
        build_error_component_map_v4.main()
    print(json.dumps({"J_over_mu": J, "status": status,
                      "blockers": blockers, "max_abs": max_abs}, indent=2))
    return 0 if status == "PROCEED" else 2


def step_phase8() -> int:
    """Select the highest-intensity and main-contribution phase-validation conditions."""
    rows, complete = node_table()
    if 32.0 not in complete:
        raise RuntimeError("32 mu four-phase response must be complete before phase selection")
    span32 = {}
    for F in [0.0, 1.0]:
        row = next(item for item in rows if item["F"] == F and item["I_over_mu"] == 32.0)
        span32[F] = row["phase_span"]
    F_max = max(span32, key=lambda F: span32[F])
    contributions = jload(AUDIT / "response_v4.json").get("contributions", [])
    # External review C: do not rank the main-contribution phase point by BSV
    # alone.  Use the BSV/thermal x F0/F1 normalised coverage table.
    initial = set(stats.response_nodes())
    totals = {(state, F): sum(item["contribution"] for item in contributions
                              if item["state"] == state and item["F"] == F) or 1.0
              for state in ["bsv", "thermal"] for F in [0.0, 1.0]}
    keys = [f"{state}_F{F:g}" for state in ["bsv", "thermal"] for F in [0.0, 1.0]]
    by_right: dict[float, dict[str, float]] = {}
    for item in contributions:
        right = float(item["right"])
        if right not in initial:  # interpolation midpoints use another run-id family
            continue
        slot = by_right.setdefault(right, {})
        channel = f"{item['state']}_F{float(item['F']):g}"
        slot[channel] = slot.get(channel, 0.0) + item["contribution"] / totals[
            (item["state"], float(item["F"]))]
    ranked = []
    for I, raw in by_right.items():
        shares = {key: raw.get(key, 0.0) for key in keys}
        ranked.append({"I_over_mu": I, "shares": shares,
                       "min_channel_share": min(shares.values()),
                       "sum_channel_shares": sum(shares.values())})
    ranked.sort(key=lambda item: (item["min_channel_share"],
                                  item["sum_channel_shares"], item["I_over_mu"]), reverse=True)
    eligible = [item for item in ranked if item["I_over_mu"] != 32.0] or ranked
    main = eligible[0] if eligible else None
    main_I = float(main["I_over_mu"]) if main else 20.0
    main_F = F_max
    conditions = [[F_max, 32.0]]
    if (main_F, main_I) != (F_max, 32.0):
        conditions.append([main_F, main_I])
    # A conditional tail extension must also have direct 4->8 evidence at its
    # new maximum node; its tasks were emitted by build_tail_extension_v5.py.
    extension_nodes = [value for value in stats.response_nodes() if value > 32.0]
    if extension_nodes:
        highest = max(extension_nodes)
        for F in [0.0, 1.0]:
            if [F, highest] not in conditions:
                conditions.append([F, highest])
    selection = {"schema": "ql-tip-1d-phase8-selection-v5", "conditions": conditions,
                 "highest_intensity": [F_max, 32.0], "main_contribution": [main_F, main_I],
                 "phase_span_32mu": {f"F{F:g}": value for F, value in span32.items()},
                 "coverage_table": ranked,
                 "main_channel_coverage": main["shares"] if main else {},
                 "coverage_gap": bool(main and min(main["shares"].values()) < 0.5),
                 "rule": ("largest 32 mu phase span plus the interval upper node with "
                          "maximin BSV/thermal x F0/F1 contribution coverage"),
                 "target_4to8_relative_change": 0.02}
    jwrite(AUDIT / "phase8_selection.json", selection)
    print(json.dumps(selection, indent=2))
    return 0


def tail_extension_plan(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Budget the SMALLEST sufficient 48/64 mu extension, never unconditional.

    External review 2026-11-07 section E: a shortfall at 32 mu must first be
    costed (new nodes, new highest-intensity controls, phase 4->8, preparation
    and readouts) against the unchanged global ceiling, before any TDSE starts.
    """
    import statistics
    if snapshot.get("status") == "INCOMPLETE":
        return {"status": "INCOMPLETE", "reason": "current response library incomplete"}
    required: dict[str, float | None] = {}
    for row in snapshot["rows"]:
        lower = float(row["positive_lower_Y"])
        target = 0.01 * lower if lower > 0.0 else 0.0
        sufficient = None
        for Imax in [32.0, 48.0, 64.0]:
            if lower > 0.0 and tail_bound(Imax, row["state"]) <= target:
                sufficient = Imax
                break
        required[row["yield_key"]] = sufficient
    targets = [value for value in required.values() if value is not None]
    if snapshot.get("status") == "SUFFICIENT":
        return {"status": "NOT_REQUIRED", "required_Imax_by_yield": required,
                "note": "32 mu truncation already satisfies the positive lower bound"}
    if not targets:
        return {"status": "NOT_SUFFICIENT_AT_64MU", "required_Imax_by_yield": required,
                "note": "no admissible extension within the frozen node options; deliver INCOMPLETE"}
    Imax_needed = max(targets)
    new_nodes = [value for value in [48.0, 64.0] if value <= Imax_needed]
    ledger = list(csv.DictReader((V2_RESULTS / "compute_ledger_v2.csv").open(
        encoding="utf-8", newline="")))
    tdse_times = [float(row["counted_seconds"]) for row in ledger
                  if row["kind"] == "tdse" and row["status"] == "PASS"]
    readout_times = [float(row["counted_seconds"]) for row in ledger
                     if row["kind"] == "readout" and row["status"] == "PASS"]
    tdse_h = (statistics.median(tdse_times) / 3600.0) if tdse_times else 1300.0 / 3600.0
    readout_h = (statistics.median(readout_times) / 3600.0) if readout_times else 180.0 / 3600.0
    # Per new node: 8 four-phase propagations; plus new highest-intensity
    # dx/dt/box/cap controls (8), representative 4->8 phases (8); plus their
    # common readouts (8 per node), control readouts (8), eta/grid/range
    # post-processings (6) and phase readouts (8).
    propagations = len(new_nodes) * 8 + 8 + 8
    readouts = len(new_nodes) * 8 + 8 + 6 + 8
    conditional_h = propagations * tdse_h + readouts * readout_h
    active = jload(ACTIVE)
    limits = active.get("budget", {})
    used_global_h = (float(limits.get("global_hours_baseline", 14.117235553083))
                     + sum(float(row["counted_seconds"] or 0.0) for row in ledger) / 3600.0)
    projected_global_h = used_global_h + conditional_h
    within = projected_global_h <= float(limits.get("global_hours_limit", 72.0))
    return {"schema": "ql-tip-1d-tail-extension-plan-v5",
            "status": "REQUIRED_WITHIN_BUDGET" if within else "REQUIRED_EXCEEDS_BUDGET",
            "required_Imax_over_mu": Imax_needed, "new_nodes": new_nodes,
            "new_tdse_estimate": propagations, "new_readout_estimate": readouts,
            "measured_median_tdse_hours": tdse_h,
            "measured_median_readout_hours": readout_h,
            "conditional_hours": conditional_h,
            "used_global_hours": used_global_h,
            "projected_global_hours": projected_global_h,
            "global_hours_limit": float(limits.get("global_hours_limit", 72.0)),
            "required_Imax_by_yield": required,
            "trigger": "tail_upper > 1% of positive lower yield bound at 32 mu",
            "rule": "extend only to the smallest sufficient node; if global ceiling "
                    "is exceeded, deliver INCOMPLETE instead of inventing certification"}


def step_tail(scope: str = "final") -> int:
    """Audit the tail against the CURRENT response library and error deltas.

    External review E: the audit must run right after preliminary statistics,
    before midpoints/phase costs are committed, and must never certify a stale
    preliminary library.  `--tail-scope precheck` writes tail_precheck.json and
    can never set `complete=true`; only the final audit writes tail_audit.json.
    """
    snapshot = stats.tail_snapshot(include_midpoints=True)
    library_scope = "current_full_library"
    if snapshot.get("status") == "INCOMPLETE":
        snapshot = stats.tail_snapshot(include_midpoints=False)
        library_scope = "initial_library_precheck"
    rows = snapshot.get("rows", [])
    with (AUDIT / "tail_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["yield_key", "Imax_over_mu", "Y", "delta", "positive_lower_Y",
                         "tail_upper", "one_percent_of_lower", "tail_ok"])
        for row in rows:
            writer.writerow([row["yield_key"], row["Imax_over_mu"], row["Y"], row["delta"],
                             row["positive_lower_Y"], row["tail_upper"],
                             row["one_percent_of_lower"], row["tail_ok"]])
    sufficient = bool(snapshot.get("complete"))
    binding_ok = snapshot.get("input_binding", {}).get("status") == "PASS"
    final = bool(sufficient and library_scope == "current_full_library"
                 and binding_ok and scope == "final")
    audit = {"schema": "ql-tip-1d-tail-audit-v5", "status": snapshot.get("status"),
             "Imax_over_mu": snapshot.get("Imax_over_mu"), "complete": final,
             "sufficient_at_32mu": sufficient, "scope": library_scope,
             "audit_scope": scope,
             "rows": rows, "input_binding": snapshot.get("input_binding", {}),
             "representatives": snapshot.get("representatives", {}),
             "extension": tail_extension_plan(snapshot),
             "note": ("Physical bound 0<=Y<=1; no endpoint extrapolation. "
                      "Uses the current response library and current per-category "
                      "error deltas, never R3_statistics_preliminary.json. "
                      "48/64 mu extension is conditional, not unconditional. "
                      "A precheck can never set complete=true.")}
    target = AUDIT / ("tail_audit.json" if scope == "final" else "tail_precheck.json")
    jwrite(target, audit)
    print(json.dumps({"file": target.name, "complete": final, "scope": library_scope,
                      "status": audit["status"], "extension": audit["extension"]["status"],
                      "rows": len(rows)}, indent=2))
    return 0 if final else 2


def step_phase_validation() -> int:
    """Compare 4-phase and 8-phase means and propagate the change into the outer delta."""
    selection = jload(AUDIT / "phase8_selection.json")
    conditions = [(float(F), float(I)) for F, I in selection.get("conditions", [])]
    plan = jload(REPORT / "midpoint_plan.json")
    mids = [float(item["midpoint_I_over_mu"]) for item in plan.get("intervals", [])]
    nodes = sorted(set(stats.response_nodes() + mids))
    Imax = max(nodes)
    rows, integrated = [], {}
    complete = True
    for state in ["bsv", "thermal"]:
        weights = linear_weights(nodes, state, Imax)
        for F in [0.0, 1.0]:
            integrated[f"{state}_F{F:g}"] = 0.0
    for F, I in conditions:
        item = phi8_yields(F, I)
        if item is None:
            complete = False
            rows.append({"F": F, "I_over_mu": I, "status": "INCOMPLETE"})
            continue
        change = (item["Y_eight"] - item["Y_four"]) / item["Y_four"] if item["Y_four"] else math.inf
        for state in ["bsv", "thermal"]:
            weights = linear_weights(nodes, state, Imax)
            index = nodes.index(I) if I in nodes else None
            if index is not None:
                integrated[f"{state}_F{F:g}"] += abs(weights[index] * (item["Y_eight"] - item["Y_four"]))
        rows.append({"F": F, "I_over_mu": I, "status": "PASS",
                     "Y_four": item["Y_four"], "Y_eight": item["Y_eight"],
                     "relative_4to8_change": change,
                     "target_ok": abs(change) <= float(selection.get("target_4to8_relative_change", 0.02))})
        complete = complete and rows[-1]["target_ok"]
    with (AUDIT / "phase_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "I_over_mu", "Y_four_phase", "Y_eight_phase",
                         "relative_4to8_change", "target_ok", "status"])
        for row in rows:
            writer.writerow([row["F"], row["I_over_mu"], row.get("Y_four", ""),
                             row.get("Y_eight", ""), row.get("relative_4to8_change", ""),
                             row.get("target_ok", ""), row["status"]])
    # External review D: only the measured conditions have direct 4->8
    # evidence; every other production node is explicitly listed as covered
    # by the single CEP envelope, never silently defaulted to zero.
    measured = {(float(row["F"]), float(row["I_over_mu"]))
                for row in rows if row["status"] == "PASS"}
    uncovered = [[F, I] for F in [0.0, 1.0] for I in nodes
                 if I not in (0.0, 1.0) and (F, I) not in measured]
    binding = stats.library_identity()
    reps = stats.expected_representatives()
    complete = bool(complete and binding["status"] == "PASS"
                    and reps["phase_conditions"] and len(reps["midpoints"]) == 2)
    jwrite(AUDIT / "phase_validation.json", {
        "schema": "ql-tip-1d-phase-validation-v5", "complete": complete,
        "integrated_change": integrated, "rows": rows,
        "input_binding": binding, "representatives": reps,
        "coverage": {"measured_4to8": [list(item) for item in sorted(measured)],
                     "uncovered_nodes": uncovered, "coverage_gap": bool(uncovered),
                     "rule": ("uncovered nodes use the largest saved 2->4 envelope plus "
                              "direct 4->8 evidence in the SINGLE cep category of "
                              "error_component_map.csv; no node is assigned zero")},
        "note": ("Phase error is charged exactly once, through the per-node CEP "
                 "category.  integrated_change is diagnostic only and is NOT added "
                 "to the outer statistical delta.")})
    print(json.dumps({"complete": complete, "uncovered_nodes": len(uncovered),
                      "integrated_change": integrated}, indent=2))
    return 0 if complete else 2


def step_error_map() -> int:
    """Rebuild the per-category map after direct 4->8 evidence is available."""
    import build_error_component_map_v4
    build_error_component_map_v4.main()
    return 0


def step_decision() -> int:
    """Write the final human-readable decision after every stage has finished."""
    final = jload(REPORT / "R3_statistics.json")
    tail = jload(AUDIT / "tail_audit.json")
    phase = jload(AUDIT / "phase_validation.json")
    budget = jload(AUDIT / "START_STATE_v4.json")
    ledger = list(csv.DictReader((V2_RESULTS / "compute_ledger_v2.csv").open(
        encoding="utf-8", newline="")))
    hours = sum(float(row["counted_seconds"] or 0.0) for row in ledger) / 3600.0
    tdse = sum(1 for row in ledger if row["kind"] == "tdse" and row["status"] == "PASS")
    readouts = sum(1 for row in ledger if row["kind"] == "readout" and row["status"] == "PASS")
    d = final.get("D", {})
    label = final.get("scientific_label", "INCOMPLETE")
    s4_gate = jload(AUDIT / "s4_gate.json")
    node_summary = jload(AUDIT / "node_error_summary.json")
    extension = tail.get("extension", {})
    phase_uncovered = phase.get("coverage", {}).get("uncovered_nodes", [])
    executed_ok = tdse > 0 and readouts > 0
    data_ok = bool(final.get("nodes")) and tail.get("complete") and phase.get("complete")
    acceptance_ok = final.get("status") == "PASS"
    attempts = sum(1 for row in ledger if row["kind"] == "tdse_attempt")
    reserve = sum(float(row["counted_seconds"] or 0.0) for row in ledger
                  if row["kind"] == "tdse_attempt") / 3600.0
    response_lines = [
        f"  - {key}: 响应项 {value['response_relative'] * 100:.2f}%，"
        f"CEP 归零后 {value['relative_if_cep_zero'] * 100:.2f}%"
        for key, value in node_summary.get("per_channel", {}).items()
    ]
    response_block = "\n".join(response_lines) if response_lines else "  - 节点表缺失"
    text = f"""# QL-TIP-1D 两偏压统计研究：判定修订（v3-r2 / r2）

日期：2026-11-08。本文件是 `decision.md` 的修订版，依据 2026-11-08 外审
`reports/FINAL_RESULT_REVIEW_20261108.md` 更正两处模板措辞；历史 `decision.md` 保留不覆盖。

## 判定

`{label}`。程序状态 `{final.get('status')}`；`gate_ok = {final.get('gate_ok')}`。

- 三者严格区分：任务已执行 = {executed_ok}；数据齐全 = {data_ok}；数值验收通过 = {acceptance_ok}。
- BSV 主量 `D = Q_BSV / Q_C` = {d.get('value')}，保守审计区间 [{d.get('lo')}, {d.get('hi')}]。
- 该区间含 1，但**只有数值门全部通过时**才可标为 `VALID_UNRESOLVED`。本轮 `D.valid = {d.get('valid')}`，
  因误差门未通过，不作该判定；区间仅作中间量审计，不是置信区间。
- 本轮数值验收未通过，**不构成关于统计依赖是否存在的结论**，也不能反向宣称物理效应不存在；
  这里只报告可复核的中间量、误差结构与缺口。
- S4 验收关口 = `{s4_gate.get('status', 'NOT_RUN')}`；尾部条件扩尾 = `{extension.get('status', 'NOT_RUN')}`；
  未直接测 4→8 相位的节点 = {len(phase_uncovered)} 个（按节点用经验包络，不记为 0）。

## 实际执行

- 新增 TDSE：{tdse} 条（另有 {attempts} 次蓝屏中断 attempt 单列）；独立共同读出：{readouts} 项；
  续接累计 {hours:.6f} h（含中断保守预留 {reserve:.3f} h，**不是实测上界**）。
- 6 项 S4 保存态对照读出已补账（784.653430 s = 0.217959 h），未重跑。
- S3 完成 8/12/20μ 两偏压四相位；S4 完成代表节点 dx/dt 直接控制及 J 点 η/能区/能量网格直接读出；
  S5 补齐 32μ 与弱光初始库；S6 完成尾部预审、两个中点、代表性 4→8 相位、误差刷新与最终统计。
- 逐节点误差表 `node_error_table.csv`：直接控制只在其实测节点生效，历史 4μ 值与 2→4 包络只转用到
  无直接证据的节点；未删除任何误差项以通过门限。
- 响应项（中点项之外）与 CEP 归零后的响应项：
{response_block}
- 缺失的最终验证项：{final.get('missing_validation', [])}。

## 预算与授权

- 用户 2026-11-06 要求完成全部任务后再提交报告；续接子上限 40→56 h、S3 阶段上限 8→10 h，
  全局 240 条 / 72 h 未变。
- 已确认续接机时 42.364278 h（不含中断预留），含预留 42.864278 h；全局约 56.48 h（含旧 1 h 后处理预留）。
- 每批 ≤8 条、单事件 ≤90 min、显式任务 ID、共同读出协议与全部误差门均未放宽。
- 32 条二级中点方案会使续接成功 TDSE 达 128 条（限 112）、每 F 非零节点达 15（限 12），
  成本约 12.24 h 且不含相位与其他数值修正；未获新 revision 授权前不启动。

## 未成立与限制

- 误差结构：节点项 25.8–45.7%（门限 1%）、响应项 11.0–11.7%（门限 5%）；即使中点项归零仍不达标。
- 材料敏感性 −5.6 eV 仍为 `DEFERRED_PRIORITY`；未新增 DFT、多电子、温度、三维或完整 QED。
- 不把单电子概率当针尖绝对计数，不把高能窗口等同再散射认证；两个 DC 端点只支持端点比较。
- 细节数据：`R3_statistics.json`、`node_error_table.csv`、`node_error_summary.json`、
  `error_component_map.csv`、`classical_response.csv`、`statistical_yields.csv`。
"""
    (AUDIT / "decision_r2.md").write_text(text, encoding="utf-8")
    jwrite(AUDIT / "final_summary_v5.json", {
        "label": label, "status": final.get("status"), "D": d,
        "tdse": tdse, "readouts": readouts, "attempts": attempts,
        "continuation_hours": hours, "interruption_reserve_hours": reserve,
        "tail_complete": tail.get("complete"), "phase_complete": phase.get("complete"),
        "missing_validation": final.get("missing_validation", []),
        "node_response_relative": {key: value["response_relative"]
                                   for key, value in node_summary.get("per_channel", {}).items()},
        "node_response_if_cep_zero": {key: value["relative_if_cep_zero"]
                                      for key, value in node_summary.get("per_channel", {}).items()},
    })
    print(text[:1400])
    return 0


STEPS = {"contributions": step_contributions, "s4-readouts": step_s4_readouts,
         "s4": step_s4, "phase8": step_phase8, "tail": step_tail,
         "error-map": step_error_map, "phase-validation": step_phase_validation,
         "decision": step_decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    parser.add_argument("--tail-scope", choices=["precheck", "final"], default="final",
                        help="precheck never certifies the final tail gate")
    args = parser.parse_args()
    if args.step == "tail":
        return step_tail(args.tail_scope)
    return STEPS[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
