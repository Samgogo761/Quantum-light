"""Budget a COMPLETE minimal acceptance path from measured costs (no TDSE).

External review 2026-11-08 (RESCUE_R2_DECISION_REVIEW) asked for one decision
page that covers both the response error and the interpolation residual, lists
only genuinely new tasks, includes preparation / propagation / readout / retry
cost, and states the required limit changes.  This script produces the numbers
that page cites; it never starts a propagation and never changes a gate.

Cost inputs are the measured medians of the existing ledger.  A 10% retry
margin is added to attempts because a failed preparation or propagation still
consumes a launch.  `hours` are planning estimates, not guarantees.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import run_continuation_v2 as runner

AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
OUT = AUDIT / "acceptance_path_estimate.json"
RETRY_MARGIN = 0.10
PROPAGATIONS_PER_NODE = 8   # two biases x four phases
READOUTS_PER_NODE = 8
CURRENT_SUCCESS = 96
CURRENT_ATTEMPTS = 97
CURRENT_NODES_PER_BIAS = 11
NODE_LIMIT = 12
ATTEMPT_LIMIT = 112
CONTINUATION_HOURS_LIMIT = 56.0
GLOBAL_HOURS_LIMIT = 72.0

SCENARIOS = {
    "interp_4nodes_only": {"new_nodes": 4, "cep_nodes": 0,
                            "covers_interpolation": True, "covers_cep": False},
    "interp_4nodes_plus_cep_20mu": {"new_nodes": 4, "cep_nodes": 1,
                                     "covers_interpolation": True, "covers_cep": False},
    "interp_4nodes_plus_cep_5nodes": {"new_nodes": 4, "cep_nodes": 5,
                                       "covers_interpolation": True, "covers_cep": True},
    "interp_8nodes_plus_cep_5nodes": {"new_nodes": 8, "cep_nodes": 5,
                                       "covers_interpolation": True, "covers_cep": True},
}


def medians() -> tuple[float, float]:
    rows = list(csv.DictReader((PROJECT / "results/bias_v2/compute_ledger_v2.csv")
                               .open(encoding="utf-8", newline="")))
    tdse = [float(row["counted_seconds"]) for row in rows
            if row["kind"] == "tdse" and row["status"] == "PASS"]
    readouts = [float(row["counted_seconds"]) for row in rows
                if row["kind"] == "readout" and row["status"] == "PASS"]
    return statistics.median(tdse) / 3600.0, statistics.median(readouts) / 3600.0


def main() -> int:
    tdse_h, readout_h = medians()
    snapshot = runner.budget_snapshot()
    continuation_now = snapshot["new_science_hours"]
    baseline = float(runner.budget_limits().get("global_hours_baseline", 14.117235553083))
    global_now = continuation_now + baseline
    result: dict[str, Any] = {
        "schema": "ql-tip-1d-acceptance-path-estimate-v5",
        "measured_median_tdse_hours": tdse_h,
        "measured_median_readout_hours": readout_h,
        "retry_margin": RETRY_MARGIN,
        "current": {"success_tdse": CURRENT_SUCCESS, "attempts": CURRENT_ATTEMPTS,
                    "nodes_per_bias": CURRENT_NODES_PER_BIAS,
                    "continuation_hours": continuation_now, "global_hours": global_now},
        "limits": {"attempts": ATTEMPT_LIMIT, "nodes_per_bias": NODE_LIMIT,
                   "continuation_hours": CONTINUATION_HOURS_LIMIT,
                   "global_hours": GLOBAL_HOURS_LIMIT},
        "scenarios": {},
    }
    for name, scenario in SCENARIOS.items():
        propagations = (scenario["new_nodes"] + scenario["cep_nodes"]) * PROPAGATIONS_PER_NODE
        readouts = (scenario["new_nodes"] + scenario["cep_nodes"]) * READOUTS_PER_NODE
        hours = propagations * tdse_h + readouts * readout_h
        success = CURRENT_SUCCESS + propagations
        attempts = CURRENT_ATTEMPTS + int(round(propagations * (1.0 + RETRY_MARGIN)))
        nodes = CURRENT_NODES_PER_BIAS + scenario["new_nodes"]
        continuation = continuation_now + hours
        global_hours = global_now + hours
        entry = {
            "new_nodes": scenario["new_nodes"], "cep_nodes": scenario["cep_nodes"],
            "new_propagations": propagations, "new_readouts": readouts,
            "added_hours": hours,
            "success_tdse": success, "attempts": attempts, "nodes_per_bias": nodes,
            "continuation_hours": continuation, "global_hours": global_hours,
            "covers_interpolation": scenario["covers_interpolation"],
            "covers_cep": scenario["covers_cep"],
            "within_attempt_limit": attempts <= ATTEMPT_LIMIT,
            "within_node_limit": nodes <= NODE_LIMIT,
            "within_continuation_hours": continuation <= CONTINUATION_HOURS_LIMIT,
            "within_global_hours": global_hours <= GLOBAL_HOURS_LIMIT,
        }
        entry["complete_path_possible"] = bool(
            entry["covers_interpolation"] and entry["covers_cep"]
            and entry["within_attempt_limit"] and entry["within_node_limit"]
            and entry["within_continuation_hours"] and entry["within_global_hours"])
        result["scenarios"][name] = entry
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"median_tdse_h": round(tdse_h, 4),
                      "median_readout_h": round(readout_h, 4),
                      "continuation_now": round(continuation_now, 6),
                      "global_now": round(global_now, 6)}, indent=2))
    for name, entry in result["scenarios"].items():
        print(f"{name}: +{entry['new_propagations']} prop, +{entry['added_hours']:.2f} h, "
              f"success={entry['success_tdse']}, attempts={entry['attempts']}, "
              f"nodes={entry['nodes_per_bias']}, cont={entry['continuation_hours']:.2f} h, "
              f"global={entry['global_hours']:.2f} h, complete={entry['complete_path_possible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
