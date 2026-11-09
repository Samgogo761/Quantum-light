"""Candidate cost scenarios for a possible acceptance path (no TDSE).

This is NOT a proof of a minimal, necessary or sufficient plan.  External
review 2026-11-08 (ACCEPTANCE_COST_REVIEW) found four defects in the first
version, all corrected here:

1. the 10% retry margin now applies to HOURS as well as attempts; applying a
   uniform 1.1 is only an order-of-magnitude scenario, not a verified cost;
2. the F1/8 mu direct 4->8 evidence is already saved, so its four extra phases
   are subtracted from any package that includes 8 mu (5 nodes: 40 -> 36);
3. a step-control (`.05->.025`) task and per-node numeric controls are NOT in
   the propagation/readout counts; they are listed as uncovered items, so no
   scenario may be called a complete acceptance path;
4. node counts are expressed as refinement LEVELS (4 / 12 / 28 cumulative
   nodes for three bisection layers), not as an unsupported "eight nodes".

No propagation, spectrum or gate is touched here.
"""
from __future__ import annotations

import csv
import json
import math
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
PROPAGATIONS_PER_NODE = 8
READOUTS_PER_NODE = 8
COMPLETED_CEP_PHASES_AT_F1_8MU = 4
CURRENT_SUCCESS = 96
CURRENT_ATTEMPTS = 97
CURRENT_NODES_PER_BIAS = 11
NODE_LIMIT = 12
ATTEMPT_LIMIT = 112
CONTINUATION_HOURS_LIMIT = 56.0
GLOBAL_HOURS_LIMIT = 72.0

# Cumulative new nodes if every 2 mu interval of 4-12 mu is bisected layer by
# layer: layer 1 adds 4 (5/7/9/11), layer 2 adds 8 more (total 12), layer 3
# adds 16 more (total 28).  These are refinement levels, not a recommendation.
LEVEL_CUMULATIVE_NODES = {1: 4, 2: 12, 3: 28}

SCENARIOS = {
    "interp_level1_only": {"level": 1, "cep_nodes": 0},
    "interp_level1_plus_cep_20mu": {"level": 1, "cep_nodes": 1},
    "interp_level1_plus_cep_5nodes": {"level": 1, "cep_nodes": 5},
    "interp_level2_plus_cep_5nodes": {"level": 2, "cep_nodes": 5},
}


def medians() -> tuple[float, float]:
    rows = list(csv.DictReader((PROJECT / "results/bias_v2/compute_ledger_v2.csv")
                               .open(encoding="utf-8", newline="")))
    tdse = [float(row["counted_seconds"]) for row in rows
            if row["kind"] == "tdse" and row["status"] == "PASS"]
    readouts = [float(row["counted_seconds"]) for row in rows
                if row["kind"] == "readout" and row["status"] == "PASS"]
    return statistics.median(tdse) / 3600.0, statistics.median(readouts) / 3600.0


def cep_propagations(cep_nodes: int) -> int:
    """Four extra phases per bias per node, minus already-saved F1/8 mu."""
    total = cep_nodes * PROPAGATIONS_PER_NODE
    if cep_nodes >= 5:  # package includes 8 mu
        total -= COMPLETED_CEP_PHASES_AT_F1_8MU
    return total


def main() -> int:
    tdse_h, readout_h = medians()
    snapshot = runner.budget_snapshot()
    continuation_now = snapshot["new_science_hours"]
    baseline = float(runner.budget_limits().get("global_hours_baseline", 14.117235553083))
    global_now = continuation_now + baseline
    result: dict[str, Any] = {
        "schema": "ql-tip-1d-acceptance-path-estimate-v6",
        "status": ("candidate_cost_scenarios_only; not a minimality or sufficiency proof"),
        "measured_median_tdse_hours": tdse_h,
        "measured_median_readout_hours": readout_h,
        "retry_margin": RETRY_MARGIN,
        "retry_note": ("applies to attempts and to hours; a uniform 1.1 is an "
                       "order-of-magnitude scenario, not a verified per-task cost"),
        "level_cumulative_new_nodes": LEVEL_CUMULATIVE_NODES,
        "completed_cep_phases_deducted": COMPLETED_CEP_PHASES_AT_F1_8MU,
        "current": {"success_tdse": CURRENT_SUCCESS, "attempts": CURRENT_ATTEMPTS,
                    "nodes_per_bias": CURRENT_NODES_PER_BIAS,
                    "continuation_hours": continuation_now, "global_hours": global_now},
        "limits": {"attempts": ATTEMPT_LIMIT, "nodes_per_bias": NODE_LIMIT,
                   "continuation_hours": CONTINUATION_HOURS_LIMIT,
                   "global_hours": GLOBAL_HOURS_LIMIT},
        "uncovered_items": [
            "step-control .05->.025: 2 propagations plus matched preparation and readouts",
            "per-node eta / box / CAP direct controls in the mid region",
            "thermal strict acceptance (currently proposed as deferred, still unverified)",
            "retry cost by task type (uniform 10% is not verified)",
            "preparation and readout overhead beyond the measured medians",
        ],
        "scenarios": {},
    }
    for name, scenario in SCENARIOS.items():
        new_nodes = LEVEL_CUMULATIVE_NODES[scenario["level"]]
        propagations = new_nodes * PROPAGATIONS_PER_NODE + cep_propagations(scenario["cep_nodes"])
        readouts = (new_nodes + scenario["cep_nodes"]) * READOUTS_PER_NODE
        nominal_hours = propagations * tdse_h + readouts * readout_h
        hours = nominal_hours * (1.0 + RETRY_MARGIN)
        success = CURRENT_SUCCESS + propagations
        attempts = CURRENT_ATTEMPTS + math.ceil(propagations * (1.0 + RETRY_MARGIN))
        nodes = CURRENT_NODES_PER_BIAS + new_nodes
        continuation = continuation_now + hours
        global_hours = global_now + hours
        result["scenarios"][name] = {
            "refinement_level": scenario["level"], "new_nodes": new_nodes,
            "cep_nodes": scenario["cep_nodes"],
            "new_propagations": propagations, "new_readouts": readouts,
            "nominal_hours": nominal_hours, "hours_with_retry": hours,
            "success_tdse": success, "attempts": attempts, "nodes_per_bias": nodes,
            "continuation_hours": continuation, "global_hours": global_hours,
            "covers_interpolation": True,
            "covers_cep_main_nodes": scenario["cep_nodes"] >= 5,
            "covers_step_and_other_numeric": False,
            "within_attempt_limit": attempts <= ATTEMPT_LIMIT,
            "within_node_limit": nodes <= NODE_LIMIT,
            "within_continuation_hours": continuation <= CONTINUATION_HOURS_LIMIT,
            "within_global_hours": global_hours <= GLOBAL_HOURS_LIMIT,
            "complete_acceptance_path": False,
        }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"median_tdse_h": round(tdse_h, 7),
                      "median_readout_h": round(readout_h, 7),
                      "continuation_now": round(continuation_now, 6),
                      "global_now": round(global_now, 6)}, indent=2))
    for name, entry in result["scenarios"].items():
        print(f"{name}: +{entry['new_propagations']} prop, nominal {entry['nominal_hours']:.2f} h, "
              f"with retry {entry['hours_with_retry']:.2f} h, success={entry['success_tdse']}, "
              f"attempts={entry['attempts']}, nodes={entry['nodes_per_bias']}, "
              f"cont={entry['continuation_hours']:.2f} h, global={entry['global_hours']:.2f} h, "
              f"complete={entry['complete_acceptance_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
