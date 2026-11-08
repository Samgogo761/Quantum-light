"""Build the CONDITIONAL 48/64 mu tail-extension task list (no TDSE here).

External review 2026-11-07 section E: when the 32 mu tail cannot satisfy the
positive lower yield bound, extend to the smallest sufficient node.  This
script only emits an explicit, frozen task list and registers the new response
nodes; every propagation is still executed by run_continuation_v2.py with
explicit allowed_task_ids.  New highest-intensity numerical controls are NOT
invented here: the audited 32 mu seven-category per-category envelope is
transferred and that limitation is recorded in the tail-extension status.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.continuation_v2 import _derive_observation

AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ACTIVE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"


def build(nodes: list[float]) -> list[dict[str, Any]]:
    """Four phases per new node per bias, plus 4->8 phases at the new maximum."""
    phases = [(0, 0.0), (90, math.pi / 2.0), (180, math.pi),
              (270, 3.0 * math.pi / 2.0)]
    tasks: list[dict[str, Any]] = []
    for intensity in nodes:
        for F in [0.0, 1.0]:
            parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_r02/config.json"
            for degrees, cep in phases:
                run_id = f"bias_v2_lib_F{F:g}_I{intensity:g}_phi{degrees:03d}_r01"
                task = _derive_observation(parent, run_id, intensity, cep,
                                           "R3_tail_extension")
                task.update({"existing": False, "F_dc_V_per_nm": F,
                             "I_over_mu": intensity, "CEP_rad": cep})
                tasks.append(task)
    highest = max(nodes)
    for F in [0.0, 1.0]:
        parent = f"results/bias_v1/bias_p1_F{F:.3f}_I1_phi000_r02/config.json"
        for degrees in [45, 135, 225, 315]:
            cep = math.radians(degrees)
            run_id = f"bias_v2_ph8_F{F:g}_I{highest:g}_phi{degrees:03d}_r01"
            task = _derive_observation(parent, run_id, highest, cep,
                                       "R3_tail_phase8")
            task.update({"existing": False, "F_dc_V_per_nm": F,
                         "I_over_mu": highest, "CEP_rad": cep})
            tasks.append(task)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True,
                        help="comma-separated admissible extension nodes, e.g. 48,64")
    args = parser.parse_args()
    nodes = sorted({float(value) for value in args.nodes.split(",") if value.strip()})
    if not nodes or any(value <= 32.0 for value in nodes):
        raise RuntimeError("Tail extension nodes must be strictly above 32 mu")
    tasks = build(nodes)
    AUDIT.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "ql-tip-1d-tail-extension-tasks-v5", "nodes": nodes,
            "task_ids": [task["run_id"] for task in tasks], "tasks": tasks,
            "control_rule": ("no new 32 mu-style control matrix is invented; the "
                             "per-category 32 mu direct envelope is transferred "
                             "to the new high-intensity nodes and recorded as a "
                             "coverage limitation"),
            "phase_rule": "4->8 direct phase validation at the new maximum node"}
    (AUDIT / "tail_extension_tasks.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    active = json.loads(ACTIVE.read_text(encoding="utf-8"))
    active["tail_extension_nodes"] = nodes
    active["tail_extension_status"] = "PLANNED"
    ACTIVE.write_text(json.dumps(active, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"nodes": nodes, "tasks": len(tasks),
                      "path": str(AUDIT / "tail_extension_tasks.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
