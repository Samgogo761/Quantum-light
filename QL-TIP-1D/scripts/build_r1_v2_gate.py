"""Build the R1 evidence-index revision without changing frozen v1 numbers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.bias_v1 import sha256
from ql_tip_1d.continuation_v2 import V2_RESULTS, readout_plan

REPORT = V2_RESULTS / "reports" / "r1_20261105_v1"
V1 = REPORT / "R1_gate_r1_20261105_v1.json"
OUT = REPORT / "R1_gate_r1_20261106_v2.json"


def main() -> int:
    """Add exact readout/state/config identities to the frozen numeric gate."""
    gate = json.loads(V1.read_text(encoding="utf-8"))
    rows = []
    missing = []
    for spec in readout_plan(include_new=True):
        summary_path = V2_RESULTS / "readouts" / spec["readout_id"] / "summary.json"
        if not summary_path.exists():
            missing.append(spec["readout_id"])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "readout_id": spec["readout_id"], "run_id": spec["run_id"],
            "state_sha256": summary["state_sha256"],
            "run_config_sha256": summary["run_config_sha256"],
            "readout_summary_sha256": sha256(summary_path),
            "eta_eV": spec["eta_eV"], "step_eV": spec["step_eV"],
            "mask_nm": [spec["mask_start_nm"], spec["mask_end_nm"]],
            "energy_max_mode": spec["energy_max_mode"],
        })
    new = dict(gate)
    new.update({
        "revision": "r1_20261106_v2",
        "parent_gate": V1.name,
        "parent_gate_sha256": sha256(V1),
        "numeric_values_unchanged": True,
        "evidence_index_complete": not missing,
        "missing_evidence": missing,
        "evidence_rows": rows,
        "limitations": [
            "v1 numeric totals are retained; this revision indexes exact readout/state/config identities.",
            "LOW_ENERGY_FLOOR_REL is a diagnostic proxy from a specific low-energy packet, not a universal absolute bound.",
            "Mixed box/CAP/mask/time aggregation remains an empirical transfer and is not a strict confidence interval.",
        ],
    })
    OUT.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (REPORT / "R1_evidence_sources_v2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} missing={len(missing)} output={OUT}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
