"""Quantify what each candidate control can reduce, in weighted error share.

No TDSE, no new spectrum, no change to any gate.  Reads the audited
`node_error_table.csv` and `node_error_summary.json` and answers, per channel:

* which (node, category) pairs dominate the weighted response error;
* how much the response term falls if the 20 mu CEP term is set to zero;
* how much it falls if the 8 mu dt term is set to zero;
* how much it falls if both are zero.

These are diagnostic arithmetic only (external review 2026-11-08 section 5):
they show the ceiling of a candidate, not a permission to delete an error term.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
TABLE = AUDIT / "node_error_table.csv"
SUMMARY = AUDIT / "node_error_summary.json"
OUT = AUDIT / "error_headroom.json"
TARGETS = {
    "20mu_cep_zero": [(20.0, "cep")],
    "8mu_dt_zero": [(8.0, "dt")],
    "both_zero": [(20.0, "cep"), (8.0, "dt")],
}


def main() -> int:
    with TABLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    output: dict[str, object] = {"schema": "ql-tip-1d-error-headroom-v5",
                                 "note": ("weighted shares are diagnostic; setting a "
                                          "term to zero here is not a scientific update"),
                                 "channels": {}}
    for F in [0.0, 1.0]:
        label = f"F{F:g}"
        for state in ["bsv", "thermal"]:
            key = f"{state}_{label}"
            channel = summary["per_channel"][key]
            y = float(channel["Y"])
            column = f"weighted_{state}"
            channel_rows = [row for row in rows if row["F"] == f"{F:g}"]
            total = sum(float(row[column]) for row in channel_rows)
            top = sorted(channel_rows, key=lambda row: float(row[column]), reverse=True)[:10]
            entry = {
                "Y": y,
                "response_relative": total / y,
                "top_contributions": [
                    {"I_over_mu": float(row["I_over_mu"]),
                     "component": row["component"],
                     "evidence_type": row["evidence_type"],
                     "weighted_absolute": float(row[column]),
                     "share_of_response": float(row[column]) / total if total else None}
                    for row in top
                ],
                "hypothetical": {},
            }
            for name, targets in TARGETS.items():
                removed = sum(float(row[column]) for row in channel_rows
                              if (float(row["I_over_mu"]), row["component"]) in targets)
                entry["hypothetical"][name] = {
                    "removed_absolute": removed,
                    "remaining_relative": (total - removed) / y,
                    "reduction_percentage_points": 100.0 * removed / y,
                }
            output["channels"][key] = entry
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, entry in output["channels"].items():
        hypothetical = entry["hypothetical"]
        print(f"{key}: response={entry['response_relative'] * 100:.3f}% "
              f"| 20mu cep=0 -> {hypothetical['20mu_cep_zero']['remaining_relative'] * 100:.3f}% "
              f"| 8mu dt=0 -> {hypothetical['8mu_dt_zero']['remaining_relative'] * 100:.3f}% "
              f"| both=0 -> {hypothetical['both_zero']['remaining_relative'] * 100:.3f}%")
        for item in entry["top_contributions"][:4]:
            print(f"    I={item['I_over_mu']:g} {item['component']:12s} "
                  f"share={item['share_of_response'] * 100:.1f}% {item['evidence_type']}")
    print(json.dumps({"path": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
