"""Summarise completed TDSE count and cumulative wall time without running science."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "results/bias_v1"


def main() -> int:
    """Print and save the current management-budget ledger."""
    runs = []
    for directory in sorted(ROOT.iterdir()) if ROOT.exists() else []:
        summary = directory / "summary.json"
        if not summary.is_file():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        runs.append({
            "run_id": directory.name,
            "run_kind": data.get("run_kind"),
            "wall_seconds": float(data.get("wall_seconds", 0.0)),
            "F_dc_V_per_nm": data.get("F_dc_V_per_nm"),
            "I_apex_W_per_cm2": data.get("I_apex_W_per_cm2"),
            "CEP_rad": data.get("CEP_rad"),
            "status": data.get("status"),
        })
    failed = []
    for directory in sorted(ROOT.iterdir()) if ROOT.exists() else []:
        if (directory / "failure.json").exists() and not (directory / "summary.json").exists():
            data = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
            failed.append({"run_id": directory.name, "error": data.get("error"),
                           "wall_seconds": float(data.get("wall_seconds", 0.0)),
                           "note": "Pre-propagation configuration failure; retained, not overwritten."})
    total_seconds = sum(item["wall_seconds"] for item in runs)
    output = {
        "completed_summary_runs": len(runs),
        "completed_tdse_propagations": len([item for item in runs if item["run_kind"] in {"preparation", "dc-only", "laser"}]),
        "cumulative_completed_wall_seconds": total_seconds,
        "cumulative_completed_wall_hours": total_seconds / 3600.0,
        "failed_prepropagation_attempts": failed,
        "limits": {"new_tdse": 240, "wall_hours": 72.0, "peak_GiB": 6.0, "single_run_minutes": 90.0},
        "runs": runs,
    }
    path = ROOT / "budget_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (ROOT / "run_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]) if runs else ["run_id"])
        writer.writeheader()
        writer.writerows(runs)
    print(json.dumps({key: output[key] for key in ["completed_summary_runs", "completed_tdse_propagations",
                                                  "cumulative_completed_wall_hours", "failed_prepropagation_attempts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
