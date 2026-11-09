"""Snapshot existing metadata only; never import or execute scientific code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "QL-TIP-1D"


def collect() -> dict:
    files = {}

    def read(path: Path):
        data = path.read_bytes()
        files[path.relative_to(ROOT).as_posix()] = {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
        }
        return data.decode("utf-8-sig")

    result = {"scope": "read-only metadata and arithmetic; no TDSE or spectrum evaluation",
              "snapshot_time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "inherited_commit": "3bba5db5"}
    base = PROJECT / "results/bias_v2"
    ledger = list(csv.DictReader(read(base / "compute_ledger_v2.csv").splitlines()))
    counts, seconds = Counter(), defaultdict(float)
    for row in ledger:
        key = (row["phase"], row["kind"])
        counts[key] += 1
        seconds[key] += float(row["counted_seconds"] or 0)
    result["ledger"] = {
        "groups": [{"phase": k[0], "kind": k[1], "events": counts[k],
                    "hours": seconds[k] / 3600} for k in sorted(counts)],
        "hours": sum(seconds.values()) / 3600,
        "duplicate_event_ids": len(ledger) - len({r["event_id"] for r in ledger}),
        "limitation": "unrecorded or interrupted work is not certified by this sum"
    }
    runs, readouts = {}, {}
    for p in sorted((base / "runs").glob("*probe*/summary.json")):
        obj = json.loads(read(p))
        cfg = p.parent / "config.json"
        read(cfg)
        fields = ["wall_seconds", "Y_H", "execution_status", "max_probability_budget_residual",
                  "right_CAP_loss_observation_100fs", "state_100fs_sha256"]
        runs[p.parent.name] = {k: obj.get(k) for k in fields}
        runs[p.parent.name]["spectral_diagnostic"] = [
            {k: s.get(k) for k in ["label", "eta_eV", "energy_step_eV", "Y_H_main"]}
            for s in obj.get("spectra", [])]
    for p in sorted((base / "readouts").glob("*probe*/summary.json")):
        obj = json.loads(read(p))
        fields = ["run_id", "eta_eV", "step_eV", "energy_range_eV", "energy_points",
                  "Y_at_E_main", "wall_seconds", "state_sha256", "run_config_sha256"]
        readouts[p.parent.name] = {k: obj.get(k) for k in fields}
    result["R2_runs"] = runs
    result["R2_readouts"] = readouts
    expected = []
    for run_id in runs:
        expected.append(f"ro2_{run_id}_t100_eta0p05_dE0p0125_m2-4")
        if run_id in [f"bias_v2_probe_F{f}_I32_phi000_r01" for f in [0, 1]]:
            expected += [f"ro2_{run_id}_t100_eta0p025_dE0p00625_m2-4",
                         f"ro2_{run_id}_t100_eta0p05_dE0p00625_m2-4",
                         f"ro2_{run_id}_t100_eta0p05_dE0p0125_m2-4_emaxplus"]
    result["R2_readouts_expected_from_existing_runs"] = len(expected)
    result["R2_readouts_missing"] = sorted(set(expected) - set(readouts))
    result["R2_completed_readout_minutes"] = {
        "min": min(s["wall_seconds"] for s in readouts.values()) / 60,
        "max": max(s["wall_seconds"] for s in readouts.values()) / 60,
        "sum": sum(s["wall_seconds"] for s in readouts.values()) / 60}
    base_y = runs["bias_v2_probe_F1_I32_phi000_r01"]["Y_H"]
    fine_y = runs["bias_v2_probe_F1_I32_phi000_dt005_r01"]["Y_H"]
    result["diagnostic_only"] = {
        "F1_I32_dt_relative": fine_y / base_y - 1,
        "q_CEP0_eta01": {str(i): runs[f"bias_v2_probe_F1_I{i}_phi000_r01"]["Y_H"] /
                                   runs[f"bias_v2_probe_F0_I{i}_phi000_r01"]["Y_H"]
                          for i in [8, 32]},
        "warning": "not four-phase means; not BSV statistics; common eta=.05 controls pending"}
    for name in ["R1", "R2"]:
        p = base / f"reports/r1_20261105_v1/{name}_gate_r1_20261105_v1.json"
        result[f"{name}_reported_gate"] = json.loads(read(p))
    dependencies = ["reports/evening_status.md", "scripts/run_continuation_v2.py",
                    "scripts/readout_continuation_v2.py", "scripts/analyze_continuation_v2.py",
                    "scripts/analyze_high_intensity_v2.py", "scripts/analyze_statistics_v2.py",
                    "src/ql_tip_1d/continuation_v2.py",
                    "plans/20261105_continuation_v2/EXECUTION_TASKBOOK.md",
                    "plans/20261105_evening_decision/DECISION_AND_AGENT_INSTRUCTIONS.md"]
    for path in dependencies:
        read(PROJECT / path)
    result["source_files"] = files
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = Path(__file__).resolve().parent
    if allowed not in output.parents:
        raise SystemExit("Snapshot output must stay inside this review package")
    result = collect()
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"path": str(output), "hours": result["ledger"]["hours"],
                      "R2_runs": len(result["R2_runs"]),
                      "missing_readouts": len(result["R2_readouts_missing"])}, ensure_ascii=False))
