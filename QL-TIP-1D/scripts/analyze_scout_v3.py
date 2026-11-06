"""Summarise the bounded S3 scout response without claiming a full library."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.continuation_v2 import V2_RESULTS

OUT = PROJECT / "results/bias_v3_audit/r1_20261106_v3"


def y(run_id: str) -> float | None:
    """Load the common eta=.05 readout for one run."""
    path = V2_RESULTS / "readouts" / f"ro2_{run_id}_t100_eta0p05_dE0p0125_m2-4" / "summary.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["Y_at_E_main"])


def ids(F: float, intensity: float) -> list[str]:
    if intensity == 8.0:
        return [f"bias_v2_probe_F{F:g}_I8_phi000_r01"] + [
            f"bias_v2_lib_F{F:g}_I8_phi{x:03d}_r01" for x in (90, 180, 270)]
    return [f"bias_v2_lib_F{F:g}_I{intensity:g}_phi{x:03d}_r01" for x in (0, 90, 180, 270)]


def main() -> int:
    """Write complete 8mu/12mu phase averages and q where both F are complete."""
    rows = []
    complete = {}
    for intensity in [8.0, 12.0]:
        values = {}
        for F in [0.0, 1.0]:
            ys = [y(run_id) for run_id in ids(F, intensity)]
            if any(item is None for item in ys):
                values[F] = None
                continue
            values[F] = float(sum(ys) / len(ys))
            rows.append([F, intensity, values[F], min(ys), max(ys),
                         (max(ys) - min(ys)) / values[F] if values[F] else None,
                         "complete_four_phase"])
        if values.get(0.0) and values.get(1.0):
            complete[intensity] = values[1.0] / values[0.0]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "s3_scout_response.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["F_dc_V_per_nm", "I_over_mu", "Y_mean", "Y_min", "Y_max",
                         "relative_phase_span", "status"])
        writer.writerows(rows)
    output = {"schema": "ql-tip-1d-s3-scout", "complete_q": complete,
              "rows": rows, "note": "Partial S3 scout; I=20mu and full library not run."}
    (OUT / "s3_scout.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
