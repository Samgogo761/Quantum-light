#!/usr/bin/env python3
"""Post-run hard gate for a layer-A SBE ensemble directory.

The validator independently reconstructs target-harmonic ICS and CS from
``HHG_nodes_modes.dat`` and compares them with ``HHG_ics_cs.dat``.  It also
ensures that every manifest node appears exactly once for every requested
harmonic and that the propagated (weight, intensity, phase) values match the
archived manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from validate_qlight_nodes_production import load_manifest


def close(a: float, b: float, rtol: float, atol: float = 1.0e-12) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def load_modes(path: Path) -> list[dict[str, float | int | complex]]:
    rows: list[dict[str, float | int | complex]] = []
    keys: set[tuple[int, int]] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(f"{path}:{lineno}: expected 10 columns, got {len(fields)}")
        node_id, order = int(fields[0]), int(fields[4])
        nums = list(map(float, fields[1:4] + fields[5:]))
        if not all(math.isfinite(x) for x in nums):
            raise ValueError(f"{path}:{lineno}: NaN/Inf")
        key = (node_id, order)
        if key in keys:
            raise ValueError(f"{path}:{lineno}: duplicate node/order {key}")
        keys.add(key)
        rows.append(
            {
                "id": node_id,
                "weight": float(fields[1]),
                "intensity": float(fields[2]),
                "phase": float(fields[3]) % (2.0 * math.pi),
                "order": order,
                "jx": complex(float(fields[5]), float(fields[6])),
                "jy": complex(float(fields[7]), float(fields[8])),
                "power": float(fields[9]),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def load_ics_cs(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"{path}:{lineno}: expected at least 5 columns")
        order, ics, cs = float(fields[0]), float(fields[2]), float(fields[3])
        if not all(math.isfinite(x) for x in (order, ics, cs)):
            raise ValueError(f"{path}:{lineno}: NaN/Inf")
        rows.append((order, ics, cs))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def moment_passed(path: Path) -> bool:
    for raw in path.read_text(encoding="utf-8").splitlines():
        compact = raw.replace(" ", "").lower()
        if compact.startswith("pass="):
            return compact.split("=", 1)[1] in {"t", "true", ".true."}
    return False


def validate(run_dir: Path, manifest_path: Path, harmonics: list[int], rtol: float) -> dict:
    required = ["HHG_nodes_modes.dat", "HHG_ics_cs.dat", "nodes_moment_check.txt", "run.log"]
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty required output: {path}")
    if not moment_passed(run_dir / "nodes_moment_check.txt"):
        raise ValueError("nodes_moment_check.txt did not report PASS")

    manifest = {int(row["id"]): row for row in load_manifest(manifest_path)}
    modes = load_modes(run_dir / "HHG_nodes_modes.dat")
    expected_keys = {(node_id, order) for node_id in manifest for order in harmonics}
    actual_keys = {(int(row["id"]), int(row["order"])) for row in modes}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        extra = sorted(actual_keys - expected_keys)[:10]
        raise ValueError(f"node/harmonic coverage mismatch; missing={missing}, extra={extra}")

    for row in modes:
        ref = manifest[int(row["id"])]
        if not close(float(row["weight"]), float(ref["weight"]), rtol):
            raise ValueError(f"node {row['id']}: propagated weight differs from manifest")
        if not close(float(row["intensity"]), float(ref["intensity"]), rtol):
            raise ValueError(f"node {row['id']}: propagated intensity differs from manifest")
        dphi = abs(
            math.atan2(
                math.sin(float(row["phase"]) - float(ref["phase"])),
                math.cos(float(row["phase"]) - float(ref["phase"])),
            )
        )
        if dphi > 2.0e-6:
            raise ValueError(f"node {row['id']}: propagated phase differs from manifest")

    grid = load_ics_cs(run_dir / "HHG_ics_cs.dat")
    comparisons: dict[str, dict[str, float]] = {}
    for order in harmonics:
        group = [row for row in modes if int(row["order"]) == order]
        scales: list[float] = []
        for row in group:
            raw_power = abs(complex(row["jx"])) ** 2 + abs(complex(row["jy"])) ** 2
            if raw_power > 1.0e-280:
                scales.append(float(row["power"]) / raw_power)
        if not scales:
            raise ValueError(f"H{order}: cannot infer spectrum scale")
        scale = sum(scales) / len(scales)
        if max(abs(x - scale) for x in scales) > rtol * max(abs(scale), 1.0e-300):
            raise ValueError(f"H{order}: inconsistent spectrum scale across nodes")
        ics = sum(float(row["weight"]) * float(row["power"]) for row in group)
        mean_jx = sum(float(row["weight"]) * complex(row["jx"]) for row in group)
        mean_jy = sum(float(row["weight"]) * complex(row["jy"]) for row in group)
        cs = scale * (abs(mean_jx) ** 2 + abs(mean_jy) ** 2)
        grid_row = min(grid, key=lambda item: abs(item[0] - order))
        if not close(ics, grid_row[1], 5.0 * rtol, atol=1.0e-280):
            raise ValueError(f"H{order}: reconstructed ICS disagrees with HHG_ics_cs.dat")
        if not close(cs, grid_row[2], 5.0 * rtol, atol=1.0e-280):
            raise ValueError(f"H{order}: reconstructed CS disagrees with HHG_ics_cs.dat")
        if ics + 1.0e-12 * max(abs(ics), 1.0e-300) < cs:
            raise ValueError(f"H{order}: ICS < CS beyond roundoff")
        comparisons[f"H{order}"] = {
            "reconstructed_ICS": ics,
            "reported_ICS": grid_row[1],
            "reconstructed_CS": cs,
            "reported_CS": grid_row[2],
            "spectrum_scale": scale,
        }

    return {
        "status": "PASS",
        "run_dir": str(run_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "n_nodes": len(manifest),
        "harmonics": harmonics,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--harmonics", default="2,5,7,9,10")
    parser.add_argument("--rtol", type=float, default=2.0e-7)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        harmonics = [int(x.strip()) for x in args.harmonics.split(",") if x.strip()]
        result = validate(args.run_dir, args.manifest, harmonics, args.rtol)
    except (OSError, ValueError) as exc:
        print(f"A0_RUN_VALIDATION=FAIL: {exc}")
        return 1
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    print("A0_RUN_VALIDATION=PASS")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
