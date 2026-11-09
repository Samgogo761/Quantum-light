#!/usr/bin/env python3
"""Production entry point for strict layer-A node validation.

This wrapper accepts the harmless 8-decimal serialization of 2*pi found in
legacy GH3 manifests, while retaining all strict moment and mapping checks
implemented in ``validate_qlight_nodes_strict``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from validate_qlight_nodes_strict import TWOPI, validate_sv


def load_manifest(path: Path, boundary_tol: float = 1.0e-6) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    seen: set[int] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}:{lineno}: expected 6 columns, got {len(fields)}")
        try:
            node_id = int(fields[0])
            weight, re_a, im_a, intensity, phase = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid numeric field") from exc
        values = (weight, re_a, im_a, intensity, phase)
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"{path}:{lineno}: NaN/Inf is forbidden")
        if node_id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate node id {node_id}")
        if weight < 0.0 or intensity < 0.0:
            raise ValueError(f"{path}:{lineno}: negative weight/intensity")
        if phase < -boundary_tol or phase > TWOPI + boundary_tol:
            raise ValueError(f"{path}:{lineno}: phase is outside [0, 2*pi)")
        phase %= TWOPI
        seen.add(node_id)
        rows.append(
            {
                "id": node_id,
                "weight": weight,
                "re": re_a,
                "im": im_a,
                "intensity": intensity,
                "phase": phase,
            }
        )
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--I-bar", type=float, required=True)
    parser.add_argument("--tol", type=float, default=2.0e-6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = validate_sv(
            load_manifest(args.manifest), args.r, args.theta_deg, args.I_bar, args.tol
        )
    except (OSError, ValueError) as exc:
        print(f"NODE_PREFLIGHT=FAIL: {exc}")
        return 1
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    print("NODE_PREFLIGHT=PASS")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
