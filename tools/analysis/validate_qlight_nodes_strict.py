#!/usr/bin/env python3
"""Strict preflight validation for layer-A quantum-light node manifests.

The Fortran solver deliberately keeps a permissive reader for development.
Production jobs should run this validator before launching any SBE trajectory.
It checks the data actually used by the solver: node identity, weights,
Husimi-Q moments, and the alpha -> (intensity, CEP) mapping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TWOPI = 2.0 * math.pi


def angular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def load_manifest(path: Path) -> list[dict[str, float | int]]:
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
        if weight < 0.0:
            raise ValueError(f"{path}:{lineno}: negative weight")
        if intensity < 0.0:
            raise ValueError(f"{path}:{lineno}: negative intensity")
        if phase < -1.0e-12 or phase >= TWOPI + 1.0e-12:
            raise ValueError(f"{path}:{lineno}: phase is outside [0, 2*pi)")
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


def sv_vq(r: float, theta: float) -> tuple[tuple[float, float], tuple[float, float]]:
    vx = 0.5 * (1.0 + math.exp(-2.0 * r))
    vp = 0.5 * (1.0 + math.exp(+2.0 * r))
    c, s = math.cos(0.5 * theta), math.sin(0.5 * theta)
    return (
        (c * c * vx + s * s * vp, c * s * (vx - vp)),
        (c * s * (vx - vp), s * s * vx + c * c * vp),
    )


def frobenius_relative(a: list[list[float]], b: tuple[tuple[float, float], tuple[float, float]]) -> float:
    num = math.sqrt(sum((a[i][j] - b[i][j]) ** 2 for i in range(2) for j in range(2)))
    den = max(1.0, math.sqrt(sum(b[i][j] ** 2 for i in range(2) for j in range(2))))
    return num / den


def validate_sv(
    rows: list[dict[str, float | int]], r: float, theta_deg: float, i_bar: float, tol: float
) -> dict[str, float | int | str]:
    if r < 0.0:
        raise ValueError("squeeze r must be non-negative")
    if i_bar <= 0.0:
        raise ValueError("I_bar must be positive")
    theta = math.radians(theta_deg)
    wsum = sum(float(row["weight"]) for row in rows)
    if abs(wsum - 1.0) > tol:
        raise ValueError(f"weight sum {wsum:.17g} differs from 1 by more than {tol:g}")

    mean = [0.0, 0.0]
    for row in rows:
        w = float(row["weight"])
        mean[0] += w * math.sqrt(2.0) * float(row["re"])
        mean[1] += w * math.sqrt(2.0) * float(row["im"])

    cov = [[0.0, 0.0], [0.0, 0.0]]
    max_i_err = 0.0
    max_phi_err = 0.0
    nbar_plus_one = math.sinh(r) ** 2 + 1.0
    i_mean = 2.0 * i_bar
    for row in rows:
        w = float(row["weight"])
        re_a, im_a = float(row["re"]), float(row["im"])
        xi = [math.sqrt(2.0) * re_a, math.sqrt(2.0) * im_a]
        dx = [xi[0] - mean[0], xi[1] - mean[1]]
        for i in range(2):
            for j in range(2):
                cov[i][j] += w * dx[i] * dx[j]

        expected_i = i_mean * (re_a * re_a + im_a * im_a) / nbar_plus_one
        i_err = abs(float(row["intensity"]) - expected_i) / max(1.0, abs(expected_i))
        expected_phi = math.atan2(im_a, re_a)
        if expected_phi < 0.0:
            expected_phi += TWOPI
        phi_err = angular_distance(float(row["phase"]), expected_phi)
        max_i_err = max(max_i_err, i_err)
        max_phi_err = max(max_phi_err, phi_err)

    mean_err = math.hypot(mean[0], mean[1])
    cov_err = frobenius_relative(cov, sv_vq(r, theta))
    if mean_err > tol:
        raise ValueError(f"Husimi-Q mean error {mean_err:.3e} exceeds {tol:.3e}")
    if cov_err > tol:
        raise ValueError(f"Husimi-Q covariance error {cov_err:.3e} exceeds {tol:.3e}")
    if max_i_err > tol:
        raise ValueError(f"alpha->intensity error {max_i_err:.3e} exceeds {tol:.3e}")
    if max_phi_err > tol:
        raise ValueError(f"alpha->phase error {max_phi_err:.3e} exceeds {tol:.3e}")

    return {
        "status": "PASS",
        "n_nodes": len(rows),
        "weight_sum": wsum,
        "mean_error": mean_err,
        "covariance_relative_error": cov_err,
        "max_alpha_to_intensity_relative_error": max_i_err,
        "max_alpha_to_phase_absolute_error_rad": max_phi_err,
        "min_intensity_Wcm2": min(float(row["intensity"]) for row in rows),
        "max_intensity_Wcm2": max(float(row["intensity"]) for row in rows),
        "weighted_mean_intensity_Wcm2": sum(
            float(row["weight"]) * float(row["intensity"]) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--state", choices=("squeezed_vacuum",), default="squeezed_vacuum")
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--I-bar", type=float, required=True)
    parser.add_argument("--tol", type=float, default=2.0e-6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        result = validate_sv(load_manifest(args.manifest), args.r, args.theta_deg, args.I_bar, args.tol)
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
