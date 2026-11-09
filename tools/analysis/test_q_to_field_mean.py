#!/usr/bin/env python3
"""Freeze Q→field benchmark mapping: sum_s w_s I_s = I_mean = 2 I_bar.

Q_NODE_MOMENTS = PASS (existing GH moment gates)
Q_TO_FIELD_BENCHMARK_MAPPING = PASS (this test)
Q_TO_FIELD_EXPERIMENTAL_CALIBRATION = NOT_DONE
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "a0_layerA_m88full"))

from make_gh_nodes import alpha_to_drive  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def load_manifest(path: Path) -> tuple[float, list[tuple[float, float]]]:
    i_bar = None
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# I_bar"):
            i_bar = float(raw.split("=", 1)[1])
            continue
        if raw.startswith("#") or not raw.strip():
            continue
        f = raw.split()
        rows.append((float(f[1]), float(f[4])))
    if i_bar is None:
        raise RuntimeError(f"no I_bar in {path}")
    return i_bar, rows


def test_manifest_mean_intensity(path: Path) -> None:
    i_bar, rows = load_manifest(path)
    i_target = 2.0 * i_bar
    wsum = sum(w for w, _i in rows)
    i_mean = sum(w * intensity for w, intensity in rows)
    require(abs(wsum - 1.0) < 1e-12, f"{path.name} sum_w={wsum}")
    rel = abs(i_mean - i_target) / i_target
    require(rel < 1e-12, f"{path.name} sum w I = {i_mean} vs 2 I_bar={i_target}")


def test_alpha_to_drive_mean_formula() -> None:
    i_bar = 1.0e11
    nbar = math.sinh(2.5) ** 2
    i_mean = 2.0 * i_bar
    alpha = -1.8815268187 - 22.9216891059j
    intensity, _phi = alpha_to_drive(alpha, i_bar, nbar)
    expected = i_mean * abs(alpha) ** 2 / (nbar + 1.0)
    require(abs(intensity - expected) / expected < 1e-14, "alpha_to_drive formula")


def main() -> int:
    test_alpha_to_drive_mean_formula()
    print("PASS test_alpha_to_drive_mean_formula")
    base = ROOT / "deploy" / "a0_layerA_m88full" / "nodes_tbv2_es25"
    files = [
        base / "gh5" / "nodes_sv_r2p5_th0_gh5.dat",
        base / "gh5" / "nodes_sv_r2p5_th180_gh5.dat",
        base / "gh3" / "nodes_sv_r2p5_th0_gh3.dat",
        base / "gh7" / "nodes_sv_r2p5_th0_gh7.dat",
    ]
    for path in files:
        test_manifest_mean_intensity(path)
        print(f"PASS mean-intensity {path.name}")
    print("Q_TO_FIELD_BENCHMARK_MAPPING=PASS")
    print("Q_TO_FIELD_EXPERIMENTAL_CALIBRATION=NOT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
