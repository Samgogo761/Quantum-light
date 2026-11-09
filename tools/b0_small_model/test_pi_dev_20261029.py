"""Unit tests for the π-deviation helper. No Hamiltonian, no QED."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOC_JSON = ROOT / "results/q_line/abc_soc_gh5_scan_20261029/soc_gh5_scan.json"


def pi_dev(arg_deg: float) -> float:
    a = ((float(arg_deg) + 180.0) % 360.0) - 180.0
    return float(180.0 - abs(a))


def old_pi_dev(arg_deg: float) -> float:
    a = abs(((float(arg_deg) + 180.0) % 360.0) - 180.0)
    return float(min(a, abs(a - 180.0)))


def main() -> int:
    cases = {
        0.0: 180.0,
        90.0: 90.0,
        180.0: 0.0,
        -180.0: 0.0,
        179.85: 0.15,
        -0.15: 179.85,
    }
    for arg, expect in cases.items():
        got = pi_dev(arg)
        if abs(got - expect) > 1e-10:
            raise SystemExit(f"pi_dev({arg}) = {got}, expected {expect}")
    if abs(old_pi_dev(0.0) - 0.0) > 1e-12:
        raise SystemExit("sanity: old helper should map 0° to 0°")
    if abs(old_pi_dev(180.0)) > 1e-12:
        raise SystemExit("sanity: old helper should map 180° to 0°")

    data = json.loads(SOC_JSON.read_text(encoding="utf-8"))
    for tag, row in data["rows"].items():
        arg = row["arg_AC_over_B_deg"]
        stored = row["pi_dev_deg"]
        if stored is None:
            continue
        new = pi_dev(arg)
        old = old_pi_dev(arg)
        if abs(new - stored) > 1e-8:
            raise SystemExit(f"{tag}: new pi_dev {new} != stored {stored}")
        if abs(old - stored) > 1e-8:
            raise SystemExit(f"{tag}: old helper already disagrees with archive")
        if new > 10.0:
            raise SystemExit(f"{tag}: unexpected large deviation {new}")
    print("pi_dev unit tests passed; archived SOC pi_dev_deg unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
