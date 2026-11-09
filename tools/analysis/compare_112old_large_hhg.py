#!/usr/bin/env python3
"""Extract H2/H5/H7 from 112_old vs 112_large classical HHG.dat runs.

Peak = max HHG_total in |order - n| <= 0.4.  Not a PASS/FAIL gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDERS = (2, 5, 7)
HALF_WIDTH = 0.4
SCHEMA = "compare_112old_large_hhg_v1"


def dumps_strict(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read_hhg_total(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append((float(parts[0]), float(parts[4])))
    return rows


def peak_at(rows: list[tuple[float, float]], order: int) -> dict:
    band = [(h, val) for h, val in rows if abs(h - order) <= HALF_WIDTH]
    if not band:
        return {"order": order, "empty": True}
    h_peak, val = max(band, key=lambda item: item[1])
    return {
        "order": order,
        "empty": False,
        "harmonic": h_peak,
        "HHG_total": val,
        "n_bins": len(band),
    }


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0.0:
        return None
    return float(a / b)


def case_report(case_dir: Path) -> dict:
    hhg = case_dir / "HHG.dat"
    status = (case_dir / "run_status.txt").read_text(encoding="utf-8", errors="replace") if (case_dir / "run_status.txt").exists() else ""
    peaks = {f"H{n}": peak_at(read_hhg_total(hhg), n) if hhg.exists() else {"order": n, "empty": True} for n in ORDERS}
    return {
        "dir": str(case_dir),
        "exists": hhg.exists(),
        "status_head": status.splitlines()[:8],
        "peaks": peaks,
    }


def pair_ratio(old: dict, large: dict) -> dict:
    out = {}
    for n in ORDERS:
        key = f"H{n}"
        vo = None if old["peaks"][key].get("empty") else old["peaks"][key]["HHG_total"]
        vl = None if large["peaks"][key].get("empty") else large["peaks"][key]["HHG_total"]
        out[key] = {"old": vo, "large": vl, "large_over_old": ratio(vl, vo)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k10-root", type=Path, required=True)
    ap.add_argument("--k20-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tags = ("I2p00e11_cep90", "I1p00e12_cep265")
    payload = {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_HHG_EVIDENCE_NO_PASS_FAIL",
        "notes": [
            "H5/H7 here are evidence columns, not a mechanism claim.",
            "plusN only; k10/k20 classical; not GH7; not 232.",
            "k10 vs k20 difference is not by itself missing Hilbert space.",
        ],
        "grids": {},
    }
    for nk, root in (("k10", args.k10_root), ("k20", args.k20_root)):
        if root is None:
            continue
        grid = {}
        for tag in tags:
            old = case_report(root / f"f112old_{tag}_k{nk[-2:]}_plusN")
            large = case_report(root / f"f112large_{tag}_k{nk[-2:]}_plusN")
            grid[tag] = {"old": old, "large": large, "ratio": pair_ratio(old, large)}
        payload["grids"][nk] = grid
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_strict(payload), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
