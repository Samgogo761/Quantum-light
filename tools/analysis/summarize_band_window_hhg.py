#!/usr/bin/env python3
"""Summarize HHG peak ratios vs full112 for band-window scan cases."""
from __future__ import annotations

import argparse
from pathlib import Path


def peak_hhg(path: Path, orders: range = range(1, 16)) -> dict[int, float]:
    if not path.exists():
        return {}
    data: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            h = float(parts[0])
            tot = float(parts[4])
        except ValueError:
            continue
        for q in orders:
            if abs(h - q) < 0.35:
                data[q] = max(data.get(q, 0.0), tot)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\26507\Documents\量子光研究\02_计算输出\新SBEs\data"),
    )
    ap.add_argument("--ref", default="output_lg_cov_fullvalence_T2_0p5cycle/lgcov_k40_nb112")
    args = ap.parse_args()

    cases = [
        ("W30", "output_lg_cov_scan_nodeph/lgcov_k40_b30"),
        ("W40", "output_lg_cov_scan_nodeph/lgcov_k40_b40"),
        ("W50", "output_lg_cov_scan_nodeph/lgcov_k40_b50"),
        ("W60", "output_lg_cov_scan_nodeph/lgcov_k40_b60"),
        ("nb1-104", "output_lg_cov_fullvalence_T2_0p5cycle/lgcov_k40_nb104"),
        ("nb1-94", "output_lg_cov_fullvalence_T2_0p5cycle/lgcov_k40_nb94"),
    ]

    ref = peak_hhg(args.data_root / args.ref / "HHG.dat")
    if not ref:
        raise SystemExit(f"Reference not found: {args.ref}")

    print("| q | " + " | ".join(n for n, _ in cases) + " | full112 |")
    print("|---:|" + "|".join(["---:"] * (len(cases) + 1)) + "|")
    for q in range(1, 16):
        row = [f"H{q}"]
        for name, rel in cases:
            d = peak_hhg(args.data_root / rel / "HHG.dat")
            if d.get(q) and ref.get(q):
                row.append(f"{d[q]/ref[q]:.3f}")
            else:
                row.append("—")
        row.append(f"{ref.get(q, 0):.3e}")
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
