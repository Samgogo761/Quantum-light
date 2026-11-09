#!/usr/bin/env python3
"""Extract harmonic peak intensities from the hBN model HHG.dat file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hhg", default="HHG.dat", help="Path to HHG.dat")
    parser.add_argument("--max-order", type=int, default=15)
    parser.add_argument("--half-width", type=float, default=0.15)
    parser.add_argument("--out", default="harmonics_summary.dat")
    args = parser.parse_args()

    hhg_path = Path(args.hhg)
    arr = np.loadtxt(hhg_path, comments="#")
    order = arr[:, 0]
    hx = arr[:, 2]
    hy = arr[:, 3]
    htot = arr[:, 4]

    lines = [
        "# order  peak_order  HHG_x  HHG_y  HHG_total",
    ]
    for n in range(1, args.max_order + 1):
        mask = (order >= n - args.half_width) & (order <= n + args.half_width)
        if not np.any(mask):
            lines.append("%5d  nan  nan  nan  nan" % n)
            continue
        idxs = np.where(mask)[0]
        idx = idxs[np.argmax(htot[mask])]
        lines.append(
            "%5d %11.5f %14.7e %14.7e %14.7e"
            % (n, order[idx], hx[idx], hy[idx], htot[idx])
        )

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="ascii")
    print("Wrote", args.out)
    print("\n".join(lines[: min(len(lines), 8)]))


if __name__ == "__main__":
    main()
