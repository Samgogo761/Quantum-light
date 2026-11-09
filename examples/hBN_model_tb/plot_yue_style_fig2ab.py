#!/usr/bin/env python3
"""Plot our hBN benchmark in a Yue/Gaarde Fig. 2(a,b)-style spectrum layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path):
    return np.loadtxt(path, comments="#")


def write_svg(rows, out: Path, gap_order: float):
    width, height = 760, 620
    ml, mr, mt, mb = 86, 34, 44, 58
    gap = 54
    pw = (width - ml - mr - gap) / 2
    ph = height - mt - mb
    ymin_order, ymax_order = 1.0, 25.5
    xmin, xmax = 1e-18, 1e-2

    def ymap(order):
        return mt + ph * (1 - (order - ymin_order) / (ymax_order - ymin_order))

    def xmap(v, x0):
        vv = max(v, xmin)
        return x0 + pw * (math.log10(vv) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))

    def path_for(vals, x0):
        pts = []
        for row, val in zip(rows, vals):
            order = row[0]
            if ymin_order <= order <= ymax_order:
                pts.append((xmap(val, x0), ymap(order)))
        if not pts:
            return ""
        # Filled curve from left baseline to data and back.
        base = xmap(xmin, x0)
        d = [f"M {base:.2f} {pts[0][1]:.2f}"]
        for x, y in pts:
            d.append(f"L {x:.2f} {y:.2f}")
        d.append(f"L {base:.2f} {pts[-1][1]:.2f} Z")
        return " ".join(d)

    hx = rows[:, 2]
    hy = rows[:, 3]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:20px;font-weight:bold}.label{font-size:14px}.small{font-size:12px}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:.8}.fill{fill:#5fb0dd;stroke:#126a8a;stroke-width:1.4}.dash{stroke:#555;stroke-dasharray:5 4;stroke-width:1.1}</style>',
        f'<text class="title" x="{width/2}" y="24" text-anchor="middle">Parameter-aligned reproduction attempt: Fig. 2(a,b)-style spectra</text>',
    ]

    panels = [("parallel Jx", hx), ("perpendicular Jy", hy)]
    for ip, (title, vals) in enumerate(panels):
        x0 = ml + ip * (pw + gap)
        parts.append(f'<text class="label" x="{x0 + pw/2:.1f}" y="{mt-10}" text-anchor="middle">{title}</text>')
        parts.append(f'<rect x="{x0}" y="{mt}" width="{pw}" height="{ph}" fill="none" class="axis"/>')
        for n in range(1, 26, 2):
            y = ymap(n)
            parts.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
            if ip == 0:
                parts.append(f'<text class="small" x="{x0-10}" y="{y+4:.2f}" text-anchor="end">{n}</text>')
        for pow10 in range(-18, -1, 4):
            x = xmap(10 ** pow10, x0)
            parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{mt}" y2="{mt+ph}" class="grid"/>')
            parts.append(f'<text class="small" x="{x:.2f}" y="{height-34}" text-anchor="middle">1e{pow10}</text>')
        parts.append(f'<path class="fill" d="{path_for(vals, x0)}"/>')
        if ymin_order <= gap_order <= ymax_order:
            yg = ymap(gap_order)
            parts.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{yg:.2f}" y2="{yg:.2f}" class="dash"/>')
        parts.append(f'<text class="small" x="{x0+pw/2:.1f}" y="{height-12}" text-anchor="middle">peak intensity, |J(omega)|^2</text>')

    parts.append(f'<text class="label" transform="translate(24,{mt+ph/2:.1f}) rotate(-90)" text-anchor="middle">harmonic order</text>')
    parts.append(f'<text class="small" x="{ml}" y="{height-2}">Parameters: 1600 nm, 35 fs, 3.5 TW/cm2, T2=5 fs, k=120. Dashed gap line: model gap order {gap_order:.2f}.</text>')
    parts.append('</svg>')
    out.write_text("\n".join(parts), encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gap-ev", type=float, default=3.92)
    parser.add_argument("--photon-ev", type=float, default=1239.841984 / 1600.0)
    args = parser.parse_args()
    rows = read_rows(Path(args.summary))
    write_svg(rows, Path(args.out), gap_order=args.gap_ev / args.photon_ev)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
