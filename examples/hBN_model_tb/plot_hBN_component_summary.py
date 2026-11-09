#!/usr/bin/env python3
"""Write a compact SVG summary of Jx/Jy harmonic peaks for one hBN run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def write_svg(rows, out: Path, title: str):
    width, height = 920, 560
    ml, mr, mt, mb = 76, 28, 54, 60
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    ymin = 1e-18
    ymax = 1e-2
    max_order = int(max(rows[:, 0]))

    def xmap(order):
        return ml + (order - 1) / (max_order - 1) * plot_w

    def ymap(v):
        vv = max(v, ymin)
        return mt + plot_h * (1 - (math.log10(vv) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)))

    def pts(vals):
        return " ".join("%.2f,%.2f" % (xmap(row[0]), ymap(v)) for row, v in zip(rows, vals))

    hx, hy = rows[:, 2], rows[:, 3]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:22px;font-weight:700}.small{font-size:13px}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:.8}.jx{stroke:#1f77b4;fill:none;stroke-width:2.4}.jy{stroke:#d62728;fill:none;stroke-width:2.4}</style>',
        f'<text class="title" x="{width/2}" y="28" text-anchor="middle">{title}</text>',
        f'<rect x="{ml}" y="{mt}" width="{plot_w}" height="{plot_h}" fill="none" class="axis"/>',
    ]
    for n in range(1, max_order + 1, 2):
        xx = xmap(n)
        parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{mt}" y2="{mt+plot_h}" class="grid"/>')
        parts.append(f'<text class="small" x="{xx:.2f}" y="{height-35}" text-anchor="middle">{n}</text>')
    for p in range(-18, -1, 2):
        yy = ymap(10 ** p)
        parts.append(f'<line x1="{ml}" x2="{ml+plot_w}" y1="{yy:.2f}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text class="small" x="{ml-8}" y="{yy+4:.2f}" text-anchor="end">1e{p}</text>')
    parts.append(f'<polyline class="jx" points="{pts(hx)}"/>')
    parts.append(f'<polyline class="jy" points="{pts(hy)}"/>')
    for row in rows:
        n, _, x, y, _ = row
        color_x = "#1f77b4" if int(n) % 2 == 1 else "#8fb9df"
        color_y = "#d62728" if int(n) % 2 == 0 else "#e89a9a"
        parts.append(f'<circle cx="{xmap(n):.2f}" cy="{ymap(x):.2f}" r="3.2" fill="{color_x}"/>')
        parts.append(f'<circle cx="{xmap(n):.2f}" cy="{ymap(y):.2f}" r="3.2" fill="{color_y}"/>')
    parts.extend([
        f'<text class="small" x="{ml + plot_w/2:.1f}" y="{height-12}" text-anchor="middle">harmonic order</text>',
        f'<text class="small" transform="translate(18,{mt+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">peak intensity, |J(omega)|^2</text>',
        f'<line x1="{width-190}" x2="{width-160}" y1="58" y2="58" class="jx"/><text class="small" x="{width-150}" y="62">parallel Jx</text>',
        f'<line x1="{width-190}" x2="{width-160}" y1="78" y2="78" class="jy"/><text class="small" x="{width-150}" y="82">perpendicular Jy</text>',
        '<text class="small" x="84" y="48">Blue odd peaks and red even peaks are the expected hBN selection pattern.</text>',
        '</svg>',
    ])
    out.write_text("\n".join(parts), encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="hBN model spectrum")
    args = parser.parse_args()
    rows = np.loadtxt(args.summary, comments="#")
    write_svg(rows, Path(args.out), args.title)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
