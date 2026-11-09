#!/usr/bin/env python3
"""Summarize hBN model k-grid scan spectra and write a small SVG plot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent


def load_hhg(run_dir: Path):
    arr = np.loadtxt(run_dir / "HHG.dat", comments="#")
    return arr[:, 0], arr[:, 2], arr[:, 3], arr[:, 4]


def peak_table(order, hx, hy, ht, max_order, half_width):
    rows = []
    for n in range(1, max_order + 1):
        mask = (order >= n - half_width) & (order <= n + half_width)
        if not np.any(mask):
            rows.append((n, math.nan, math.nan, math.nan, math.nan))
            continue
        idxs = np.where(mask)[0]
        idx = idxs[np.argmax(ht[mask])]
        rows.append((n, float(order[idx]), float(hx[idx]), float(hy[idx]), float(ht[idx])))
    return rows


def write_summary(cases, max_order, half_width, outdir):
    loaded = {}
    peak_rows = {}
    for nk, run_dir in cases:
        order, hx, hy, ht = load_hhg(run_dir)
        loaded[nk] = (order, hx, hy, ht)
        peak_rows[nk] = peak_table(order, hx, hy, ht, max_order, half_width)

    largest_nk = max(loaded)
    lines = ["# order " + " ".join(f"Htot_k{nk}" for nk in loaded) + f" ratio_to_k{largest_nk}"]
    for i in range(max_order):
        vals = [peak_rows[nk][i][4] for nk in loaded]
        ref = peak_rows[largest_nk][i][4]
        ratios = [v / ref if ref > 0 else math.nan for v in vals]
        lines.append(
            "%5d " % (i + 1)
            + " ".join("%14.7e" % v for v in vals)
            + " "
            + " ".join("%10.5f" % r for r in ratios)
        )
    (outdir / "k_scan_harmonics_summary.dat").write_text("\n".join(lines) + "\n", encoding="ascii")

    return loaded, peak_rows


def polyline(points):
    return " ".join("%.2f,%.2f" % p for p in points)


def write_svg(loaded, outdir, max_plot_order=25.0):
    width, height = 980, 620
    margin_l, margin_r, margin_t, margin_b = 70, 24, 34, 58
    panel_gap = 44
    panel_w = (width - margin_l - margin_r - panel_gap) / 2
    panel_h = height - margin_t - margin_b
    colors = {30: "#1f77b4", 60: "#d62728", 90: "#2ca02c", 120: "#9467bd"}

    all_vals = []
    for _, hx, hy, _ in loaded.values():
        mask_len = min(len(hx), len(hy))
        all_vals.extend([v for v in hx[:mask_len] if v > 0])
        all_vals.extend([v for v in hy[:mask_len] if v > 0])
    ymin = max(min(all_vals), 1e-14)
    ymax = max(all_vals)
    ymin = 10 ** math.floor(math.log10(ymin))
    ymax = 10 ** math.ceil(math.log10(ymax))

    def map_x(x, x0):
        return x0 + (x / max_plot_order) * panel_w

    def map_y(y):
        yy = max(y, ymin)
        return margin_t + panel_h * (1.0 - (math.log10(yy) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px}.title{font-size:18px;font-weight:bold}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:0.8}.legend{font-size:12px}</style>',
        f'<text class="title" x="{width/2:.1f}" y="22" text-anchor="middle">hBN two-band model: k-grid convergence, T2=5 fs</text>',
    ]

    for panel, comp in enumerate(["parallel Jx", "perpendicular Jy"]):
        x0 = margin_l + panel * (panel_w + panel_gap)
        parts.append(f'<text x="{x0 + panel_w/2:.1f}" y="{margin_t-8}" text-anchor="middle">{comp}</text>')
        parts.append(f'<rect x="{x0}" y="{margin_t}" width="{panel_w}" height="{panel_h}" fill="none" class="axis"/>')
        for h in range(0, int(max_plot_order) + 1, 5):
            xx = map_x(h, x0)
            parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{margin_t}" y2="{margin_t+panel_h}" class="grid"/>')
            parts.append(f'<text x="{xx:.2f}" y="{height-35}" text-anchor="middle">{h}</text>')
        p = math.floor(math.log10(ymin))
        while 10 ** p <= ymax:
            yy = map_y(10 ** p)
            parts.append(f'<line x1="{x0}" x2="{x0+panel_w}" y1="{yy:.2f}" y2="{yy:.2f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{yy+4:.2f}" text-anchor="end">1e{p}</text>')
            p += 2
        parts.append(f'<text x="{x0 + panel_w/2:.1f}" y="{height-12}" text-anchor="middle">harmonic order</text>')

        for nk, (order, hx, hy, _) in sorted(loaded.items()):
            vals = hx if panel == 0 else hy
            mask = (order >= 0.0) & (order <= max_plot_order)
            pts = [(map_x(float(o), x0), map_y(float(v))) for o, v in zip(order[mask], vals[mask]) if v > 0]
            if pts:
                parts.append(f'<polyline fill="none" stroke="{colors.get(nk, "#555")}" stroke-width="1.5" points="{polyline(pts)}"/>')

    lx = width - 160
    ly = 48
    for i, nk in enumerate(sorted(loaded)):
        y = ly + i * 18
        parts.append(f'<line x1="{lx}" x2="{lx+24}" y1="{y}" y2="{y}" stroke="{colors.get(nk, "#555")}" stroke-width="2"/>')
        parts.append(f'<text class="legend" x="{lx+32}" y="{y+4}">nk={nk}</text>')
    parts.append("</svg>")
    (outdir / "k_scan_spectrum_T2_5fs.svg").write_text("\n".join(parts), encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(EXAMPLE_DIR / "runs"))
    parser.add_argument("--k", nargs="+", type=int, default=[30, 60, 90])
    parser.add_argument("--t2-label", default="5p0")
    parser.add_argument("--max-order", type=int, default=25)
    parser.add_argument("--half-width", type=float, default=0.15)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    outdir = EXAMPLE_DIR / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    cases = []
    for nk in args.k:
        run_dir = runs_dir / f"lgcov_k{nk}_T2_{args.t2_label}fs"
        if not (run_dir / "HHG.dat").exists():
            raise FileNotFoundError(run_dir / "HHG.dat")
        cases.append((nk, run_dir))

    loaded, _ = write_summary(cases, args.max_order, args.half_width, outdir)
    write_svg(loaded, outdir, max_plot_order=float(args.max_order))
    print("Wrote", outdir / "k_scan_harmonics_summary.dat")
    print("Wrote", outdir / "k_scan_spectrum_T2_5fs.svg")


if __name__ == "__main__":
    main()
