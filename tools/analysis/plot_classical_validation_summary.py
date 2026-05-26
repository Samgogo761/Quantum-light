"""Generate lightweight SVG summary figures for CrI3 SBE validation.

This script intentionally uses only the Python standard library so it can run
inside the Codex bundled Python without extra plotting dependencies.
"""

from __future__ import annotations

import math
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(r"C:\Users\26507\Documents\量子光研究\新SBEs\data")
RUNS_ROOT = Path(r"C:\Users\26507\Documents\量子光研究\新SBEs\runs")
OUT_DIR = REPO / "docs" / "figures" / "classical_validation"

LG_SCAN = DATA_ROOT / "output_lg_cov_scan_nodeph"
LG_VALDEPTH = DATA_ROOT / "output_lg_cov_valence_depth_nodeph"
LG_FULLVAL = DATA_ROOT / "output_lg_cov_fullvalence_nodeph"
LG_FULLVAL_T2 = DATA_ROOT / "output_lg_cov_fullvalence_T2_0p5cycle"
LG_K60_T2 = DATA_ROOT / "output_lg_cov_k60_b112_T2_0p5cycle"
LG_T2_KEY = DATA_ROOT / "output_lg_cov_t2_key_cases"
LG_VTRIM104_T2 = DATA_ROOT / "output_lg_cov_valence_trim_104_T2_0p5cycle"
PEIERLS_LOCAL = REPO / "local_runs" / "peierls_window_20260517_nodeph"
PHASE0 = RUNS_ROOT / "output_phase0_gauge_10x10_112_nodeph"

ORDERS_LOW = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ORDERS_ODD = [1, 3, 5, 7, 9, 11, 13, 15]


def read_hhg(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append((float(parts[0]), float(parts[4])))
        except ValueError:
            continue
    return rows


def nearest_hhg(case_dir: Path, order: float) -> float:
    rows = read_hhg(case_dir / "HHG.dat")
    if not rows:
        return float("nan")
    return min(rows, key=lambda item: abs(item[0] - order))[1]


def has_hhg(case_dir: Path) -> bool:
    return (case_dir / "HHG.dat").exists()


def read_j0(case_dir: Path) -> float:
    path = case_dir / "Jt.dat"
    if not path.exists():
        return float("nan")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                jx = float(parts[2])
                jy = float(parts[3])
                return math.hypot(jx, jy)
            except ValueError:
                pass
    return float("nan")


def read_real_minutes(case_dir: Path) -> float:
    path = case_dir / "run.log"
    if not path.exists():
        return float("nan")
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^real\s+(\d+)m([0-9.]+)s", text, flags=re.MULTILINE)
    if match:
        return int(match.group(1)) + float(match.group(2)) / 60.0
    match = re.search(r"Total wall time:\s+([0-9.]+)\s+seconds", text)
    if match:
        return float(match.group(1)) / 60.0
    return float("nan")


def read_band_ranges(case_dir: Path) -> dict[int, tuple[float, float]]:
    path = case_dir / "bands.dat"
    ranges: dict[int, tuple[float, float]] = {}
    if not path.exists():
        return ranges
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            band = int(parts[2])
            energy = float(parts[3])
        except ValueError:
            continue
        if band not in ranges:
            ranges[band] = (energy, energy)
        else:
            lo, hi = ranges[band]
            ranges[band] = (min(lo, energy), max(hi, energy))
    return ranges


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#202124">{escaped}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def log_scale(values: list[float], y_min: float | None = None, y_max: float | None = None):
    finite = [v for v in values if v > 0 and math.isfinite(v)]
    if not finite:
        finite = [1e-12, 1.0]
    lo = math.log10(y_min if y_min else min(finite))
    hi = math.log10(y_max if y_max else max(finite))
    if hi == lo:
        hi = lo + 1.0
    pad = 0.15 * (hi - lo)
    return lo - pad, hi + pad


def palette() -> list[str]:
    return ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def line_chart(
    path: Path,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    log_y: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 980,
    height: int = 560,
) -> None:
    margin = dict(left=86, right=34, top=72, bottom=92)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_values = [v for _, vals, _ in series for v in vals if math.isfinite(v)]
    if log_y:
        lo, hi = log_scale(all_values, y_min, y_max)

        def ymap(v: float) -> float:
            return margin["top"] + (hi - math.log10(max(v, 1e-300))) / (hi - lo) * plot_h

        tick_exp_min = math.ceil(lo)
        tick_exp_max = math.floor(hi)
        ticks = [(10.0**e, f"1e{e}") for e in range(tick_exp_min, tick_exp_max + 1)]
    else:
        finite = all_values or [0.0, 1.0]
        lo = y_min if y_min is not None else min(finite)
        hi = y_max if y_max is not None else max(finite)
        if hi == lo:
            hi = lo + 1
        pad = 0.1 * (hi - lo)
        lo -= pad
        hi += pad

        def ymap(v: float) -> float:
            return margin["top"] + (hi - v) / (hi - lo) * plot_h

        ticks = [(lo + (hi - lo) * i / 5, f"{lo + (hi - lo) * i / 5:.2g}") for i in range(6)]

    def xmap(i: int) -> float:
        if len(x_labels) == 1:
            return margin["left"] + plot_w / 2
        return margin["left"] + i / (len(x_labels) - 1) * plot_w

    body: list[str] = [
        svg_text(width / 2, 32, title, 20, "middle", "700"),
        svg_text(width / 2, height - 16, "window / case", 12, "middle"),
        svg_text(16, margin["top"] + plot_h / 2, y_label, 12, "middle"),
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
    ]
    for val, label in ticks:
        y = ymap(val)
        body.append(f'<line x1="{margin["left"] - 5}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, label, 11, "end"))
    for i, label in enumerate(x_labels):
        x = xmap(i)
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"] + plot_h}" x2="{x:.1f}" y2="{margin["top"] + plot_h + 5}" stroke="#333"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 22, label, 11, "middle"))
    legend_x = margin["left"] + 10
    legend_y = 54
    for idx, (name, vals, color) in enumerate(series):
        points = []
        for i, v in enumerate(vals):
            if not math.isfinite(v) or v <= 0 and log_y:
                continue
            points.append((xmap(i), ymap(v)))
        if len(points) >= 2:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            body.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1"/>')
        lx = legend_x + idx * 150
        body.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        body.append(svg_text(lx + 30, legend_y + 4, name, 12))
    write_svg(path, width, height, body)


def grouped_bar_chart(
    path: Path,
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
    width: int = 1000,
    height: int = 560,
) -> None:
    margin = dict(left=82, right=32, top=78, bottom=90)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_val = y_max if y_max is not None else max([v for _, vals, _ in series for v in vals] + [1.0])

    def ymap(v: float) -> float:
        return margin["top"] + (max_val - v) / (max_val - y_min) * plot_h

    body: list[str] = [
        svg_text(width / 2, 32, title, 20, "middle", "700"),
        svg_text(18, margin["top"] + plot_h / 2, y_label, 12, "middle"),
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        val = y_min + (max_val - y_min) * i / 5
        y = ymap(val)
        body.append(f'<line x1="{margin["left"] - 5}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"{val:.2g}", 11, "end"))
    group_w = plot_w / len(groups)
    bar_w = min(26, group_w / (len(series) + 1.2))
    for gi, group in enumerate(groups):
        gx = margin["left"] + gi * group_w + group_w / 2
        body.append(svg_text(gx, margin["top"] + plot_h + 24, group, 11, "middle"))
        for si, (_, vals, color) in enumerate(series):
            x = gx - (len(series) * bar_w) / 2 + si * bar_w
            v = vals[gi]
            y = ymap(v)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{margin["top"] + plot_h - y:.1f}" fill="{color}"/>')
    for idx, (name, _, color) in enumerate(series):
        lx = margin["left"] + idx * 170
        body.append(f'<rect x="{lx}" y="52" width="18" height="12" fill="{color}"/>')
        body.append(svg_text(lx + 24, 63, name, 12))
    write_svg(path, width, height, body)


def figure_peierls_failure() -> None:
    cases = [
        ("20", PEIERLS_LOCAL / "k10_b20"),
        ("30", PEIERLS_LOCAL / "k10_b30"),
        ("40", PEIERLS_LOCAL / "k10_b40"),
        ("50", PEIERLS_LOCAL / "k10_b50"),
        ("60", PEIERLS_LOCAL / "k10_b60"),
        ("112", PHASE0 / "peierls_vg"),
    ]
    colors = palette()
    series = []
    for idx, order in enumerate([1, 3, 5, 7, 9]):
        vals = [nearest_hhg(case_dir, order) for _, case_dir in cases]
        series.append((f"H{order}", vals, colors[idx]))
    line_chart(
        OUT_DIR / "fig01_peierls_vg_symmetric_truncation_failure.svg",
        "Peierls VG symmetric windows: low-order artifacts",
        [label for label, _ in cases],
        series,
        "HHG total (log)",
        log_y=True,
    )


def figure_lgcov_valdepth() -> None:
    cases = [
        ("70-112", LG_VALDEPTH / "lgcov_k40_nb70_112"),
        ("60-112", LG_VALDEPTH / "lgcov_k40_nb60_112"),
        ("50-112", LG_VALDEPTH / "lgcov_k40_nb50_112"),
        ("30-112", LG_VALDEPTH / "lgcov_k40_nb30_112"),
    ]
    full = LG_SCAN / "lgcov_k40_b112"
    colors = palette()
    series = []
    for idx, order in enumerate([1, 3, 4, 5, 6, 7, 8, 9]):
        full_val = nearest_hhg(full, order)
        vals = [nearest_hhg(case_dir, order) / full_val for _, case_dir in cases]
        series.append((f"H{order}", vals, colors[idx % len(colors)]))
    line_chart(
        OUT_DIR / "fig02_lgcov_valence_depth_ratios.svg",
        "LG-covariant valence-depth scan: ratio to full112",
        [label for label, _ in cases],
        series,
        "ratio to full112",
        log_y=False,
        y_min=0.0,
        y_max=2.8,
    )


def figure_full112_kconv() -> None:
    k30 = LG_SCAN / "lgcov_k30_b112"
    k40 = LG_SCAN / "lgcov_k40_b112"
    orders = ORDERS_ODD
    ratios = [nearest_hhg(k40, order) / nearest_hhg(k30, order) for order in orders]
    grouped_bar_chart(
        OUT_DIR / "fig03_lgcov_full112_k_convergence.svg",
        "LG-covariant full112 k-grid convergence",
        [f"H{o}" for o in orders],
        [("k40/k30", ratios, "#1f77b4")],
        "ratio",
        y_min=0.9,
        y_max=1.08,
        width=900,
    )


def figure_energy_windows() -> None:
    ranges = read_band_ranges(LG_SCAN / "lgcov_k40_b112")
    ef = 0.0843
    windows = [
        ("75-94", 75, 94),
        ("70-99", 70, 99),
        ("65-104", 65, 104),
        ("53-112", 53, 112),
        ("30-112", 30, 112),
        ("1-112", 1, 112),
    ]
    width, height = 980, 420
    margin = dict(left=120, right=34, top=64, bottom=70)
    plot_w = width - margin["left"] - margin["right"]
    lo, hi = -5.2, 3.4

    def xmap(v: float) -> float:
        return margin["left"] + (v - lo) / (hi - lo) * plot_w

    body = [
        svg_text(width / 2, 32, "Band-window energy coverage relative to Ef", 20, "middle", "700"),
        f'<line x1="{xmap(lo):.1f}" y1="{height-margin["bottom"]}" x2="{xmap(hi):.1f}" y2="{height-margin["bottom"]}" stroke="#333"/>',
        f'<line x1="{xmap(0):.1f}" y1="{margin["top"]-8}" x2="{xmap(0):.1f}" y2="{height-margin["bottom"]+10}" stroke="#111" stroke-dasharray="4,4"/>',
        svg_text(xmap(0), margin["top"] - 14, "Ef", 12, "middle", "700"),
        svg_text(width / 2, height - 22, "energy - Ef (eV)", 12, "middle"),
    ]
    for tick in range(-5, 4):
        x = xmap(tick)
        body.append(f'<line x1="{x:.1f}" y1="{height-margin["bottom"]}" x2="{x:.1f}" y2="{height-margin["bottom"]+5}" stroke="#333"/>')
        body.append(svg_text(x, height - margin["bottom"] + 22, str(tick), 11, "middle"))
    for i, (label, start, end) in enumerate(windows):
        mn = min(ranges[b][0] for b in range(start, end + 1)) - ef
        mx = max(ranges[b][1] for b in range(start, end + 1)) - ef
        y = margin["top"] + i * 42
        body.append(svg_text(margin["left"] - 16, y + 5, label, 12, "end"))
        body.append(f'<line x1="{xmap(mn):.1f}" y1="{y:.1f}" x2="{xmap(mx):.1f}" y2="{y:.1f}" stroke="#1f77b4" stroke-width="12" stroke-linecap="round"/>')
        body.append(svg_text(xmap(mx) + 8, y + 4, f"{mn:.2f} to {mx:.2f}", 11))
    write_svg(OUT_DIR / "fig04_band_window_energy_coverage.svg", width, height, body)


def figure_runtime() -> None:
    cases = [
        ("70-112", LG_VALDEPTH / "lgcov_k40_nb70_112"),
        ("60-112", LG_VALDEPTH / "lgcov_k40_nb60_112"),
        ("50-112", LG_VALDEPTH / "lgcov_k40_nb50_112"),
        ("30-112", LG_VALDEPTH / "lgcov_k40_nb30_112"),
        ("1-112", LG_SCAN / "lgcov_k40_b112"),
    ]
    vals = [read_real_minutes(path) for _, path in cases]
    grouped_bar_chart(
        OUT_DIR / "fig05_runtime_lgcov_valence_depth.svg",
        "Server wall-clock time, lg_cov 40x40",
        [label for label, _ in cases],
        [("minutes", vals, "#2ca02c")],
        "minutes",
        y_min=0,
        y_max=max(vals) * 1.15,
        width=900,
    )


def resample_spectrum(case_dir: Path, xmax: float = 30.0) -> list[tuple[float, float]]:
    rows = read_hhg(case_dir / "HHG.dat")
    return [(x, y) for x, y in rows if 0.0 <= x <= xmax and y > 0 and math.isfinite(y)]


def spectrum_chart(
    path: Path,
    title: str,
    spectra: list[tuple[str, list[tuple[float, float]], str, float]],
    xmax: float = 30.0,
    width: int = 1100,
    height: int = 680,
) -> None:
    margin = dict(left=90, right=36, top=74, bottom=82)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_y = [y for _, rows, _, _ in spectra for _, y in rows if y > 0]
    lo, hi = log_scale(all_y, y_min=max(min(all_y) if all_y else 1e-18, 1e-18))

    def xmap(x: float) -> float:
        return margin["left"] + x / xmax * plot_w

    def ymap(y: float) -> float:
        return margin["top"] + (hi - math.log10(max(y, 1e-300))) / (hi - lo) * plot_h

    body: list[str] = [
        svg_text(width / 2, 32, title, 20, "middle", "700"),
        svg_text(width / 2, height - 22, "harmonic order", 12, "middle"),
        svg_text(18, margin["top"] + plot_h / 2, "HHG total (log)", 12, "middle"),
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
    ]
    for x_tick in range(0, int(xmax) + 1, 5):
        x = xmap(float(x_tick))
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="#f1f3f4"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 22, str(x_tick), 11, "middle"))
    for e in range(math.ceil(lo), math.floor(hi) + 1):
        y = ymap(10.0**e)
        body.append(f'<line x1="{margin["left"] - 5}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"1e{e}", 11, "end"))
    for idx, (label, rows, color, stroke_width) in enumerate(spectra):
        pts = [(xmap(x), ymap(y)) for x, y in rows]
        if len(pts) >= 2:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            body.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="{stroke_width:.1f}" opacity="0.92"/>'
            )
        lx = margin["left"] + (idx % 4) * 230
        ly = 52 + (idx // 4) * 18
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" stroke-width="{stroke_width:.1f}"/>')
        body.append(svg_text(lx + 32, ly + 4, label, 12))
    write_svg(path, width, height, body)


def spectrum_ratio_chart(
    path: Path,
    title: str,
    ref_rows: list[tuple[float, float]],
    spectra: list[tuple[str, list[tuple[float, float]], str]],
    xmax: float = 15.0,
    width: int = 1100,
    height: int = 520,
) -> None:
    margin = dict(left=90, right=36, top=72, bottom=80)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    ref_by_index = ref_rows

    def ref_at(x: float) -> float:
        if not ref_by_index:
            return float("nan")
        return min(ref_by_index, key=lambda item: abs(item[0] - x))[1]

    ratio_series: list[tuple[str, list[tuple[float, float]], str]] = []
    all_ratios: list[float] = []
    for label, rows, color in spectra:
        ratios = []
        for x, y in rows:
            if x > xmax:
                continue
            r = ref_at(x)
            if r > 0 and math.isfinite(r):
                val = y / r
                if math.isfinite(val) and val > 0:
                    ratios.append((x, val))
                    if 0.5 <= x <= xmax:
                        all_ratios.append(val)
        ratio_series.append((label, ratios, color))

    y_min, y_max = 0.0, max(3.0, min(6.0, max(all_ratios + [1.0]) * 1.1))

    def xmap(x: float) -> float:
        return margin["left"] + x / xmax * plot_w

    def ymap(y: float) -> float:
        return margin["top"] + (y_max - y) / (y_max - y_min) * plot_h

    body: list[str] = [
        svg_text(width / 2, 32, title, 20, "middle", "700"),
        svg_text(width / 2, height - 22, "harmonic order", 12, "middle"),
        svg_text(18, margin["top"] + plot_h / 2, "ratio to full112", 12, "middle"),
        f'<line x1="{margin["left"]}" y1="{ymap(1):.1f}" x2="{margin["left"] + plot_w}" y2="{ymap(1):.1f}" stroke="#111" stroke-dasharray="5,5"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#333"/>',
    ]
    for x_tick in range(0, int(xmax) + 1, 3):
        x = xmap(float(x_tick))
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="#f1f3f4"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 22, str(x_tick), 11, "middle"))
    for val in [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]:
        if y_min <= val <= y_max:
            y = ymap(val)
            body.append(f'<line x1="{margin["left"] - 5}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            body.append(svg_text(margin["left"] - 10, y + 4, f"{val:g}", 11, "end"))
    for idx, (label, rows, color) in enumerate(ratio_series):
        pts = [(xmap(x), ymap(min(y, y_max))) for x, y in rows]
        if len(pts) >= 2:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.9"/>')
        lx = margin["left"] + idx * 210
        ly = 52
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" stroke-width="2.6"/>')
        body.append(svg_text(lx + 32, ly + 4, label, 12))
    write_svg(path, width, height, body)


def figure_spectra_valdepth() -> None:
    colors = {
        "70-112": "#a6cee3",
        "60-112": "#1f78b4",
        "50-112": "#33a02c",
        "30-112": "#e31a1c",
        "full112": "#111111",
    }
    specs = [
        ("70-112", resample_spectrum(LG_VALDEPTH / "lgcov_k40_nb70_112"), colors["70-112"], 1.5),
        ("60-112", resample_spectrum(LG_VALDEPTH / "lgcov_k40_nb60_112"), colors["60-112"], 1.7),
        ("50-112", resample_spectrum(LG_VALDEPTH / "lgcov_k40_nb50_112"), colors["50-112"], 1.7),
        ("30-112", resample_spectrum(LG_VALDEPTH / "lgcov_k40_nb30_112"), colors["30-112"], 2.4),
        ("full112", resample_spectrum(LG_SCAN / "lgcov_k40_b112"), colors["full112"], 2.8),
    ]
    spectrum_chart(
        OUT_DIR / "fig06_hhg_spectrum_lgcov_valence_depth.svg",
        "HHG spectra: lg_cov valence-depth windows vs full112",
        specs,
        xmax=30.0,
    )
    ref = resample_spectrum(LG_SCAN / "lgcov_k40_b112", xmax=15.0)
    spectrum_ratio_chart(
        OUT_DIR / "fig07_hhg_spectrum_ratio_lgcov_valence_depth.svg",
        "HHG spectral ratio to full112: valence-depth windows",
        ref,
        [(label, rows, color) for label, rows, color, _ in specs if label != "full112"],
        xmax=15.0,
    )


def figure_spectra_symmetric_lgcov() -> None:
    colors = {
        "30 sym": "#1f77b4",
        "40 sym": "#ff7f0e",
        "50 sym": "#2ca02c",
        "60 sym": "#9467bd",
        "full112": "#111111",
    }
    specs = [
        ("30 sym", resample_spectrum(LG_SCAN / "lgcov_k40_b30"), colors["30 sym"], 1.7),
        ("40 sym", resample_spectrum(LG_SCAN / "lgcov_k40_b40"), colors["40 sym"], 1.7),
        ("50 sym", resample_spectrum(LG_SCAN / "lgcov_k40_b50"), colors["50 sym"], 1.7),
        ("60 sym", resample_spectrum(LG_SCAN / "lgcov_k40_b60"), colors["60 sym"], 1.7),
        ("full112", resample_spectrum(LG_SCAN / "lgcov_k40_b112"), colors["full112"], 2.8),
    ]
    spectrum_chart(
        OUT_DIR / "fig08_hhg_spectrum_lgcov_symmetric_windows.svg",
        "HHG spectra: lg_cov symmetric windows vs full112",
        specs,
        xmax=30.0,
    )


def figure_spectra_full112_kgrid() -> None:
    colors = {
        "k10 full112": "#a6cee3",
        "k30 full112": "#1f78b4",
        "k40 full112": "#111111",
    }
    specs = [
        ("k10 full112", resample_spectrum(LG_SCAN / "lgcov_k10_b112"), colors["k10 full112"], 1.5),
        ("k30 full112", resample_spectrum(LG_SCAN / "lgcov_k30_b112"), colors["k30 full112"], 2.0),
        ("k40 full112", resample_spectrum(LG_SCAN / "lgcov_k40_b112"), colors["k40 full112"], 2.8),
    ]
    spectrum_chart(
        OUT_DIR / "fig09_hhg_spectrum_lgcov_full112_kgrid.svg",
        "HHG spectra: lg_cov full112 k-grid scan",
        specs,
        xmax=30.0,
    )
    ref = resample_spectrum(LG_SCAN / "lgcov_k40_b112", xmax=15.0)
    spectrum_ratio_chart(
        OUT_DIR / "fig10_hhg_spectrum_ratio_lgcov_full112_kgrid.svg",
        "HHG spectral ratio to k40 full112: k-grid scan",
        ref,
        [(label, rows, color) for label, rows, color, _ in specs if label != "k40 full112"],
        xmax=15.0,
    )


def figure_spectra_fullvalence() -> None:
    colors = {
        "nb1-94": "#0072b2",
        "nb1-104": "#009e73",
        "nb30-112": "#d55e00",
        "full112": "#111111",
    }
    candidate_specs = [
        ("nb1-94", LG_FULLVAL / "lgcov_k40_nb94", colors["nb1-94"], 2.2),
        ("nb1-104", LG_FULLVAL / "lgcov_k40_nb104", colors["nb1-104"], 2.2),
        ("nb30-112", LG_VALDEPTH / "lgcov_k40_nb30_112", colors["nb30-112"], 1.7),
        ("full112", LG_SCAN / "lgcov_k40_b112", colors["full112"], 2.8),
    ]
    specs = [
        (label, resample_spectrum(path), color, width)
        for label, path, color, width in candidate_specs
        if has_hhg(path)
    ]
    spectrum_chart(
        OUT_DIR / "fig11_hhg_spectrum_lgcov_fullvalence.svg",
        "HHG spectra: lg_cov full-valence conduction-cut windows",
        specs,
        xmax=30.0,
    )


def figure_even_harmonic_ratios() -> None:
    ref = LG_SCAN / "lgcov_k40_b112"
    orders = [2, 4, 6, 8, 10]
    candidate_cases = [
        ("nb1-94", LG_FULLVAL / "lgcov_k40_nb94", "#0072b2"),
        ("nb1-104", LG_FULLVAL / "lgcov_k40_nb104", "#009e73"),
        ("nb30-112", LG_VALDEPTH / "lgcov_k40_nb30_112", "#d55e00"),
    ]
    series = []
    for label, path, color in candidate_cases:
        if not has_hhg(path):
            continue
        vals = []
        for order in orders:
            ref_val = nearest_hhg(ref, order)
            vals.append(nearest_hhg(path, order) / ref_val if ref_val > 0 else float("nan"))
        series.append((label, vals, color))
    grouped_bar_chart(
        OUT_DIR / "fig12_even_harmonic_ratios_lgcov_windows.svg",
        "Even harmonics: ratio to lg_cov full112 k40",
        [f"H{o}" for o in orders],
        series,
        "ratio to full112",
        y_min=0.0,
        y_max=1.8,
        width=900,
    )


def figure_spectra_fullvalence_t2() -> None:
    colors = {
        "nb1-104": "#009e73",
        "full112": "#111111",
    }
    cases = [
        ("nb1-104", LG_FULLVAL_T2 / "lgcov_k40_nb104", colors["nb1-104"], 2.3),
        ("full112", LG_FULLVAL_T2 / "lgcov_k40_nb112", colors["full112"], 2.8),
    ]
    specs = [
        (label, resample_spectrum(path), color, width)
        for label, path, color, width in cases
        if has_hhg(path)
    ]
    if len(specs) < 2:
        return
    spectrum_chart(
        OUT_DIR / "fig13_hhg_spectrum_lgcov_fullvalence_T2_0p5cycle.svg",
        "HHG spectra: lg_cov nb1-104 vs full112, T2=0.5 cycle",
        specs,
        xmax=30.0,
    )

    full = LG_FULLVAL_T2 / "lgcov_k40_nb112"
    nb104 = LG_FULLVAL_T2 / "lgcov_k40_nb104"
    orders = list(range(1, 12))
    vals = []
    for order in orders:
        ref_val = nearest_hhg(full, order)
        vals.append(nearest_hhg(nb104, order) / ref_val if ref_val > 0 else float("nan"))
    grouped_bar_chart(
        OUT_DIR / "fig14_hhg_ratio_lgcov_nb104_full112_T2_0p5cycle.svg",
        "nb1-104 / full112: lg_cov 40x40, T2=0.5 cycle",
        [f"H{o}" for o in orders],
        [("nb1-104", vals, colors["nb1-104"])],
        "ratio to full112",
        y_min=0.9,
        y_max=1.12,
        width=980,
    )


def figure_full112_kconv_t2() -> None:
    k40 = LG_FULLVAL_T2 / "lgcov_k40_nb112"
    k60 = LG_K60_T2 / "lgcov_k60_b112_T2_0p5cycle"
    if not has_hhg(k40) or not has_hhg(k60):
        return
    specs = [
        ("k40 full112", resample_spectrum(k40), "#111111", 2.8),
        ("k60 full112", resample_spectrum(k60), "#0072b2", 2.2),
    ]
    spectrum_chart(
        OUT_DIR / "fig15_hhg_spectrum_lgcov_full112_k40_k60_T2_0p5cycle.svg",
        "HHG spectra: lg_cov full112 k40 vs k60, T2=0.5 cycle",
        specs,
        xmax=30.0,
    )
    spectrum_ratio_chart(
        OUT_DIR / "fig16_hhg_ratio_lgcov_full112_k60_k40_T2_0p5cycle.svg",
        "k60 / k40 ratio: lg_cov full112, T2=0.5 cycle",
        resample_spectrum(k40, xmax=15.0),
        [("k60/k40", resample_spectrum(k60, xmax=15.0), "#0072b2")],
        xmax=15.0,
    )


def figure_t2_scan() -> None:
    cases = [
        ("0.5 fs", LG_T2_KEY / "lgcov_k40_b112_T2_0p5fs", "#d55e00"),
        ("1 fs", LG_T2_KEY / "lgcov_k40_b112_T2_1fs", "#cc79a7"),
        ("2 fs", LG_T2_KEY / "lgcov_k40_b112_T2_2fs", "#0072b2"),
        ("5 fs", LG_T2_KEY / "lgcov_k40_b112_T2_5fs", "#009e73"),
        ("0.5 cycle", LG_FULLVAL_T2 / "lgcov_k40_nb112", "#111111"),
        ("nodeph", LG_SCAN / "lgcov_k40_b112", "#999999"),
    ]
    specs = [
        (label, resample_spectrum(path), color, 2.2 if label != "nodeph" else 2.8)
        for label, path, color in cases
        if has_hhg(path)
    ]
    if len(specs) >= 2:
        spectrum_chart(
            OUT_DIR / "fig17_hhg_spectrum_lgcov_full112_T2_scan.svg",
            "HHG spectra: lg_cov full112 T2 scan",
            specs,
            xmax=30.0,
            width=1200,
        )

    ref = LG_SCAN / "lgcov_k40_b112"
    labels = ["0.5fs", "1fs", "2fs", "5fs", "0.5cyc"]
    paths = [path for _, path, _ in cases[:5]]
    colors = palette() + ["#111111"]
    series = []
    for idx, order in enumerate([1, 2, 3, 4, 5, 6, 8, 10, 11]):
        ref_val = nearest_hhg(ref, order)
        vals = [nearest_hhg(path, order) / ref_val if ref_val > 0 and has_hhg(path) else float("nan") for path in paths]
        series.append((f"H{order}", vals, colors[idx % len(colors)]))
    line_chart(
        OUT_DIR / "fig18_hhg_ratio_lgcov_full112_T2_to_nodeph.svg",
        "T2 scan: ratio to no-dephasing full112",
        labels,
        series,
        "ratio to nodeph",
        log_y=True,
        y_min=1.0e-5,
        y_max=5.0,
        width=1320,
    )

    even_series = []
    for idx, order in enumerate([2, 4, 6, 8, 10]):
        vals = [nearest_hhg(path, order) if has_hhg(path) else float("nan") for _, path, _ in cases]
        even_series.append((f"H{order}", vals, colors[idx % len(colors)]))
    line_chart(
        OUT_DIR / "fig19_even_harmonics_lgcov_full112_T2_scan.svg",
        "Even harmonics vs T2: lg_cov full112",
        ["0.5fs", "1fs", "2fs", "5fs", "0.5cyc", "nodeph"],
        even_series,
        "HHG total (log)",
        log_y=True,
        y_min=1.0e-13,
        y_max=1.0e-5,
        width=1120,
    )


def figure_band_economy_t2() -> None:
    full = LG_FULLVAL_T2 / "lgcov_k40_nb112"
    if not has_hhg(full):
        return
    colors = {
        "nb1-94": "#0072b2",
        "nb1-104": "#009e73",
        "nb10-104": "#d55e00",
        "nb20-104": "#cc79a7",
        "full112": "#111111",
    }
    cases = [
        ("nb1-94", LG_FULLVAL_T2 / "lgcov_k40_nb94", colors["nb1-94"], 2.1),
        ("nb1-104", LG_FULLVAL_T2 / "lgcov_k40_nb104", colors["nb1-104"], 2.4),
        ("nb10-104", LG_VTRIM104_T2 / "lgcov_k40_nb10_104", colors["nb10-104"], 1.9),
        ("nb20-104", LG_VTRIM104_T2 / "lgcov_k40_nb20_104", colors["nb20-104"], 1.9),
        ("full112", full, colors["full112"], 2.8),
    ]
    specs = [
        (label, resample_spectrum(path), color, width)
        for label, path, color, width in cases
        if has_hhg(path)
    ]
    if len(specs) >= 2:
        spectrum_chart(
            OUT_DIR / "fig20_hhg_spectrum_lgcov_band_economy_T2_0p5cycle.svg",
            "HHG spectra: band-window economy tests, T2=0.5 cycle",
            specs,
            xmax=30.0,
            width=1200,
        )

    orders = list(range(1, 12))
    ratio_series = []
    for label, path, color, _ in cases:
        if label == "full112" or not has_hhg(path):
            continue
        vals = []
        for order in orders:
            ref_val = nearest_hhg(full, order)
            vals.append(nearest_hhg(path, order) / ref_val if ref_val > 0 else float("nan"))
        ratio_series.append((label, vals, color))
    grouped_bar_chart(
        OUT_DIR / "fig21_hhg_ratio_lgcov_band_economy_T2_0p5cycle.svg",
        "Band-window ratios to full112: T2=0.5 cycle",
        [f"H{o}" for o in orders],
        ratio_series,
        "ratio to full112",
        y_min=0.0,
        y_max=3.0,
        width=1180,
    )

    j0_cases = [
        ("nb1-94", LG_FULLVAL_T2 / "lgcov_k40_nb94"),
        ("nb1-104", LG_FULLVAL_T2 / "lgcov_k40_nb104"),
        ("nb10-104", LG_VTRIM104_T2 / "lgcov_k40_nb10_104"),
        ("nb20-104", LG_VTRIM104_T2 / "lgcov_k40_nb20_104"),
        ("full112", full),
    ]
    vals = [read_j0(path) for _, path in j0_cases]
    line_chart(
        OUT_DIR / "fig22_initial_current_lgcov_band_economy_T2_0p5cycle.svg",
        "Initial current check: band-window economy tests",
        [label for label, _ in j0_cases],
        [("|J(0)|", vals, "#d55e00")],
        "|J(0)| (a.u., log)",
        log_y=True,
        y_min=1.0e-19,
        y_max=1.0e-6,
        width=980,
    )


def write_readme() -> None:
    text = """# Classical Validation Figures

Generated by `tools/analysis/plot_classical_validation_summary.py`.

Figures:

- `fig01_peierls_vg_symmetric_truncation_failure.svg`: Peierls VG symmetric-window low-order artifacts.
- `fig02_lgcov_valence_depth_ratios.svg`: lg_cov fixed-conduction valence-depth ratios to full112.
- `fig03_lgcov_full112_k_convergence.svg`: full112 k-grid convergence, 30x30 to 40x40.
- `fig04_band_window_energy_coverage.svg`: band-window energy coverage relative to Ef.
- `fig05_runtime_lgcov_valence_depth.svg`: server wall-clock time for candidate windows.
- `fig06_hhg_spectrum_lgcov_valence_depth.svg`: full HHG spectra for valence-depth windows.
- `fig07_hhg_spectrum_ratio_lgcov_valence_depth.svg`: spectral ratio to full112 for valence-depth windows.
- `fig08_hhg_spectrum_lgcov_symmetric_windows.svg`: full HHG spectra for symmetric lg_cov windows.
- `fig09_hhg_spectrum_lgcov_full112_kgrid.svg`: full HHG spectra for full112 k-grid scan.
- `fig10_hhg_spectrum_ratio_lgcov_full112_kgrid.svg`: spectral ratio to k40 full112 for full112 k-grid scan.
- `fig11_hhg_spectrum_lgcov_fullvalence.svg`: full HHG spectra for full-valence conduction-cut windows.
- `fig12_even_harmonic_ratios_lgcov_windows.svg`: even-harmonic ratios for candidate lg_cov windows.
- `fig13_hhg_spectrum_lgcov_fullvalence_T2_0p5cycle.svg`: full-valence spectra for nb1-104 vs full112 at T2=0.5 cycle.
- `fig14_hhg_ratio_lgcov_nb104_full112_T2_0p5cycle.svg`: H1-H11 ratio for nb1-104/full112 at T2=0.5 cycle.
- `fig15_hhg_spectrum_lgcov_full112_k40_k60_T2_0p5cycle.svg`: full112 spectra for k40 vs k60 at T2=0.5 cycle.
- `fig16_hhg_ratio_lgcov_full112_k60_k40_T2_0p5cycle.svg`: k60/k40 spectral ratio for full112 at T2=0.5 cycle.
- `fig17_hhg_spectrum_lgcov_full112_T2_scan.svg`: full112 HHG spectra for the T2 scan.
- `fig18_hhg_ratio_lgcov_full112_T2_to_nodeph.svg`: T2-scan harmonic ratios to no-dephasing full112.
- `fig19_even_harmonics_lgcov_full112_T2_scan.svg`: even harmonics across T2 values.
- `fig20_hhg_spectrum_lgcov_band_economy_T2_0p5cycle.svg`: spectra for band-window economy tests at T2=0.5 cycle.
- `fig21_hhg_ratio_lgcov_band_economy_T2_0p5cycle.svg`: H1-H11 band-window ratios to full112 at T2=0.5 cycle.
- `fig22_initial_current_lgcov_band_economy_T2_0p5cycle.svg`: initial-current check for band-window economy tests.

Rerun this script after downloading new server outputs to overwrite the figures.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_peierls_failure()
    figure_lgcov_valdepth()
    figure_full112_kconv()
    figure_energy_windows()
    figure_runtime()
    figure_spectra_valdepth()
    figure_spectra_symmetric_lgcov()
    figure_spectra_full112_kgrid()
    figure_spectra_fullvalence()
    figure_even_harmonic_ratios()
    figure_spectra_fullvalence_t2()
    figure_full112_kconv_t2()
    figure_t2_scan()
    figure_band_economy_t2()
    write_readme()
    print(f"Wrote figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
