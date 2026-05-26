"""AFM Neel-vector symmetry check from two SBE current traces.

This script compares two classical-light runs whose Wannier TB files represent
opposite AFM order parameters, N and -N, with otherwise identical input.

It works with complex current amplitudes, not only HHG intensities:

  J_evenN(t) = 0.5 * (J_N(t) + J_-N(t))
  J_oddN(t)  = 0.5 * (J_N(t) - J_-N(t))

For the ideal selection rule J_n(-N)=(-1)^(n+1) J_n(N):

  odd harmonic n  -> evenN component allowed, oddN component forbidden
  even harmonic n -> oddN component allowed, evenN component forbidden

Outputs:
  - afm_neel_symmetry_harmonics.csv
  - Jt_evenN.dat
  - Jt_oddN.dat
"""

from __future__ import annotations

import argparse
import cmath
import csv
import math
from pathlib import Path


AU_TO_FS = 0.024188843265857
FS_TO_AU = 1.0 / AU_TO_FS
TWOPI = 2.0 * math.pi


def read_jt(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(rows) < 2:
        raise ValueError(f"Not enough rows in {path}")
    return rows


def infer_omega0_from_hhg(path: Path) -> float:
    hhg = path / "HHG.dat"
    if not hhg.exists():
        raise FileNotFoundError(f"Cannot infer omega0; missing {hhg}")
    vals = []
    for line in hhg.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        h = float(parts[0])
        omega = float(parts[1])
        if abs(h) > 1.0e-8:
            vals.append(omega / h)
        if len(vals) >= 16:
            break
    if not vals:
        raise ValueError(f"Could not infer omega0 from {hhg}")
    return sum(vals) / len(vals)


def parse_orders(text: str) -> list[int]:
    orders: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            orders.extend(range(int(a), int(b) + 1))
        else:
            orders.append(int(part))
    return sorted(set(orders))


def check_same_grid(plus: list[tuple[float, float, float]], minus: list[tuple[float, float, float]]) -> None:
    if len(plus) != len(minus):
        raise ValueError(f"Current traces have different lengths: {len(plus)} vs {len(minus)}")
    max_dt = max(abs(a[0] - b[0]) for a, b in zip(plus, minus))
    if max_dt > 1.0e-9:
        raise ValueError(f"Current traces have different time grids; max dt(fs) mismatch = {max_dt:g}")


def windowed_amp(rows: list[tuple[float, float, float]], order: int, omega0: float) -> tuple[complex, complex]:
    nt = len(rows)
    dt_au = (rows[1][0] - rows[0][0]) * FS_TO_AU
    ax = 0.0j
    ay = 0.0j
    for it, (t_fs, jx, jy) in enumerate(rows):
        w = 0.5 * (1.0 - math.cos(TWOPI * it / (nt - 1)))
        phase = cmath.exp(1j * order * omega0 * t_fs * FS_TO_AU)
        ax += dt_au * w * jx * phase
        ay += dt_au * w * jy * phase
    return ax, ay


def yield_xy(ax: complex, ay: complex) -> float:
    return abs(ax) ** 2 + abs(ay) ** 2


def write_current(path: Path, rows: list[tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("# it   time(fs)   Jx(a.u.)   Jy(a.u.)\n")
        for idx, (t, jx, jy) in enumerate(rows, start=1):
            f.write(f"{idx:8d} {t:18.10E} {jx:18.10E} {jy:18.10E}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus-dir", required=True, type=Path, help="Run directory for AFM N, containing Jt.dat")
    parser.add_argument("--minus-dir", required=True, type=Path, help="Run directory for AFM -N, containing Jt.dat")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--orders", default="1-15", help="Comma/range list, e.g. 1-15 or 1,2,3,4")
    parser.add_argument("--omega0", type=float, default=None, help="Fundamental frequency in a.u.; inferred from plus HHG.dat if omitted")
    args = parser.parse_args()

    plus = read_jt(args.plus_dir / "Jt.dat")
    minus = read_jt(args.minus_dir / "Jt.dat")
    check_same_grid(plus, minus)

    omega0 = args.omega0 if args.omega0 is not None else infer_omega0_from_hhg(args.plus_dir)
    orders = parse_orders(args.orders)
    args.outdir.mkdir(parents=True, exist_ok=True)

    even_rows = []
    odd_rows = []
    for p, m in zip(plus, minus):
        t = p[0]
        even_rows.append((t, 0.5 * (p[1] + m[1]), 0.5 * (p[2] + m[2])))
        odd_rows.append((t, 0.5 * (p[1] - m[1]), 0.5 * (p[2] - m[2])))
    write_current(args.outdir / "Jt_evenN.dat", even_rows)
    write_current(args.outdir / "Jt_oddN.dat", odd_rows)

    with (args.outdir / "afm_neel_symmetry_harmonics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "order",
                "expected_allowed",
                "yield_plus",
                "yield_minus",
                "yield_evenN",
                "yield_oddN",
                "leakage_ratio",
                "evenN_Jx_re",
                "evenN_Jx_im",
                "evenN_Jy_re",
                "evenN_Jy_im",
                "oddN_Jx_re",
                "oddN_Jx_im",
                "oddN_Jy_re",
                "oddN_Jy_im",
            ]
        )
        for order in orders:
            px, py = windowed_amp(plus, order, omega0)
            mx, my = windowed_amp(minus, order, omega0)
            ex, ey = windowed_amp(even_rows, order, omega0)
            ox, oy = windowed_amp(odd_rows, order, omega0)
            y_plus = yield_xy(px, py)
            y_minus = yield_xy(mx, my)
            y_even = yield_xy(ex, ey)
            y_odd = yield_xy(ox, oy)
            if order % 2 == 0:
                expected = "oddN"
                leakage = y_even / y_odd if y_odd > 0 else float("nan")
            else:
                expected = "evenN"
                leakage = y_odd / y_even if y_even > 0 else float("nan")
            writer.writerow(
                [
                    order,
                    expected,
                    f"{y_plus:.12e}",
                    f"{y_minus:.12e}",
                    f"{y_even:.12e}",
                    f"{y_odd:.12e}",
                    f"{leakage:.12e}",
                    f"{ex.real:.12e}",
                    f"{ex.imag:.12e}",
                    f"{ey.real:.12e}",
                    f"{ey.imag:.12e}",
                    f"{ox.real:.12e}",
                    f"{ox.imag:.12e}",
                    f"{oy.real:.12e}",
                    f"{oy.imag:.12e}",
                ]
            )

    print(f"Wrote {args.outdir / 'afm_neel_symmetry_harmonics.csv'}")


if __name__ == "__main__":
    main()
