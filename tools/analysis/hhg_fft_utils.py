"""HHG FFT bin helpers matching src/mod_hhg.f90 ``harmonic_fft_index``."""
from __future__ import annotations

import math
import re
from pathlib import Path

TWOPI = 2.0 * math.pi
NM_TO_BOHR = 10.0 * (1.0 / 0.529177210903)
C_AU = 137.035999084


def fortran_nint(x: float) -> int:
    """Match Fortran ``nint`` for non-negative values used in FFT bin lookup."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def omega0_from_wvl_nm(wvl_nm: float) -> float:
    return TWOPI * C_AU / (wvl_nm * NM_TO_BOHR)


def harmonic_fft_index(nt: int, dt: float, omega0: float, order: int) -> int:
    """Fortran ``harmonic_fft_index`` (1-based FFT bin index). ``dt`` in a.u."""
    n_omega = nt // 2 + 1
    domega = TWOPI / (float(nt) * dt)
    target_omega = float(order) * omega0
    iw = fortran_nint(target_omega / domega) + 1
    return max(1, min(n_omega, iw))


def harmonic_order_at_index(nt: int, dt: float, omega0: float, iw: int) -> float:
    domega = TWOPI / (float(nt) * dt)
    omega_n = float(iw - 1) * domega
    return omega_n / omega0


def parse_input_nml(path: Path) -> dict[str, float | int]:
    """Read laser/grid/dephasing from input.nml. ``dt`` is atomic units (same as Fortran)."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    def f(name: str, default: float | None = None) -> float:
        m = re.search(rf"^\s*{re.escape(name)}\s*=\s*([0-9.eE+\-]+)", text, re.M)
        if not m:
            if default is None:
                raise ValueError(f"{path}: missing {name}")
            return float(default)
        return float(m.group(1))

    wvl = f("wvl_nm", 3200.0)
    dt_au = f("dt", 0.35)
    ncyc = f("ncyc", 4.0)
    omega0 = omega0_from_wvl_nm(wvl)
    t_cycle = TWOPI / omega0
    t_total = ncyc * t_cycle
    nt = int(math.ceil(t_total / dt_au)) + 1
    return {
        "wvl_nm": wvl,
        "dt_au": dt_au,
        "ncyc": ncyc,
        "omega0": omega0,
        "nt": nt,
    }


def load_ics_cs_rows(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4:
            continue
        h = float(f[0])
        ics = float(f[2])
        cs = float(f[3])
        rows.append((h, ics, cs))
    if not rows:
        raise ValueError(f"{path}: no ICS/CS rows")
    return rows


def pick_ics_cs_at_order(
    path: Path,
    order: int,
    *,
    nt: int,
    dt: float,
    omega0: float,
) -> dict[str, float]:
    """Pick ICS/CS at the same FFT bin Fortran uses for ``order``."""
    iw = harmonic_fft_index(nt, dt, omega0, order)
    target_h = harmonic_order_at_index(nt, dt, omega0, iw)
    rows = load_ics_cs_rows(path)
    best = min(rows, key=lambda item: abs(item[0] - target_h))
    h_bin, ics, cs = best
    return {
        "ICS": ics,
        "CS": cs,
        "harmonic_order_bin": h_bin,
        "target_h": target_h,
        "fft_index": iw,
        "delta_order": abs(h_bin - target_h),
    }
