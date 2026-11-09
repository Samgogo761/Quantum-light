#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finite_bandwidth_modes import HARMONICS, build_modes, inner_product, project_current
from hhg_fft_utils import omega0_from_wvl_nm


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def test_orthonormal() -> None:
    omega0 = omega0_from_wvl_nm(3200.0)
    omega = np.linspace(0.0, 12.0 * omega0, 4001)
    dw = float(omega[1] - omega[0])
    modes = build_modes(omega, omega0)
    for n in HARMONICS:
        require(n in modes, f"missing H{n}")
        require(abs(inner_product(modes[n], modes[n], dw) - 1.0) < 1e-8, f"norm H{n}")
    for i, n in enumerate(HARMONICS):
        for m in HARMONICS[i + 1 :]:
            ov = inner_product(modes[n], modes[m], dw)
            require(abs(ov) < 1e-8, f"overlap H{n}·H{m}={ov}")


def test_projects_synthetic_peak() -> None:
    omega0 = omega0_from_wvl_nm(3200.0)
    omega = np.linspace(0.0, 12.0 * omega0, 4001)
    dw = float(omega[1] - omega[0])
    modes = build_modes(omega, omega0)
    jx = modes[5].real.copy()
    jy = np.zeros_like(jx)
    amp = project_current(modes, jx, jy, dw)
    require(abs(amp[5]) > 0.3, f"H5 projection too small: {amp[5]}")
    for n in (2, 7, 9):
        require(abs(amp[n]) < 0.05, f"leakage into H{n}: {amp[n]}")


def test_single_bin_is_not_a_mode() -> None:
    """A Kronecker bin is not orthonormal to a finite-width F_n."""
    omega0 = omega0_from_wvl_nm(3200.0)
    omega = np.linspace(0.0, 12.0 * omega0, 4001)
    dw = float(omega[1] - omega[0])
    modes = build_modes(omega, omega0)
    bin5 = np.zeros_like(omega)
    idx = int(np.argmin(np.abs(omega - 5.0 * omega0)))
    bin5[idx] = 1.0 / math.sqrt(dw)
    ov = inner_product(modes[5], bin5, dw)
    require(abs(abs(ov) - 1.0) > 0.2, "single FFT bin must not coincide with F_5")


def main() -> int:
    test_orthonormal()
    print("PASS test_orthonormal")
    test_projects_synthetic_peak()
    print("PASS test_projects_synthetic_peak")
    test_single_bin_is_not_a_mode()
    print("PASS test_single_bin_is_not_a_mode")
    print("A1_MODE_DEFINITION=FINITE_BANDWIDTH_F2_F5_F7_F9")
    print("FFT_CONVENTION=e^{-i omega t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
