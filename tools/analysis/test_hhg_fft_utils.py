#!/usr/bin/env python3
"""Unit tests for hhg_fft_utils (Fortran nint / harmonic_fft_index)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hhg_fft_utils import fortran_nint, harmonic_fft_index, harmonic_order_at_index, omega0_from_wvl_nm


class TestHhgFftUtils(unittest.TestCase):
    def test_fortran_nint_positive(self) -> None:
        self.assertEqual(fortran_nint(0.4), 0)
        self.assertEqual(fortran_nint(0.5), 1)
        self.assertEqual(fortran_nint(1.49), 1)
        self.assertEqual(fortran_nint(2.5), 3)

    def test_harmonic_fft_index_roundtrip(self) -> None:
        wvl = 3200.0
        dt = 0.35
        ncyc = 4.0
        omega0 = omega0_from_wvl_nm(wvl)
        t_total = (2.0 * math.pi / omega0) * ncyc
        nt = int(math.ceil(t_total / dt)) + 1
        for order in (2, 5, 7, 9, 10):
            iw = harmonic_fft_index(nt, dt, omega0, order)
            h = harmonic_order_at_index(nt, dt, omega0, iw)
            self.assertLess(abs(h - float(order)), 0.05, msg=f"H{order} bin mismatch")


if __name__ == "__main__":
    unittest.main()
