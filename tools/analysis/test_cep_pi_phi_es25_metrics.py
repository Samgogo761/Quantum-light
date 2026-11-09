#!/usr/bin/env python3
"""Unit test: e_abs uses global A_n; e_rel uses local den — they must differ."""
from __future__ import annotations

import math
import unittest


class TestAbsRelSeparation(unittest.TestCase):
    def test_definitions_differ_when_local_weaker_than_An(self):
        d = 1.0e-10
        A_n = 1.0e-2  # ensemble scale
        den_local = 1.0e-3  # weaker local node
        e_abs = d / A_n
        e_rel = d / den_local
        self.assertAlmostEqual(e_abs, 1.0e-8)
        self.assertAlmostEqual(e_rel, 1.0e-7)
        self.assertGreater(e_rel, e_abs)
        self.assertNotAlmostEqual(e_abs, e_rel)

    def test_time_metric_definition(self):
        # ||J+||+|J-|| / 2 peak vs sum peak
        jp = 2.0
        jm = 2.0
        A_t = 0.5 * (jp + jm)
        sum_norm = abs(jp - jm)  # perfect cancel if opposite; here same sign example
        # just sanity on formula
        self.assertEqual(A_t, 2.0)
        e = sum_norm / A_t
        self.assertTrue(math.isfinite(e))


if __name__ == "__main__":
    unittest.main()
