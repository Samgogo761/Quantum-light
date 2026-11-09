#!/usr/bin/env python3
"""Static checks for the 20261103 DS nmax=12 truncation driver. No long run."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_shared_kk_ds_nmax12_20261103 as tr
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds

ROOT = Path(__file__).resolve().parents[2]


class Nmax12StaticTests(unittest.TestCase):
    def test_jobs_are_ds_both_neel_only(self):
        self.assertEqual(tr.JOBS, (("ds_p", 1), ("ds_m", -1)))
        self.assertEqual(tr.NMAX, 12)
        self.assertEqual(tr.SAMPLE_EVERY, ds.SAMPLE_EVERY)
        self.assertEqual(tr.WALL_CAP_S, 600.0)
        self.assertEqual(ds.THETA, 0.0)

    def test_does_not_reopen_theta_or_cn(self):
        self.assertNotIn("coh_p", dict(tr.JOBS))
        self.assertEqual(tr.TARGET_ABS, 1e-8)

    def test_archive8_exists_and_is_protected(self):
        self.assertTrue(tr.ARCHIVE_8.is_file())
        self.assertIn(tr.ARCHIVE_8, tr.OLD_ARCHIVES)
        dest = ROOT / "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/trunc.json"
        self.assertNotEqual(tr.ARCHIVE_8.resolve(), dest.resolve())

    def test_x_peierls_coherent_vacuum_limit(self):
        mom = tr.x_peierls_moments(0.5 + 0j, 0.25, 0.25 + 0j)
        self.assertAlmostEqual(mom["mean_X2"], 2.0, places=12)
        self.assertAlmostEqual(mom["var_X"], 1.0, places=12)

    def test_interp_j_identity(self):
        t = np.array([0.0, 1.0, 2.0])
        j = np.array([[1 + 0j, 0.2j], [2, 0.4j], [3, 0.6j]], complex)
        out = tr.interp_j(t, j, t)
        np.testing.assert_allclose(out.real, j.real)
        np.testing.assert_allclose(out.imag, j.imag)

    def test_even_odd_reused(self):
        jp = np.array([[1.0, 0.2]], complex)
        jm = np.array([[0.4, -0.2]], complex)
        even, odd = ds.even_odd(jp, jm)
        np.testing.assert_allclose(even, [[0.7, 0.0]])
        np.testing.assert_allclose(odd, [[0.3, 0.2]])

    def test_load_archive8_series(self):
        arch = tr.load_archive8(tr.ARCHIVE_8)
        self.assertEqual(arch["n_max"], 8)
        self.assertEqual(arch["sample_every"], 32)
        n = arch["t"].size
        self.assertGreater(n, 10)
        self.assertEqual(arch["J_odd_ds"].shape, (n, 2))
        self.assertEqual(arch["dJ_odd"].shape, (n, 2))
        self.assertAlmostEqual(arch["max_abs"]["dJ_odd_x"], 1.2230424761838776e-07, places=16)


if __name__ == "__main__":
    unittest.main()
