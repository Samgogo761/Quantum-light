#!/usr/bin/env python3
"""Finite-coupling symmetry and vacuum-input checks. No long propagation."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]


class SharedKKDynPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart(sh.K_FRAC)
        cls.rng = np.random.default_rng(20261030)
        cls.model = sh.build_shared(
            cls.geo, R0, cls.k, sh.NEEL, n_max=2, omega0=sh.OMEGA0, g=sh.G_ABS)
        cls.ek, cls.emk, cls.masses = pf.ground_pair(cls.geo, cls.k)
        cls.psi, cls.raw_n = pf.vacuum_product(cls.model, cls.ek, cls.emk)

    def test_dims_do_not_form_dense_h(self):
        self.assertEqual(36 * 36 * 9, 11664)
        self.assertEqual(36 * 36 * 13, 16848)
        self.assertEqual(self.model.dim, 36 * 36 * 3)
        self.assertLess(self.model.dim * 16, 1_000_000)

    def test_vacuum_input_is_normalized(self):
        self.assertAlmostEqual(self.raw_n, 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(self.psi)), 1.0, places=12)
        self.assertAlmostEqual(self.masses["norm_k"], 1.0, places=12)
        self.assertAlmostEqual(self.masses["norm_mk"], 1.0, places=12)
        ph = sh.photon_observables(self.psi, sh.EL36, self.model.d1)
        self.assertLess(abs(ph["n_ph"]), 1e-15)
        self.assertAlmostEqual(ph["trace"], 1.0, places=12)

    def test_truncated_coherent_is_not_untruncated_nbar(self):
        c2 = pf.truncated_coherent(2, 0.5)
        c8 = pf.truncated_coherent(8, 0.5)
        c12 = pf.truncated_coherent(12, 0.5)
        self.assertAlmostEqual(c2["n_ph"], 0.24390243902439024, places=8)
        self.assertNotAlmostEqual(c2["n_ph"], 0.25, places=4)
        self.assertLess(abs(c8["n_ph"] - 0.25), abs(c2["n_ph"] - 0.25))
        self.assertLess(abs(c12["n_ph"] - 0.25), 1e-10)
        self.assertAlmostEqual(c2["norm"], 1.0)
        self.assertAlmostEqual(c8["norm"], 1.0)

    def test_finite_coupling_joint_map(self):
        report = pf.finite_coupling_symmetry(self.geo, self.k, n_max=2, rng=self.rng)
        r0 = report["residuals"]["s_0.0"]
        r1 = report["residuals"]["s_1.0"]
        r04 = report["residuals"]["s_0.4"]
        self.assertLess(r0["H_N_g_vs_H_N_minusg"], 1e-12)
        self.assertLess(r0["PoxP_H_N_g_vs_swap_H_minusN_g"], 1e-10)
        self.assertLess(r0["PoxP_Pi_H_N_g_vs_swap_H_minusN_same_g"], 1e-10)
        for row in (r04, r1):
            self.assertLess(row["Pi_H_N_g_vs_H_N_minusg"], 1e-10)
            self.assertLess(row["PoxP_H_N_g_vs_swap_H_minusN_minusg"], 1e-10)
            self.assertLess(row["PoxP_Pi_H_N_g_vs_swap_H_minusN_same_g"], 1e-10)
            self.assertGreater(row["PoxP_H_N_g_vs_swap_H_minusN_g"], 1e-4)
            self.assertGreater(row["PoxP_Pi_H_N_g_vs_swap_H_minusN_minusg"], 1e-4)
            self.assertGreater(row["H_N_g_vs_H_N_minusg"], 1e-4)

    def test_taylor_step_preserves_norm_on_tiny_space(self):
        dt = 0.02
        out, last = pf.taylor_step(self.model, self.psi, 0.4, dt)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(abs(float(np.linalg.norm(out)) - 1.0), 1e-8)
        self.assertLess(last, 1e-8)
        hv = sh.apply_h(self.model, self.psi, 0.4)
        herm = abs(np.vdot(self.psi, hv) - np.vdot(hv, self.psi).conjugate())
        self.assertLess(herm, 1e-10)

    def test_snapshot_splits_photon_number(self):
        snap = pf.snapshot(self.model, self.psi, 0.0)
        self.assertAlmostEqual(snap["n_ph"], 0.0, places=12)
        self.assertAlmostEqual(snap["n_coherent"], 0.0, places=12)
        self.assertAlmostEqual(snap["n_incoherent"], 0.0, places=12)
        self.assertAlmostEqual(snap["n_e"], 8.0)
        self.assertLess(abs(snap["F_linear_minus_g_e_dot_J"]), 1e-10)

    def test_old_archives_untouched(self):
        for rel in (
            "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
            "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json",
            "results/q_line/qinput_a_j0_drive_diag_20261028/diag.json",
            "results/q_line/qinput_a_shared_kk_static_20261029/static.json",
        ):
            self.assertTrue((ROOT / rel).is_file(), rel)


if __name__ == "__main__":
    unittest.main()
