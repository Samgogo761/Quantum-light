#!/usr/bin/env python3
"""Static checks for the vacuum switch close-out. No long propagation."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
import q8_m0_qinput_shared_kk_vacuum_switch_20261101 as sw
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]


class VacuumSwitchStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = build_geometry()
        cls.k = cls.geo.k_cart(sh.K_FRAC)
        cls.env = sw.switch_envelope()
        cls.model = sh.build_shared(
            cls.geo, R0, cls.k, sh.NEEL, n_max=2, omega0=sh.OMEGA0, g=sh.G_ABS)
        cls.ek, cls.emk, _ = pf.ground_pair(cls.geo, cls.k)
        cls.psi, _ = pf.vacuum_product(cls.model, cls.ek, cls.emk)

    def test_no_plateau_envelope(self):
        env = self.env
        self.assertEqual(env.flat_cycles, 0.0)
        self.assertAlmostEqual(env.t_flat, 0.0)
        self.assertAlmostEqual(env.t_total, 2.0 * env.t_ramp)
        self.assertAlmostEqual(env.s(0.0), 0.0)
        self.assertAlmostEqual(env.s(env.t_ramp), 1.0)
        self.assertAlmostEqual(env.s(env.t_total), 0.0)
        mid_up = env.s(0.5 * env.t_ramp)
        mid_dn = env.s(1.5 * env.t_ramp)
        self.assertGreater(mid_up, 0.4)
        self.assertLess(mid_up, 0.6)
        self.assertAlmostEqual(mid_up, mid_dn, places=12)
        self.assertLess(env.s(env.t_ramp + 0.25 * env.t_ramp), 0.9)
        self.assertGreater(env.s(env.t_ramp + 0.25 * env.t_ramp), 0.4)

    def test_s0_hamiltonian_independent_of_g(self):
        dec = sw.decouple_check(self.model, self.psi, self.geo, self.k)
        self.assertTrue(dec["strictly_decoupled"])
        self.assertLess(dec["max_abs_H_s0_g_vs_g0"], 1e-12)
        self.assertLess(dec["max_abs_peierls_s0_minus_I"], 1e-15)

    def test_no_joint_state_checkpoint(self):
        hits = sw.find_joint_state_files()
        self.assertEqual(hits, [])

    def test_dt_plan_halves_archive(self):
        env = self.env
        n_cycle = sw.ARCHIVE_STEPS * sw.DT_REFINE
        dt = env.t_ramp / n_cycle
        self.assertEqual(n_cycle, 2420)
        self.assertAlmostEqual(dt * 2.0, sw.ARCHIVE_DT, places=12)
        self.assertEqual(sw.SAMPLE_EVERY, 8)
        self.assertEqual(sw.NMAX, 8)

    def test_archive_cycle1_readable(self):
        arch = sw.archive_cycle1()
        p = arch["packed"]
        self.assertEqual(arch["steps"], 1210)
        self.assertAlmostEqual(p["W_sq"], -0.005629413886161171, places=12)
        self.assertAlmostEqual(p["purity"], 0.9991916380066825, places=12)
        self.assertLess(p["var_min"], 0.25)

    def test_old_preflight_exists_and_is_not_this_report(self):
        self.assertTrue(sw.OLD_PREFLIGHT.is_file())
        data = json.loads(sw.OLD_PREFLIGHT.read_text(encoding="utf-8"))
        self.assertEqual(data["record_date"], "2026-10-30")
        self.assertNotEqual(data["model_id"], sw.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
